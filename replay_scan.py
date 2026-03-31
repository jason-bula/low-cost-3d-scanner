#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy", "pyserial", "scipy"]
# ///
"""Replay a raw LiDAR PCAP capture and produce a 3D PCD.

Reads a .pcap file recorded by raw_lidar_capture.py, applies motor rotation
and geometric pipeline, and writes a PCD. No hardware required — fully offline.
Uses fully vectorized numpy parsing (chunked) for fast processing.

Usage:
    uv run replay_scan.py scan_raw.pcap --total-angle 360 --block-interp -o scan.pcd
    uv run replay_scan.py scan_raw.pcap --total-angle 360 --block-interp --split-lidar-coords --min-dist 2 --channel 10 -o scans/scan.pcd
    uv run replay_scan.py scan_raw.pcap --total-angle 360 --block-interp --split-channels -o scans/scan.pcd
"""

import argparse
import os
import struct
import sys

import numpy as np

from record_lidar import (
    BLOCK_SIZE, BLOCKS_PER_PKT, CHANNELS,
    PCD_DTYPE, VALID_SIZES,
    load_calibration, write_pcd,
)
from scan_lidar import load_calibration_file

CHUNK_SIZE = 50_000  # packets per chunk


# ── PCAP reader ───────────────────────────────────────────────────────────

def read_pcap(filepath):
    """Parse a libpcap file and extract raw LiDAR UDP payloads.

    Supports microsecond and nanosecond variants, little- and big-endian.
    Strips Ethernet (14 B) + IP (variable) + UDP (8 B) headers.

    Returns (packets_with_ts, t_start):
        packets_with_ts : list of (t_wall_float, payload_bytes)
        t_start         : wall timestamp of the first packet (seconds)
    """
    with open(filepath, 'rb') as f:
        hdr = f.read(24)
        if len(hdr) < 24:
            raise ValueError(f"Truncated PCAP file: {filepath}")

        magic_le = struct.unpack('<I', hdr[:4])[0]
        if   magic_le == 0xa1b2c3d4:  endian, nano = '<', False
        elif magic_le == 0xd4c3b2a1:  endian, nano = '>', False
        elif magic_le == 0xa1b23c4d:  endian, nano = '<', True
        elif magic_le == 0x4d3cb2a1:  endian, nano = '>', True
        else:
            raise ValueError(
                f"Unrecognized PCAP magic 0x{magic_le:08x} "
                f"— is this a PCAP-NG file?"
            )

        link_type = struct.unpack(f'{endian}I', hdr[20:24])[0]

        packets = []
        t_start = None

        while True:
            rec = f.read(16)
            if len(rec) < 16:
                break
            ts_sec, ts_sub, incl_len, _ = struct.unpack(f'{endian}IIII', rec)
            frame = f.read(incl_len)
            if len(frame) < incl_len:
                break

            t_wall = ts_sec + (ts_sub / 1_000_000_000 if nano else ts_sub / 1_000_000)
            if t_start is None:
                t_start = t_wall

            if link_type == 1:        # LINKTYPE_ETHERNET
                if len(frame) < 34:
                    continue
                ihl = (frame[14] & 0x0f) * 4
                if frame[14 + 9] != 17:
                    continue
                payload = frame[14 + ihl + 8:]
            elif link_type == 101:    # LINKTYPE_RAW
                if len(frame) < 28:
                    continue
                ihl = (frame[0] & 0x0f) * 4
                if frame[9] != 17:
                    continue
                payload = frame[ihl + 8:]
            elif link_type == 228:    # LINKTYPE_IPV4 / raw UDP payload
                payload = frame
            else:
                payload = frame[42:]

            if len(payload) in VALID_SIZES:
                packets.append((t_wall, bytes(payload)))

    if not packets:
        raise ValueError(f"No valid LiDAR packets found in {filepath}")

    duration = packets[-1][0] - packets[0][0]
    print(f"Read {len(packets):,} packets from {filepath} "
          f"({duration:.1f}s, {len(packets) / max(duration, 0.001):.0f} pkt/s)")
    return packets, t_start


# ── Vectorized parser ─────────────────────────────────────────────────────

def fast_parse(packets_with_ts, t_start, cal, rpm,
               channels=None, block_interp=True,
               min_dist=0.0, split_lidar_coords=False,
               return_channels=False,
               R_calib=None, t_calib=None,
               R_mount=None, t_mount=None):
    """Vectorized numpy parser — processes packets in chunks of CHUNK_SIZE.

    Args:
        channels:           list of channel indices to keep (None = all 40)
        min_dist:           discard points closer than this (meters)
        split_lidar_coords: if True, return (pts_front, pts_back) split by x_L sign
        return_channels:    if True, also return uint8 channel-index arrays
        R_mount:            (3,3) mounting correction applied BEFORE motor rotation
        t_mount:            (3,)  translation (meters) applied BEFORE motor rotation

    Returns (depending on flags):
        default:            pts
        split_lidar_coords: (pts_front, pts_back)
        return_channels:    (pts, ch_arr)  or  (pts_front, ch_f, pts_back, ch_b)
    """
    N_pkt = len(packets_with_ts)

    timestamps = np.array([t for t, _ in packets_with_ts])
    elapsed = timestamps - t_start

    ch_sel = np.array(channels, dtype=np.intp) if channels is not None else np.arange(CHANNELS)
    n_ch   = len(ch_sel)

    cos_e  = cal['cos_elev'][ch_sel]
    sin_e  = cal['sin_elev'][ch_sel]
    az_off = cal['az_off_rad'][ch_sel]

    # Block time offsets: block b captured at t_pkt - (BLOCKS_PER_PKT-1-b)/18000
    block_dt = ((BLOCKS_PER_PKT - 1 - np.arange(BLOCKS_PER_PKT)) / 18000.0
                if block_interp else np.zeros(BLOCKS_PER_PKT))

    # Byte indices for selected channels within a block: shape (n_ch, 3)
    ch_byte_idx = (4 + ch_sel * 3)[:, np.newaxis] + np.arange(3)[np.newaxis, :]

    out_pts_f, out_pts_b, out_pts = [], [], []
    out_ch_f,  out_ch_b,  out_ch  = [], [], []

    n_chunks = (N_pkt + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"Parsing {N_pkt:,} packets in {n_chunks} chunks "
          f"({'block-interp' if block_interp else 'per-packet'}) ...")

    for start in range(0, N_pkt, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, N_pkt)
        n = end - start

        # ── Raw block bytes: (n, BLOCKS_PER_PKT, BLOCK_SIZE) ─────────────
        raw_chunk = np.array([
            np.frombuffer(raw, dtype=np.uint8)[:BLOCKS_PER_PKT * BLOCK_SIZE]
            .reshape(BLOCKS_PER_PKT, BLOCK_SIZE)
            for _, raw in packets_with_ts[start:end]
        ])

        # ── SOB validity: (n, 10) ─────────────────────────────────────────
        valid_sob = (raw_chunk[:, :, 0] == 0xFF) & (raw_chunk[:, :, 1] == 0xEE)

        # ── Azimuth (radians): (n, 10) ────────────────────────────────────
        az_base = (
            raw_chunk[:, :, 2].astype(np.uint32) |
            (raw_chunk[:, :, 3].astype(np.uint32) << 8)
        ).astype(np.float64) * (0.01 * np.pi / 180.0)

        # ── Channel bytes: (n, 10, n_ch, 3) ──────────────────────────────
        ch_bytes = raw_chunk[:, :, ch_byte_idx]
        dist_raw = (ch_bytes[:, :, :, 0].astype(np.uint16) |
                    (ch_bytes[:, :, :, 1].astype(np.uint16) << 8))
        intens = ch_bytes[:, :, :, 2]

        # ── Motor angle per block (radians): (n, 10) ─────────────────────
        elapsed_blocks = elapsed[start:end, np.newaxis] - block_dt[np.newaxis, :]
        theta = np.deg2rad(-elapsed_blocks * rpm * 6.0)

        # ── LiDAR geometry: (n, 10, n_ch) ────────────────────────────────
        dist_m = dist_raw.astype(np.float64) * 0.004
        az = az_base[:, :, np.newaxis] + az_off

        x_l = dist_m * cos_e * np.sin(az)
        y_l = dist_m * cos_e * np.cos(az)
        z_l = dist_m * sin_e

        # ── Mounting correction (before motor rotation) ───────────────────
        if R_mount is not None or t_mount is not None:
            sh = x_l.shape
            xyz_m = np.stack([x_l.ravel(), y_l.ravel(), z_l.ravel()])
            if R_mount is not None:
                xyz_m = R_mount @ xyz_m
            if t_mount is not None:
                xyz_m = xyz_m + t_mount[:, np.newaxis]
            x_l = xyz_m[0].reshape(sh)
            y_l = xyz_m[1].reshape(sh)
            z_l = xyz_m[2].reshape(sh)

        # ── Motor rotation Ry(-θ) ─────────────────────────────────────────
        cos_t = np.cos(theta)[:, :, np.newaxis]
        sin_t = np.sin(theta)[:, :, np.newaxis]

        x_f = x_l * cos_t + z_l * sin_t
        y_f = y_l
        z_f = -x_l * sin_t + z_l * cos_t

        # ── Calibration correction ────────────────────────────────────────
        if R_calib is not None:
            sh = x_f.shape
            pts_in  = np.stack([x_f.ravel(), y_f.ravel(), z_f.ravel()], axis=1)
            pts_out = pts_in @ R_calib.T + t_calib
            x_f = pts_out[:, 0].reshape(sh)
            y_f = pts_out[:, 1].reshape(sh)
            z_f = pts_out[:, 2].reshape(sh)

        # ── Valid mask ────────────────────────────────────────────────────
        valid = valid_sob[:, :, np.newaxis] & (dist_raw > 0)
        if min_dist > 0:
            valid &= (x_f**2 + y_f**2 + z_f**2) >= (min_dist ** 2)

        # ── Channel index array (for return_channels) ─────────────────────
        if return_channels:
            ch_idx_arr = np.broadcast_to(
                ch_sel.astype(np.uint8)[np.newaxis, np.newaxis, :],
                x_f.shape,
            )

        # ── Assemble PCD points ───────────────────────────────────────────
        def make_pts(mask):
            pts = np.empty(int(mask.sum()), dtype=PCD_DTYPE)
            pts['x']         = x_f[mask].astype(np.float32)
            pts['y']         = y_f[mask].astype(np.float32)
            pts['z']         = z_f[mask].astype(np.float32)
            pts['intensity'] = intens[mask]
            return pts

        if split_lidar_coords:
            mf = valid & (x_l > 0)
            mb = valid & (x_l < 0)
            out_pts_f.append(make_pts(mf))
            out_pts_b.append(make_pts(mb))
            if return_channels:
                out_ch_f.append(ch_idx_arr[mf])
                out_ch_b.append(ch_idx_arr[mb])
        else:
            out_pts.append(make_pts(valid))
            if return_channels:
                out_ch.append(ch_idx_arr[valid])

    # ── Concatenate chunks ────────────────────────────────────────────────
    if split_lidar_coords:
        pts_f = np.concatenate(out_pts_f) if out_pts_f else np.empty(0, dtype=PCD_DTYPE)
        pts_b = np.concatenate(out_pts_b) if out_pts_b else np.empty(0, dtype=PCD_DTYPE)
        print(f"Parsed {len(pts_f) + len(pts_b):,} points "
              f"(front: {len(pts_f):,}, back: {len(pts_b):,})")
        if return_channels:
            ch_f = np.concatenate(out_ch_f) if out_ch_f else np.empty(0, dtype=np.uint8)
            ch_b = np.concatenate(out_ch_b) if out_ch_b else np.empty(0, dtype=np.uint8)
            return pts_f, ch_f, pts_b, ch_b
        return pts_f, pts_b
    else:
        pts = np.concatenate(out_pts) if out_pts else np.empty(0, dtype=PCD_DTYPE)
        print(f"Parsed {len(pts):,} points")
        if return_channels:
            ch = np.concatenate(out_ch) if out_ch else np.empty(0, dtype=np.uint8)
            return pts, ch
        return pts


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Replay a LiDAR PCAP and produce a 3D PCD (offline, vectorized)")
    ap.add_argument("pcap", help="input PCAP file")
    ap.add_argument("-o", "--output", default="scans/replay.pcd",
                    help="output PCD file (default: scans/replay.pcd)")
    ap.add_argument("--rpm", type=float, default=None,
                    help="motor speed in RPM (alternative to --total-angle)")
    ap.add_argument("--total-angle", type=float, default=None,
                    help="total angle swept in degrees; RPM computed from PCAP duration")
    ap.add_argument("--ip", default="192.168.20.13",
                    help="LiDAR IP for PTC calibration (default: 192.168.20.13)")
    ap.add_argument("--decimate", type=int, default=1,
                    help="keep 1 every N points (default: 1)")
    ap.add_argument("--block-interp", action="store_true",
                    help="interpolate motor angle per block (higher precision)")
    ap.add_argument("--split-front-back", action="store_true",
                    help="split into front (x>0) and back (x<0) PCD files")
    ap.add_argument("--split-lidar-coords", action="store_true",
                    help="split by x_L sign (before motor rotation)")
    ap.add_argument("--split-channels", action="store_true",
                    help="write one PCD per LiDAR channel (40 files)")
    ap.add_argument("--channel", type=int, nargs='+', default=None, metavar="N",
                    help="export only channel(s) N (0-39), e.g. --channel 10 11")
    ap.add_argument("--min-dist", type=float, default=0.0,
                    help="discard points closer than this distance in meters")
    ap.add_argument("--mount-correction", type=float, nargs=3, default=None,
                    metavar=("ROLL", "PITCH", "YAW"),
                    help="mounting correction in degrees applied before motor rotation (e.g. 0.72 0.15 -1.28)")
    ap.add_argument("--mount-translation", type=float, nargs=3, default=None,
                    metavar=("TX", "TY", "TZ"),
                    help="mounting translation in meters applied before motor rotation (e.g. 0.01 -0.005 0.02)")
    ap.add_argument("--apply-calibration", action="store_true",
                    help="apply alignment correction from calibration.json")
    ap.add_argument("--calibration-file", default="calibration.json",
                    help="path to calibration.json (default: calibration.json)")
    ap.add_argument("--ascii", action="store_true",
                    help="write ASCII PCD instead of binary")
    args = ap.parse_args()

    if args.channel is not None:
        bad = [c for c in args.channel if not (0 <= c < CHANNELS)]
        if bad:
            ap.error(f"--channel out of range (0-{CHANNELS-1}): {bad}")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 1. LiDAR angle calibration
    cal = load_calibration(args.ip)

    # 2. Read PCAP
    packets_with_ts, t_start = read_pcap(args.pcap)

    # 3. Resolve RPM
    if args.total_angle is not None:
        duration = packets_with_ts[-1][0] - packets_with_ts[0][0]
        rpm = args.total_angle / 6.0 / duration
        print(f"Total angle {args.total_angle}° over {duration:.2f}s → {rpm:.4f} RPM")
    elif args.rpm is not None:
        rpm = args.rpm
    else:
        rpm = 0.3333
        print(f"No --rpm or --total-angle, using default {rpm} RPM")

    # 4. Mounting correction
    R_mount = None
    t_mount = None
    if args.mount_correction is not None:
        from scipy.spatial.transform import Rotation
        R_mount = Rotation.from_euler('xyz', args.mount_correction, degrees=True).as_matrix()
        r, p, y = args.mount_correction
        print(f"Mount correction: Roll={r}° Pitch={p}° Yaw={y}°")
    if args.mount_translation is not None:
        t_mount = np.array(args.mount_translation)
        tx, ty, tz = args.mount_translation
        print(f"Mount translation: tx={tx*1000:.1f}mm ty={ty*1000:.1f}mm tz={tz*1000:.1f}mm")

    # 5. Alignment calibration
    R_calib = t_calib = None
    if args.apply_calibration:
        R_calib, t_calib = load_calibration_file(args.calibration_file)
        if R_calib is None:
            print(f"ERROR: cannot load {args.calibration_file}", file=sys.stderr)
            sys.exit(1)
        print("Alignment calibration loaded.")

    # 5. Parse (vectorized)
    need_channels = args.split_channels or (args.channel is not None)

    result = fast_parse(
        packets_with_ts, t_start, cal, rpm,
        channels=args.channel,
        block_interp=args.block_interp,
        min_dist=args.min_dist,
        split_lidar_coords=args.split_lidar_coords,
        return_channels=need_channels and args.split_channels,
        R_calib=R_calib,
        t_calib=t_calib,
        R_mount=R_mount,
        t_mount=t_mount,
    )

    ch_arr = ch_arr_front = ch_arr_back = None
    pts = pts_front = pts_back = None

    if args.split_lidar_coords and args.split_channels:
        pts_front, ch_arr_front, pts_back, ch_arr_back = result
    elif args.split_lidar_coords:
        pts_front, pts_back = result
    elif args.split_channels:
        pts, ch_arr = result
    else:
        pts = result

    # 6. Decimate
    if args.decimate > 1:
        if args.split_lidar_coords:
            pts_front = pts_front[::args.decimate]
            pts_back  = pts_back[::args.decimate]
            if ch_arr_front is not None:
                ch_arr_front = ch_arr_front[::args.decimate]
                ch_arr_back  = ch_arr_back[::args.decimate]
            print(f"Decimated to {len(pts_front) + len(pts_back):,} points (1/{args.decimate})")
        else:
            pts = pts[::args.decimate]
            if ch_arr is not None:
                ch_arr = ch_arr[::args.decimate]
            print(f"Decimated to {len(pts):,} points (1/{args.decimate})")

    # 7. Write PCD
    base, ext = os.path.splitext(args.output)
    ch_suffix = ("_ch" + "-".join(f"{c:02d}" for c in args.channel)) if args.channel else ""

    if args.split_lidar_coords and args.split_channels:
        for ch in range(CHANNELS):
            write_pcd(f"{base}_front_ch{ch:02d}{ext}", pts_front[ch_arr_front == ch], args.ascii)
            write_pcd(f"{base}_back_ch{ch:02d}{ext}",  pts_back[ch_arr_back  == ch], args.ascii)
        print(f"Written {CHANNELS * 2} files: "
              f"{base}_front_ch00{ext} … {base}_back_ch{CHANNELS-1:02d}{ext}")
    elif args.split_lidar_coords:
        write_pcd(f"{base}_front{ch_suffix}{ext}", pts_front, args.ascii)
        write_pcd(f"{base}_back{ch_suffix}{ext}",  pts_back,  args.ascii)
    elif args.split_front_back:
        write_pcd(f"{base}_front{ext}", pts[pts['x'] > 0], args.ascii)
        write_pcd(f"{base}_back{ext}",  pts[pts['x'] < 0], args.ascii)
    elif args.split_channels:
        for ch in range(CHANNELS):
            write_pcd(f"{base}_ch{ch:02d}{ext}", pts[ch_arr == ch], args.ascii)
        print(f"Written {CHANNELS} files: {base}_ch00{ext} … {base}_ch{CHANNELS-1:02d}{ext}")
    else:
        write_pcd(args.output, pts, args.ascii)


if __name__ == "__main__":
    main()
