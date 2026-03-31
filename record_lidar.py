#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy"]
# ///
"""Record Pandar 40P point cloud to PCD file.

Two-phase approach:
  Phase 1 (time-critical): tight recvfrom loop, store raw packets
  Phase 2 (offline):       numpy-vectorized parse + PCD write

Calibration angles are downloaded from the LiDAR via PTC (port 9347),
with hardcoded defaults as fallback.
"""

import argparse
import os
import signal
import socket
import struct
import sys
import time

import numpy as np

# ── Pandar40P protocol constants ─────────────────────────────────────
BLOCK_SIZE = 124          # 2 SOB + 2 azimuth + 40*(2 dist + 1 intensity)
BLOCKS_PER_PKT = 10
CHANNELS = 40
SOB_FLAG = 0xFFEE
VALID_SIZES = (1262, 1266)
PTC_PORT = 9347

# Default elevation angles per channel (degrees) — Hesai Pandar40P SDK
DEFAULT_ELEVATION = [
     15.0,   11.0,    8.0,    5.0,    3.0,    2.0,    1.67,   1.33,
      1.0,    0.67,   0.33,   0.0,   -0.33,  -0.67,  -1.0,   -1.33,
     -1.67,  -2.0,   -2.33,  -2.67,  -3.0,   -3.33,  -3.67,  -4.0,
     -4.33,  -4.67,  -5.0,   -5.33,  -5.67,  -6.0,   -7.0,   -8.0,
     -9.0,  -10.0,  -12.0,  -14.0,  -16.0,  -19.0,  -23.0,  -25.0,
]

# Default azimuth offsets per channel (degrees)
DEFAULT_AZIMUTH_OFFSET = [
    -1.042, -1.042, -1.042, -1.042, -1.042, -1.042,  3.125, -5.208,
    -1.042,  3.125, -5.208, -1.042,  3.125, -5.208, -1.042,  3.125,
    -5.208, -1.042,  3.125, -5.208, -1.042,  3.125, -5.208, -1.042,
     3.125, -5.208, -1.042,  3.125, -5.208, -1.042, -1.042, -1.042,
    -1.042, -1.042, -1.042, -1.042, -1.042, -1.042, -1.042, -1.042,
]

# Structured dtype for PCD binary output: 13 bytes/point
PCD_DTYPE = np.dtype([
    ('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', 'u1')
])

_stop = False


def _sigint_handler(sig, frame):
    global _stop
    _stop = True


# ── PTC: download calibration from LiDAR ────────────────────────────

def ptc_get_corrections(lidar_ip, timeout=3.0):
    """Download angle corrections from Pandar via PTC protocol.

    Returns (elevation_deg[40], azimuth_offset_deg[40]) or None on failure.
    """
    # PTC request: magic(2) + cmd(1) + reserved(1) + payload_len(4)
    # Command 0x05 = GET_CALIBRATION, no payload
    request = struct.pack(">2sBBI", b'\x47\x74', 0x05, 0x00, 0)

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((lidar_ip, PTC_PORT))
        sock.sendall(request)

        # Read response header: magic(2) + cmd(1) + return_code(1) + payload_len(4)
        hdr = b''
        while len(hdr) < 8:
            chunk = sock.recv(8 - len(hdr))
            if not chunk:
                break
            hdr += chunk

        if len(hdr) < 8:
            print(f"PTC: incomplete header ({len(hdr)} bytes)", file=sys.stderr)
            sock.close()
            return None

        _, cmd, ret_code, payload_len = struct.unpack(">2sBBI", hdr)
        if ret_code != 0:
            print(f"PTC: error code {ret_code}", file=sys.stderr)
            sock.close()
            return None

        # Read payload
        payload = b''
        while len(payload) < payload_len:
            chunk = sock.recv(min(4096, payload_len - len(payload)))
            if not chunk:
                break
            payload += chunk
        sock.close()

        if len(payload) < payload_len:
            print(f"PTC: incomplete payload ({len(payload)}/{payload_len})",
                  file=sys.stderr)
            return None

        # Parse CSV: "channel,elevation,azimuth_offset\n" per line
        text = payload.decode('ascii', errors='replace')
        elev = [0.0] * CHANNELS
        az_off = [0.0] * CHANNELS
        for line in text.strip().splitlines():
            parts = line.split(',')
            if len(parts) >= 3:
                try:
                    ch = int(parts[0]) - 1  # 1-indexed → 0-indexed
                    if 0 <= ch < CHANNELS:
                        elev[ch] = float(parts[1])
                        az_off[ch] = float(parts[2])
                except ValueError:
                    continue

        return elev, az_off

    except (OSError, socket.timeout) as e:
        print(f"PTC: connection failed ({e})", file=sys.stderr)
        return None


def load_calibration(lidar_ip):
    """Load calibration from LiDAR, fall back to defaults."""
    result = ptc_get_corrections(lidar_ip)
    if result is not None:
        elev, az_off = result
        print(f"Loaded calibration from {lidar_ip}:{PTC_PORT} "
              f"(elev range: {min(elev):.2f}° to {max(elev):.2f}°)")
    else:
        print("Using default calibration angles")
        elev = DEFAULT_ELEVATION
        az_off = DEFAULT_AZIMUTH_OFFSET

    elev_rad = np.deg2rad(np.array(elev, dtype=np.float64))
    az_off_rad = np.deg2rad(np.array(az_off, dtype=np.float64))
    return {
        'cos_elev': np.cos(elev_rad),
        'sin_elev': np.sin(elev_rad),
        'az_off_rad': az_off_rad,
    }


# ── Phase 1: Capture ─────────────────────────────────────────────────

def capture_packets(port, duration):
    """Tight recvfrom loop — no parsing, just store raw bytes."""
    global _stop
    _stop = False
    prev_handler = signal.signal(signal.SIGINT, _sigint_handler)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(2.0)

    packets = []
    t0 = time.monotonic()
    deadline = t0 + duration
    warned = False

    print(f"Capturing on UDP :{port} for {duration}s ...")

    while not _stop and time.monotonic() < deadline:
        try:
            data, _ = sock.recvfrom(2048)
        except socket.timeout:
            if not packets and not warned:
                print("WARNING: no packets received after 2s — "
                      "check network/IP config", file=sys.stderr)
                warned = True
            continue
        if len(data) in VALID_SIZES:
            packets.append(data)

    sock.close()
    signal.signal(signal.SIGINT, prev_handler)

    elapsed = time.monotonic() - t0
    pps = len(packets) / max(elapsed, 0.001)
    print(f"Captured {len(packets):,} packets in {elapsed:.1f}s ({pps:.0f} pkt/s)")
    if _stop and packets:
        print("(interrupted — saving partial data)")
    return packets


# ── Phase 2: Parse ───────────────────────────────────────────────────

def parse_packets(packets, cal):
    """Vectorized parse of raw packets → structured numpy array."""
    if not packets:
        return np.empty(0, dtype=PCD_DTYPE)

    cos_elev = cal['cos_elev']
    sin_elev = cal['sin_elev']
    az_off_rad = cal['az_off_rad']

    max_pts = len(packets) * BLOCKS_PER_PKT * CHANNELS
    out = np.empty(max_pts, dtype=PCD_DTYPE)
    idx = 0

    print("Parsing packets ...")

    for raw in packets:
        for b in range(BLOCKS_PER_PKT):
            off = b * BLOCK_SIZE
            if raw[off] != 0xFF or raw[off + 1] != 0xEE:
                continue
            az_raw = struct.unpack_from("<H", raw, off + 2)[0]
            az_base = np.float64(az_raw) * 0.01 * (np.pi / 180.0)

            # Unpack 40 channels at once: each is 2B distance LE + 1B intensity
            ch_data = np.frombuffer(raw, dtype=np.uint8,
                                    count=CHANNELS * 3, offset=off + 4)
            ch = ch_data.reshape(CHANNELS, 3)
            dist_raw = (ch[:, 0].astype(np.uint16)
                        | (ch[:, 1].astype(np.uint16) << 8))
            intens = ch[:, 2]

            # Skip zero-distance returns (no echo)
            mask = dist_raw > 0
            if not mask.any():
                continue

            dist_m = dist_raw[mask].astype(np.float64) * 0.004
            az = az_base + az_off_rad[mask]
            cos_e = cos_elev[mask]
            sin_e = sin_elev[mask]

            n = dist_m.shape[0]
            out['x'][idx:idx + n] = (dist_m * cos_e * np.sin(az))
            out['y'][idx:idx + n] = (dist_m * cos_e * np.cos(az))
            out['z'][idx:idx + n] = (dist_m * sin_e)
            out['intensity'][idx:idx + n] = intens[mask]
            idx += n

    out = out[:idx]
    print(f"Parsed {idx:,} points")
    return out


# ── PCD Writer ───────────────────────────────────────────────────────

def write_pcd(filename, points, ascii_mode=False):
    """Write structured array to PCD file (binary or ASCII)."""
    n = len(points)
    if n == 0:
        print("No points to write.", file=sys.stderr)
        return

    header = (
        "# .PCD v0.7 - Point Cloud Data\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 1\n"
        "TYPE F F F U\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
    )

    if ascii_mode:
        header += "DATA ascii\n"
        with open(filename, "w") as f:
            f.write(header)
            for i in range(n):
                p = points[i]
                f.write(f"{p['x']:.6f} {p['y']:.6f} {p['z']:.6f} "
                        f"{p['intensity']}\n")
    else:
        header += "DATA binary\n"
        with open(filename, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(points.tobytes())

    sz = os.path.getsize(filename) / (1024 * 1024)
    print(f"Wrote {filename} ({n:,} points, {sz:.1f} MB)")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Record Pandar 40P point cloud to PCD file")
    ap.add_argument("-d", "--duration", type=float, default=10,
                    help="capture duration in seconds (default: 10)")
    ap.add_argument("-o", "--output", default="scan.pcd",
                    help="output PCD file (default: scan.pcd)")
    ap.add_argument("-p", "--port", type=int, default=2368,
                    help="UDP port (default: 2368)")
    ap.add_argument("--ip", default="192.168.20.13",
                    help="LiDAR IP for PTC calibration (default: 192.168.20.13)")
    ap.add_argument("--ascii", action="store_true",
                    help="write ASCII PCD instead of binary")
    args = ap.parse_args()

    cal = load_calibration(args.ip)
    pkts = capture_packets(args.port, args.duration)
    if not pkts:
        print("No packets captured — nothing to write.", file=sys.stderr)
        sys.exit(1)

    pts = parse_packets(pkts, cal)
    write_pcd(args.output, pts, args.ascii)


if __name__ == "__main__":
    main()
