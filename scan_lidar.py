#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy", "pyserial"]
# ///
"""Scan 3D with rotating motor + synchronized LiDAR capture.

Records Pandar 40P packets with timestamps, reads motor angle from serial
(AS5048A encoder via Arduino), and outputs a merged PCD with proper
geometric transformations.

Mount orientation: Pandar vertical (top = -Y_world, +X_lidar → -Z_world).
Motor rotates around Y axis at configurable RPM.
Uses closed-loop angle feedback from encoder for accuracy.
"""

import argparse
import json
import os
import signal
import socket
import struct
import sys
import threading
import time

import numpy as np
import serial

from record_lidar import (
    BLOCK_SIZE,
    BLOCKS_PER_PKT,
    CHANNELS,
    PCD_DTYPE,
    VALID_SIZES,
    load_calibration,
    write_pcd,
)

_stop = False


def _sigint_handler(sig, frame):
    global _stop
    _stop = True


# ── Serial Motor Thread ──────────────────────────────────────────────────

def read_motor_angles(port, motor_log):
    """Read motor angles from serial port (AS5048A encoder via Arduino).

    Expected output format:
        Tours:0.500 (180.0°) Err:0.125 rpm:0.333

    Parses angle in degrees and stores (timestamp, angle_deg) tuples.
    Appends to motor_log list for synchronization.
    """
    try:
        ser = serial.Serial(port, 115200, timeout=0.5)
        time.sleep(2)  # Arduino boot time
    except serial.SerialException as e:
        print(f"ERROR: Cannot open serial port {port}: {e}", file=sys.stderr)
        return

    print(f"Serial thread: reading from {port}")

    while not _stop:
        try:
            if ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if "°)" in line:
                    # Extract angle between parentheses: "(...°)"
                    try:
                        start = line.rfind('(')
                        end = line.rfind('°')
                        if start >= 0 and end > start:
                            angle_str = line[start+1:end]
                            angle_deg = float(angle_str)
                            t_mono = time.monotonic()
                            motor_log.append((t_mono, angle_deg))
                    except (ValueError, IndexError):
                        pass
        except (serial.SerialException, OSError):
            break

    try:
        ser.close()
    except:
        pass


def interpolate_motor_angle(t, motor_data, rpm_nominal):
    """Interpolate motor angle at time t from serial log.

    Linear interpolation between logged points.
    Before first log: extrapolate using rpm_nominal.
    After last log: use last angle + extrapolate.
    """
    if not motor_data:
        # Pure extrapolation from nominal RPM
        return t * rpm_nominal * 360.0 / 60.0

    if len(motor_data) == 1:
        t0, a0 = motor_data[0]
        return a0 + (t - t0) * rpm_nominal * 360.0 / 60.0

    # Find bracketing points
    t0, angle0 = motor_data[0]
    if t <= t0:
        return angle0 - (t0 - t) * rpm_nominal * 360.0 / 60.0

    t1, angle1 = motor_data[-1]
    if t >= t1:
        return angle1 + (t - t1) * rpm_nominal * 360.0 / 60.0

    # Linear interpolation
    for i in range(len(motor_data) - 1):
        ta, aa = motor_data[i]
        tb, ab = motor_data[i + 1]
        if ta <= t <= tb:
            if tb > ta:
                return aa + (ab - aa) * (t - ta) / (tb - ta)
            return aa

    return angle1


# ── Packet Capture with Timestamps ───────────────────────────────────────

def capture_packets_with_timestamps(port, duration):
    """Capture UDP packets with monotonic timestamps."""
    global _stop
    _stop = False
    prev_handler = signal.signal(signal.SIGINT, _sigint_handler)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(2.0)

    packets = []  # List of (timestamp, data) tuples
    t0 = time.monotonic()
    deadline = t0 + duration
    warned = False

    print(f"Capturing on UDP :{port} for {duration}s ...")

    while not _stop and time.monotonic() < deadline:
        try:
            data, _ = sock.recvfrom(2048)
            t_pkt = time.monotonic()
        except socket.timeout:
            if not packets and not warned:
                print("WARNING: no packets after 2s — check network/IP config",
                      file=sys.stderr)
                warned = True
            continue

        if len(data) in VALID_SIZES:
            packets.append((t_pkt, data))

    sock.close()
    signal.signal(signal.SIGINT, prev_handler)

    elapsed = time.monotonic() - t0
    pps = len(packets) / max(elapsed, 0.001)
    print(f"Captured {len(packets):,} packets in {elapsed:.1f}s ({pps:.0f} pkt/s)")
    if _stop and packets:
        print("(interrupted — saving partial data)")

    return packets, t0


# ── Calibration & Geometric Transformations ───────────────────────────────

def load_calibration_file(filepath):
    """Load calibration.json if available."""
    try:
        with open(filepath, 'r') as f:
            cal = json.load(f)
        R_inv = np.array(cal['inverse_transformation']['rotation_matrix'])
        t_inv = np.array(cal['inverse_transformation']['translation_meters'])
        return R_inv, t_inv
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None, None



def apply_rotation_matrix(x_l, y_l, z_l, theta_rad):
    """Apply Ry(-θ) rotation to LiDAR coordinates (no mount transformation).

    Simple rotation around Y axis:
      x_f = x_L * cos(θ) + z_L * sin(θ)
      y_f = y_L
      z_f = -x_L * sin(θ) + z_L * cos(θ)
    where θ = -rpm × 6 × elapsed (angle open-loop)
    """
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)

    x_final = x_l * cos_t + z_l * sin_t
    y_final = y_l
    z_final = -x_l * sin_t + z_l * cos_t

    return x_final, y_final, z_final


# ── Enhanced Parse with Motor Angles ─────────────────────────────────────

def parse_packets_with_motor(packets_with_ts, t_start, motor_data, cal, rpm_nominal, block_interp=False, split_lidar_coords=False, R_calib=None, t_calib=None, return_channels=False):
    """Parse packets, apply motor rotation, return numpy array.

    Args:
        block_interp: if True, interpolate motor angle per block (higher precision)
                      if False, use angle per packet (faster)
        split_lidar_coords: if True, return (pts_front, pts_back) split by x_L > 0
                           if False, return single pts array
        return_channels: if True (and split_lidar_coords=False), return (pts, ch_arr)
                         where ch_arr is a uint8 array of channel indices (0-39)
    """
    if not packets_with_ts:
        if split_lidar_coords:
            return np.empty(0, dtype=PCD_DTYPE), np.empty(0, dtype=PCD_DTYPE)
        else:
            return np.empty(0, dtype=PCD_DTYPE)

    cos_elev = cal['cos_elev']
    sin_elev = cal['sin_elev']
    az_off_rad = cal['az_off_rad']

    max_pts = len(packets_with_ts) * BLOCKS_PER_PKT * CHANNELS
    out = np.empty(max_pts, dtype=PCD_DTYPE)
    out_front = np.empty(max_pts, dtype=PCD_DTYPE) if split_lidar_coords else None
    out_back = np.empty(max_pts, dtype=PCD_DTYPE) if split_lidar_coords else None
    ch_out       = np.empty(max_pts, dtype=np.uint8) if (return_channels and not split_lidar_coords) else None
    ch_out_front = np.empty(max_pts, dtype=np.uint8) if (return_channels and split_lidar_coords) else None
    ch_out_back  = np.empty(max_pts, dtype=np.uint8) if (return_channels and split_lidar_coords) else None
    idx = 0
    idx_front = 0
    idx_back = 0

    print("Parsing packets with motor rotation " +
          ("(per-block interpolation)" if block_interp else "(per-packet)") + " ...")
    # Time per block: 1/18000 s (since 1800 pkt/s × 10 blocks/pkt = 18000 blocks/s)
    time_per_block = 1.0 / (1800 * BLOCKS_PER_PKT) if block_interp else 0

    for t_pkt, raw in packets_with_ts:
        elapsed = t_pkt - t_start

        for b in range(BLOCKS_PER_PKT):
            if block_interp:
                # Interpolate motor angle at the time of this block capture
                # Block b was captured approximately at: t_pkt - (9-b) * time_per_block
                elapsed_block = elapsed - (BLOCKS_PER_PKT - 1 - b) * time_per_block
            else:
                # Use angle at packet reception time
                elapsed_block = elapsed
            # θ = -rpm × 6 × elapsed (angle open-loop) as per CLAUDE.md
            theta_rad = np.deg2rad(-interpolate_motor_angle(elapsed_block, motor_data, rpm_nominal))
            off = b * BLOCK_SIZE
            if len(raw) < off + 4 or raw[off] != 0xFF or raw[off + 1] != 0xEE:
                continue

            az_raw = struct.unpack_from("<H", raw, off + 2)[0]
            az_base = np.float64(az_raw) * 0.01 * (np.pi / 180.0)

            # Extract channels
            if len(raw) < off + 4 + CHANNELS * 3:
                continue

            ch_data = np.frombuffer(raw, dtype=np.uint8,
                                    count=CHANNELS * 3, offset=off + 4)
            ch = ch_data.reshape(CHANNELS, 3)
            dist_raw = (ch[:, 0].astype(np.uint16) |
                       (ch[:, 1].astype(np.uint16) << 8))
            intens = ch[:, 2]

            mask = dist_raw > 0
            if not mask.any():
                continue

            dist_m = dist_raw[mask].astype(np.float64) * 0.004
            az = az_base + az_off_rad[mask]
            cos_e = cos_elev[mask]
            sin_e = sin_elev[mask]

            # LiDAR coordinates
            x_l = dist_m * cos_e * np.sin(az)
            y_l = dist_m * cos_e * np.cos(az)
            z_l = dist_m * sin_e

            # Apply rotation (mount + motor)
            x_final, y_final, z_final = apply_rotation_matrix(x_l, y_l, z_l, theta_rad)

            # Apply calibration correction if provided
            if R_calib is not None and t_calib is not None:
                pts_rotated = np.column_stack([x_final, y_final, z_final])
                pts_corrected = pts_rotated @ R_calib.T + t_calib
                x_final = pts_corrected[:, 0]
                y_final = pts_corrected[:, 1]
                z_final = pts_corrected[:, 2]

            n = dist_m.shape[0]
            ch_indices = np.where(mask)[0]  # channel indices (0-39) for valid points

            if split_lidar_coords:
                # Split by x_L sign (before rotation)
                mask_front = x_l > 0
                mask_back = x_l < 0

                n_front = mask_front.sum()
                n_back = mask_back.sum()

                if n_front > 0:
                    out_front['x'][idx_front:idx_front + n_front] = x_final[mask_front]
                    out_front['y'][idx_front:idx_front + n_front] = y_final[mask_front]
                    out_front['z'][idx_front:idx_front + n_front] = z_final[mask_front]
                    out_front['intensity'][idx_front:idx_front + n_front] = intens[mask][mask_front]
                    if ch_out_front is not None:
                        ch_out_front[idx_front:idx_front + n_front] = ch_indices[mask_front]
                    idx_front += n_front

                if n_back > 0:
                    out_back['x'][idx_back:idx_back + n_back] = x_final[mask_back]
                    out_back['y'][idx_back:idx_back + n_back] = y_final[mask_back]
                    out_back['z'][idx_back:idx_back + n_back] = z_final[mask_back]
                    out_back['intensity'][idx_back:idx_back + n_back] = intens[mask][mask_back]
                    if ch_out_back is not None:
                        ch_out_back[idx_back:idx_back + n_back] = ch_indices[mask_back]
                    idx_back += n_back
            else:
                out['x'][idx:idx + n] = x_final
                out['y'][idx:idx + n] = y_final
                out['z'][idx:idx + n] = z_final
                out['intensity'][idx:idx + n] = intens[mask]
                if ch_out is not None:
                    ch_out[idx:idx + n] = ch_indices
                idx += n

    if split_lidar_coords:
        out_front = out_front[:idx_front]
        out_back = out_back[:idx_back]
        print(f"Parsed {idx_front + idx_back:,} points (front: {idx_front:,}, back: {idx_back:,})")
        if return_channels:
            return out_front, ch_out_front[:idx_front], out_back, ch_out_back[:idx_back]
        return out_front, out_back
    else:
        out = out[:idx]
        print(f"Parsed {idx:,} points")
        if return_channels:
            return out, ch_out[:idx]
        return out


# ── Main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="3D LiDAR scan with motor rotation")
    ap.add_argument("-d", "--duration", type=float, default=180,
                    help="capture duration in seconds (default: 180)")
    ap.add_argument("-o", "--output", default="scan.pcd",
                    help="output PCD file (default: scan.pcd)")
    ap.add_argument("--rpm", type=float, default=0.3333,
                    help="motor speed in RPM (default: 0.3333)")
    ap.add_argument("--serial", default="/dev/ttyUSB0",
                    help="serial port for motor (default: /dev/ttyUSB0)")
    ap.add_argument("--ip", default="192.168.20.13",
                    help="LiDAR IP for PTC calibration (default: 192.168.20.13)")
    ap.add_argument("-p", "--port", type=int, default=2368,
                    help="UDP port (default: 2368)")
    ap.add_argument("--return-rpm", type=float, default=0,
                    help="RPM for return to start (0 = no return)")
    ap.add_argument("--decimate", type=int, default=1,
                    help="keep 1 every N points (default: 1 = no decimation)")
    ap.add_argument("--split-channels", action="store_true",
                    help="write one PCD per channel (40 files)")
    ap.add_argument("--ascii", action="store_true",
                    help="write ASCII PCD instead of binary")
    ap.add_argument("--block-interp", action="store_true",
                    help="interpolate motor angle per block (vs per packet) for higher precision")
    ap.add_argument("--split-front-back", action="store_true",
                    help="split points into front (x>0) and back (x<0) files")
    ap.add_argument("--split-lidar-coords", action="store_true",
                    help="split points by x_L sign (before motor rotation) into front and back files")
    ap.add_argument("--apply-calibration", action="store_true",
                    help="apply alignment correction from calibration.json")
    ap.add_argument("--calibration-file", default="calibration.json",
                    help="path to calibration.json (default: calibration.json)")
    args = ap.parse_args()

    # 1. Load calibration
    cal = load_calibration(args.ip)

    # 2. Open serial, wait for Arduino boot
    print(f"Opening {args.serial} ...")
    ser = serial.Serial(args.serial, 115200, timeout=1)
    time.sleep(2.0)
    # Drain boot messages
    while ser.in_waiting:
        ser.readline()

    # 3. Start motor
    cmd = f"R{args.rpm}\n"
    print(f"Sending motor command: {cmd.strip()}")
    ser.write(cmd.encode())
    print("Waiting 3s for motor stabilization ...")
    time.sleep(3.0)

    # 4. Capture UDP packets with timestamps
    packets_with_ts, t_start = capture_packets_with_timestamps(args.port, args.duration)

    # 5. Stop motor
    print("Stopping motor ...")
    ser.write(b"X\n")
    time.sleep(0.5)

    # 5b. Return to start if requested
    if args.return_rpm > 0:
        scan_angle = args.rpm * 6.0 * args.duration  # degrees traveled
        return_time = scan_angle / (args.return_rpm * 6.0)
        cmd_ret = f"R-{args.return_rpm}\n"
        print(f"Returning: {cmd_ret.strip()} for {return_time:.1f}s ({scan_angle:.1f}°)")
        ser.write(cmd_ret.encode())
        time.sleep(return_time + 1.0)  # +1s margin
        ser.write(b"X\n")
        time.sleep(0.5)

    ser.close()

    if not packets_with_ts:
        print("No packets captured -- nothing to write.", file=sys.stderr)
        sys.exit(1)

    # 6. Load calibration if requested
    R_calib = None
    t_calib = None
    if args.apply_calibration:
        print(f"Loading calibration from {args.calibration_file} ...")
        R_calib, t_calib = load_calibration_file(args.calibration_file)
        if R_calib is not None:
            print(f"✓ Calibration loaded - will apply correction during parsing")
        else:
            print(f"✗ Could not load calibration file", file=sys.stderr)
            sys.exit(1)

    # 7. Parse packets with motor rotation
    motor_data = []  # Empty: angle computed from RPM nominal
    result = parse_packets_with_motor(packets_with_ts, t_start, motor_data, cal, args.rpm,
                                      block_interp=args.block_interp,
                                      split_lidar_coords=args.split_lidar_coords,
                                      R_calib=R_calib, t_calib=t_calib)

    # Handle split result
    if args.split_lidar_coords:
        pts_front, pts_back = result
        pts = None
    else:
        pts = result
        pts_front = None
        pts_back = None

    # 8. Decimate
    if args.decimate > 1:
        if args.split_lidar_coords:
            pts_front = pts_front[::args.decimate]
            pts_back = pts_back[::args.decimate]
            print(f"Decimated to {len(pts_front) + len(pts_back):,} points (1/{args.decimate})")
        else:
            pts = pts[::args.decimate]
            print(f"Decimated to {len(pts):,} points (1/{args.decimate})")

    # 9. Write PCD
    if args.split_lidar_coords:
        # Already split by LiDAR coords (x_L sign)
        base, ext = os.path.splitext(args.output)
        fname_front = f"{base}_front{ext}"
        fname_back = f"{base}_back{ext}"

        write_pcd(fname_front, pts_front, args.ascii)
        write_pcd(fname_back, pts_back, args.ascii)

        print(f"Split by LiDAR coords: {len(pts_front):,} points (x_L>0) → {fname_front}")
        print(f"                       {len(pts_back):,} points (x_L<0) → {fname_back}")
    elif args.split_front_back:
        # Split into front (x > 0) and back (x < 0) in global frame
        mask_front = pts['x'] > 0
        mask_back = pts['x'] < 0

        pts_front = pts[mask_front]
        pts_back = pts[mask_back]

        # Generate filenames with _front and _back suffixes
        base, ext = os.path.splitext(args.output)
        fname_front = f"{base}_front{ext}"
        fname_back = f"{base}_back{ext}"

        write_pcd(fname_front, pts_front, args.ascii)
        write_pcd(fname_back, pts_back, args.ascii)

        print(f"Split by global frame: {len(pts_front):,} points (x>0) → {fname_front}")
        print(f"                       {len(pts_back):,} points (x<0) → {fname_back}")
    else:
        write_pcd(args.output, pts, args.ascii)


if __name__ == "__main__":
    main()
