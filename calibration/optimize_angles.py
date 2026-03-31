#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy", "scipy", "pyserial"]
# ///
"""Optimize LiDAR mounting correction to minimize front/back point cloud distance.

Pipeline:
  1. Parse PCAP once → extract raw LiDAR coords (x_L, y_L, z_L) + motor angles θ
  2. Nelder-Mead loop: apply R_mount(roll,pitch,yaw) + t_mount(tx,ty,tz) + Ry(-θ),
     measure median nearest-neighbor distance between:
       - cloud A: channel CH_BACK,  back half  (x_L < 0)
       - cloud B: channel CH_FRONT, front half (x_L > 0)
  3. Write optimized PCD files

Bounds: ±MAX_DEG degrees per rotation axis, ±MAX_SHIFT meters per translation axis.

Usage:
    uv run optimize_angles.py
    uv run optimize_angles.py --init-angles 0.723 0.152 -1.281
    uv run optimize_angles.py --channel-back 0 --channel-front 39
    uv run optimize_angles.py --init-angles 0.723 0.152 -1.281 --init-translation 0 0 0
"""

import argparse
import os
import sys
import time

# Allow imports from project root and from this directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from record_lidar import (
    BLOCK_SIZE, BLOCKS_PER_PKT, CHANNELS,
    PCD_DTYPE, load_calibration, write_pcd,
)
from replay_scan import read_pcap

# ── Defaults ───────────────────────────────────────────────────────────────
PCAP         = 'raw/scan_360_6min.pcap'
TOTAL_ANGLE  = 360.0
CH_BACK      = 0    # channel used for back half  (x_L < 0)
CH_FRONT     = 39   # channel used for front half (x_L > 0)
MIN_DIST     = 2.0
BLOCK_INTERP = True
MAX_DEG      = 5.0
MAX_SHIFT    = 0.01  # ±1 cm per translation axis
MAX_ANGLE_DELTA = 1.0  # ±1° on total rotation angle
SUBSAMPLE    = 10   # use 1/N points in optimization loop (speed vs accuracy)
CHUNK_SIZE   = 50_000


# ── Step 1: Parse PCAP → raw LiDAR coords + motor angles ──────────────────

def parse_raw(packets_with_ts, t_start, cal, rpm, channels, block_interp, min_dist, max_dist=0.0):
    """Vectorized parse — returns pre-rotation LiDAR frame arrays.

    Returns:
        x_L, y_L, z_L : LiDAR-frame coordinates  (float32, shape N)
        theta          : motor angle per point     (float32, shape N, radians)
        intens         : intensity                 (uint8,   shape N)
        ch_idx         : LiDAR channel number      (uint8,   shape N)
    """
    N_pkt = len(packets_with_ts)
    timestamps = np.array([t for t, _ in packets_with_ts])
    elapsed = timestamps - t_start

    ch_sel = np.array(channels, dtype=np.intp)
    cos_e  = cal['cos_elev'][ch_sel]
    sin_e  = cal['sin_elev'][ch_sel]
    az_off = cal['az_off_rad'][ch_sel]

    block_dt = ((BLOCKS_PER_PKT - 1 - np.arange(BLOCKS_PER_PKT)) / 18000.0
                if block_interp else np.zeros(BLOCKS_PER_PKT))
    ch_byte_idx = (4 + ch_sel * 3)[:, np.newaxis] + np.arange(3)[np.newaxis, :]

    # Channel label broadcast array: value = actual channel number (0-39)
    ch_label = ch_sel.astype(np.uint8)  # shape (n_ch,)

    out_xl, out_yl, out_zl, out_th, out_in, out_ch = [], [], [], [], [], []

    n_chunks = (N_pkt + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"Parsing {N_pkt:,} packets in {n_chunks} chunks ...")

    for start in range(0, N_pkt, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, N_pkt)

        raw_chunk = np.array([
            np.frombuffer(raw, dtype=np.uint8)[:BLOCKS_PER_PKT * BLOCK_SIZE]
            .reshape(BLOCKS_PER_PKT, BLOCK_SIZE)
            for _, raw in packets_with_ts[start:end]
        ])

        valid_sob = (raw_chunk[:, :, 0] == 0xFF) & (raw_chunk[:, :, 1] == 0xEE)

        az_base = (
            raw_chunk[:, :, 2].astype(np.uint32) |
            (raw_chunk[:, :, 3].astype(np.uint32) << 8)
        ).astype(np.float64) * (0.01 * np.pi / 180.0)

        ch_bytes = raw_chunk[:, :, ch_byte_idx]
        dist_raw = (ch_bytes[:, :, :, 0].astype(np.uint16) |
                    (ch_bytes[:, :, :, 1].astype(np.uint16) << 8))
        intens = ch_bytes[:, :, :, 2]

        elapsed_blocks = elapsed[start:end, np.newaxis] - block_dt[np.newaxis, :]
        theta = np.deg2rad(-elapsed_blocks * rpm * 6.0)  # (n, 10)

        dist_m = dist_raw.astype(np.float64) * 0.004
        az     = az_base[:, :, np.newaxis] + az_off

        x_l = dist_m * cos_e * np.sin(az)
        y_l = dist_m * cos_e * np.cos(az)
        z_l = dist_m * sin_e

        valid = valid_sob[:, :, np.newaxis] & (dist_raw > 0)
        r2 = x_l**2 + y_l**2 + z_l**2
        if min_dist > 0:
            valid &= r2 >= (min_dist ** 2)
        if max_dist > 0:
            valid &= r2 <= (max_dist ** 2)

        # Expand theta and channel label to (n, BLOCKS_PER_PKT, n_ch)
        theta_exp = np.broadcast_to(theta[:, :, np.newaxis], x_l.shape)
        ch_exp    = np.broadcast_to(ch_label[np.newaxis, np.newaxis, :], x_l.shape)

        out_xl.append(x_l[valid].astype(np.float32))
        out_yl.append(y_l[valid].astype(np.float32))
        out_zl.append(z_l[valid].astype(np.float32))
        out_th.append(theta_exp[valid].astype(np.float32))
        out_in.append(intens[valid])
        out_ch.append(ch_exp[valid])

    x_L = np.concatenate(out_xl)
    y_L = np.concatenate(out_yl)
    z_L = np.concatenate(out_zl)
    th  = np.concatenate(out_th)
    ins = np.concatenate(out_in)
    chs = np.concatenate(out_ch)
    print(f"Parsed {len(x_L):,} points  (front: {(x_L>0).sum():,}, back: {(x_L<0).sum():,})")
    return x_L, y_L, z_L, th, ins, chs


# ── Step 2: Apply R_mount + t_mount + Ry(-θ) ──────────────────────────────

def apply_transform(x_L, y_L, z_L, theta, R_mount, t_mount=None):
    """Apply mounting correction R_mount + t_mount then motor rotation Ry(-θ) per point.

    R_mount:  (3, 3) fixed for all points
    t_mount:  (3,)   translation in meters, or None
    theta:    (N,)   per-point motor angle (radians)
    """
    # Mounting correction (same for all points)
    xyz_m = R_mount @ np.stack([x_L, y_L, z_L])  # (3, N)
    if t_mount is not None:
        xyz_m = xyz_m + t_mount[:, np.newaxis]
    xm, ym, zm = xyz_m

    # Motor rotation Ry(-θ): varies per point
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x_f = xm * cos_t + zm * sin_t
    y_f = ym
    z_f = -xm * sin_t + zm * cos_t
    return x_f, y_f, z_f


# ── Step 3: Objective function ─────────────────────────────────────────────

def make_objective(x_L, y_L, z_L, theta, idx_f, idx_b,
                   max_deg, max_shift, max_angle_delta):
    """Build Nelder-Mead objective. idx_f/idx_b are pre-subsampled indices.

    params: [roll_deg, pitch_deg, yaw_deg, tx_m, ty_m, tz_m, total_angle_deg]
    theta is pre-computed for total_angle=360°; rescaled via total_angle/360.
    """
    _state = {'n': 0, 't0': time.monotonic()}

    def objective(params):
        roll_deg, pitch_deg, yaw_deg, tx, ty, tz, total_angle = params
        _state['n'] += 1

        # Hard boundary penalty
        if max(abs(roll_deg), abs(pitch_deg), abs(yaw_deg)) > max_deg:
            return 1e6
        if max(abs(tx), abs(ty), abs(tz)) > max_shift:
            return 1e6
        if abs(total_angle - 360.0) > max_angle_delta:
            return 1e6

        R_mount = Rotation.from_euler('xyz', [roll_deg, pitch_deg, yaw_deg],
                                      degrees=True).as_matrix()
        t_mount = np.array([tx, ty, tz])
        theta_s = theta * (total_angle / 360.0)
        x_f, y_f, z_f = apply_transform(x_L, y_L, z_L, theta_s, R_mount, t_mount)

        pts_f = np.stack([x_f[idx_f], y_f[idx_f], z_f[idx_f]], axis=1)
        pts_b = np.stack([x_f[idx_b], y_f[idx_b], z_f[idx_b]], axis=1)

        tree = cKDTree(pts_b)
        dists, _ = tree.query(pts_f, workers=-1)
        cost = float(np.median(dists))

        n = _state['n']
        if n % 10 == 0 or n == 1:
            elapsed = time.monotonic() - _state['t0']
            print(f"  [{n:4d}] roll={roll_deg:+.3f}° pitch={pitch_deg:+.3f}° "
                  f"yaw={yaw_deg:+.3f}°  tx={tx*1000:+.1f}mm ty={ty*1000:+.1f}mm "
                  f"tz={tz*1000:+.1f}mm  ang={total_angle:.3f}° → {cost*1000:.1f} mm  ({elapsed:.0f}s)")

        return cost

    return objective


# ── Step 4: Write PCD files ────────────────────────────────────────────────

def write_result(x_L, y_L, z_L, theta, intens, mask_f, mask_b, R_mount, t_mount, out_dir, tag):
    x_f, y_f, z_f = apply_transform(x_L, y_L, z_L, theta, R_mount, t_mount)

    def make_pcd(mask):
        pts = np.empty(int(mask.sum()), dtype=PCD_DTYPE)
        pts['x']         = x_f[mask].astype(np.float32)
        pts['y']         = y_f[mask].astype(np.float32)
        pts['z']         = z_f[mask].astype(np.float32)
        pts['intensity'] = intens[mask]
        return pts

    fn_f = os.path.join(out_dir, f"opt_front_{tag}.pcd")
    fn_b = os.path.join(out_dir, f"opt_back_{tag}.pcd")
    write_pcd(fn_f, make_pcd(mask_f))
    write_pcd(fn_b, make_pcd(mask_b))


# ── Measure distance ───────────────────────────────────────────────────────

def measure_distance(x_f, y_f, z_f, mask_f, mask_b, label=""):
    pts_f = np.stack([x_f[mask_f], y_f[mask_f], z_f[mask_f]], axis=1)
    pts_b = np.stack([x_f[mask_b], y_f[mask_b], z_f[mask_b]], axis=1)
    tree = cKDTree(pts_b)
    dists, _ = tree.query(pts_f, workers=-1)
    med = np.median(dists)
    mean = dists.mean()
    if label:
        print(f"{label}: median={med*1000:.1f} mm  mean={mean*1000:.1f} mm")
    return med


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Optimize LiDAR mounting angles + translation")
    ap.add_argument("--pcap",             default=PCAP)
    ap.add_argument("--total-angle",      type=float, default=TOTAL_ANGLE,
                    help="total motor angle swept (degrees)")
    ap.add_argument("--channel-back",     type=int, default=CH_BACK,
                    help=f"channel for back half  (x_L<0), default {CH_BACK}")
    ap.add_argument("--channel-front",    type=int, default=CH_FRONT,
                    help=f"channel for front half (x_L>0), default {CH_FRONT}")
    ap.add_argument("--min-dist",         type=float, default=MIN_DIST)
    ap.add_argument("--max-deg",          type=float, default=MAX_DEG,
                    help="max rotation correction per axis (degrees)")
    ap.add_argument("--max-shift",        type=float, default=MAX_SHIFT,
                    help="max translation correction per axis (meters)")
    ap.add_argument("--max-angle-delta",  type=float, default=MAX_ANGLE_DELTA,
                    help="max deviation from 360° for total rotation angle (degrees)")
    ap.add_argument("--subsample",        type=int, default=SUBSAMPLE,
                    help="use 1/N points in optimization loop")
    ap.add_argument("--output-dir",       default="scans")
    ap.add_argument("--ip",               default="192.168.20.13")
    ap.add_argument("--init-angles",      type=float, nargs=3,
                    metavar=('ROLL', 'PITCH', 'YAW'), default=[0.0, 0.0, 0.0],
                    help="initial roll/pitch/yaw in degrees (default: 0 0 0)")
    ap.add_argument("--init-translation", type=float, nargs=3,
                    metavar=('TX', 'TY', 'TZ'), default=[0.0, 0.0, 0.0],
                    help="initial tx/ty/tz in meters (default: 0 0 0)")
    args = ap.parse_args()

    ch_b = args.channel_back
    ch_f = args.channel_front
    channels = sorted(set([ch_b, ch_f]))

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load & parse ───────────────────────────────────────────────────
    cal = load_calibration(args.ip)
    packets, t_start = read_pcap(args.pcap)
    duration = packets[-1][0] - packets[0][0]
    rpm = args.total_angle / 6.0 / duration
    print(f"Total angle {args.total_angle}° over {duration:.2f}s → {rpm:.4f} RPM")
    print(f"Cloud A: channel {ch_b} back (x_L<0)   Cloud B: channel {ch_f} front (x_L>0)")

    x_L, y_L, z_L, theta, intens, ch_idx = parse_raw(
        packets, t_start, cal, rpm,
        channels, BLOCK_INTERP, args.min_dist,
    )

    # Masks: channel-specific + half-space
    mask_f = (ch_idx == ch_f) & (x_L > 0)   # channel ch_f, front half
    mask_b = (ch_idx == ch_b) & (x_L < 0)   # channel ch_b, back half
    print(f"Cloud A (ch{ch_b} back):  {mask_b.sum():,} points")
    print(f"Cloud B (ch{ch_f} front): {mask_f.sum():,} points")

    # ── 2. Baseline (identity) ────────────────────────────────────────────
    R_id = np.eye(3)
    x_f0, y_f0, z_f0 = apply_transform(x_L, y_L, z_L, theta, R_id)
    print("\n--- Baseline ---")
    d_baseline = measure_distance(x_f0, y_f0, z_f0, mask_f, mask_b, "Baseline")

    # ── 3. Optimize ───────────────────────────────────────────────────────
    idx_f = np.where(mask_f)[0][::args.subsample]
    idx_b = np.where(mask_b)[0][::args.subsample]
    x0 = np.array(args.init_angles + args.init_translation + [360.0])
    print(f"\nOptimizing with {len(idx_f):,} front + {len(idx_b):,} back points "
          f"(1/{args.subsample} subsample, bounds ±{args.max_deg}° / ±{args.max_shift*100:.0f}cm / "
          f"±{args.max_angle_delta}° total angle) ...")
    print(f"Initial params: roll={x0[0]:+.3f}° pitch={x0[1]:+.3f}° yaw={x0[2]:+.3f}°  "
          f"tx={x0[3]*1000:+.1f}mm ty={x0[4]*1000:+.1f}mm tz={x0[5]*1000:+.1f}mm  ang={x0[6]:.3f}°")

    objective = make_objective(x_L, y_L, z_L, theta, idx_f, idx_b,
                               args.max_deg, args.max_shift, args.max_angle_delta)

    t0 = time.monotonic()
    result = minimize(
        objective,
        x0=x0,
        method='Nelder-Mead',
        options={
            'xatol':   1e-4,
            'fatol':   1e-5,     # 0.01 mm convergence
            'maxiter': 3000,
            'adaptive': True,
            'disp':    False,
        },
    )
    print(f"\nOptimization finished in {time.monotonic() - t0:.1f}s "
          f"({result.nit} iterations, success={result.success})")

    roll, pitch, yaw, tx, ty, tz, total_angle = result.x
    print(f"Optimal rotation:    Roll={roll:+.3f}°  Pitch={pitch:+.3f}°  Yaw={yaw:+.3f}°")
    print(f"Optimal translation: tx={tx*1000:+.1f}mm  ty={ty*1000:+.1f}mm  tz={tz*1000:+.1f}mm")
    print(f"Optimal total angle: {total_angle:.4f}°  (delta={total_angle-360:.4f}°)")

    # ── 4. Final measurement (full resolution) ────────────────────────────
    R_opt = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()
    t_opt = np.array([tx, ty, tz])
    theta_opt = theta * (total_angle / 360.0)
    x_fo, y_fo, z_fo = apply_transform(x_L, y_L, z_L, theta_opt, R_opt, t_opt)
    print("\n--- Results (full resolution) ---")
    d_opt = measure_distance(x_fo, y_fo, z_fo, mask_f, mask_b, "Optimized")
    improvement = (d_baseline - d_opt) / d_baseline * 100
    print(f"Improvement: {improvement:.1f}%  "
          f"({d_baseline*1000:.1f} mm → {d_opt*1000:.1f} mm)")
    print(f"\nTo replay with these parameters:")
    print(f"  --mount-correction {roll:.3f} {pitch:.3f} {yaw:.3f} "
          f"--mount-translation {tx:.4f} {ty:.4f} {tz:.4f} "
          f"--total-angle {total_angle:.4f}")

    # ── 5. Write PCD files ────────────────────────────────────────────────
    tag = (f"cb{ch_b}_cf{ch_f}"
           f"_r{roll:.2f}_p{pitch:.2f}_y{yaw:.2f}"
           f"_tx{tx*100:.1f}_ty{ty*100:.1f}_tz{tz*100:.1f}"
           f"_ang{total_angle:.2f}")
    write_result(x_L, y_L, z_L, theta_opt, intens, mask_f, mask_b, R_opt, t_opt,
                 args.output_dir, tag)
    print(f"Written {args.output_dir}/opt_front_{tag}.pcd")
    print(f"Written {args.output_dir}/opt_back_{tag}.pcd")


if __name__ == "__main__":
    main()
