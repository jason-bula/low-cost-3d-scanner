#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy", "scipy", "pyserial"]
# ///
"""Optimize LiDAR mounting correction by minimizing point cloud roughness.

Pipeline:
  1. Parse PCAP once → raw LiDAR coords (x_L, y_L, z_L) + motor angles θ
  2. Nelder-Mead loop: apply R_mount(roll,pitch,yaw) + t_mount(tx,ty,tz) + Ry(-θ),
     compute roughness on the combined point cloud via local PCA
  3. Write optimized PCD file

Roughness metric: for each query point, find k nearest neighbors, compute the
local covariance matrix, and take its smallest eigenvalue λ₀.  A perfectly flat
surface gives λ₀ = 0.  Misalignment produces double-wall artifacts → large λ₀.
The objective is median(λ₀) over a fixed random subsample of the cloud.

Usage:
    uv run optimize_roughness.py
    uv run optimize_roughness.py --init-angles -0.871 0.248 -1.264 --init-translation 0.0052 -0.0064 -0.0096
    uv run optimize_roughness.py --channel 10 11 12 --n-opt 30000 --k 15
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
    PCD_DTYPE, load_calibration, write_pcd,
)
from replay_scan import read_pcap
from optimize_angles import parse_raw, apply_transform

# ── Defaults ───────────────────────────────────────────────────────────────
PCAP         = 'raw/scan_360_6min.pcap'
TOTAL_ANGLE  = 360.0
CHANNEL      = None          # None = all 40 channels
MIN_DIST     = 2.0
BLOCK_INTERP = True
MAX_DEG      = 5.0
MAX_SHIFT    = 0.02          # ±2 cm per translation axis
MAX_ANGLE_DELTA = 1.0        # ±1° on total rotation angle
K_NEIGHBORS  = 15            # neighbors for local PCA
N_OPT        = 20_000        # points used in optimization loop (tree + query)
RANDOM_SEED  = 42


# ── Roughness metric ───────────────────────────────────────────────────────

def roughness(x_f, y_f, z_f, idx_sample, k):
    """Median smallest PCA eigenvalue over idx_sample points.

    Builds a KDTree on the sample itself, queries k neighbors per point,
    computes the 3×3 local covariance, returns median(λ₀) in meters².
    """
    pts = np.stack([x_f[idx_sample], y_f[idx_sample], z_f[idx_sample]], axis=1)

    tree = cKDTree(pts)
    _, nn_idx = tree.query(pts, k=k + 1, workers=-1)  # +1: first hit = self
    nn_idx = nn_idx[:, 1:]                             # drop self

    neighbors = pts[nn_idx]                            # (N, k, 3)
    centered  = neighbors - neighbors.mean(axis=1, keepdims=True)
    cov       = np.einsum('nki,nkj->nij', centered, centered) / k  # (N, 3, 3)
    eigvals   = np.linalg.eigvalsh(cov)                # (N, 3), ascending

    return float(np.median(eigvals[:, 0]))


# ── Objective ──────────────────────────────────────────────────────────────

def make_objective(x_L, y_L, z_L, theta, idx_sample, k,
                   max_deg, max_shift, max_angle_delta):
    """Build Nelder-Mead objective.

    params: [roll_deg, pitch_deg, yaw_deg, tx_m, ty_m, tz_m, total_angle_deg]
    """
    _state = {'n': 0, 't0': time.monotonic()}

    def objective(params):
        roll_deg, pitch_deg, yaw_deg, tx, ty, tz, total_angle = params
        _state['n'] += 1

        if max(abs(roll_deg), abs(pitch_deg), abs(yaw_deg)) > max_deg:
            return 1e6
        if max(abs(tx), abs(ty), abs(tz)) > max_shift:
            return 1e6
        if abs(total_angle - 360.0) > max_angle_delta:
            return 1e6

        theta_s = theta * (total_angle / 360.0)
        R = Rotation.from_euler('xyz', [roll_deg, pitch_deg, yaw_deg],
                                degrees=True).as_matrix()
        t = np.array([tx, ty, tz])
        x_f, y_f, z_f = apply_transform(x_L, y_L, z_L, theta_s, R, t)

        cost = roughness(x_f, y_f, z_f, idx_sample, k)

        n = _state['n']
        if n % 10 == 0 or n == 1:
            elapsed = time.monotonic() - _state['t0']
            print(f"  [{n:4d}] roll={roll_deg:+.3f}° pitch={pitch_deg:+.3f}° "
                  f"yaw={yaw_deg:+.3f}°  tx={tx*1000:+.1f}mm ty={ty*1000:+.1f}mm "
                  f"tz={tz*1000:+.1f}mm  Δangle={total_angle-360:+.3f}° "
                  f"→ λ₀={cost*1e6:.2f} µm²  ({elapsed:.0f}s)")

        return cost

    return objective


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Optimize LiDAR mounting by minimizing point cloud roughness")
    ap.add_argument("--pcap",             default=PCAP)
    ap.add_argument("--total-angle",      type=float, default=TOTAL_ANGLE)
    ap.add_argument("--channel",          type=int, nargs='+', default=CHANNEL,
                    help="channels to use (default: all 40)")
    ap.add_argument("--min-dist",         type=float, default=MIN_DIST)
    ap.add_argument("--max-dist",         type=float, default=0.0,
                    help="discard points farther than this distance in meters (0 = no limit)")
    ap.add_argument("--max-deg",          type=float, default=MAX_DEG)
    ap.add_argument("--max-shift",        type=float, default=MAX_SHIFT,
                    help="max translation per axis (m)")
    ap.add_argument("--k",                type=int, default=K_NEIGHBORS,
                    help="neighbors for local PCA (default: 15)")
    ap.add_argument("--n-opt",            type=int, default=N_OPT,
                    help="points used in optimization loop (default: 20000)")
    ap.add_argument("--subsample",        type=int, default=1,
                    help="parse 1/N packets (pre-filter, default: 1 = all)")
    ap.add_argument("--output-dir",       default="scans")
    ap.add_argument("--ip",               default="192.168.20.13")
    ap.add_argument("--init-angles",      type=float, nargs=3,
                    metavar=('ROLL', 'PITCH', 'YAW'), default=[0.0, 0.0, 0.0])
    ap.add_argument("--init-translation", type=float, nargs=3,
                    metavar=('TX', 'TY', 'TZ'), default=[0.0, 0.0, 0.0])
    ap.add_argument("--max-angle-delta", type=float, default=MAX_ANGLE_DELTA,
                    help="max deviation from total-angle (°, default: 1.0)")
    ap.add_argument("--init-total-angle", type=float, default=None,
                    help="initial total angle (°, default: same as --total-angle)")
    args = ap.parse_args()

    channels = args.channel  # None → all 40 inside parse_raw
    if channels is None:
        from record_lidar import CHANNELS
        channels = list(range(CHANNELS))

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Parse ──────────────────────────────────────────────────────────
    cal = load_calibration(args.ip)
    packets, t_start = read_pcap(args.pcap)
    if args.subsample > 1:
        packets = packets[::args.subsample]
        print(f"Pre-subsampled packets: {len(packets):,} (1/{args.subsample})")
    duration = packets[-1][0] - packets[0][0]
    rpm = args.total_angle / 6.0 / duration
    print(f"Total angle {args.total_angle}° over {duration:.2f}s → {rpm:.4f} RPM")
    print(f"Channels: {channels}  ({len(channels)} total)")

    x_L, y_L, z_L, theta, intens, _ = parse_raw(
        packets, t_start, cal, rpm,
        channels, BLOCK_INTERP, args.min_dist, args.max_dist,
    )
    N = len(x_L)
    print(f"Total points: {N:,}")

    # Fixed random subsample used every iteration
    rng = np.random.RandomState(RANDOM_SEED)
    n_opt = min(args.n_opt, N)
    idx_sample = rng.choice(N, size=n_opt, replace=False)
    print(f"Optimization sample: {n_opt:,} points, k={args.k} neighbors")

    # ── 2. Baseline ───────────────────────────────────────────────────────
    R_id = np.eye(3)
    x_f0, y_f0, z_f0 = apply_transform(x_L, y_L, z_L, theta, R_id)
    r0 = roughness(x_f0, y_f0, z_f0, idx_sample, args.k)
    print(f"\n--- Baseline ---")
    print(f"Roughness λ₀ = {r0*1e6:.2f} µm²  (std ≈ {np.sqrt(r0)*1000:.2f} mm)")

    # ── 3. Optimize ───────────────────────────────────────────────────────
    init_total_angle = args.init_total_angle if args.init_total_angle is not None \
                       else args.total_angle
    x0 = np.array(args.init_angles + args.init_translation + [init_total_angle])
    print(f"\nOptimizing (bounds ±{args.max_deg}° / ±{args.max_shift*100:.0f}cm"
          f" / total_angle ±{args.max_angle_delta}°) ...")
    print(f"Initial: roll={x0[0]:+.3f}° pitch={x0[1]:+.3f}° yaw={x0[2]:+.3f}°  "
          f"tx={x0[3]*1000:+.1f}mm ty={x0[4]*1000:+.1f}mm tz={x0[5]*1000:+.1f}mm  "
          f"total_angle={x0[6]:.3f}°")

    obj = make_objective(x_L, y_L, z_L, theta, idx_sample, args.k,
                         args.max_deg, args.max_shift, args.max_angle_delta)

    t0 = time.monotonic()
    result = minimize(
        obj, x0=x0, method='Nelder-Mead',
        options={
            'xatol':   1e-4,
            'fatol':   1e-8,
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
    print(f"Optimal total_angle: {total_angle:.4f}°  (Δ={total_angle-360:+.4f}°)")

    # ── 4. Final measurement (full resolution) ────────────────────────────
    theta_opt = theta * (total_angle / 360.0)
    R_opt = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()
    t_opt = np.array([tx, ty, tz])
    x_fo, y_fo, z_fo = apply_transform(x_L, y_L, z_L, theta_opt, R_opt, t_opt)
    r_opt = roughness(x_fo, y_fo, z_fo, idx_sample, args.k)

    print(f"\n--- Results ---")
    print(f"Baseline:  λ₀ = {r0*1e6:.2f} µm²  (≈ {np.sqrt(r0)*1000:.2f} mm)")
    print(f"Optimized: λ₀ = {r_opt*1e6:.2f} µm²  (≈ {np.sqrt(r_opt)*1000:.2f} mm)")
    improvement = (r0 - r_opt) / r0 * 100
    print(f"Improvement: {improvement:.1f}%")
    print(f"\nTo replay with these parameters:")
    print(f"  --total-angle {total_angle:.4f} --mount-correction {roll:.3f} {pitch:.3f} {yaw:.3f} "
          f"--mount-translation {tx:.4f} {ty:.4f} {tz:.4f}")

    # ── 5. Write PCD ──────────────────────────────────────────────────────
    tag = (f"rough_r{roll:.2f}_p{pitch:.2f}_y{yaw:.2f}"
           f"_tx{tx*100:.1f}_ty{ty*100:.1f}_tz{tz*100:.1f}")
    fn = os.path.join(args.output_dir, f"opt_{tag}.pcd")
    pts = np.empty(N, dtype=PCD_DTYPE)
    pts['x']         = x_fo.astype(np.float32)
    pts['y']         = y_fo.astype(np.float32)
    pts['z']         = z_fo.astype(np.float32)
    pts['intensity'] = intens
    write_pcd(fn, pts)
    print(f"Written {fn}")


if __name__ == "__main__":
    main()
