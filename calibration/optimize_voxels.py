#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy", "scipy", "pyserial"]
# ///
"""Optimize LiDAR mounting correction by maximizing voxel point density.

Metric: -sum(N_v²) / N²  where N_v = points per occupied voxel.

A well-calibrated cloud has front and back scans overlapping on the same
surfaces → many points per voxel → large sum(N_v²).
A miscalibrated cloud has double-wall artifacts → points split across
adjacent voxels → fewer points per voxel → smaller sum(N_v²).

Mathematically: for a surface with N_total points split into two copies
of N/2 (miscalibrated), sum(N_v²) ≈ N²/2.
With perfect overlap (calibrated), sum(N_v²) ≈ N².
The metric improves 2× from worst to best case.

Advantages over other methods:
  - No KDTree, no plane detection → extremely fast per call (~0.1 ms)
  - Potentially works cold-start (unlike v2 roughness and v3 planes)
  - Smooth-ish: with N=20k points many cross voxel boundaries per step
  - Sensitive to voxel_size: use ~half the expected wall separation

Voxel size tuning:
  - Too large (> wall separation): both walls in same voxel → no signal
  - Too small (< point spacing): each point alone → no signal
  - Optimal: ~half the expected residual (default 1 cm for ~2 cm errors)

Usage:
    uv run optimize_voxels.py --channel $(seq 10 30)
    uv run optimize_voxels.py --channel $(seq 10 30) --voxel-size 0.015
    uv run optimize_voxels.py --channel $(seq 10 30) --init-angles -0.871 0.248 -1.264 --init-translation 0.0052 -0.0064 -0.0096
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
from scipy.spatial.transform import Rotation

from record_lidar import PCD_DTYPE, load_calibration, write_pcd
from replay_scan import read_pcap
from optimize_angles import parse_raw, apply_transform

# ── Defaults ───────────────────────────────────────────────────────────────
PCAP            = 'raw/scan_360_6min.pcap'
TOTAL_ANGLE     = 360.0
CHANNEL         = None
MIN_DIST        = 2.0
BLOCK_INTERP    = True
MAX_DEG         = 5.0
MAX_SHIFT       = 0.02
VOXEL_SIZE      = 0.01   # 1 cm — half of the ~2 cm residual error
N_OPT           = 20_000
RANDOM_SEED     = 42


# ── Voxel density metric ───────────────────────────────────────────────────

def voxel_density(pts, voxel_size):
    """Return -sum(N_v²) / N² (minimise → maximise point clustering).

    pts        : (N, 3) float array, already subsampled.
    voxel_size : grid cell size in meters.

    Uses np.unique on compact int64 voxel indices — O(N log N), no large
    histogram allocation, O(N) memory.
    """
    idx = np.floor(pts / voxel_size).astype(np.int64)
    # Shift to non-negative (avoids sign issues in flat index)
    idx -= idx.min(axis=0)
    dims = idx.max(axis=0) + 1
    flat = idx[:, 0] * dims[1] * dims[2] + idx[:, 1] * dims[2] + idx[:, 2]
    _, counts = np.unique(flat, return_counts=True)
    N = len(pts)
    return -float(np.dot(counts, counts)) / (N * N)


# ── Objective ──────────────────────────────────────────────────────────────

def make_objective(x_Ls, y_Ls, z_Ls, theta_s, voxel_size, max_deg, max_shift):
    """Build Nelder-Mead objective.

    x_Ls/y_Ls/z_Ls/theta_s : pre-extracted subsample arrays.
    params: [roll_deg, pitch_deg, yaw_deg, tx_m, ty_m, tz_m]
    """
    _state = {'n': 0, 't0': time.monotonic()}

    def objective(params):
        roll_deg, pitch_deg, yaw_deg, tx, ty, tz = params
        _state['n'] += 1

        if max(abs(roll_deg), abs(pitch_deg), abs(yaw_deg)) > max_deg:
            return 0.0   # worst possible (we minimise a negative number)
        if max(abs(tx), abs(ty), abs(tz)) > max_shift:
            return 0.0

        R = Rotation.from_euler('xyz', [roll_deg, pitch_deg, yaw_deg],
                                degrees=True).as_matrix()
        t = np.array([tx, ty, tz])
        x_f, y_f, z_f = apply_transform(x_Ls, y_Ls, z_Ls, theta_s, R, t)
        pts = np.stack([x_f, y_f, z_f], axis=1)

        cost = voxel_density(pts, voxel_size)

        n = _state['n']
        if n % 20 == 0 or n == 1:
            elapsed = time.monotonic() - _state['t0']
            print(f"  [{n:4d}] roll={roll_deg:+.3f}° pitch={pitch_deg:+.3f}° "
                  f"yaw={yaw_deg:+.3f}°  tx={tx*1000:+.1f}mm ty={ty*1000:+.1f}mm "
                  f"tz={tz*1000:+.1f}mm → density={-cost:.4f}  ({elapsed:.1f}s)")
        return cost

    return objective


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Optimize LiDAR mounting by maximizing voxel point density")
    ap.add_argument("--pcap",             default=PCAP)
    ap.add_argument("--total-angle",      type=float, default=TOTAL_ANGLE)
    ap.add_argument("--channel",          type=int, nargs='+', default=CHANNEL)
    ap.add_argument("--min-dist",         type=float, default=MIN_DIST)
    ap.add_argument("--max-dist",         type=float, default=0.0)
    ap.add_argument("--max-deg",          type=float, default=MAX_DEG)
    ap.add_argument("--max-shift",        type=float, default=MAX_SHIFT)
    ap.add_argument("--voxel-size",       type=float, default=VOXEL_SIZE,
                    help="voxel grid cell size in meters (default: 0.01 = 1 cm)")
    ap.add_argument("--n-opt",            type=int, default=N_OPT)
    ap.add_argument("--subsample",        type=int, default=1)
    ap.add_argument("--output-dir",       default="scans")
    ap.add_argument("--ip",               default="192.168.20.13")
    ap.add_argument("--init-angles",      type=float, nargs=3,
                    metavar=('ROLL', 'PITCH', 'YAW'), default=[0.0, 0.0, 0.0])
    ap.add_argument("--init-translation", type=float, nargs=3,
                    metavar=('TX', 'TY', 'TZ'), default=[0.0, 0.0, 0.0])
    args = ap.parse_args()

    channels = args.channel
    if channels is None:
        from record_lidar import CHANNELS
        channels = list(range(CHANNELS))

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.RandomState(RANDOM_SEED)

    # ── 1. Parse ──────────────────────────────────────────────────────────
    cal = load_calibration(args.ip)
    packets, t_start = read_pcap(args.pcap)
    if args.subsample > 1:
        packets = packets[::args.subsample]
        print(f"Pre-subsampled: {len(packets):,} packets (1/{args.subsample})")
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

    # Pre-extract subsample
    n_opt = min(args.n_opt, N)
    idx_sample = rng.choice(N, size=n_opt, replace=False)
    x_Ls = x_L[idx_sample]; y_Ls = y_L[idx_sample]
    z_Ls = z_L[idx_sample]; theta_s = theta[idx_sample]
    print(f"Optimization sample: {n_opt:,} points  voxel_size={args.voxel_size*100:.1f}cm")

    # ── 2. Baseline ───────────────────────────────────────────────────────
    pts_zero = np.stack(apply_transform(x_Ls, y_Ls, z_Ls, theta_s, np.eye(3)), axis=1)
    d0 = voxel_density(pts_zero, args.voxel_size)

    R_init = Rotation.from_euler('xyz', args.init_angles, degrees=True).as_matrix()
    t_init = np.array(args.init_translation)
    pts_init = np.stack(apply_transform(x_Ls, y_Ls, z_Ls, theta_s, R_init, t_init), axis=1)
    d_ini = voxel_density(pts_init, args.voxel_size)

    print(f"\n--- Baseline ---")
    print(f"No correction:  density = {-d0:.4f}  (unique voxels ≈ {n_opt / (-d0 * n_opt**2 / n_opt + 1e-9):.0f})")
    print(f"Warm-start:     density = {-d_ini:.4f}")

    # ── 3. Optimize ───────────────────────────────────────────────────────
    x0 = np.array(args.init_angles + args.init_translation)
    print(f"\nOptimizing (bounds ±{args.max_deg}° / ±{args.max_shift*100:.0f}cm) ...")
    print(f"Initial: roll={x0[0]:+.3f}° pitch={x0[1]:+.3f}° yaw={x0[2]:+.3f}°  "
          f"tx={x0[3]*1000:+.1f}mm ty={x0[4]*1000:+.1f}mm tz={x0[5]*1000:+.1f}mm")

    obj = make_objective(x_Ls, y_Ls, z_Ls, theta_s,
                         args.voxel_size, args.max_deg, args.max_shift)

    t0 = time.monotonic()
    result = minimize(
        obj, x0=x0, method='Nelder-Mead',
        options={
            'xatol':   1e-4,
            'fatol':   1e-7,
            'maxiter': 5000,
            'adaptive': True,
            'disp':    False,
        },
    )
    print(f"\nOptimization finished in {time.monotonic() - t0:.1f}s "
          f"({result.nit} iterations, success={result.success})")

    roll, pitch, yaw, tx, ty, tz = result.x
    print(f"Optimal rotation:    Roll={roll:+.3f}°  Pitch={pitch:+.3f}°  Yaw={yaw:+.3f}°")
    print(f"Optimal translation: tx={tx*1000:+.1f}mm  ty={ty*1000:+.1f}mm  tz={tz*1000:+.1f}mm")

    # ── 4. Final measurement ──────────────────────────────────────────────
    R_opt = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()
    t_opt = np.array([tx, ty, tz])
    pts_opt = np.stack(apply_transform(x_Ls, y_Ls, z_Ls, theta_s, R_opt, t_opt), axis=1)
    d_opt = voxel_density(pts_opt, args.voxel_size)

    improvement = (-d_opt - (-d0)) / (-d0) * 100
    print(f"\n--- Results ---")
    print(f"No correction:  density = {-d0:.4f}")
    print(f"Warm-start:     density = {-d_ini:.4f}")
    print(f"Optimized:      density = {-d_opt:.4f}")
    print(f"Improvement vs baseline: {improvement:.1f}%")
    print(f"\nTo replay with these parameters:")
    print(f"  --mount-correction {roll:.3f} {pitch:.3f} {yaw:.3f} "
          f"--mount-translation {tx:.4f} {ty:.4f} {tz:.4f}")

    # ── 5. Write PCD (full cloud) ──────────────────────────────────────────
    x_fo, y_fo, z_fo = apply_transform(x_L, y_L, z_L, theta, R_opt, t_opt)
    tag = (f"vox_r{roll:.2f}_p{pitch:.2f}_y{yaw:.2f}"
           f"_tx{tx*100:.1f}_ty{ty*100:.1f}_tz{tz*100:.1f}")
    fn = os.path.join(args.output_dir, f"opt_{tag}.pcd")
    pts_out = np.empty(N, dtype=PCD_DTYPE)
    pts_out['x']         = x_fo.astype(np.float32)
    pts_out['y']         = y_fo.astype(np.float32)
    pts_out['z']         = z_fo.astype(np.float32)
    pts_out['intensity'] = intens
    write_pcd(fn, pts_out)
    print(f"Written {fn}")


if __name__ == "__main__":
    main()
