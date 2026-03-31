#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy", "scipy", "pyserial"]
# ///
"""Optimize LiDAR mounting correction by minimizing point-to-plane distances.

Pipeline:
  1. Parse PCAP once → raw LiDAR coords (x_L, y_L, z_L) + motor angles θ
  2. Apply warm-start transform (or identity) → detect dominant planes via RANSAC
  3. Nelder-Mead loop: apply R_mount(roll,pitch,yaw) + t_mount(tx,ty,tz) + Ry(-θ),
     compute median point-to-nearest-plane distance on a pre-extracted subsample
  4. Write optimized PCD file

Speed note: apply_transform runs only on the N_OPT subsample (not on all points),
making each objective call ~1800× faster than operating on the full 36M-point cloud.

Plane metric: planes are detected once (from the initial cloud) and stored as
fixed (normal, offset) pairs in world space. Each iteration, every sampled point
is assigned to its nearest plane and the median distance is returned (in mm).

Usage:
    uv run optimize_planes.py
    uv run optimize_planes.py --channel $(seq 10 30) --init-angles -0.871 0.248 -1.264 --init-translation 0.0052 -0.0064 -0.0096
    uv run optimize_planes.py --n-planes 8 --ransac-thresh 0.03
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
N_PLANES        = 6
RANSAC_ITER     = 500
RANSAC_THRESH   = 0.02
MIN_INLIERS     = 200
N_OPT           = 20_000
RANDOM_SEED     = 42


# ── RANSAC plane detection ─────────────────────────────────────────────────

def ransac_planes(pts, n_planes, n_iter, thresh, min_inliers, rng):
    """Detect up to n_planes dominant planes via RANSAC + SVD refit.

    Sequential RANSAC: each detected plane's inliers are removed before
    searching for the next one.

    Returns list of (normal, offset) tuples where  normal · x = offset.
    """
    remaining = pts.copy()
    planes = []

    for plane_i in range(n_planes):
        if len(remaining) < max(min_inliers, 3):
            break

        best_n, best_d, best_mask, best_count = None, 0.0, None, 0

        for _ in range(n_iter):
            idx3 = rng.choice(len(remaining), 3, replace=False)
            p0, p1, p2 = remaining[idx3]
            n = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(n)
            if norm < 1e-6:
                continue
            n /= norm
            d = float(n @ p0)
            mask = np.abs(remaining @ n - d) < thresh
            count = int(mask.sum())
            if count > best_count:
                best_count, best_n, best_d, best_mask = count, n.copy(), d, mask.copy()

        if best_n is None or best_count < min_inliers:
            print(f"  Plane {plane_i + 1}: {best_count} inliers < {min_inliers}, stopping")
            break

        # SVD refit on all inliers
        inliers  = remaining[best_mask]
        centroid = inliers.mean(axis=0)
        _, _, Vt = np.linalg.svd(inliers - centroid, full_matrices=False)
        n_fit = Vt[-1]
        n_fit *= np.sign(n_fit[np.argmax(np.abs(n_fit))])  # canonical sign
        d_fit = float(n_fit @ centroid)

        resid = np.abs(inliers @ n_fit - d_fit).mean()
        print(f"  Plane {plane_i + 1}: {best_count:,} inliers  "
              f"n=({n_fit[0]:+.3f},{n_fit[1]:+.3f},{n_fit[2]:+.3f})  "
              f"d={d_fit:+.2f}m  residual={resid*1000:.1f}mm")

        planes.append((n_fit, d_fit))
        remaining = remaining[~best_mask]

    return planes


# ── Plane metric ───────────────────────────────────────────────────────────

def plane_metric(pts, planes):
    """Median point-to-nearest-plane distance.

    pts : (N, 3) array (already subsampled — no indexing done here).
    Returns median of min-distance-to-any-plane in meters.
    """
    min_dists = np.full(len(pts), np.inf)
    for n, d in planes:
        np.minimum(min_dists, np.abs(pts @ n - d), out=min_dists)
    return float(np.median(min_dists))


# ── Objective ──────────────────────────────────────────────────────────────

def make_objective(x_Ls, y_Ls, z_Ls, theta_s, planes, max_deg, max_shift):
    """Build Nelder-Mead objective.

    x_Ls/y_Ls/z_Ls/theta_s are pre-extracted subsample arrays → fast per call.
    params: [roll_deg, pitch_deg, yaw_deg, tx_m, ty_m, tz_m]
    """
    _state = {'n': 0, 't0': time.monotonic()}

    def objective(params):
        roll_deg, pitch_deg, yaw_deg, tx, ty, tz = params
        _state['n'] += 1

        if max(abs(roll_deg), abs(pitch_deg), abs(yaw_deg)) > max_deg:
            return 1e6
        if max(abs(tx), abs(ty), abs(tz)) > max_shift:
            return 1e6

        R = Rotation.from_euler('xyz', [roll_deg, pitch_deg, yaw_deg],
                                degrees=True).as_matrix()
        t = np.array([tx, ty, tz])
        x_f, y_f, z_f = apply_transform(x_Ls, y_Ls, z_Ls, theta_s, R, t)
        pts = np.stack([x_f, y_f, z_f], axis=1)

        cost = plane_metric(pts, planes)

        n = _state['n']
        if n % 10 == 0 or n == 1:
            elapsed = time.monotonic() - _state['t0']
            print(f"  [{n:4d}] roll={roll_deg:+.3f}° pitch={pitch_deg:+.3f}° "
                  f"yaw={yaw_deg:+.3f}°  tx={tx*1000:+.1f}mm ty={ty*1000:+.1f}mm "
                  f"tz={tz*1000:+.1f}mm → d={cost*1000:.2f}mm  ({elapsed:.0f}s)")
        return cost

    return objective


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Optimize LiDAR mounting by minimizing point-to-plane distances")
    ap.add_argument("--pcap",             default=PCAP)
    ap.add_argument("--total-angle",      type=float, default=TOTAL_ANGLE)
    ap.add_argument("--channel",          type=int, nargs='+', default=CHANNEL)
    ap.add_argument("--min-dist",         type=float, default=MIN_DIST)
    ap.add_argument("--max-dist",         type=float, default=0.0)
    ap.add_argument("--max-deg",          type=float, default=MAX_DEG)
    ap.add_argument("--max-shift",        type=float, default=MAX_SHIFT)
    ap.add_argument("--n-planes",         type=int, default=N_PLANES)
    ap.add_argument("--ransac-iter",      type=int, default=RANSAC_ITER)
    ap.add_argument("--ransac-thresh",    type=float, default=RANSAC_THRESH,
                    help="RANSAC inlier threshold (m, default: 0.02)")
    ap.add_argument("--min-inliers",      type=int, default=MIN_INLIERS)
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

    # Pre-extract subsample — only these points are transformed each iteration
    n_opt = min(args.n_opt, N)
    idx_sample = rng.choice(N, size=n_opt, replace=False)
    x_Ls, y_Ls, z_Ls, theta_s = (x_L[idx_sample], y_L[idx_sample],
                                   z_L[idx_sample], theta[idx_sample])
    print(f"Optimization sample: {n_opt:,} points")

    # ── 2. Apply warm-start → detect planes ───────────────────────────────
    R_init = Rotation.from_euler('xyz', args.init_angles, degrees=True).as_matrix()
    t_init = np.array(args.init_translation)
    x_i, y_i, z_i = apply_transform(x_Ls, y_Ls, z_Ls, theta_s, R_init, t_init)
    pts_init = np.stack([x_i, y_i, z_i], axis=1)

    print(f"\n--- RANSAC plane detection ---")
    print(f"  {args.n_planes} planes, {args.ransac_iter} iter, "
          f"thresh={args.ransac_thresh*100:.1f}cm, min_inliers={args.min_inliers}")
    t_r = time.monotonic()
    planes = ransac_planes(pts_init, args.n_planes, args.ransac_iter,
                           args.ransac_thresh, args.min_inliers, rng)
    print(f"Detected {len(planes)} planes in {time.monotonic() - t_r:.1f}s")

    if not planes:
        print("ERROR: no planes detected. Try --min-inliers lower or --ransac-thresh higher.")
        return

    # ── 3. Baseline ───────────────────────────────────────────────────────
    pts_zero = np.stack(apply_transform(x_Ls, y_Ls, z_Ls, theta_s, np.eye(3)), axis=1)
    r0    = plane_metric(pts_zero,  planes)
    r_ini = plane_metric(pts_init,  planes)
    print(f"\n--- Baseline ---")
    print(f"No correction:  d = {r0*1000:.2f} mm")
    print(f"Warm-start:     d = {r_ini*1000:.2f} mm")

    # ── 4. Optimize ───────────────────────────────────────────────────────
    x0 = np.array(args.init_angles + args.init_translation)
    print(f"\nOptimizing (bounds ±{args.max_deg}° / ±{args.max_shift*100:.0f}cm) ...")
    print(f"Initial: roll={x0[0]:+.3f}° pitch={x0[1]:+.3f}° yaw={x0[2]:+.3f}°  "
          f"tx={x0[3]*1000:+.1f}mm ty={x0[4]*1000:+.1f}mm tz={x0[5]*1000:+.1f}mm")

    obj = make_objective(x_Ls, y_Ls, z_Ls, theta_s, planes, args.max_deg, args.max_shift)

    t0 = time.monotonic()
    result = minimize(
        obj, x0=x0, method='Nelder-Mead',
        options={
            'xatol':   1e-4,
            'fatol':   1e-5,
            'maxiter': 3000,
            'adaptive': True,
            'disp':    False,
        },
    )
    print(f"\nOptimization finished in {time.monotonic() - t0:.1f}s "
          f"({result.nit} iterations, success={result.success})")

    roll, pitch, yaw, tx, ty, tz = result.x
    print(f"Optimal rotation:    Roll={roll:+.3f}°  Pitch={pitch:+.3f}°  Yaw={yaw:+.3f}°")
    print(f"Optimal translation: tx={tx*1000:+.1f}mm  ty={ty*1000:+.1f}mm  tz={tz*1000:+.1f}mm")

    # ── 5. Final measurement ──────────────────────────────────────────────
    R_opt = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()
    t_opt = np.array([tx, ty, tz])
    pts_opt = np.stack(apply_transform(x_Ls, y_Ls, z_Ls, theta_s, R_opt, t_opt), axis=1)
    r_opt = plane_metric(pts_opt, planes)

    print(f"\n--- Results ---")
    print(f"No correction:  d = {r0*1000:.2f} mm")
    print(f"Warm-start:     d = {r_ini*1000:.2f} mm")
    print(f"Optimized:      d = {r_opt*1000:.2f} mm")
    print(f"Improvement vs baseline: {(r0 - r_opt) / r0 * 100:.1f}%")
    print(f"\nTo replay with these parameters:")
    print(f"  --mount-correction {roll:.3f} {pitch:.3f} {yaw:.3f} "
          f"--mount-translation {tx:.4f} {ty:.4f} {tz:.4f}")

    # ── 6. Write PCD (full cloud) ──────────────────────────────────────────
    x_fo, y_fo, z_fo = apply_transform(x_L, y_L, z_L, theta, R_opt, t_opt)
    tag = (f"planes_r{roll:.2f}_p{pitch:.2f}_y{yaw:.2f}"
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
