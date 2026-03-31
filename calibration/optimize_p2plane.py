#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy", "scipy", "pyserial"]
# ///
"""Optimize LiDAR mounting via point-to-plane ICP between front/back half-scans.

Pipeline:
  1. Parse PCAP once → raw LiDAR coords + motor angles
  2. Split into front half (x_L > 0) and back half (x_L < 0)
  3. Nelder-Mead loop: apply mounting transform to both halves,
     for each back point find nearest front neighbour and compute
     the point-to-plane distance using the front's local surface normal.
  4. Write optimized PCD

Why point-to-plane over point-to-point (v1)?
  Point-to-point distance penalises tangential sliding along a surface,
  creating a flat landscape in those directions.  Point-to-plane ignores
  tangential motion and focuses only on the normal direction → steeper
  gradient, faster Nelder-Mead convergence, same accuracy.

Both halves receive the same mounting transform each iteration so the
metric is symmetric and cold-start capable.

Normals are estimated once from the warm-started front cloud (PCA of
k nearest neighbours) and kept fixed throughout optimisation.  Near the
warm-start this is an excellent approximation; the bias is second-order
in the parameter change.

Usage:
    uv run optimize_p2plane.py --channel $(seq 10 30)
    uv run optimize_p2plane.py --channel $(seq 10 30) --init-angles -0.871 0.248 -1.264 --init-translation 0.0052 -0.0064 -0.0096
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
K_NORMAL        = 15    # neighbours for PCA normal estimation
N_FRONT         = 5_000  # front subsample size (KDTree + normals)
N_BACK          = 15_000 # back subsample size  (queries)
RANDOM_SEED     = 42


# ── Point-to-plane metric ──────────────────────────────────────────────────

def compute_normals(pts, k):
    """PCA surface normals for each point in pts via k nearest neighbours."""
    tree = cKDTree(pts)
    _, nn_idx = tree.query(pts, k=k + 1, workers=-1)
    nn_idx = nn_idx[:, 1:]                              # drop self
    nbrs = pts[nn_idx]                                  # (N, k, 3)
    centered = nbrs - nbrs.mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', centered, centered) / k  # (N, 3, 3)
    return np.linalg.eigh(cov)[1][:, :, 0], tree       # smallest eigenvec


def p2plane_metric(pts_b, tree_f, pts_f, normals_f):
    """Median point-to-plane distance: back cloud → fixed front reference.

    pts_b     : (N_b, 3) transformed back-half points
    tree_f    : cKDTree built on the reference front cloud (fixed)
    pts_f     : (N_f, 3) reference front cloud points (fixed)
    normals_f : (N_f, 3) surface normals of the reference front cloud (fixed)
    """
    _, idx = tree_f.query(pts_b, k=1, workers=-1)
    diffs  = pts_b - pts_f[idx]
    p2pl   = np.abs(np.einsum('ni,ni->n', diffs, normals_f[idx]))
    return float(np.median(p2pl))


# ── Objective ──────────────────────────────────────────────────────────────

def make_objective(x_Ls_b, y_Ls_b, z_Ls_b, theta_s_b,
                   tree_f, pts_f_ref, normals_f,
                   max_deg, max_shift):
    """Build Nelder-Mead objective.

    The front reference (tree_f, pts_f_ref, normals_f) is fixed from the
    warm-start transform.  Only the back subsample is transformed each call.

    params: [roll_deg, pitch_deg, yaw_deg, tx_m, ty_m, tz_m]
    """
    _state = {'n': 0, 't0': time.monotonic()}

    def objective(params):
        roll, pitch, yaw, tx, ty, tz = params
        _state['n'] += 1

        if max(abs(roll), abs(pitch), abs(yaw)) > max_deg:
            return 1e6
        if max(abs(tx), abs(ty), abs(tz)) > max_shift:
            return 1e6

        R = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()
        t = np.array([tx, ty, tz])
        pts_b = np.stack(
            apply_transform(x_Ls_b, y_Ls_b, z_Ls_b, theta_s_b, R, t), axis=1)

        cost = p2plane_metric(pts_b, tree_f, pts_f_ref, normals_f)

        n = _state['n']
        if n % 10 == 0 or n == 1:
            elapsed = time.monotonic() - _state['t0']
            print(f"  [{n:4d}] roll={roll:+.3f}° pitch={pitch:+.3f}° "
                  f"yaw={yaw:+.3f}°  tx={tx*1000:+.1f}mm ty={ty*1000:+.1f}mm "
                  f"tz={tz*1000:+.1f}mm → d={cost*1000:.2f}mm  ({elapsed:.1f}s)")
        return cost

    return objective


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Optimize LiDAR mounting via point-to-plane ICP (front vs back)")
    ap.add_argument("--pcap",             default=PCAP)
    ap.add_argument("--total-angle",      type=float, default=TOTAL_ANGLE)
    ap.add_argument("--channel",          type=int, nargs='+', default=CHANNEL)
    ap.add_argument("--min-dist",         type=float, default=MIN_DIST)
    ap.add_argument("--max-dist",         type=float, default=0.0)
    ap.add_argument("--max-deg",          type=float, default=MAX_DEG)
    ap.add_argument("--max-shift",        type=float, default=MAX_SHIFT)
    ap.add_argument("--k-normal",         type=int, default=K_NORMAL,
                    help="neighbours for surface normal PCA (default: 15)")
    ap.add_argument("--n-front",          type=int, default=N_FRONT,
                    help="front subsample size for KDTree + normals (default: 5000)")
    ap.add_argument("--n-back",           type=int, default=N_BACK,
                    help="back subsample size for queries (default: 15000)")
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
    duration = packets[-1][0] - packets[0][0]
    rpm = args.total_angle / 6.0 / duration
    print(f"Total angle {args.total_angle}° over {duration:.2f}s → {rpm:.4f} RPM")
    print(f"Channels: {channels}  ({len(channels)} total)")

    x_L, y_L, z_L, theta, intens, _ = parse_raw(
        packets, t_start, cal, rpm,
        channels, BLOCK_INTERP, args.min_dist, args.max_dist,
    )
    N = len(x_L)
    print(f"Total points: {N:,}  (front: {(x_L > 0).sum():,}, back: {(x_L < 0).sum():,})")

    # ── 2. Split front / back, subsample each half ────────────────────────
    idx_f = np.where(x_L > 0)[0]
    idx_b = np.where(x_L < 0)[0]

    n_f = min(args.n_front, len(idx_f))
    n_b = min(args.n_back,  len(idx_b))
    samp_f = rng.choice(idx_f, size=n_f, replace=False)
    samp_b = rng.choice(idx_b, size=n_b, replace=False)

    x_Ls_f, y_Ls_f, z_Ls_f, theta_s_f = (x_L[samp_f], y_L[samp_f],
                                            z_L[samp_f], theta[samp_f])
    x_Ls_b, y_Ls_b, z_Ls_b, theta_s_b = (x_L[samp_b], y_L[samp_b],
                                            z_L[samp_b], theta[samp_b])
    print(f"Subsamples: front={n_f:,}  back={n_b:,}  k_normal={args.k_normal}")

    # ── 3. Build reference front (warm-start) + compute normals ───────────
    R_init = Rotation.from_euler('xyz', args.init_angles, degrees=True).as_matrix()
    t_init = np.array(args.init_translation)
    pts_f_ref = np.stack(
        apply_transform(x_Ls_f, y_Ls_f, z_Ls_f, theta_s_f, R_init, t_init), axis=1)

    print(f"\nComputing surface normals on {n_f:,}-point front reference ...")
    t_n = time.monotonic()
    normals_f, tree_f = compute_normals(pts_f_ref, args.k_normal)
    print(f"Done in {time.monotonic() - t_n:.2f}s")

    # ── 4. Baseline ───────────────────────────────────────────────────────
    pts_b_zero = np.stack(
        apply_transform(x_Ls_b, y_Ls_b, z_Ls_b, theta_s_b, np.eye(3)), axis=1)
    pts_b_init = np.stack(
        apply_transform(x_Ls_b, y_Ls_b, z_Ls_b, theta_s_b, R_init, t_init), axis=1)

    r0    = p2plane_metric(pts_b_zero, tree_f, pts_f_ref, normals_f)
    r_ini = p2plane_metric(pts_b_init, tree_f, pts_f_ref, normals_f)
    print(f"\n--- Baseline ---")
    print(f"No correction:  d = {r0*1000:.2f} mm")
    print(f"Warm-start:     d = {r_ini*1000:.2f} mm")

    # ── 5. Optimize ───────────────────────────────────────────────────────
    x0 = np.array(args.init_angles + args.init_translation)
    print(f"\nOptimizing (bounds ±{args.max_deg}° / ±{args.max_shift*100:.0f}cm) ...")
    print(f"Initial: roll={x0[0]:+.3f}° pitch={x0[1]:+.3f}° yaw={x0[2]:+.3f}°  "
          f"tx={x0[3]*1000:+.1f}mm ty={x0[4]*1000:+.1f}mm tz={x0[5]*1000:+.1f}mm")

    obj = make_objective(x_Ls_b, y_Ls_b, z_Ls_b, theta_s_b,
                         tree_f, pts_f_ref, normals_f,
                         args.max_deg, args.max_shift)

    t0 = time.monotonic()
    result = minimize(
        obj, x0=x0, method='Nelder-Mead',
        options={
            'xatol':   1e-4,
            'fatol':   1e-6,
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

    # ── 6. Final measurement ──────────────────────────────────────────────
    R_opt = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()
    t_opt = np.array([tx, ty, tz])
    pts_b_opt = np.stack(
        apply_transform(x_Ls_b, y_Ls_b, z_Ls_b, theta_s_b, R_opt, t_opt), axis=1)
    r_opt = p2plane_metric(pts_b_opt, tree_f, pts_f_ref, normals_f)

    print(f"\n--- Results ---")
    print(f"No correction:  d = {r0*1000:.2f} mm")
    print(f"Warm-start:     d = {r_ini*1000:.2f} mm")
    print(f"Optimized:      d = {r_opt*1000:.2f} mm")
    print(f"Improvement vs baseline: {(r0 - r_opt) / r0 * 100:.1f}%")
    print(f"\nTo replay with these parameters:")
    print(f"  --mount-correction {roll:.3f} {pitch:.3f} {yaw:.3f} "
          f"--mount-translation {tx:.4f} {ty:.4f} {tz:.4f}")

    # ── 7. Write PCD ──────────────────────────────────────────────────────
    x_fo, y_fo, z_fo = apply_transform(x_L, y_L, z_L, theta, R_opt, t_opt)
    tag = (f"p2pl_r{roll:.2f}_p{pitch:.2f}_y{yaw:.2f}"
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
