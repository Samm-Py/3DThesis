"""Coupled velocity-trajectory optimization of the single track
(TRAJECTORY_OPTIMIZER_PLAN.md Phase 4).

One Gauss-Newton problem over the whole velocity profile v = (v_1 .. v_N),
holding power and beam width at nominal:

    min  0.5 ||r(v)/d*||^2 + 0.5 l_s ||D1 vt||^2 + 0.5 l_m ||vt - vt_nom||^2
    r_i = d_i(v) - d*,   vt = v / V_NOM (normalized controls)

with the banded lower-triangular influence matrix J_ij = dd_i/dv_j from OTI
(traj_jacobian.py at baseline; refreshed at the current profile when the
merit stalls). This replaces the greedy per-segment controller, which is
structurally ill-posed at 0.25 mm segments (each row of J is dominated by
its OFF-diagonal lag-2..3 entries -- see the plan's Phase 3 table).

Regularizers (user-approved from the measured table): l_s = 1e-2 smoothness
(first differences; permits the physical startup ramp, suppresses
oscillation), l_m = 1e-3 anchor (pins the residual-free developed tail at
nominal instead of letting it drift in the null space). Trust region 1 m/s
per iteration, backtracking on the fully re-simulated merit.

Outputs:
  results/trajectory_zero.csv    -- per-segment schedule + achieved depths
  results/optimized_zero.csv     -- same schedule in the optimizer schema
                                    (pmod=1, sigma nominal, vel=v_i) so
                                    verify_optimized.py replays it unchanged;
                                    the greedy divergent run is preserved as
                                    optimized_zero_greedy_divergent.csv.

Usage (from single_track/):  python run_trajectory.py
"""
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mp_lib as R

V_NOM = R.V
V_LO, V_HI = 0.5, 8.0
LAM_S, LAM_M = 1e-2, 1e-3      # normalized-unit regularizers (plan Phase 3)
TRUST = 1.0                    # m/s per iteration, per segment
MAX_ITER = 8
TOL = 0.01                     # max |r_i|/d* convergence
BACKTRACK = (1.0, 0.5, 0.25, 0.125)
STALL = 0.30                   # merit reduction below this -> refresh J
N_EXACT = 12                   # startup rows re-measured on refresh
BAND = 6


def simulate_depths(div, info, vels_map):
    """Depths at every controlled segment end for a given velocity profile
    (N truncated double-solver snapshots, full profile in the path)."""
    d = np.zeros(len(info))
    w = np.zeros(len(info))
    for k, e in enumerate(info):
        pmods = {info[m]['idx']: 1.0 for m in range(k + 1)}
        vels = {info[m]['idx']: vels_map[info[m]['idx']] for m in range(k + 1)}
        R.run_segment(div, e['idx'], pmods, R.BIN_DBL, vels=vels)
        wk, dk, _a, clip = R.measure_dims()
        if clip:
            print(f"      WARNING: clip at obs {k}: {clip}", flush=True)
        d[k], w[k] = dk, wk
    return d, w


def measure_J(div, info, vels_map, n_exact=N_EXACT, band=BAND, stencil_k=14):
    """Banded influence matrix at the given profile: exact startup rows +
    one developed Toeplitz stencil (see traj_jacobian.py)."""
    N = len(info)
    J = np.zeros((N, N))
    stencil = np.zeros(band + 1)
    runs = 0
    for k in list(range(n_exact)) + [stencil_k]:
        for lag in range(min(k + 1, band + 1)):
            i = info[k]['idx']
            pmods = {info[m]['idx']: 1.0 for m in range(k + 1)}
            vels = {info[m]['idx']: vels_map[info[m]['idx']] for m in range(k + 1)}
            R.run_segment(div, i, pmods, R.BIN_OTI, sigma=R.SIGMA, sig_hist={},
                          vels=vels, seed_idx=i - lag)
            dv = R.sens()['ddepth_dv']
            runs += 1
            if k == stencil_k:
                stencil[lag] = dv
            else:
                J[k, k - lag] = dv
    for i in range(n_exact, N):
        for lag in range(min(i + 1, band + 1)):
            J[i, i - lag] = stencil[lag]
    return J, runs


def merit(r_n, vt):
    d1 = np.diff(vt)
    return 0.5 * float(r_n @ r_n) + 0.5 * LAM_S * float(d1 @ d1) \
        + 0.5 * LAM_M * float((vt - 1.0) @ (vt - 1.0))


if __name__ == "__main__":
    t0 = time.time()
    div, powered, hop_after = R.build_path(0.0)
    info = R.seg_info(div, powered)
    N = len(info)
    idxs = [e['idx'] for e in info]

    # target: developed single-track depth (baseline is exactly on it)
    base = pd.read_csv(R.data_path("baseline_zero.csv"))
    d_star = float(base['d'].iloc[-1])
    print(f"N = {N} segments, d* = {d_star*1e6:.2f} um, "
          f"lambda_s = {LAM_S}, lambda_m = {LAM_M}, trust {TRUST} m/s")

    # baseline Jacobian from Phase 3 (refreshed on stall)
    jz = np.load(R.data_path("traj_jacobian.npz"))
    J = jz['J']
    n_runs_oti, n_runs_dbl = 0, 0

    # first-difference operator for the smoothness regularizer
    D1 = np.diff(np.eye(N), axis=0)

    v = np.full(N, V_NOM)
    vmap = dict(zip(idxs, v))
    d, w = simulate_depths(div, info, vmap)
    n_runs_dbl += N
    r_n = (d - d_star) / d_star
    phi = merit(r_n, v / V_NOM)
    print(f"it0: max|r| = {np.abs(r_n).max()*100:.2f}%  merit = {phi:.5f}  "
          f"(baseline)", flush=True)

    for it in range(MAX_ITER):
        if np.abs(r_n).max() < TOL:
            print("converged")
            break
        # normalized GN step
        vt = v / V_NOM
        Jn = J * (V_NOM / d_star)                    # d r_n / d vt
        lhs = Jn.T @ Jn + LAM_S * (D1.T @ D1) + LAM_M * np.eye(N)
        rhs = -(Jn.T @ r_n + LAM_S * (D1.T @ (D1 @ vt)) + LAM_M * (vt - 1.0))
        dvt = np.linalg.solve(lhs, rhs)
        dv = np.clip(dvt * V_NOM, -TRUST, TRUST)

        accepted = False
        for alpha in BACKTRACK:
            v_try = np.clip(v + alpha * dv, V_LO, V_HI)
            vmap = dict(zip(idxs, v_try))
            d_try, w_try = simulate_depths(div, info, vmap)
            n_runs_dbl += N
            r_try = (d_try - d_star) / d_star
            phi_try = merit(r_try, v_try / V_NOM)
            print(f"it{it+1}: alpha={alpha:g}  max|r| = "
                  f"{np.abs(r_try).max()*100:.2f}%  merit = {phi_try:.5f}",
                  flush=True)
            if phi_try < phi:
                improvement = (phi - phi_try) / phi
                v, d, w, r_n, phi = v_try, d_try, w_try, r_try, phi_try
                accepted = True
                if improvement < STALL and np.abs(r_n).max() >= TOL:
                    print(f"      merit reduction {improvement*100:.0f}% < "
                          f"{STALL*100:.0f}%: refreshing J at current profile",
                          flush=True)
                    J, nr = measure_J(div, info, dict(zip(idxs, v)))
                    n_runs_oti += nr
                break
        if not accepted:
            print("      no improving step; refreshing J at current profile",
                  flush=True)
            J, nr = measure_J(div, info, dict(zip(idxs, v)))
            n_runs_oti += nr

    # ---- report + outputs ----
    print(f"\nfinal profile (m/s):")
    for k in range(N):
        mark = " *" if abs(v[k] - V_NOM) > 1e-3 else ""
        if k < N_EXACT or mark:
            print(f"  seg {k:2d} (s={info[k]['s_mm']:5.2f} mm): v = {v[k]:.3f}"
                  f"  d = {d[k]*1e6:6.2f} um  r = {r_n[k]*100:+6.2f}%{mark}")
    cv = lambda x: 100.0 * np.std(x) / np.mean(x)
    print(f"\nsnapshot depth: baseline CV {cv(base['d'].values):.2f}%  ->  "
          f"optimized CV {cv(d):.2f}%")
    print(f"max|r|: baseline {np.abs((base['d'].values - d_star)/d_star).max()*100:.2f}%"
          f"  ->  optimized {np.abs(r_n).max()*100:.2f}%")
    print(f"runs: {n_runs_dbl} double + {n_runs_oti} OTI, "
          f"wall {time.time()-t0:.0f} s")

    rows = [dict(**info[k], pmod=1.0, sigma=R.SIGMA, vel=v[k],
                 w=w[k], d=d[k], r_rel=r_n[k]) for k in range(N)]
    out = R.data_path("trajectory_zero.csv", write=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")

    # replayable schedule in the optimizer schema; keep the greedy divergent
    # run as the posedness diagnostic it is
    opt_path = R.data_path("optimized_zero.csv", write=True)
    keep = R.data_path("optimized_zero_greedy_divergent.csv", write=True)
    if os.path.exists(opt_path) and not os.path.exists(keep):
        shutil.copy2(opt_path, keep)
        print(f"preserved greedy divergent run as {keep}")
    pd.DataFrame(rows)[['idx', 'x', 'y', 'line', 's_mm', 'pmod', 'sigma',
                        'vel', 'w', 'd']].to_csv(opt_path, index=False)
    print(f"wrote {opt_path} (replay with: python verify_optimized.py)")
