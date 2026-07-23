"""Assemble the velocity influence table (trajectory Jacobian) for the single
track: J[i, j] = d(depth at end of controlled segment i) / d(v_j), from OTI
runs with the generalized seed segment (TRAJECTORY_OPTIMIZER_PLAN.md Phase 3).

Structure exploited (measured in common/validate_traj.py):
  * causality: j <= i (lower triangular);
  * band: influence peaks at lag ~3 (the deepest point trails the beam by
    ~0.8 mm) and is super-exponentially dead past lag ~5 -> band B;
  * Toeplitz: in the developed region the lag stencil is translation-
    invariant to 0.00%, so rows past the startup block reuse one stencil.

Exact startup block: observations k = 0..N_EXACT-1, all lags <= B, one OTI
run each ("seed segment j, truncate at i"). Developed rows: the stencil
measured at observation K_STENCIL. `--full` measures every banded pair
instead (audit mode). All runs are at the BASELINE control point (Pmod 1,
sigma nominal, v = 3 m/s everywhere), with the full control history written
into the path (mp_lib.run_segment pmods/vels/sig_hist).

Usage (from single_track/):  python traj_jacobian.py [--full]
  ->  results/traj_jacobian.npz  (J, J_width, s_mm, band, meta)
      results/traj_jacobian.csv  (readable table, um per m/s)
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mp_lib as R

BAND = 6          # lags kept (validate_traj: dead by lag 5-6)
N_EXACT = 12      # exact-row block (startup + one pool footprint, 3 mm)
K_STENCIL = 14    # developed observation for the Toeplitz stencil


def measure(div, info, k, lag):
    """One OTI run: observe at controlled segment k, seed at lag behind."""
    i = info[k]['idx']
    j = i - lag
    pmods = {info[m]['idx']: 1.0 for m in range(k + 1)}
    vels = {info[m]['idx']: R.V for m in range(k + 1)}
    R.run_segment(div, i, pmods, R.BIN_OTI, sigma=R.SIGMA, sig_hist={},
                  vels=vels, seed_idx=j)
    s = R.sens()
    return s['ddepth_dv'], s['dwidth_dv'], s['depth'], s['width']


if __name__ == "__main__":
    full = "--full" in sys.argv
    div, powered, hop_after = R.build_path(0.0)
    info = R.seg_info(div, powered)
    N = len(info)
    J = np.zeros((N, N))          # ddepth_i / dv_j
    JW = np.zeros((N, N))         # dwidth_i / dv_j (monitor)
    d0 = np.zeros(N)              # baseline depth at each observation
    w0 = np.zeros(N)

    t0, runs = time.time(), 0
    if full:
        rows = [(k, lag) for k in range(N) for lag in range(min(k + 1, BAND + 1))]
    else:
        rows = [(k, lag) for k in range(N_EXACT) for lag in range(min(k + 1, BAND + 1))]
        rows += [(K_STENCIL, lag) for lag in range(BAND + 1)]

    stencil = np.zeros(BAND + 1)
    for k, lag in rows:
        dv, wv, d, w = measure(div, info, k, lag)
        runs += 1
        i = k
        J[i, i - lag] = dv
        JW[i, i - lag] = wv
        d0[i], w0[i] = d, w
        if k == K_STENCIL:
            stencil[lag] = dv
        print(f"  obs {k:2d} (s={info[k]['s_mm']:5.2f} mm)  lag {lag}: "
              f"dd/dv = {dv:+.4e}  dw/dv = {wv:+.4e}", flush=True)

    if not full:
        # developed rows (beyond the exact block) reuse the stencil; their
        # baseline depth is the developed value measured at K_STENCIL
        for i in range(N_EXACT, N):
            for lag in range(min(i + 1, BAND + 1)):
                J[i, i - lag] = stencil[lag]
            if d0[i] == 0.0:
                d0[i], w0[i] = d0[K_STENCIL], w0[K_STENCIL]
        # cross-check: exact block's last row vs the stencil (both developed)
        ref = J[N_EXACT - 1, N_EXACT - 1 - np.arange(BAND + 1)]
        dev = np.abs(ref - stencil) / np.maximum(np.abs(stencil), 1e-30)
        print(f"\nToeplitz cross-check (exact row {N_EXACT-1} vs stencil row "
              f"{K_STENCIL}): max deviation {dev.max()*100:.2f}%")

    s_mm = np.array([e['s_mm'] for e in info])
    idx = np.array([e['idx'] for e in info])
    out = R.data_path("traj_jacobian.npz", write=True)
    np.savez(out, J=J, J_width=JW, s_mm=s_mm, div_idx=idx, d0=d0, w0=w0,
             band=BAND, n_exact=(N if full else N_EXACT),
             k_stencil=K_STENCIL, v0=R.V, mode=("full" if full else "toeplitz"))

    # readable table: um of depth per (m/s), rows = observation, cols = lag
    tab = pd.DataFrame(
        {f"lag{lag}": [J[i, i - lag] * 1e6 if lag <= i else np.nan
                       for i in range(N)] for lag in range(BAND + 1)},
        index=[f"seg{k} @{s_mm[k]:.2f}mm" for k in range(N)])
    tab.insert(0, "d0_um", d0 * 1e6)
    csv = R.data_path("traj_jacobian.csv", write=True)
    tab.to_csv(csv, float_format="%.4e")

    print(f"\n{runs} OTI runs, wall {time.time()-t0:.0f} s")
    print(f"wrote {out}\nwrote {csv}")
    print("\ninfluence table (um of depth per m/s), startup block:")
    with pd.option_context('display.width', 140, 'display.float_format',
                           lambda v: f"{v:9.3f}"):
        print(tab.head(N_EXACT))
