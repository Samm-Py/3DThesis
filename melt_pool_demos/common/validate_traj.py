"""Validation of the GENERALIZED seed-segment sensitivities (seed segment j,
observe at the end of segment i) on the real single-track path -- the Phase 2
gate of TRAJECTORY_OPTIMIZER_PLAN.md. Complements verify_analytic_moving.py
(continuum check; exact for shallow lags, quadrature-coarseness-limited for
deep history) with checks against the DISCRETE solver itself, which is what
OTI differentiates and what the trajectory optimizer consumes.

  1. field FD-by-lag:  central FD of T w.r.t. v_j (double-identical values)
     vs dT_dv seeded at j, at the 300 highest-signal grid points.
  2. depth FD-by-lag:  central FD of the ISOTHERM CROSSING DEPTH ALONG THE
     Z-COLUMN through the unperturbed deepest support point, vs ddepth_dv.
     NOT the max-depth: the developed pool's depth profile is a near-flat
     plateau, so a localized history perturbation relocates the argmax and
     FD-of-max picks up a large second-order envelope term (observed:
     15-60% apparent 'errors' that vanish at a fixed column). The fixed
     column isolates the implicit-function-theorem derivative, which is
     what the OTI support sensitivity is.
  3. Q/sigma cross-checks at one lag (the moved seeding guard): field FD on
     the history segment's Pmod (T is LINEAR in Q_j -> near-exact) and width
     factor, plus the column-depth FD for both.
  4. band decay: one full column j (all observations i >= j) of ddepth_dv.
  5. Toeplitz check: the same-lag stencil measured at two developed
     observations must agree (translation invariance).

Run from single_track/:  python ../common/validate_traj.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mp_lib as R

V0 = R.V
R.RES_Z = 1e-6          # fine z-grid so column-depth interpolation is clean
LAGS = (0, 1, 2, 3, 4)
EPS_V = 0.15            # 5% of 3 m/s (history signals are small)
EPS_P = 0.05            # Pmod
EPS_W = 0.05            # width factor


def field(csv=None):
    return pd.read_csv(csv or R.CSV)


def run(div, i, pmods, vels, binary, seed_idx=None, sig_hist=None):
    R.run_segment(div, i, pmods, binary, sigma=R.SIGMA,
                  sig_hist=sig_hist or {}, vels=vels, seed_idx=seed_idx)


def support_point(d):
    """(x0, y0) of the deepest point of the T_LIQ isosurface."""
    from skimage import measure as MEAS
    xs, ys, zs = np.unique(d.x), np.unique(d.y), np.unique(d.z)
    T3 = d['T'].values.reshape(xs.size, ys.size, zs.size)
    verts, _, _, _ = MEAS.marching_cubes(T3, R.T_LIQ)
    k = np.argmin(verts[:, 2])
    return (xs[0] + verts[k, 0] * (xs[1] - xs[0]),
            ys[0] + verts[k, 1] * (ys[1] - ys[0]))


def col_depth(d, x0, y0):
    """Isotherm crossing depth along the z-column nearest (x0, y0) -- the
    fixed-point IFT quantity the OTI support sensitivity differentiates."""
    gx = np.unique(d.x.values)[np.argmin(np.abs(np.unique(d.x.values) - x0))]
    gy = np.unique(d.y.values)[np.argmin(np.abs(np.unique(d.y.values) - y0))]
    c = d[np.isclose(d.x, gx) & np.isclose(d.y, gy)].sort_values('z')
    z, T = c['z'].values, c['T'].values
    k0 = np.where(T >= R.T_LIQ)[0].min()
    if k0 == 0:
        return -z[0]
    z1, z2, T1, T2 = z[k0 - 1], z[k0], T[k0 - 1], T[k0]
    return -(z1 + (R.T_LIQ - T1) * (z2 - z1) / (T2 - T1))


if __name__ == "__main__":
    div, powered, hop_after = R.build_path(0.0)
    info = R.seg_info(div, powered)
    i = info[8]['idx']                       # observe at ~2.25 mm (developed)
    hist = list(range(9))
    pmods = {info[k]['idx']: 1.0 for k in hist}
    vels0 = {info[k]['idx']: V0 for k in hist}

    # frozen support point of the unperturbed pool (x ~ -0.8 mm: the deepest
    # point trails the beam; all column-depth FDs are taken at this column)
    run(div, i, pmods, vels0, R.BIN_OTI)
    x0, y0 = support_point(field())
    print(f"unperturbed deepest support point: x = {x0*1e3:.3f} mm, "
          f"y = {y0*1e6:.1f} um behind segment end")

    print("\n== 1+2. FD vs OTI by lag (field, and column depth at the "
          "frozen support point) ==")
    print(f"{'lag':>4} {'field med%':>11} {'field p90%':>11} "
          f"{'ddepth_dv OTI':>14} {'column FD':>14} {'rel%':>8}")
    for lag in LAGS:
        j = i - lag
        run(div, i, pmods, vels0, R.BIN_OTI, seed_idx=j)
        oti_f = field()['dT_dv'].values
        dd_oti = R.sens()['ddepth_dv']
        vp = dict(vels0); vp[j] = V0 + EPS_V
        run(div, i, pmods, vp, R.BIN_OTI)          # OTI binary, default seed:
        d = field(); Tp, dp = d['T'].values, col_depth(d, x0, y0)
        vm = dict(vels0); vm[j] = V0 - EPS_V
        run(div, i, pmods, vm, R.BIN_OTI)
        d = field(); Tm, dm = d['T'].values, col_depth(d, x0, y0)
        fd_f = (Tp - Tm) / (2 * EPS_V)
        idx = np.argsort(-np.abs(fd_f))[:300]
        rel_f = np.abs(oti_f[idx] - fd_f[idx]) / np.maximum(np.abs(fd_f[idx]), 1e-12)
        dd_fd = (dp - dm) / (2 * EPS_V)
        rel_d = (abs(dd_oti - dd_fd) / abs(dd_fd) * 100) if abs(dd_fd) > 1e-12 else float('nan')
        print(f"{lag:>4} {np.median(rel_f)*100:>10.4f}% {np.percentile(rel_f,90)*100:>10.4f}% "
              f"{dd_oti:>14.4e} {dd_fd:>14.4e} {rel_d:>7.2f}%")

    print("\n== 3. Q / sigma cross-checks at lag 2 (moved seeding guard) ==")
    j = i - 2
    run(div, i, pmods, vels0, R.BIN_OTI, seed_idx=j)
    s = R.sens()
    oq, osg = field()['dT_dQ'].values, field()['dT_dsig'].values
    # power (T is LINEAR in Q_j -> field FD must be near-exact)
    pp = dict(pmods); pp[j] = 1.0 + EPS_P
    run(div, i, pp, vels0, R.BIN_OTI)
    d = field(); Tp, d_p = d['T'].values, col_depth(d, x0, y0)
    pm = dict(pmods); pm[j] = 1.0 - EPS_P
    run(div, i, pm, vels0, R.BIN_OTI)
    d = field(); Tm, d_m = d['T'].values, col_depth(d, x0, y0)
    fd_f = (Tp - Tm) / (2 * EPS_P * R.P_BASE)
    idx = np.argsort(-np.abs(fd_f))[:300]
    rel_f = np.abs(oq[idx] - fd_f[idx]) / np.abs(fd_f[idx])
    fd_q = (d_p - d_m) / (2 * EPS_P * R.P_BASE)
    print(f"  dT_dQ field: median {np.median(rel_f)*100:.4f}%   "
          f"ddepth_dQ: OTI {s['ddepth_dQ']:+.4e}  FD {fd_q:+.4e}  "
          f"rel {abs(s['ddepth_dQ']-fd_q)/abs(fd_q)*100:.2f}%")
    # sigma: FD on the history segment's width factor (sigma_j = wmod_j * SIGMA)
    run(div, i, pmods, vels0, R.BIN_OTI, sig_hist={j: R.SIGMA * (1 + EPS_W)})
    d = field(); Tp, d_p = d['T'].values, col_depth(d, x0, y0)
    run(div, i, pmods, vels0, R.BIN_OTI, sig_hist={j: R.SIGMA * (1 - EPS_W)})
    d = field(); Tm, d_m = d['T'].values, col_depth(d, x0, y0)
    fd_f = (Tp - Tm) / (2 * EPS_W * R.SIGMA)
    idx = np.argsort(-np.abs(fd_f))[:300]
    rel_f = np.abs(osg[idx] - fd_f[idx]) / np.abs(fd_f[idx])
    fd_s = (d_p - d_m) / (2 * EPS_W * R.SIGMA)
    print(f"  dT_dsig field: median {np.median(rel_f)*100:.4f}%   "
          f"ddepth_dsig: OTI {s['ddepth_dsig']:+.4e}  FD {fd_s:+.4e}  "
          f"rel {abs(s['ddepth_dsig']-fd_s)/abs(fd_s)*100:.2f}%")

    print("\n== 4. band decay: column j (all observations i >= j) ==")
    jcol = info[2]['idx']                     # a startup segment (~0.75 mm)
    print(f"  seeded segment idx {jcol} (s = {info[2]['s_mm']:.2f} mm)")
    for k in range(2, min(2 + 10, len(info))):
        io = info[k]['idx']
        pm_h = {info[m]['idx']: 1.0 for m in range(k + 1)}
        vl_h = {info[m]['idx']: V0 for m in range(k + 1)}
        run(div, io, pm_h, vl_h, R.BIN_OTI, seed_idx=jcol)
        s = R.sens()
        print(f"    obs seg {io} (lag {io - jcol:>2}): "
              f"ddepth_dv = {s['ddepth_dv']:+.4e}  dwidth_dv = {s['dwidth_dv']:+.4e}")

    print("\n== 5. Toeplitz: lag-2 stencil at two developed observations ==")
    vals = []
    for k in (8, 12):
        io = info[k]['idx']
        pm_h = {info[m]['idx']: 1.0 for m in range(k + 1)}
        vl_h = {info[m]['idx']: V0 for m in range(k + 1)}
        run(div, io, pm_h, vl_h, R.BIN_OTI, seed_idx=io - 2)
        s = R.sens()
        vals.append(s['ddepth_dv'])
        print(f"    obs seg {io}: ddepth_dv(lag 2) = {s['ddepth_dv']:+.4e}")
    print(f"    translation-invariance deviation: "
          f"{abs(vals[0]-vals[1])/abs(vals[0])*100:.2f}%")
