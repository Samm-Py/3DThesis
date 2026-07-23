"""Uniformity optimization of a melt-pool demo (square, triangle, or
single_track, following the working directory), for one dwell policy. Port of
melt_pool_demos/raster/run_optimized.py to the shared physics (see that file
for the controller rationale).

Sequential per-segment damped Newton on the control vector u:

    rasters (square/triangle):  u = (Pmod, sigma)          -- 2x2
    single_track:               u = (Pmod, sigma, v)        -- 2x3 regularized

    r(u) = (w - w*, d - d*),  J from ONE OTI snapshot per iterate,
    scaled + Tikhonov-regularized step, per-component trust region,
    backtracking merit check, best-evaluated-state fallback.

Two residuals with three controls (single_track) is underdetermined: the
Tikhonov term selects the minimum-effort step, distributing the correction
across power, beam width, and scan speed (a neutral regularizer). Velocity is
the natural handle for the single track's one non-uniformity -- the startup
transient -- which segment-end power/sigma regulation cannot see. dT_dv is the
validated OTI velocity sensitivity (see common/validate_dv.py,
common/verify_analytic_moving.py).

Target = the developed single-track pool (last segment of line 0 of the
baseline): the well-formed traveling steady state, same for both policies.
At zero dwell the target may be unreachable at hot line starts / cold starts --
that shows up honestly as bound-pinned segments (reported), never by
re-measuring.

Usage: python run_optimized.py {zero|dwell}
   ->  optimized_{policy}.csv, target_{policy}.json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mp_lib as R

# scan speed joins the control set only where it is the operative handle (the
# single track); the rasters keep the published 2x2 (Pmod, sigma) exactly.
VEL_CONTROL = os.environ.get(
    "MP_VEL_CONTROL", "1" if R.GEOM == "single_track" else "0") == "1"

PMOD_LO, PMOD_HI = 0.05, 3.0          # 7.5 - 450 W
SIG_LO, SIG_HI = 80e-6, 400e-6
VEL_LO, VEL_HI = 0.5, 8.0             # m/s (nominal 3.0)
TOL, MAX_ITER = 0.01, 8
J_MIN = 2e-6
STEP_REG = 2.5e-2
BACKTRACK = (1.0, 0.5, 0.25, 0.125, 0.0625)
MIN_MERIT_IMPROVEMENT = 1e-3
BOUND_ATOL = 1e-9

if VEL_CONTROL:
    U_LO = np.array([PMOD_LO, SIG_LO, VEL_LO])
    U_HI = np.array([PMOD_HI, SIG_HI, VEL_HI])
    CONTROL_SCALE = np.array([1.0, 200e-6, 3.0])
    DU_MAX = np.array([0.6, 100e-6, 1.5])   # trust region per iteration
    U0 = np.array([R.PMOD0, R.SIGMA, R.V])
else:
    U_LO = np.array([PMOD_LO, SIG_LO])
    U_HI = np.array([PMOD_HI, SIG_HI])
    CONTROL_SCALE = np.array([1.0, 200e-6])
    DU_MAX = np.array([0.6, 100e-6])
    U0 = np.array([R.PMOD0, R.SIGMA])


def _ufmt(u):
    s = f"Pmod={u[0]:.3f} sig={u[1]*1e6:5.1f}um"
    if VEL_CONTROL:
        s += f" v={u[2]:.2f}"
    return s


def _residual(s, w_star, d_star):
    return np.array([s['width'] - w_star, s['depth'] - d_star])


def _rel_residual(r, w_star, d_star):
    return np.max(np.abs(r) / np.array([w_star, d_star]))


def _merit(r, w_star, d_star):
    rn = r / np.array([w_star, d_star])
    return 0.5 * float(rn @ rn)


def _simulate_segment(div, i, pmods_hist, sig_hist, vel_hist, u):
    pmods = dict(pmods_hist)
    pmods[i] = u[0]
    vels = None
    if VEL_CONTROL:
        vels = dict(vel_hist)
        vels[i] = u[2]
    R.run_segment(div, i, pmods, R.BIN_OTI, sigma=u[1], sig_hist=sig_hist, vels=vels)
    s = R.sens()
    clip = R.clip_check()
    if clip:
        print(f"      WARNING: liquid clipped at box faces {clip}", flush=True)
    return s


def _jacobian(s):
    """QoI Jacobian w.r.t. the control vector: [width; depth] x u. Power column
    is converted from d/dQ (W) to d/dPmod (P_BASE * d/dQ)."""
    J = np.array([[s['dwidth_dQ'] * R.P_BASE, s['dwidth_dsig']],
                  [s['ddepth_dQ'] * R.P_BASE, s['ddepth_dsig']]])
    if VEL_CONTROL:
        J = np.column_stack([J, [s['dwidth_dv'], s['ddepth_dv']]])
    return J


def _regularized_step(s, r, w_star, d_star):
    J = _jacobian(s)
    # no thermal signal in the power/width columns (e.g. barely-melting cold
    # start): push power up, and -- with velocity -- slow down to add energy.
    if np.linalg.norm(J[:, 0]) < J_MIN and np.linalg.norm(J[:, 1]) < 0.2:
        direction = -np.sign(r[0] / w_star + r[1] / d_star)
        du = np.zeros(len(U_LO))
        du[0] = 0.5 * DU_MAX[0] * direction
        if VEL_CONTROL:
            du[2] = -0.5 * DU_MAX[2] * direction
        return du
    target = np.array([w_star, d_star])
    Jn = J / target[:, None]
    rn = r / target
    Js = Jn * CONTROL_SCALE[None, :]
    lhs = Js.T @ Js + STEP_REG * np.eye(len(U_LO))
    rhs = -(Js.T @ rn)
    dz = np.linalg.solve(lhs, rhs)
    return np.clip(CONTROL_SCALE * dz, -DU_MAX, DU_MAX)


def newton_segment(div, i, pmods_hist, sig_hist, vel_hist, u0, w_star, d_star):
    u = np.clip(u0, U_LO, U_HI)
    n, s = 0, None
    best_u, best_s, best_phi = u.copy(), None, np.inf
    cache_u, cache_s = None, None
    for it in range(MAX_ITER):
        try:
            if cache_u is not None and np.allclose(u, cache_u, rtol=0.0, atol=1e-15):
                s = cache_s
            else:
                s = _simulate_segment(div, i, pmods_hist, sig_hist, vel_hist, u)
                n += 1
        except ValueError:
            print(f"      it{it}: {_ufmt(u)} -> no melt; raising power, "
                  f"tightening beam" + (", slowing" if VEL_CONTROL else ""),
                  flush=True)
            u_new = u.copy()
            u_new[0] = max(2.0 * u[0], 1.0)
            u_new[1] = max(0.5 * u[1], R.SIGMA)
            if VEL_CONTROL:
                u_new[2] = 0.5 * u[2]
            u = np.clip(u_new, U_LO, U_HI)
            cache_u, cache_s = None, None
            continue
        r = _residual(s, w_star, d_star)
        rel = _rel_residual(r, w_star, d_star)
        phi = _merit(r, w_star, d_star)
        if phi < best_phi:
            best_u, best_s, best_phi = u.copy(), s, phi
        print(f"      it{it}: {_ufmt(u)} -> "
              f"w={s['width']*1e6:6.2f} d={s['depth']*1e6:6.2f} relres={rel:.4f}",
              flush=True)
        if rel < TOL:
            break
        du = _regularized_step(s, r, w_star, d_star)
        accepted = False
        for alpha in BACKTRACK:
            u_try = np.clip(u + alpha * du, U_LO, U_HI)
            if np.max(np.abs(u_try - u) / np.maximum(np.abs(u), 1e-12)) < 1e-3:
                continue
            try:
                s_try = _simulate_segment(div, i, pmods_hist, sig_hist, vel_hist, u_try)
                n += 1
            except ValueError:
                continue
            r_try = _residual(s_try, w_star, d_star)
            phi_try = _merit(r_try, w_star, d_star)
            if phi_try < best_phi:
                best_u, best_s, best_phi = u_try.copy(), s_try, phi_try
            if phi_try < phi * (1.0 - MIN_MERIT_IMPROVEMENT):
                print(f"           accept alpha={alpha:g}: {_ufmt(u_try)}",
                      flush=True)
                u = u_try
                cache_u, cache_s = u_try.copy(), s_try
                accepted = True
                break
        if not accepted:
            print("      no improving backtracked step; accepting best evaluated state",
                  flush=True)
            break
    if s is None:
        u = U0.copy()
        s = _simulate_segment(div, i, pmods_hist, sig_hist, vel_hist, u)
        n += 1
        best_u, best_s = u, s
    return best_u, best_s, n


def pinned(u):
    flags = []
    if abs(u[0] - PMOD_LO) < BOUND_ATOL: flags.append("Plo")
    if abs(u[0] - PMOD_HI) < BOUND_ATOL: flags.append("Phi")
    if abs(u[1] - SIG_LO) < BOUND_ATOL: flags.append("Slo")
    if abs(u[1] - SIG_HI) < BOUND_ATOL: flags.append("Shi")
    if VEL_CONTROL:
        if abs(u[2] - VEL_LO) < BOUND_ATOL: flags.append("Vlo")
        if abs(u[2] - VEL_HI) < BOUND_ATOL: flags.append("Vhi")
    return "+".join(flags)


if __name__ == "__main__":
    policy = sys.argv[1]
    dwell = 0.0 if policy == "zero" else R.dwell_from_json()
    base = pd.read_csv(R.data_path(f"baseline_{policy}.csv"))

    # target: developed single-track pool = last segment of line 0
    l0 = base[base['line'] == 0]
    w_star, d_star = float(l0['w'].iloc[-1]), float(l0['d'].iloc[-1])
    ctrl = "(Pmod, sigma, v)" if VEL_CONTROL else "(Pmod, sigma)"
    print(f"{policy}: dwell {dwell*1e3:.3f} ms; controls {ctrl}; targets "
          f"w*={w_star*1e6:.2f} d*={d_star*1e6:.2f} um (developed single track)")
    with open(R.data_path(f"target_{policy}.json", write=True), "w") as f:
        json.dump({"w_star": w_star, "d_star": d_star,
                   "seg_idx": int(l0['idx'].iloc[-1]),
                   "anchor": "developed single-track (end of line 0)"}, f,
                  indent=1)

    div, powered, hop_after = R.build_path(dwell)
    info = R.seg_info(div, powered)

    rows, opt_pmods, opt_sigmas, opt_vels, total = [], {}, {}, {}, 0
    # zero-dwell: powered hops inherit the controls of the segment they follow
    u = U0.copy()
    by_pos, pos, cur_line = {}, -1, -1
    t0 = time.time()
    for e in info:
        pos = 0 if e['line'] != cur_line else pos + 1
        cur_line = e['line']
        u0 = by_pos.get(pos, u)
        print(f"   seg {e['idx']} (line {e['line']}, s={e['s_mm']:.1f} mm):",
              flush=True)
        u, s, n = newton_segment(div, e['idx'], opt_pmods, opt_sigmas, opt_vels,
                                 u0, w_star, d_star)
        by_pos[pos] = u
        opt_pmods[e['idx']] = u[0]
        opt_sigmas[e['idx']] = u[1]
        if VEL_CONTROL:
            opt_vels[e['idx']] = u[2]
        if e['idx'] in hop_after:
            opt_pmods[hop_after[e['idx']]] = u[0]
            opt_sigmas[hop_after[e['idx']]] = u[1]
            if VEL_CONTROL:
                opt_vels[hop_after[e['idx']]] = u[2]
        total += n
        rows.append(dict(**e, pmod=u[0], sigma=u[1],
                         vel=(u[2] if VEL_CONTROL else R.V),
                         w=s['width'], d=s['depth'], asym=s['asym'],
                         pinned=pinned(u)))
    pd.DataFrame(rows).to_csv(R.data_path(f"optimized_{policy}.csv", write=True), index=False)

    bw, bd = base['w'].values, base['d'].values
    ow = np.array([r['w'] for r in rows]); od = np.array([r['d'] for r in rows])
    cv = lambda v: 100.0 * np.std(v) / np.mean(v)
    npin = sum(1 for r in rows if r['pinned'])
    print(f"\ntotal simulations: {total} ({total/len(info):.1f} per segment), "
          f"wall {time.time()-t0:.0f} s")
    print(f"bound-pinned segments: {npin}/{len(rows)}")
    print(f"{'':>10} {'width CV%':>10} {'depth CV%':>10} {'w range(um)':>12} {'d range(um)':>12}")
    print(f"{'baseline':>10} {cv(bw):10.2f} {cv(bd):10.2f} "
          f"{(bw.max()-bw.min())*1e6:12.2f} {(bd.max()-bd.min())*1e6:12.2f}")
    print(f"{'optimised':>10} {cv(ow):10.2f} {cv(od):10.2f} "
          f"{(ow.max()-ow.min())*1e6:12.2f} {(od.max()-od.min())*1e6:12.2f}")
    print(f"wrote {R.data_path(f'optimized_{policy}.csv')}")
