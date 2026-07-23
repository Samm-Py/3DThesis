"""Power + beam-width uniformity optimisation of the serpentine raster.

Sequential per-segment 2x2 Newton on u = (Pmod, sigma): power alone cannot
hold the centre-of-square target pool (from cold ground even 200 W falls
short, and once the background is hot even ~0 W overshoots -- the power-only
schedule degenerates into a spike-and-coast limit cycle). Beam width supplies
the missing degree of freedom: it reshapes the pool (wide-vs-deep) at a given
power instead of only scaling it.

    r(u) = (w - w*, d - d*),   J = [[dw/dPmod, dw/dsig],
                                    [dd/dPmod, dd/dsig]],   du = -J^{-1} r

damped with a scaled, regularized step and a backtracking merit check to avoid
bound slamming and Pmod/sigma ping-pong.
Both the residual and the exact Jacobian come from one OTI-build run per
iteration (simulate.ExtractMPSensitivities; derivatives seeded on the
current segment only). History segments replay their optimised Pmods AND
sigmas (7th path column, factor sigma_j / current base sigma).

Targets are the DEVELOPED SINGLE-TRACK pool: the last segment of line 0,
where the first track has reached its steady size on cold ground. A uniform
target must be a sustainable fixed point of the controlled process --
reachable from a fresh track start at moderate power and holdable mid-line
inside the control bounds. Heat-soaked baseline pools (e.g. the centre of the
square) fail both ways and drive the controller into a spike-and-coast limit
cycle. Writes optimized.csv (incl. sigma) and target.json."""
import json
import numpy as np
import pandas as pd
import raster_lib as R
from common import measurement as Sim

PMOD_LO, PMOD_HI = 0.05, 3.0      # ceiling 120 W: a higher cap lets the first
                                  # segment of a line "reach" the target with a
                                  # ~175 W spike whose leftover lake then owns
                                  # the next 2-3 segments (zero control
                                  # authority) -- worse overall than a small
                                  # development transient
SIG_LO,  SIG_HI  = 5.0e-6, 60.0e-6
TOL, MAX_ITER = 0.01, 8
DU_MAX = np.array([0.6, 10e-6])   # trust region: (Pmod, sigma) per iteration
J_MIN = 2e-6                      # residual-sensitivity floor (m per unit)
CONTROL_SCALE = np.array([1.0, 10e-6])
STEP_REG = 2.5e-2                 # Tikhonov regularization in scaled controls
BACKTRACK = (1.0, 0.5, 0.25, 0.125, 0.0625)
SEG_CTRL = 0.5                    # see note in main: 0.25 mm causes period-2 cycling
MIN_MERIT_IMPROVEMENT = 1e-3


def _residual(s, w_star, d_star):
    return np.array([s['width'] - w_star, s['depth'] - d_star])


def _rel_residual(r, w_star, d_star):
    target = np.array([w_star, d_star])
    return np.max(np.abs(r) / target)


def _merit(r, w_star, d_star):
    target = np.array([w_star, d_star])
    rn = r / target
    return 0.5 * float(rn @ rn)


def _simulate_segment(div, i, pmods_hist, sig_hist, u):
    pmods = dict(pmods_hist)
    pmods[i] = u[0]
    R.run_segment(div, i, pmods, R.BIN_OTI, sigma=u[1], sig_hist=sig_hist)
    return Sim.ExtractMPSensitivities(inFile=R.CSV)


def _regularized_step(s, r, w_star, d_star):
    J = np.array([[s['dwidth_dQ'] * R.P_BASE, s['dwidth_dsig']],
                  [s['ddepth_dQ'] * R.P_BASE, s['ddepth_dsig']]])
    if np.linalg.norm(J[:, 0]) < J_MIN and np.linalg.norm(J[:, 1]) < 0.2:
        # Pool extremes are dominated by inherited heat, so the local
        # derivative has little authority. Take a small power-only correction
        # and let the merit check decide whether it is actually useful.
        direction = -np.sign(r[0] / w_star + r[1] / d_star)
        return np.array([0.5 * DU_MAX[0] * direction, 0.0])

    target = np.array([w_star, d_star])
    Jn = J / target[:, None]
    rn = r / target
    Js = Jn * CONTROL_SCALE[None, :]
    lhs = Js.T @ Js + STEP_REG * np.eye(2)
    rhs = -(Js.T @ rn)
    dz = np.linalg.solve(lhs, rhs)
    du = CONTROL_SCALE * dz
    return np.clip(du, -DU_MAX, DU_MAX)


def newton_segment(div, i, pmods_hist, sig_hist, u0, w_star, d_star):
    u = np.array([np.clip(u0[0], PMOD_LO, PMOD_HI),
                  np.clip(u0[1], SIG_LO, SIG_HI)])
    n, s = 0, None
    best_u, best_s, best_phi = u.copy(), None, np.inf
    cache_u, cache_s = None, None
    for it in range(MAX_ITER):
        try:
            if cache_u is not None and np.allclose(u, cache_u, rtol=0.0, atol=1e-15):
                s = cache_s
            else:
                s = _simulate_segment(div, i, pmods_hist, sig_hist, u)
                n += 1
        except ValueError:
            # no melt at all: this candidate is far too weak for fresh ground
            print(f"      it{it}: Pmod={u[0]:.3f} sig={u[1]*1e6:5.2f}um -> "
                  f"no melt; raising power, tightening beam")
            u = np.array([np.clip(max(2.0 * u[0], 1.0), PMOD_LO, PMOD_HI),
                          np.clip(max(0.5 * u[1], R.SIGMA), SIG_LO, SIG_HI)])
            cache_u, cache_s = None, None
            continue
        r = _residual(s, w_star, d_star)
        rel = _rel_residual(r, w_star, d_star)
        phi = _merit(r, w_star, d_star)
        if phi < best_phi:
            best_u, best_s, best_phi = u.copy(), s, phi
        print(f"      it{it}: Pmod={u[0]:.3f} sig={u[1]*1e6:5.2f}um -> "
              f"w={s['width']*1e6:6.2f} d={s['depth']*1e6:6.2f} relres={rel:.4f}")
        if rel < TOL:
            break
        du = _regularized_step(s, r, w_star, d_star)
        accepted = False
        for alpha in BACKTRACK:
            u_try = np.array([np.clip(u[0] + alpha * du[0], PMOD_LO, PMOD_HI),
                              np.clip(u[1] + alpha * du[1], SIG_LO, SIG_HI)])
            if np.max(np.abs(u_try - u) / np.maximum(np.abs(u), 1e-12)) < 1e-3:
                continue
            try:
                s_try = _simulate_segment(div, i, pmods_hist, sig_hist, u_try)
                n += 1
            except ValueError:
                continue
            r_try = _residual(s_try, w_star, d_star)
            phi_try = _merit(r_try, w_star, d_star)
            if phi_try < best_phi:
                best_u, best_s, best_phi = u_try.copy(), s_try, phi_try
            if phi_try < phi * (1.0 - MIN_MERIT_IMPROVEMENT):
                print(f"           accept alpha={alpha:g}: "
                      f"Pmod={u_try[0]:.3f} sig={u_try[1]*1e6:5.2f}um")
                u = u_try
                cache_u, cache_s = u_try.copy(), s_try
                accepted = True
                break
        if not accepted:
            print("      no improving backtracked step; accepting best evaluated state")
            break
    if s is None:                     # never melted: fall back to the baseline
        u = np.array([R.PMOD0, R.SIGMA])
        s = _simulate_segment(div, i, pmods_hist, sig_hist, u)
        n += 1
        best_u, best_s = u, s
    return best_u, best_s, n


if __name__ == "__main__":
    dwell = json.load(open(R.data_path("min_dwell.json")))["dwell_s"]
    base = pd.read_csv(R.data_path("baseline.csv"))
    # target: the developed single-track pool -- the LAST segment of line 0,
    # where the first track has reached its steady size but no neighbour or
    # background preheat exists yet
    l0 = base[base['line'] == 0]
    w_star, d_star = float(l0['w'].iloc[-1]), float(l0['d'].iloc[-1])
    print(f"dwell {dwell*1e3:.3f} ms; targets w*={w_star*1e6:.2f} "
          f"d*={d_star*1e6:.2f} um (developed single track, "
          f"seg idx {l0['idx'].iloc[-1]})")
    with open(R.data_path("target.json", write=True), "w") as f:
        json.dump({"w_star": w_star, "d_star": d_star,
                   "seg_idx": int(l0['idx'].iloc[-1]),
                   "anchor": "developed single-track (end of line 0)"}, f,
                  indent=1)

    # 0.5 mm control segments (~2 pool relaxation times at 0.7 m/s): each
    # segment reaches its own quasi-steady pool before it is measured, so the
    # end-of-segment snapshot reflects the CURRENT controls rather than the
    # previous segment's superheat afterglow (0.25 mm segments produce a
    # period-2 hit/overshoot cycle no causal per-segment controller can fix)
    div, powered = R.build_path(dwell, seg=SEG_CTRL)
    info = R.seg_info(div, powered)

    # warm starts: the raster is periodic line to line, so the best prior for
    # a segment is the SAME within-line position one line below (a line-end
    # control is a terrible guess for the next line's fresh-ground start).
    rows, opt_pmods, opt_sigmas, total = [], {}, {}, 0
    u = np.array([R.PMOD0, R.SIGMA])
    by_pos, pos, cur_line = {}, -1, -1
    for e in info:
        pos = 0 if e['line'] != cur_line else pos + 1
        cur_line = e['line']
        u0 = by_pos.get(pos, u)
        print(f"   seg {e['idx']} (line {e['line']}, s={e['s_mm']:.2f} mm):")
        u, s, n = newton_segment(div, e['idx'], opt_pmods, opt_sigmas, u0,
                                 w_star, d_star)
        by_pos[pos] = u
        opt_pmods[e['idx']] = u[0]
        opt_sigmas[e['idx']] = u[1]
        total += n
        rows.append(dict(**e, pmod=u[0], sigma=u[1],
                         w=s['width'], d=s['depth'], asym=s['asym']))
    out = R.data_path("optimized.csv", write=True)
    pd.DataFrame(rows).to_csv(out, index=False)

    bw, bd = base['w'].values, base['d'].values
    ow = np.array([r['w'] for r in rows]); od = np.array([r['d'] for r in rows])
    cv = lambda v: 100.0 * np.std(v) / np.mean(v)
    print(f"\ntotal simulations: {total} ({total/len(info):.1f} per segment)")
    print(f"{'':>10} {'width CV%':>10} {'depth CV%':>10} {'w range(um)':>12} {'d range(um)':>12}")
    print(f"{'baseline':>10} {cv(bw):10.2f} {cv(bd):10.2f} "
          f"{(bw.max()-bw.min())*1e6:12.2f} {(bd.max()-bd.min())*1e6:12.2f}")
    print(f"{'optimised':>10} {cv(ow):10.2f} {cv(od):10.2f} "
          f"{(ow.max()-ow.min())*1e6:12.2f} {(od.max()-od.min())*1e6:12.2f}")
    print(f"wrote {out}")
