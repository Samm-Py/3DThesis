"""Validate the OTI velocity sensitivity dT_dv (DesignVar DV_V) against central
finite differences, at fixed path geometry.

The current (last) path segment's scan speed v is perturbed by rewriting the
last Path.txt row's Vel column; because the snapshot is taken at scan end, the
observation time moves with v exactly as the analytic derivative assumes. Two
checks per segment:

  * field level  -- (T(v+d) - T(v-d)) / 2d  vs the dT_dv column, on the
                    highest-signal grid points (measurement-free, tightest);
  * support fn   -- FD of the melt-pool half-width / depth (the same
                    ExtractIsoSupportSensitivities the optimizer uses) vs the
                    OTI dwidth_dv / ddepth_dv.

A single-track-like segment exercises the current-segment channel; a mid-raster
segment additionally exercises the history channel (observation-time shift
through frozen upstream nodes) -- it FAILS if only the kinematics were wired.

Run from a demo dir, e.g. melt_pool_demos/square/:
    python ../common/validate_dv.py [eps]        (default eps = 1e-2)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mp_lib as R
from common import measurement as Sim

EPS = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-2
PATH_FILE = os.path.join(R.CASE, "Path.txt")


def _set_last_vel(v):
    """Rewrite the Vel/Time column (index 5) of the last powered path row."""
    lines = open(PATH_FILE).read().splitlines()
    for i in range(len(lines) - 1, 0, -1):
        parts = lines[i].split("\t")
        if parts and parts[0].strip() in ("0", "1"):
            parts[5] = "%.12g" % v
            lines[i] = "\t".join(parts)
            break
    with open(PATH_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")


def _support_wd(csv):
    h, _dh, _c = Sim.ExtractIsoSupportSensitivities(csv, R.T_LIQ)
    return 0.5 * (h["y+"] + h["y-"]), h["z-"]


def _run_v(v):
    _set_last_vel(v)
    R.run_thesis(R.BIN_OTI)
    df = pd.read_csv(R.CSV)
    w, d = _support_wd(R.CSV)
    return df, w, d


def validate(div, pmods, seg, eps, label):
    # baseline OTI snapshot + analytic sensitivities
    R.run_segment(div, seg, pmods, R.BIN_OTI)
    oti = pd.read_csv(R.CSV)
    ana = R.sens()
    v0 = R.V
    dv = v0 * eps

    dfp, wp, dp = _run_v(v0 * (1.0 + eps))
    dfm, wm, dm = _run_v(v0 * (1.0 - eps))
    _set_last_vel(v0)                       # restore

    xyz = ["x", "y", "z"]
    assert np.allclose(oti[xyz].values, dfp[xyz].values) and \
           np.allclose(oti[xyz].values, dfm[xyz].values), "grid drifted"

    # --- field-level check on the highest-signal points ---
    ana_f = oti["dT_dv"].values
    fd_f = (dfp["T"].values - dfm["T"].values) / (2.0 * dv)
    top = np.argsort(-np.abs(ana_f))[:300]
    rel = np.abs(fd_f[top] - ana_f[top]) / np.abs(ana_f[top])
    field_med, field_max = float(np.median(rel)), float(np.max(rel))

    # --- support-function check (what the optimizer consumes) ---
    dwidth_fd = (wp - wm) / (2.0 * dv)
    ddepth_fd = (dp - dm) / (2.0 * dv)
    def relerr(fd, an):
        return abs(fd - an) / abs(an) if an != 0 else float("nan")
    w_rel = relerr(dwidth_fd, ana["dwidth_dv"])
    d_rel = relerr(ddepth_fd, ana["ddepth_dv"])

    print(f"\n=== {label}  (seg {seg}, eps {eps:g}, dv {dv:.4g} m/s) ===")
    print(f"  pool: width {ana['width']*1e6:.2f} um, depth {ana['depth']*1e6:.2f} um")
    print(f"  field dT_dv (top 300 pts): median rel {field_med*100:.3f}% , "
          f"max rel {field_max*100:.3f}%")
    print(f"  dwidth_dv: OTI {ana['dwidth_dv']:.6g}  FD {dwidth_fd:.6g}  "
          f"rel {w_rel*100:.3f}%")
    print(f"  ddepth_dv: OTI {ana['ddepth_dv']:.6g}  FD {ddepth_fd:.6g}  "
          f"rel {d_rel*100:.3f}%")
    # physics sanity
    ros = v0 * abs(ana["ddepth_dv"]) / (0.5 * ana["depth"])   # ~1 if d ~ v^-1/2
    print(f"  signs: ddepth_dv {'<0 OK' if ana['ddepth_dv']<0 else '>=0 FAIL'}, "
          f"dwidth_dv {'<0 OK' if ana['dwidth_dv']<0 else '>=0 FAIL'};  "
          f"Rosenthal v|dd/dv|/(d/2) = {ros:.2f}")
    return dict(field_med=field_med, field_max=field_max,
                w_rel=w_rel, d_rel=d_rel)


if __name__ == "__main__":
    div, powered, hop_after = R.build_path(0.0)
    pmods = {j: R.PMOD0 for j in powered}
    pmods.update({h: R.PMOD0 for h in hop_after.values()})

    # single-track-like (first segment, no history) and mid-raster (history
    # channel). Mid index chosen ~1/4 into the build for background heat.
    seg_single = powered[0]
    seg_mid = powered[min(len(powered) // 4, len(powered) - 1)]

    r1 = validate(div, pmods, seg_single, EPS, "single-track (current-seg channel)")
    r2 = validate(div, pmods, seg_mid, EPS, "mid-raster (history channel)")

    # Solver-correctness gate: the FIELD dT_dv is the exact quantity the solver
    # produces (measurement-free), and depth is extracted on the fine 5 um
    # z-grid. Width is extracted on the 50 um y-grid, so its central-FD estimate
    # is quantization-limited (24% at 50 um -> ~2% at 25 um, converging to the
    # grid-independent analytic dwidth_dv); it is reported but NOT a hard gate.
    ok = all(r["field_max"] < 0.01 and r["d_rel"] < 0.03 for r in (r1, r2))
    print(f"\n{'PASS' if ok else 'FAIL'}: field dT_dv < 1% and depth-FD < 3% on "
          f"both configurations: {ok}")
    print("note: dwidth_dv FD is y-grid-quantization-limited at 50 um "
          "(refine RES_XY to converge); the analytic value is grid-independent.")
