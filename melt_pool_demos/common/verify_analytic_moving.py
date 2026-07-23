"""Analytic verification of the OTI field derivatives on a MOVING source (the
raster-relevant case), extending verify_analytic_derivs.py from the stationary
spot.

A straight track of N equal segments runs at speed v0 from x0 to the origin;
the snapshot is at scan end. For a moving source the temperature is still a
1-D integral over the deposition time t', with the beam at x_b(t') = x0 + v0 t':

    T - T0 = A \\int_0^{t_end} K(t') dt',
    K = (phix phiy phiz)^{-1/2} exp(-3[ (xp-x_b)^2/phix + yp^2/phiy + zp^2/phiz ]),
    phi_i = 12 alpha (t_end - t') + sigma_i^2,   A = 2 eta Q / (rho c (pi/3)^{3/2}).

The OTI seeding rules dictate the analytic integration ranges. With the SEEDED
segment j spanning deposition times [t_{j-1}, t_j] (default: j = last):
  * dT/dx,dy,dz  -- full path [0, t_end]           (evaluation point is global)
  * dT/dQ,dsig   -- seeded segment only [t_{j-1}, t_j]
  * dT/dv        -- Dt = L/v of the seeded segment perturbs, in three zones:
                    nodes BEFORE j see the observation time shift (H channel,
                    [0, t_{j-1}]); nodes INSIDE j stretch with the segment
                    (W weight channel + C conduction channel weighted by
                    (t_j - t')); nodes AFTER j are deposited later and observed
                    later by the same amount -- exactly zero contribution.
                    d/dv = -(L/v^2) d/dDt; scipy evaluates all channels. For
                    j = last (t_j = t_end) this reduces to the original check.
                    This is the analytic analogue of the SeedCtx / dv_tau
                    three-zone construction in Calc.cpp, and a stronger check
                    than the finite-difference validation in validate_dv.py.

Run: python common/verify_analytic_moving.py
"""
import os
import sys
import subprocess

import numpy as np
import pandas as pd
from scipy.integrate import quad

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.case import ensure_snapshot_case

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN_OTI = os.environ.get("THESIS_BIN_OTI",
                         os.path.join(_ROOT, "build-oti", "bin", "3DThesis"))
CASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "_analytic_moving_case")
CSV = os.path.join(CASE, "Data", "TestSim.Snapshot.00.csv")

K_COND, CP, RHO, T0 = 26.6, 600.0, 7451.0, 1273.0
ALPHA = K_COND / (RHO * CP)
P = 100.0
SIG_XY, SIG_Z = 40e-6, 20e-6
V0 = 0.5                      # m/s
L_SEG = 0.25e-3               # m, per segment
N_SEG = 8                     # -> 2 mm track
RES = 5e-6
A_PREF = 2.0 * 1.0 * P / (RHO * CP * (np.pi / 3.0) ** 1.5)

X0 = -N_SEG * L_SEG           # track start (m); ends at origin
T_END = (N_SEG * L_SEG) / V0
DT_LAST = L_SEG / V0          # last-segment duration
SEED_J = None                 # powered segment to seed (1..N_SEG); None = last
TJ0, TJ1 = T_END - DT_LAST, T_END   # seeded segment's deposition window


def write_case():
    ensure_snapshot_case(CASE, initial_temperature=T0, conductivity=K_COND,
                         heat_capacity=CP, density=RHO)
    if SEED_J is not None:
        # SeedSegment = 0-based Path.txt data row; row 0 is the unpowered
        # spot, so powered segment j is data row j.
        open(os.path.join(CASE, "Settings.txt"), "w").write(
            f"Compute\n{{\n\tMaxThreads\t4\n\tSeedSegment\t{SEED_J}\n}}\n")
    open(os.path.join(CASE, "Beam.txt"), "w").write(
        f"Shape\n{{\n\tWidth_X\t\t{SIG_XY}\n\tWidth_Y\t\t{SIG_XY}\n"
        f"\tDepth_Z\t\t{SIG_Z}\n}}\n"
        f"Intensity\n{{\n\tPower\t\t{P}\n\tEfficiency\t1.0\n}}\n")
    with open(os.path.join(CASE, "Path.txt"), "w") as f:
        f.write("Mode\tX(mm)\tY(mm)\tZ(mm)\tPmod\tVel(m/s)/Time\n")
        f.write(f"1\t{X0*1e3:.6f}\t0\t0\t0\t0\n")            # unpowered start
        for j in range(1, N_SEG + 1):
            xj = (X0 + j * L_SEG) * 1e3
            f.write(f"0\t{xj:.6f}\t0\t0\t1.0\t{V0}\n")
    # domain over the trailing pool near the origin
    open(os.path.join(CASE, "Domain.txt"), "w").write(
        f"X\n{{\n\tMin\t{-0.5e-3:.7f}\n\tMax\t{0.05e-3:.7f}\n\tRes\t{RES}\n}}\n"
        f"Y\n{{\n\tMin\t{-0.15e-3:.7f}\n\tMax\t{0.15e-3:.7f}\n\tRes\t{RES}\n}}\n"
        f"Z\n{{\n\tMin\t{-0.15e-3:.7f}\n\tMax\t0\n\tRes\t{RES}\n}}\n")


def _xb(tp):
    return X0 + V0 * tp                       # beam x at deposition time tp


def _phi(tp):                                  # widths at conduction time t_end-tp
    p_xy = 12.0 * ALPHA * (T_END - tp) + SIG_XY ** 2
    p_z = 12.0 * ALPHA * (T_END - tp) + SIG_Z ** 2
    return p_xy, p_xy, p_z


def _K(tp, x, y, z):
    px, py, pz = _phi(tp)
    rx = x - _xb(tp)
    return (px * py * pz) ** -0.5 * np.exp(
        -3.0 * (rx * rx / px + y * y / py + z * z / pz))


def _logderiv_sum(tp, x, y, z):
    """Sum_i [3 r_i^2/phi_i^2 - 1/(2 phi_i)] -- d(ln K)/d(phi-shift common to all
    axes), used for the v channels (a uniform shift of all phi_i)."""
    px, py, pz = _phi(tp)
    rx = x - _xb(tp)
    return (3 * rx * rx / px ** 2 - 0.5 / px
            + 3 * y * y / py ** 2 - 0.5 / py
            + 3 * z * z / pz ** 2 - 0.5 / pz)


def analytic(x, y, z):
    Q = lambda g, a, b: quad(g, a, b, limit=200)[0]
    T = T0 + A_PREF * Q(lambda t: _K(t, x, y, z), 0.0, T_END)
    dTdx = A_PREF * Q(lambda t: _K(t, x, y, z) * (-6.0 * (x - _xb(t)) / _phi(t)[0]), 0.0, T_END)
    dTdy = A_PREF * Q(lambda t: _K(t, x, y, z) * (-6.0 * y / _phi(t)[1]), 0.0, T_END)
    dTdz = A_PREF * Q(lambda t: _K(t, x, y, z) * (-6.0 * z / _phi(t)[2]), 0.0, T_END)
    # Q, sigma: seeded segment only
    dTdQ = A_PREF / P * Q(lambda t: _K(t, x, y, z), TJ0, TJ1)
    dTdsig = A_PREF * Q(lambda t: _K(t, x, y, z) * SIG_XY * (
        6.0 * ((x - _xb(t)) ** 2 + y * y) / _phi(t)[0] ** 2 - 2.0 / _phi(t)[0]),
        TJ0, TJ1)
    # velocity: d/dv = -(L/v^2) d/dDt, three zones of d/dDt. Nodes after the
    # seeded segment contribute exactly zero (deposition and observation shift
    # together), so the integrals stop at TJ1.
    H = 12.0 * ALPHA * Q(lambda t: _K(t, x, y, z) * _logderiv_sum(t, x, y, z), 0.0, TJ0)
    W = (1.0 / DT_LAST) * Q(lambda t: _K(t, x, y, z), TJ0, TJ1)
    C = (12.0 * ALPHA / DT_LAST) * Q(
        lambda t: _K(t, x, y, z) * _logderiv_sum(t, x, y, z) * (TJ1 - t),
        TJ0, TJ1)
    dTdv = -(L_SEG / V0 ** 2) * A_PREF * (H + W + C)
    return dict(T=T, dT_dx=dTdx, dT_dy=dTdy, dT_dz=dTdz,
                dT_dQ=dTdQ, dT_dsig=dTdsig, dT_dv=dTdv)


def configure(n_seg, v0, seed_j=None):
    """Set the track geometry (a given observation time t_end = n_seg*L/v0)
    and the seeded powered segment (1..n_seg; None = last)."""
    global N_SEG, V0, X0, T_END, DT_LAST, SEED_J, TJ0, TJ1
    N_SEG, V0 = n_seg, v0
    X0 = -n_seg * L_SEG
    T_END = (n_seg * L_SEG) / v0
    DT_LAST = L_SEG / v0
    SEED_J = seed_j
    j = n_seg if seed_j is None else seed_j
    TJ0, TJ1 = (j - 1) * DT_LAST, j * DT_LAST


def run_check():
    write_case()
    subprocess.run([BIN_OTI, "./ParamInput.txt"], cwd=CASE,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    d = pd.read_csv(CSV)
    dT = d["T"].values - T0
    cand = np.where(dT > 20.0)[0]
    pick = np.random.default_rng(1).choice(cand, size=min(20, cand.size),
                                           replace=False)
    cols = ["T", "dT_dx", "dT_dy", "dT_dz", "dT_dQ", "dT_dsig", "dT_dv"]
    ref = {c: [] for c in cols}
    oti = {c: [] for c in cols}
    for i in pick:
        a = analytic(d.x[i], d.y[i], d.z[i])
        for c in cols:
            ref[c].append(a[c])
            oti[c].append(d[c].values[i])
    seed_lbl = "last" if SEED_J is None else f"j={SEED_J}/{N_SEG}"
    print(f"\nmoving track: v0={V0} m/s, {N_SEG}x{L_SEG*1e3:.2f} mm segments, "
          f"t_end={T_END*1e3:.2f} ms, seed {seed_lbl}, {len(pick)} sample points")
    print("  relative error, OTI vs scipy analytic (max excludes zero-crossings, "
          "|ref| < 10% of peak):")
    for c in cols:
        r, o = np.array(ref[c]), np.array(oti[c])
        rel = np.abs(o - r) / np.maximum(np.abs(r), 1e-25)
        meaningful = np.abs(r) > 0.1 * np.abs(r).max()   # drop zero-crossings
        tag = ("[full path]" if c in ("dT_dx", "dT_dy", "dT_dz") else
               "[seeded segment]" if c in ("dT_dQ", "dT_dsig") else
               "[3-zone v]" if c == "dT_dv" else "")
        print(f"    {c:8s}: median {np.median(rel)*100:8.4f}%   "
              f"max {rel[meaningful].max()*100:8.4f}%   {tag}")


if __name__ == "__main__":
    # seed=None regressions (the pre-existing last-segment checks), then
    # middle-segment seeds exercising all three zones of the generalized rule.
    for n_seg, v0, seed_j in [(8, 0.5, None), (4, 1.0, None),
                              (8, 0.5, 6), (8, 0.5, 4), (8, 0.5, 2)]:
        configure(n_seg, v0, seed_j)
        run_check()
