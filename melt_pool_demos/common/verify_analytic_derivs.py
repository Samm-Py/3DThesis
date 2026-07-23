"""Verify the OTI field derivatives against CLOSED-FORM analytic expressions
(a stronger check than finite differences: no step size, no truncation, no
subtractive cancellation).

Configuration: a single STATIONARY Gaussian spot at the origin, powered for a
time t_on, snapshot at t_on. One path segment => every quadrature node is the
"current" segment, so the OTI dT_dQ / dT_dsig columns are the FULL derivatives
(not per-segment-restricted), and the path integral is a 1-D integral over the
conduction time u = t - t' that scipy evaluates as the analytic reference:

    T(x) - T0 = A \\int_0^{t_on} K(u) du,
    K(u) = (phix phiy phiz)^{-1/2}
           exp(-3[ x^2/phix + y^2/phiy + z^2/phiz ]),
    phi_i(u) = 12 alpha u + sigma_i^2,   A = 2 eta Q / (rho c (pi/3)^{3/2}).

Analytic derivatives (all differentiate UNDER the integral -- x,y,z,Q,sigma
never touch the nodes or limits, cf. Appendix B of Stump & Plotkowski for the
spatial gradients G_x,G_y,G_z; the sigma form is derived here since sigma_x =
sigma_y = sigma is one variable and sigma_z is held fixed):

    dT/dx = A \\int K (-6 x / phix) du                       (B.1, dimensional)
    dT/dz = A \\int K (-6 z / phiz) du                       (B.4, dimensional)
    dT/dQ = (T - T0) / Q                                     (Q is linear)
    dT/dsig = A \\int K sigma [ 6(x^2+y^2)/phix^2 - 2/phix ] du   (phix=phiy)

Run: python common/verify_analytic_derivs.py
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
                    "_analytic_check_case")
CSV = os.path.join(CASE, "Data", "TestSim.Snapshot.00.csv")

# material + process (η = 1); alpha = k / (rho c)
K_COND, CP, RHO, T0 = 26.6, 600.0, 7451.0, 1273.0
ALPHA = K_COND / (RHO * CP)
P = 100.0                      # W
SIG_XY, SIG_Z = 40e-6, 20e-6   # m
T_ON = 1.0e-3                  # s the spot is powered
RES = 5e-6
HALF = 2.0e-4                  # domain half-extent (m)
A_PREF = 2.0 * 1.0 * P / (RHO * CP * (np.pi / 3.0) ** 1.5)


def write_case():
    ensure_snapshot_case(CASE, initial_temperature=T0, conductivity=K_COND,
                         heat_capacity=CP, density=RHO)
    open(os.path.join(CASE, "Beam.txt"), "w").write(
        f"Shape\n{{\n\tWidth_X\t\t{SIG_XY}\n\tWidth_Y\t\t{SIG_XY}\n"
        f"\tDepth_Z\t\t{SIG_Z}\n}}\n"
        f"Intensity\n{{\n\tPower\t\t{P}\n\tEfficiency\t1.0\n}}\n")
    # one stationary spot (Mode 1) at the origin, powered for T_ON seconds
    with open(os.path.join(CASE, "Path.txt"), "w") as f:
        f.write("Mode\tX(mm)\tY(mm)\tZ(mm)\tPmod\tVel(m/s)/Time\n")
        f.write(f"1\t0\t0\t0\t1.0\t{T_ON:.8f}\n")
    open(os.path.join(CASE, "Domain.txt"), "w").write(
        f"X\n{{\n\tMin\t{-HALF:.7f}\n\tMax\t{HALF:.7f}\n\tRes\t{RES}\n}}\n"
        f"Y\n{{\n\tMin\t{-HALF:.7f}\n\tMax\t{HALF:.7f}\n\tRes\t{RES}\n}}\n"
        f"Z\n{{\n\tMin\t{-HALF:.7f}\n\tMax\t0\n\tRes\t{RES}\n}}\n")


def _phi(u):
    p_xy = 12.0 * ALPHA * u + SIG_XY ** 2
    p_z = 12.0 * ALPHA * u + SIG_Z ** 2
    return p_xy, p_xy, p_z


def _K(u, x, y, z):
    px, py, pz = _phi(u)
    return (px * py * pz) ** -0.5 * np.exp(
        -3.0 * (x * x / px + y * y / py + z * z / pz))


def analytic(x, y, z):
    I = lambda g: quad(g, 0.0, T_ON, limit=200)[0]
    T = T0 + A_PREF * I(lambda u: _K(u, x, y, z))
    dTdx = A_PREF * I(lambda u: _K(u, x, y, z) * (-6.0 * x / _phi(u)[0]))
    dTdy = A_PREF * I(lambda u: _K(u, x, y, z) * (-6.0 * y / _phi(u)[1]))
    dTdz = A_PREF * I(lambda u: _K(u, x, y, z) * (-6.0 * z / _phi(u)[2]))
    dTdsig = A_PREF * I(lambda u: _K(u, x, y, z) * SIG_XY * (
        6.0 * (x * x + y * y) / _phi(u)[0] ** 2 - 2.0 / _phi(u)[0]))
    dTdQ = (T - T0) / P
    return dict(T=T, dT_dx=dTdx, dT_dy=dTdy, dT_dz=dTdz, dT_dsig=dTdsig,
                dT_dQ=dTdQ)


if __name__ == "__main__":
    write_case()
    subprocess.run([BIN_OTI, "./ParamInput.txt"], cwd=CASE,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    d = pd.read_csv(CSV)

    # pick well-signalled points spread over the octant (avoid the origin,
    # where the odd gradients vanish by symmetry)
    dT = d["T"].values - T0
    r = np.sqrt(d.x ** 2 + d.y ** 2 + d.z ** 2).values
    cand = np.where((dT > 5.0) & (r > 1.5 * SIG_XY))[0]
    rng = np.random.default_rng(0)
    pick = rng.choice(cand, size=min(24, cand.size), replace=False)

    cols = ["T", "dT_dx", "dT_dy", "dT_dz", "dT_dsig", "dT_dQ"]
    err = {c: [] for c in cols}
    print(f"stationary spot: alpha={ALPHA:.3e} m^2/s, sigma_xy={SIG_XY*1e6:.0f} "
          f"um, t_on={T_ON*1e3:.1f} ms, {len(pick)} sample points")
    print("relative error, OTI (analytic-differentiated quadrature) vs scipy "
          "analytic integral:")
    for i in pick:
        a = analytic(d.x[i], d.y[i], d.z[i])
        for c in cols:
            oti = d[c].values[i]
            ref = a[c]
            if abs(ref) > 1e-30:
                err[c].append(abs(oti - ref) / abs(ref))
    for c in cols:
        e = np.array(err[c])
        print(f"  {c:8s}: median {np.median(e)*100:8.4f}%   max {e.max()*100:8.4f}%")
    # Q is a pure linear prefactor, so dT/dQ = (T-T0)/Q holds against the
    # solver's OWN T, to machine precision (no quadrature error enters).
    # Q is a pure linear prefactor, so dT/dQ = (T-T0)/Q holds ALGEBRAICALLY
    # against the solver's own T; the only residual is the snapshot's ~6-7 sig
    # fig text precision, which shrinks with signal magnitude.
    qref = (d["T"].values - T0) / P
    qrel = np.abs(d["dT_dQ"].values - qref) / np.abs(qref)
    strong = dT > 0.25 * dT.max()
    print(f"\ndT_dQ vs (T-T0)/Q on the solver's own T (exact linear relation; "
          f"residual = snapshot text precision):")
    print(f"  max rel: {qrel[strong].max():.1e} (strong signal) -> "
          f"{qrel[dT>0.9*dT.max()].max():.1e} (highest signal)")
