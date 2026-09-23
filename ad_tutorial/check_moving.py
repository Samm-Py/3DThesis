"""Step 4 of the tutorial: every design variable on a moving source.

Three cases, each compared with the closed form in analytic.py:

A. Seed the LAST segment of an 8-segment track (the solver default, and what
   a sequential controller uses). Q, sigma and v of that segment, plus the
   spatial gradient. Exercises the "inside" and "before" zones of the
   velocity rule in Calc.cpp (seed_ctx / dv_tau).

B. Seed a beam-off DWELL between two lines. Only the aging of the heat laid
   down before the dwell survives, so dT_dtau is the only nonzero control
   column.

C. Seed an OLDER segment (row 2 of 8). OTI now disagrees with the closed form
   by a few tenths of a percent to a few percent. That is not an AD error: the
   solver integrates old heat with coarse quadrature, and AD differentiates
   the solver's integral, not the exact one. The proof is exact linearity in
   power: with only that segment switched on, the plain-double solver's
   temperature rise equals Q * dT_dQ to every printed digit, and both miss the
   closed form by the same amount.

Run:  python check_moving.py
"""
import numpy as np

import analytic
import thesis_case as tc

N_PICK = 15
LIMIT = 5e-4       # 0.05 %, as in check_stationary.py
rng = np.random.default_rng(0)
beam = tc.Beam(width=40e-6, depth=20e-6, power=100.0)
V, SEG, N = 0.5, 0.25, 8                       # m/s, mm per segment, segments


def track(pmods=None):
    pmods = pmods or [1.0] * N
    return ([tc.spot(-N * SEG, 0.0)]
            + [tc.line(-N * SEG + (j + 1) * SEG, V, pmod=pmods[j]) for j in range(N)])


def compare(df, gate, material, columns, limit, label, info=False):
    """Compare OTI columns with the closed form at points where each matters."""
    for col, exact in columns.items():
        offset = material.T0 if col == "T" else 0.0
        ours = df[col].values - offset
        where = np.where(np.abs(ours) >= 0.1 * np.abs(ours).max())[0]
        pick = rng.choice(where, size=min(N_PICK, where.size), replace=False)
        ref = np.array([exact(df.x[i], df.y[i], df.z[i]) for i in pick]) - offset
        err = tc.max_rel_error(ours[pick], ref)
        if info:
            gate.info(label + col, err)
        else:
            gate.check(label + col, err, limit)


def zero(df, gate, columns, why):
    for col in columns:
        gate.zero(f"{col} == 0 ({why})", df[col])


gate = tc.Gate()
full_domain = tc.grid((-2.05e-3, 0.05e-3), (-0.15e-3, 0.15e-3), (-0.15e-3, 0.0), 10e-6)

# --- A: current segment ------------------------------------------------------
material = tc.Material()
path = track()
ref = analytic.Reference(material, beam, path)
df = tc.run(tc.write_case("moving_last", path=path, domain=full_domain,
                          material=material, beam=beam))
print("A. 8 x 0.25 mm track at 0.5 m/s, last segment seeded")
compare(df, gate, material, {
    "T": ref.T,
    "dT_dx": lambda *p: ref.grad(*p)[0],
    "dT_dy": lambda *p: ref.grad(*p)[1],
    "dT_dz": lambda *p: ref.grad(*p)[2],
    "dT_dQ": ref.dQ,
    "dT_dsig": ref.dsig,
    "dT_dv": ref.dv,
}, LIMIT, "")
zero(df, gate, ["dT_dtau"], "seeded row is a line")

# --- B: dwell ------------------------------------------------------------------
DWELL_ROW = 2
path = [tc.spot(-1.0, 0.0), tc.line(-0.5, V), tc.spot(-0.5, 0.3e-3), tc.line(-0.25, V)]
ref = analytic.Reference(material, beam, path)
df = tc.run(tc.write_case(
    "moving_dwell", path=path, seed_segment=DWELL_ROW, material=material, beam=beam,
    domain=tc.grid((-1.05e-3, -0.2e-3), (-0.15e-3, 0.15e-3), (-0.15e-3, 0.0), 10e-6)))
print("\nB. 0.5 mm line, 0.3 ms beam-off dwell, 0.25 mm line; dwell row seeded")
compare(df, gate, material, {"dT_dtau": lambda *p: ref.dtau(*p, DWELL_ROW)}, LIMIT, "")
zero(df, gate, ["dT_dQ", "dT_dsig", "dT_dv"], "seeded row is beam-off")

# --- C: older segment ----------------------------------------------------------
# T0 = 0 so the CSV's six significant digits resolve the rise, not 1273 K.
OLD = 2
material0 = tc.Material(T0=0.0)
path = track()
ref = analytic.Reference(material0, beam, path)
df = tc.run(tc.write_case("moving_old", path=path, domain=full_domain, seed_segment=OLD,
                          material=material0, beam=beam))
print(f"\nC. same track, segment {OLD} of {N} seeded (heat ~3 ms old at the snapshot)")
compare(df, gate, material0, {
    "dT_dsig": lambda *p: ref.dsig(*p, OLD),
    "dT_dv": lambda *p: ref.dv(*p, OLD),
}, None, "OTI vs closed form: ", info=True)

# Power is linear: segment OLD's share of the temperature is Q * dT_dQ.
only = track([1.0 if j + 1 == OLD else 0.0 for j in range(N)])
solo = tc.run(tc.write_case("moving_old_only", path=only, domain=full_domain,
                            material=material0, beam=beam), oti=False)["T"].values
ad = beam.power * df["dT_dQ"].values
hot = solo > 0.1 * solo.max()
pick = rng.choice(np.where(hot)[0], size=N_PICK, replace=False)
exact = np.array([beam.power * ref.dQ(df.x[i], df.y[i], df.z[i], OLD) for i in pick])
gate.info("OTI vs closed form: Q * dT_dQ", tc.max_rel_error(ad[pick], exact))
gate.info(f"T(only segment {OLD} on) vs closed form", tc.max_rel_error(solo[pick], exact))
gate.check(f"Q * dT_dQ == T(only segment {OLD} on)",
           float(np.max(np.abs(ad[hot] - solo[hot]) / solo[hot])), 1e-5,
           "  <- AD is exact for the solver")
gate.exit()
