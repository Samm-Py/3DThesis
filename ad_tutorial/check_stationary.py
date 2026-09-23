"""Step 2 of the tutorial: the simplest possible AD check.

One stationary Gaussian spot at the origin is on for 1 ms and the snapshot is
taken when it switches off. With a single path row, the seeded segment is the
whole history, so dT_dQ and dT_dsig are the full derivatives of T. Every
column the OTI build writes is compared with the closed-form reference in
analytic.py (scipy quadrature, no step sizes).

Also checks one exact identity: T - T0 is linear in the power, so
Q * dT_dQ must equal T - T0 at every point.

Run:  python check_stationary.py
"""
import numpy as np

import analytic
import thesis_case as tc

GRID_POINTS = 24   # analytic references are a few scipy integrals per point
LIMIT = 5e-4       # 0.05 %: quadrature and 6-digit CSV output, far below FD noise

material = tc.Material()
beam = tc.Beam(width=40e-6, depth=20e-6, power=100.0)
path = [tc.spot(0.0, 1e-3, pmod=1.0)]       # on for 1 ms, then snapshot
domain = tc.grid((-0.1e-3, 0.1e-3), (-0.1e-3, 0.1e-3), (-0.1e-3, 0.0), 10e-6)

df = tc.run(tc.write_case("stationary", path=path, domain=domain,
                          material=material, beam=beam))
ref = analytic.Reference(material, beam, path)

rise = df["T"].values - material.T0
rng = np.random.default_rng(1)
pts = rng.choice(np.where(rise > 20.0)[0], size=GRID_POINTS, replace=False)
xyz = [(df.x[i], df.y[i], df.z[i]) for i in pts]

exact = {
    "T": [ref.T(*p) - material.T0 for p in xyz],
    "dT_dx": [ref.grad(*p)[0] for p in xyz],
    "dT_dy": [ref.grad(*p)[1] for p in xyz],
    "dT_dz": [ref.grad(*p)[2] for p in xyz],
    "dT_dQ": [ref.dQ(*p) for p in xyz],
    "dT_dsig": [ref.dsig(*p) for p in xyz],
}

print(f"stationary spot, {GRID_POINTS} points: OTI column vs closed form")
gate = tc.Gate()
for col, values in exact.items():
    ours = df[col].values[pts] - (material.T0 if col == "T" else 0.0)
    gate.check(col, tc.max_rel_error(ours, values), LIMIT)

# A spot has no scan speed and a powered spot is not a dwell, so these two
# design variables are never seeded here: their columns must be exactly zero.
for col in ("dT_dv", "dT_dtau"):
    gate.zero(f"{col} == 0 (not seeded)", df[col])

hot = rise > 0.5 * rise.max()
identity = np.abs(beam.power * df["dT_dQ"].values[hot] - rise[hot]) / rise[hot]
gate.check("Q * dT_dQ == T - T0", float(identity.max()), 1e-5,
           "  (CSV keeps 6 digits)")
gate.exit()
