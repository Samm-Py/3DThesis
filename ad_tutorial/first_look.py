"""Step 1 of the tutorial: one case, both binaries.

The same inputs go through the plain-double build and the OTI build. The OTI
build writes seven extra columns next to T, and a second OTI run with
Compute/SeedSegment pointed at the dwell row shows that the controls belong to
one path row at a time.

Run:  python first_look.py
"""
import time

import thesis_case as tc

DWELL_ROW = 2
V = tc.SPEED_316H
path = [tc.spot(-3.0, 0.0),            # row 0: place the beam
        tc.line(-1.0, V),              # row 1: 2 mm at 1.35 m/s
        tc.spot(-1.0, 0.2e-3),         # row 2: 0.2 ms beam-off dwell
        tc.line(0.0, V)]               # row 3: 1 mm at 1.35 m/s (last row)
domain = tc.grid((-1.5e-3, 0.2e-3), (-0.25e-3, 0.25e-3), (-0.15e-3, 0.0), 10e-6)
setup = dict(path=path, domain=domain, material=tc.MATERIAL_316H, beam=tc.BEAM_316H)
case = tc.write_case("first_look", **setup)

timings = {}
for oti in (False, True):
    start = time.perf_counter()
    df = tc.run(case, oti=oti)
    timings[oti] = time.perf_counter() - start
    print(f"{'OTI' if oti else 'plain'} build columns: {', '.join(df.columns)}")
print(f"\n{len(df)} grid points; OTI run took {timings[True] / timings[False]:.1f}x "
      f"as long ({timings[False]:.2f} s vs {timings[True]:.2f} s)")

units = {"T": "K", "dT_dx": "K/m", "dT_dy": "K/m", "dT_dz": "K/m",
         "dT_dQ": "K/W", "dT_dsig": "K/m", "dT_dv": "K/(m/s)", "dT_dtau": "K/s"}
dwell = tc.run(tc.write_case("first_look_dwell", seed_segment=DWELL_ROW, **setup))
# A point beside the track, 0.5 mm behind the beam and 50 um below the surface.
row = ((df.x + 0.5e-3) ** 2 + (df.y - 0.1e-3) ** 2 + (df.z + 0.05e-3) ** 2).idxmin()
print(f"\n316H, at (x, y, z) = ({df.x[row] * 1e3:.2f}, {df.y[row] * 1e3:.2f}, "
      f"{df.z[row] * 1e3:.2f}) mm:")
print(f"  {'column':<8s} {'last row seeded':>16s} {'dwell seeded':>16s}")
for col, unit in units.items():
    print(f"  {col:<8s} {df[col][row]:16.6g} {dwell[col][row]:16.6g}  {unit}")
