"""Generate a 3DThesis case for the single-track base case.

Same calibrated physics as the square/triangle raster demos (Stump &
Plotkowski Sec. 3.5): IN718 at 1273 K preheat, 3 m/s, sigma_xy = 200 um,
sigma_z = 10 um, 150 W absorbed, melt isotherm 1610 K, 50 um in-plane grid.
The geometry is one straight bead-on-plate line of length `--length` mm at
y = 0 -- the simplest geometry in the shared pipeline, and the base case the
per-segment velocity optimizer is developed on. Surface tracking records the
full 3D pool (G/V/tSol + the RDF event list), so one run feeds fig17 (L/D/V vs
time), fig16 (G maps), the calibration, and the whole-track statistics.

Outputs a self-contained case directory runnable as
    (cd <case>) && <3DThesis binary> ./ParamInput.txt
"""
import argparse
import os

# ---- physics shared with the square/triangle demos (calibrated match) ----
LENGTH = 10.0      # mm, track length
V      = 3.0       # m/s
POWER  = 150.0     # W absorbed; Efficiency 1.0
SIGMA  = 200e-6    # m, sigma_xy
SIGZ   = 10e-6     # m, penetration sigma_z
T0     = 1273.0    # K preheat
TLIQ   = 1610.0    # K, IN718 liquidus (Table 1)
KCOND  = 26.6      # W/mK
CP     = 600.0     # J/kgK
RHO    = 7451.0    # kg/m^3
RESXY  = 50e-6     # m
RESZ   = 5e-6      # m (z-converged, matches calibration)
TSTEP  = 5e-5      # s tracking timestep
MARGIN = 1.0       # mm domain margin ahead/behind the track (x)
HALF_Y = 0.6       # mm domain half-width in y (track is ~0.3 mm wide)


def write(path, text):
    with open(path, "w") as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cases/case_base", help="case directory")
    ap.add_argument("--name", default="Track")
    ap.add_argument("--length", type=float, default=LENGTH, help="track length (mm)")
    ap.add_argument("--power", type=float, default=POWER)
    ap.add_argument("--preheat", type=float, default=T0)
    ap.add_argument("--resz", type=float, default=RESZ, help="z resolution (m)")
    ap.add_argument("--timestep", type=float, default=TSTEP)
    ap.add_argument("--zmin", type=float, default=-0.4e-3,
                    help="domain depth (m); developed pool depth ~ -0.06e-3")
    ap.add_argument("--threads", type=int, default=14)
    args = ap.parse_args()

    case = os.path.abspath(args.out)
    os.makedirs(os.path.join(case, "Data"), exist_ok=True)
    L = args.length

    # Path: spot row sets the start (beam off), then one powered line to (L, 0).
    write(os.path.join(case, "Path.txt"),
          "Mode\tX(mm)\tY(mm)\tZ(mm)\tPmod\tVel/Time\n"
          f"1\t0.000000\t0.000000\t0\t0\t0\n"
          f"0\t{L:.6f}\t0.000000\t0\t1\t{V}\n")

    write(os.path.join(case, "Beam.txt"), f"""Shape
{{
\tWidth_X\t\t{SIGMA:g}
\tWidth_Y\t\t{SIGMA:g}
\tDepth_Z\t\t{SIGZ:g}
}}
Intensity
{{
\tPower\t\t{args.power:g}
\tEfficiency\t1.0
}}
""")

    write(os.path.join(case, "Material.txt"), f"""Constants
{{
\tT_0\t{args.preheat:g}
\tT_L\t{TLIQ:g}
\tk\t{KCOND:g}
\tc\t{CP:g}
\tp\t{RHO:g}
}}
""")

    write(os.path.join(case, "Mode.txt"), f"""Solidification
{{
\tTracking\tSurface
\tTimestep\t{args.timestep:g}
\tOutputFrequency\t1000000
}}
""")

    write(os.path.join(case, "Domain.txt"), f"""X
{{
\tMin\t{(-MARGIN) * 1e-3:.6f}
\tMax\t{(L + MARGIN) * 1e-3:.6f}
\tRes\t{RESXY:g}
}}
Y
{{
\tMin\t{(-HALF_Y) * 1e-3:.6f}
\tMax\t{(HALF_Y) * 1e-3:.6f}
\tRes\t{RESXY:g}
}}
Z
{{
\tMin\t{args.zmin:g}
\tMax\t0
\tRes\t{args.resz:g}
}}
""")

    write(os.path.join(case, "Output.txt"), """Grid
{
\tx\t1
\ty\t1
\tz\t1
}
Temperature
{
\tT\t0
\tT_hist\t0
}
Solidification
{
\ttSol\t1
\tG\t1
\tV\t1
\tdepth\t1
\tnumMelt\t1
\tRDF\t1
\tMP_Stats\t1
}
Solidification+
{
\tH\t0
\tHx\t0
\tHy\t0
\tHz\t0
}
""")

    write(os.path.join(case, "Settings.txt"), f"""Compute
{{
\tMaxThreads\t{args.threads}
}}
""")

    write(os.path.join(case, "ParamInput.txt"), f"""Simulation
{{
\tName\t\t{args.name}
\tMode\t\tMode.txt
\tMaterial\tMaterial.txt
\tBeam\t\tBeam.txt
\tPath\t\tPath.txt
}}
Options
{{
\tDomain\t\tDomain.txt
\tOutput\t\tOutput.txt
\tSettings\tSettings.txt
}}
""")

    scan_s = L * 1e-3 / V
    print(f"case: {case}")
    print(f"single track: {L:.1f} mm at {V} m/s -> scan time {scan_s*1e3:.2f} ms "
          f"({int(scan_s / args.timestep)} timesteps)")


if __name__ == "__main__":
    main()
