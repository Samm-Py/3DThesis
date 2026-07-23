"""Generate 3DThesis cases for the 10 mm square raster demo.

Same physics as the triangle demo (Stump & Plotkowski Sec. 3.5 calibrated
match): IN718 at 1273 K preheat, 3 m/s, sigma_xy = 200 um, sigma_z = 10 um,
150 W absorbed, melt isotherm 1610 K, 50 um in-plane grid. The geometry is a
10 mm square rastered bottom-to-top with a 0.1 mm serpentine hatch — the
constant-line-length control case for the triangle optimization work.

`--single-track` emits one 10 mm line instead (bead-on-plate calibration:
developed pool size, startup transient, solidification tail).

Outputs a self-contained case directory runnable as
    (cd <case>) && <3DThesis binary> ./ParamInput.txt
"""
import argparse
import math
import os

# ---- physics shared with the triangle demo (calibrated published match) ----
SIDE   = 10.0      # mm, square side
HATCH  = 0.1       # mm
V      = 3.0       # m/s
POWER  = 150.0     # W absorbed (recovered Sec. 3.5 value); Efficiency 1.0
SIGMA  = 200e-6    # m, sigma_xy
SIGZ   = 10e-6     # m, penetration sigma_z
T0     = 1273.0    # K preheat
TLIQ   = 1610.0    # K, IN718 liquidus (Table 1)
KCOND  = 26.6      # W/mK
CP     = 600.0     # J/kgK
RHO    = 7451.0    # kg/m^3
RESXY  = 50e-6     # m
RESZ   = 12.5e-6   # m
TSTEP  = 1e-4      # s tracking timestep
MARGIN = 1.0       # mm domain margin around the scanned area


def square_serpentine(side, hatch):
    """Serpentine raster lines covering a square [0,side]x[0,side], bottom to
    top. Returns [(x_start, x_end, y)] with alternating direction."""
    lines, j = [], 0
    nlines = int(round(side / hatch)) + 1
    for j in range(nlines):
        y = j * hatch
        if y > side + 1e-9:
            break
        if j % 2 == 0:
            lines.append((0.0, side, y))
        else:
            lines.append((side, 0.0, y))
    return lines


def write(path, text):
    with open(path, "w") as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="case", help="case directory")
    ap.add_argument("--side", type=float, default=SIDE, help="square side (mm)")
    ap.add_argument("--hatch", type=float, default=HATCH, help="hatch spacing (mm)")
    ap.add_argument("--timestep", type=float, default=TSTEP)
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--name", default="Square")
    ap.add_argument("--turn-dwell", type=float, default=0.0,
                    help="beam-off dwell (s) parked at the start of each line")
    ap.add_argument("--power", type=float, default=POWER)
    ap.add_argument("--preheat", type=float, default=T0)
    ap.add_argument("--resx", type=float, default=RESXY,
                    help="x resolution (m)")
    ap.add_argument("--resy", type=float, default=RESXY,
                    help="y resolution (m)")
    ap.add_argument("--resz", type=float, default=RESZ, help="z resolution (m)")
    ap.add_argument("--zmin", type=float, default=-0.4e-3,
                    help="domain depth (m); pool depth stays above -0.13e-3")
    ap.add_argument("--margin", type=float, default=MARGIN,
                    help="lateral domain margin around scan geometry (mm)")
    ap.add_argument("--single-track", action="store_true",
                    help="one 10 mm line at y=0 (bead-on-plate calibration)")
    ap.add_argument("--nlines", type=int, default=0,
                    help="truncate the raster after this many lines (0 = all)")
    args = ap.parse_args()

    case = os.path.abspath(args.out)
    os.makedirs(os.path.join(case, "Data"), exist_ok=True)

    if args.single_track:
        lines = [(0.0, args.side, 0.0)]
    else:
        lines = square_serpentine(args.side, args.hatch)
        if args.nlines > 0:
            lines = lines[:args.nlines]
    ymax = max(y for _, _, y in lines)

    # ---- Path: spot row sets the start point (beam off), then powered
    # line moves; the beam stays ON through the serpentine hops unless a
    # turnaround dwell is requested.
    rows = []
    x0, x1, y = lines[0]
    rows.append(f"1\t{x0:.6f}\t{y:.6f}\t0\t0\t0")
    rows.append(f"0\t{x1:.6f}\t{y:.6f}\t0\t1\t{V}")
    for x0, x1, y in lines[1:]:
        if args.turn_dwell > 0.0:
            rows.append(f"1\t{x0:.6f}\t{y:.6f}\t0\t0\t{args.turn_dwell:g}")
        else:
            rows.append(f"0\t{x0:.6f}\t{y:.6f}\t0\t1\t{V}")   # hop, beam on
        rows.append(f"0\t{x1:.6f}\t{y:.6f}\t0\t1\t{V}")       # scan line
    write(os.path.join(case, "Path.txt"),
          "Mode\tX(mm)\tY(mm)\tZ(mm)\tPmod\tVel/Time\n" + "\n".join(rows) + "\n")

    path_mm = sum(abs(x1 - x0) for x0, x1, y in lines)
    hops_mm = sum(math.hypot(lines[i + 1][0] - lines[i][1],
                             lines[i + 1][2] - lines[i][2])
                  for i in range(len(lines) - 1))
    scan_s = (path_mm + hops_mm) * 1e-3 / V + args.turn_dwell * (len(lines) - 1)

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

    # Surface tracking records the full 3D pool (see triangle demo notes):
    # one run yields G/V/tSol at every melted (x,y,z) plus the RDF event list.
    write(os.path.join(case, "Mode.txt"), f"""Solidification
{{
\tTracking\tSurface
\tTimestep\t{args.timestep:g}
\tOutputFrequency\t1000000
}}
""")

    m = args.margin
    write(os.path.join(case, "Domain.txt"), f"""X
{{
\tMin\t{(-m) * 1e-3:.6f}
\tMax\t{(args.side + m) * 1e-3:.6f}
\tRes\t{args.resx:g}
}}
Y
{{
\tMin\t{(-m) * 1e-3:.6f}
\tMax\t{(ymax + m) * 1e-3:.6f}
\tRes\t{args.resy:g}
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
\tGx\t0
\tGy\t0
\tGz\t0
\tV\t1
\tdTdt\t0
\teqFrac\t0
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

    print(f"case: {case}")
    print(f"raster lines: {len(lines)}  powered path: {path_mm:.1f} mm "
          f"+ hops {hops_mm:.1f} mm  -> scan time {scan_s * 1e3:.2f} ms "
          f"({int(scan_s / args.timestep)} timesteps)")


if __name__ == "__main__":
    main()
