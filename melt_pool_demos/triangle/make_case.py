"""Generate 3DThesis cases for the 10 mm triangle raster demo.

The triangle uses the same process and material basis as the square and
single-track control studies: IN718 at 1273 K preheat, 150 W absorbed power,
3 m/s, sigma_xy = 200 um, sigma_z = 10 um, and a 0.1 mm hatch.  Continuous
serpentine and beam-off turnaround-dwell policies are both supported.

Outputs a self-contained case directory runnable as
    (cd <case>) && <3DThesis binary> ./ParamInput.txt
with Solidification/Volume tracking, and RDF + MP_Stats + G/V/tSol/depth
outputs so post-processing can rebuild the paper's Figures 16 and 17.
"""
import argparse
import math
import os

SQRT3 = math.sqrt(3.0)

# ---- physics shared with the square/control studies ----
SIDE   = 10.0      # mm, equilateral side
HATCH  = 0.1       # mm
V      = 3.0       # m/s (Sec. 3.2 IN718 EBM)
POWER  = 150.0     # W absorbed; must match common.mp_lib.P_BASE
SIGMA  = 200e-6    # m, sigma_xy
SIGZ   = 10e-6     # m, penetration sigma_z
T0     = 1273.0    # K preheat
TLIQ   = 1610.0    # K, IN718 liquidus
KCOND  = 26.6      # W/mK
CP     = 600.0     # J/kgK
RHO    = 7451.0    # kg/m^3
RESXY  = 50e-6     # m
RESZ   = 12.5e-6   # m
TSTEP  = 1e-4      # s tracking timestep (3DThesis solidification default)
MARGIN = 1.0       # mm domain margin around the triangle


def triangle_serpentine(side, hatch):
    """Serpentine raster lines covering an equilateral triangle with its base
    on y=0 from (0,0) to (side,0) and apex at (side/2, side*sqrt(3)/2).
    Returns [(x_start, x_end, y)] with alternating direction, base->apex."""
    height = side * SQRT3 / 2.0
    lines, j = [], 0
    while True:
        y = j * hatch
        if y >= height:
            break
        half = y / SQRT3                    # edge inset at this height
        x0, x1 = half, side - half
        if x1 - x0 <= 0.0:
            break
        if j % 2 == 0:
            lines.append((x0, x1, y))
        else:
            lines.append((x1, x0, y))
        j += 1
    return lines


def write(path, text):
    with open(path, "w") as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="case", help="case directory")
    ap.add_argument("--side", type=float, default=SIDE, help="triangle side (mm)")
    ap.add_argument("--hatch", type=float, default=HATCH, help="hatch spacing (mm)")
    ap.add_argument("--timestep", type=float, default=TSTEP)
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--name", default="Triangle")
    ap.add_argument("--turn-dwell", type=float, default=0.0,
                    help="beam-off dwell (s) at the start of each raster line")
    ap.add_argument("--power", type=float, default=POWER,
                    help="absorbed base power (W)")
    ap.add_argument("--preheat", type=float, default=T0)
    ap.add_argument("--resx", type=float, default=RESXY,
                    help="x resolution (m)")
    ap.add_argument("--resy", type=float, default=RESXY,
                    help="y resolution (m)")
    ap.add_argument("--resz", type=float, default=RESZ,
                    help="z resolution (m); 5e-6 is nearly converged for scalar "
                         "metrics, while 1e-6 gives a publication-smooth depth trace")
    ap.add_argument("--zmin", type=float, default=-0.4e-3,
                    help="minimum domain z coordinate (m)")
    ap.add_argument("--margin", type=float, default=MARGIN,
                    help="lateral domain margin around scan geometry (mm)")
    ap.add_argument("--nlines", type=int, default=0,
                    help="truncate after this many lines (0 = all)")
    args = ap.parse_args()

    case = os.path.abspath(args.out)
    os.makedirs(os.path.join(case, "Data"), exist_ok=True)

    lines = triangle_serpentine(args.side, args.hatch)
    if args.nlines > 0:
        lines = lines[:args.nlines]
    ymax = max(y for _, _, y in lines)

    # ---- Path: spot row sets the start point (beam off), then powered
    # line moves; the beam stays on through serpentine hops unless a
    # turnaround dwell is requested.
    rows = []
    x0, x1, y = lines[0]
    rows.append(f"1\t{x0:.6f}\t{y:.6f}\t0\t0\t0")
    rows.append(f"0\t{x1:.6f}\t{y:.6f}\t0\t1\t{V}")
    for x0, x1, y in lines[1:]:
        if args.turn_dwell > 0.0:
            # beam-off dwell parked at the next line's start
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
    scan_s = ((path_mm + hops_mm) * 1e-3 / V
              + args.turn_dwell * (len(lines) - 1))

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

    # Surface tracking: fastest, and it still records the FULL 3D pool --
    # depth columns are probed each step, so the solidification grid carries
    # G/V/tSol at every melted (x,y,z) and the RDF has melt/solidify event
    # pairs at all z levels (verified against the melted-grid z histogram).
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
