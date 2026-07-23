"""Generate a two-track 3DThesis experiment.

The geometry is exactly the first two lines of the 10 mm square raster:

    line 1: (0.0, 0.0) -> (10.0, 0.0) mm
    line 2: (10.0, 0.1) -> (0.0, 0.1) mm

``continuous`` keeps the beam on during the 0.1 mm serpentine turnaround.
``dwell`` turns the beam off, moves to the second-line start, and waits there
for ``--turn-dwell`` seconds before scanning the second line.
"""

from __future__ import annotations

import argparse
import math
import os


SIDE = 10.0       # mm
HATCH = 0.1       # mm
VELOCITY = 3.0    # m/s
POWER = 150.0     # W absorbed
SIGMA = 200e-6    # m
SIGMA_Z = 10e-6   # m
T0 = 1273.0       # K
T_LIQ = 1610.0    # K
KCOND = 26.6      # W/(m K)
CP = 600.0        # J/(kg K)
RHO = 7451.0      # kg/m^3
RES_X = 50e-6     # m
RES_Y = 50e-6     # m
DEFAULT_RES_Z = 5e-6
DEFAULT_TIMESTEP = 5e-5
MARGIN = 1.0      # mm
Y_MARGIN = 0.3    # mm; comfortably outside the two-track melt region


def write(path: str, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)


def path_rows(policy: str, dwell_s: float, side_mm: float,
              hatch_mm: float, velocity: float) -> list[str]:
    """Return the two-line path using the square demo's turnaround semantics."""
    rows = [
        "Mode\tX(mm)\tY(mm)\tZ(mm)\tPmod\tVel/Time",
        "1\t0.000000\t0.000000\t0\t0\t0",
        f"0\t{side_mm:.6f}\t0.000000\t0\t1\t{velocity:g}",
    ]
    if policy == "continuous":
        rows.append(
            f"0\t{side_mm:.6f}\t{hatch_mm:.6f}\t0\t1\t{velocity:g}")
    else:
        rows.append(
            f"1\t{side_mm:.6f}\t{hatch_mm:.6f}\t0\t0\t{dwell_s:g}")
    rows.append(f"0\t0.000000\t{hatch_mm:.6f}\t0\t1\t{velocity:g}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="case", help="output case directory")
    ap.add_argument("--policy", choices=("continuous", "dwell"),
                    default="continuous")
    ap.add_argument("--turn-dwell", type=float, default=0.0,
                    help="beam-off time before track two (dwell policy only)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--side", type=float, default=SIDE)
    ap.add_argument("--hatch", type=float, default=HATCH)
    ap.add_argument("--velocity", type=float, default=VELOCITY)
    ap.add_argument("--power", type=float, default=POWER)
    ap.add_argument("--preheat", type=float, default=T0)
    ap.add_argument("--timestep", type=float, default=DEFAULT_TIMESTEP)
    ap.add_argument("--resx", type=float, default=RES_X)
    ap.add_argument("--resy", type=float, default=RES_Y)
    ap.add_argument("--resz", type=float, default=DEFAULT_RES_Z)
    ap.add_argument("--zmin", type=float, default=-0.4e-3)
    ap.add_argument("--ymargin", type=float, default=Y_MARGIN,
                    help="transverse domain margin on each side in mm")
    ap.add_argument("--threads", type=int, default=14)
    args = ap.parse_args()

    if args.turn_dwell < 0.0:
        ap.error("--turn-dwell must be nonnegative")
    if args.policy == "continuous" and args.turn_dwell != 0.0:
        ap.error("--turn-dwell applies only to --policy dwell")

    name = args.name or ("TwoContinuous" if args.policy == "continuous"
                         else "TwoDwell")
    case = os.path.abspath(args.out)
    os.makedirs(os.path.join(case, "Data"), exist_ok=True)

    rows = path_rows(args.policy, args.turn_dwell, args.side,
                     args.hatch, args.velocity)
    write(os.path.join(case, "Path.txt"), "\n".join(rows) + "\n")

    write(os.path.join(case, "Beam.txt"), f"""Shape
{{
\tWidth_X\t\t{SIGMA:g}
\tWidth_Y\t\t{SIGMA:g}
\tDepth_Z\t\t{SIGMA_Z:g}
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
\tT_L\t{T_LIQ:g}
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

    m = MARGIN
    my = args.ymargin
    write(os.path.join(case, "Domain.txt"), f"""X
{{
\tMin\t{-m * 1e-3:.6f}
\tMax\t{(args.side + m) * 1e-3:.6f}
\tRes\t{args.resx:g}
}}
Y
{{
\tMin\t{-my * 1e-3:.6f}
\tMax\t{(args.hatch + my) * 1e-3:.6f}
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
\tName\t\t{name}
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

    line_time = 2.0 * args.side * 1e-3 / args.velocity
    turnaround_time = (args.hatch * 1e-3 / args.velocity
                       if args.policy == "continuous" else args.turn_dwell)
    total_time = line_time + turnaround_time
    powered_mm = 2.0 * args.side + (
        args.hatch if args.policy == "continuous" else 0.0)
    print(f"case: {case}")
    print(f"policy: {args.policy}; tracks: 2; powered path: {powered_mm:.3f} mm")
    print(f"turnaround: {turnaround_time*1e3:.3f} ms; "
          f"path time: {total_time*1e3:.3f} ms "
          f"({math.ceil(total_time / args.timestep)} tracking steps)")
    print(f"resolution: x={args.resx*1e6:g}, y={args.resy*1e6:g}, "
          f"z={args.resz*1e6:g} um")


if __name__ == "__main__":
    main()
