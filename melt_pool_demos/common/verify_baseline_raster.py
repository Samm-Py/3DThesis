"""Tracked refined-grid replay of a square/triangle nominal baseline."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys


COMMON = os.path.dirname(os.path.abspath(__file__))
DEMOS = os.path.dirname(COMMON)
DEMO = os.path.abspath(os.getcwd())
GEOMETRY = os.path.basename(DEMO)
if GEOMETRY not in ("square", "triangle"):
    raise RuntimeError(
        "run verify_baseline_raster.py from square/ or triangle/")
sys.path.insert(0, COMMON)
sys.path.insert(0, DEMOS)
os.environ.setdefault("MP_GEOM", GEOMETRY)

from common import mp_lib as R  # noqa: E402
from greedy_raster import dwell_for, result_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("zero", "dwell"), required=True)
    parser.add_argument("--resx", type=float, default=50e-6)
    parser.add_argument("--resy", type=float, default=10e-6)
    parser.add_argument("--resz", type=float, default=1e-6)
    parser.add_argument("--margin", type=float, default=0.3,
                        help="lateral domain padding (mm)")
    parser.add_argument("--zmin", type=float, default=-0.16e-3,
                        help="minimum domain z coordinate (m)")
    args = parser.parse_args()

    resolution = (
        f"X{args.resx*1e6:g}Y{args.resy*1e6:g}Z{args.resz*1e6:g}"
        .replace(".", "p"))
    prefix = "Sq" if GEOMETRY == "square" else "Tri"
    name = f"{prefix}Baseline{resolution}{args.policy.capitalize()}"
    case = os.path.join(
        DEMO, "cases", f"case_baseline_{resolution.lower()}_{args.policy}")
    shutil.rmtree(case, ignore_errors=True)

    command = [
        sys.executable,
        os.path.join(DEMO, "make_case.py"),
        "--out", case,
        "--name", name,
        "--resx", str(args.resx),
        "--resy", str(args.resy),
        "--resz", str(args.resz),
        "--timestep", "5e-5",
        "--power", str(R.P_BASE),
        "--margin", str(args.margin),
        "--zmin", str(args.zmin),
    ]
    if args.policy == "dwell":
        command.extend(("--turn-dwell", str(dwell_for(args.policy))))
    subprocess.run(command, cwd=DEMO, check=True)

    env = os.environ.copy()
    env.setdefault("I_MPI_FABRICS", env.get("THESIS_MPI_FABRICS", "shm"))
    subprocess.run(
        [R.BIN_DBL, "./ParamInput.txt"], cwd=case, env=env, check=True)
    subprocess.run(
        [sys.executable, os.path.join(COMMON, "fullfield_stats.py"),
         case, name],
        cwd=DEMO, check=True)

    stats_file = result_path(
        os.path.join("fullfield", f"{name}_fullfield.json"))
    with open(stats_file) as f:
        print(json.dumps({"name": name, "stats": json.load(f)}, indent=2))


if __name__ == "__main__":
    main()
