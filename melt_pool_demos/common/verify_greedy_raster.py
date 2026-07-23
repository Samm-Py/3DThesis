"""Tracked replay of a square/triangle greedy schedule without optimization."""

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
    raise RuntimeError("run verify_greedy_raster.py from square/ or triangle/")
sys.path.insert(0, COMMON)
sys.path.insert(0, DEMOS)

os.environ.setdefault("MP_GEOM", GEOMETRY)

from common import mp_lib as R  # noqa: E402
from common import scan as Scan  # noqa: E402
from greedy_raster import dwell_for, result_path  # noqa: E402


def load_schedule(schedule_file: str, policy: str) -> tuple[dict, list[str]]:
    with open(schedule_file) as f:
        saved = json.load(f)
    if saved["policy"] != policy:
        raise RuntimeError(
            f"schedule policy {saved['policy']!r} does not match {policy!r}")
    nlines = int(saved["nlines"])
    dwell = dwell_for(policy)
    div, powered, hop_after = R.build_path(
        dwell, seg=float(saved["segment_mm"]), nlines=nlines, balanced=True)
    info = R.seg_info(div, powered)
    if len(info) != len(saved["schedule"]):
        raise RuntimeError("saved schedule does not match reconstructed path")

    pmods, sigmas, velocities = {}, {}, {}
    for segment, control in zip(info, saved["schedule"]):
        idx = segment["idx"]
        pmods[idx] = float(control["power_w"]) / R.P_BASE
        sigmas[idx] = float(control["sigma_um"]) * 1e-6
        velocities[idx] = float(control["velocity_m_per_s"])
    for before, hop in hop_after.items():
        pmods[hop], sigmas[hop], velocities[hop] = (
            pmods[before], sigmas[before], velocities[before])
    wmods = {idx: sigma / R.SIGMA for idx, sigma in sigmas.items()}
    return saved, R.set_pmods(div, pmods, wmods, vels=velocities)


def run_replay(schedule_file: str, method: str, policy: str, resx: float,
               resy: float, resz: float, margin: float,
               zmin: float) -> dict:
    saved, rows = load_schedule(schedule_file, policy)
    nlines = int(saved["nlines"])
    label = policy.capitalize()
    resolution = (
        f"X{resx*1e6:g}Y{resy*1e6:g}Z{resz*1e6:g}".replace(".", "p"))
    prefix = "Sq" if GEOMETRY == "square" else "Tri"
    name = f"{prefix}{method}{resolution}{label}"
    case = os.path.join(
        DEMO, "cases", f"case_{method.lower()}_{resolution.lower()}_{policy}")
    shutil.rmtree(case, ignore_errors=True)

    command = [
        sys.executable, os.path.join(DEMO, "make_case.py"),
        "--out", case,
        "--name", name,
        "--nlines", str(nlines),
        "--resx", str(resx),
        "--resy", str(resy),
        "--resz", str(resz),
        "--timestep", "5e-5",
        "--power", str(R.P_BASE),
        "--margin", str(margin),
        "--zmin", str(zmin),
    ]
    dwell = dwell_for(policy)
    if policy == "dwell":
        command.extend(("--turn-dwell", str(dwell)))
    subprocess.run(command, cwd=DEMO, check=True)
    Scan.ExportScan(rows, outFile=os.path.join(case, "Path.txt"))

    env = os.environ.copy()
    env.setdefault("I_MPI_FABRICS", env.get("THESIS_MPI_FABRICS", "shm"))
    subprocess.run(
        [R.BIN_DBL, "./ParamInput.txt"], cwd=case, env=env, check=True)
    subprocess.run(
        [sys.executable, os.path.join(DEMOS, "common", "fullfield_stats.py"),
         case, name],
        cwd=DEMO, check=True)
    stats_file = result_path(
        os.path.join("fullfield", f"{name}_fullfield.json"))
    with open(stats_file) as f:
        return {"name": name, "case": case, "stats": json.load(f)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("schedule", help="schedule JSON produced by optimizer")
    parser.add_argument("method", help="case/name tag, e.g. Greedy1")
    parser.add_argument("--policy", choices=("zero", "dwell"), required=True)
    parser.add_argument("--resx", type=float, default=50e-6)
    parser.add_argument("--resy", type=float, default=50e-6)
    parser.add_argument("--resz", type=float, default=1e-6)
    parser.add_argument("--margin", type=float, default=0.3,
                        help="lateral domain padding (mm)")
    parser.add_argument("--zmin", type=float, default=-0.16e-3,
                        help="minimum domain z coordinate (m)")
    args = parser.parse_args()
    result = run_replay(
        args.schedule, args.method, args.policy,
        args.resx, args.resy, args.resz, args.margin, args.zmin)
    print(json.dumps({"name": result["name"], "stats": result["stats"]},
                     indent=2))


if __name__ == "__main__":
    main()
