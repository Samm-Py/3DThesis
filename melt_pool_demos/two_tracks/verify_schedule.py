"""High-resolution tracked replay of an arbitrary optimized schedule.

Takes any schedule JSON produced by ``optimize_greedy.py`` and replays it
without re-optimization on the presentation tracking grid, writing the
compact full-field statistics under the given method name.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
DEMOS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, DEMOS)

os.environ.setdefault("MP_GEOM", "two_tracks")

from common import mp_lib as R  # noqa: E402
from common import scan as Scan  # noqa: E402
from optimize_greedy import dwell_for, result_path  # noqa: E402


def schedule_rows(schedule_file: str, policy: str) -> list[str]:
    with open(schedule_file) as f:
        saved = json.load(f)
    dwell = dwell_for(policy)
    div, powered, hop_after = R.build_path(dwell, seg=saved["segment_mm"])
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
    rows = R.set_pmods(div, pmods, wmods, vels=velocities)
    turn_dwells = saved.get("turnaround_dwells_s", [])
    if turn_dwells:
        line_starts = [
            next(segment for segment in info if segment["line"] == line)
            for line in sorted({segment["line"] for segment in info})
            if line > 0
        ]
        if len(turn_dwells) != len(line_starts):
            raise RuntimeError("saved turnaround dwell count does not match path")
        rows = R.apply_dwells(rows, {
            segment["idx"] - 1: float(dt)
            for segment, dt in zip(line_starts, turn_dwells)
        })
    return rows


def run_replay(schedule_file: str, method: str, policy: str,
               resx: float, resy: float, resz: float) -> dict:
    with open(schedule_file) as f:
        saved = json.load(f)
    label = policy.capitalize()
    resolution = f"X{resx*1e6:g}Y{resy*1e6:g}Z{resz*1e6:g}".replace(".", "p")
    name = f"Two{method}{resolution}{label}"
    case = os.path.join(HERE, "cases", f"case_{method.lower()}_hires_{policy}")
    shutil.rmtree(case, ignore_errors=True)
    dwell = dwell_for(policy)
    if saved.get("turnaround_dwells_s"):
        dwell = float(saved["turnaround_dwells_s"][0])
    command = [
        sys.executable, os.path.join(HERE, "make_case.py"),
        "--out", case, "--policy", policy, "--name", name,
        "--resx", str(resx), "--resy", str(resy), "--resz", str(resz),
    ]
    if policy == "dwell":
        command.extend(("--turn-dwell", str(dwell)))
    subprocess.run(command, cwd=HERE, check=True)
    Scan.ExportScan(schedule_rows(schedule_file, policy),
                    outFile=os.path.join(case, "Path.txt"))

    env = os.environ.copy()
    env.setdefault("I_MPI_FABRICS", env.get("THESIS_MPI_FABRICS", "shm"))
    subprocess.run([R.BIN_DBL, "./ParamInput.txt"], cwd=case, env=env,
                   check=True)
    subprocess.run([sys.executable,
                    os.path.join(DEMOS, "common", "fullfield_stats.py"),
                    case, name], cwd=HERE, check=True)
    with open(result_path(os.path.join("fullfield",
                                       f"{name}_fullfield.json"))) as f:
        return {"name": name, "case": case, "stats": json.load(f)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("schedule", help="schedule JSON under results/")
    parser.add_argument("method", help="case/name tag, e.g. Rh1w3 or Greedy1")
    parser.add_argument("--policy", choices=("continuous", "dwell"),
                        required=True)
    parser.add_argument("--resx", type=float, default=50e-6)
    parser.add_argument("--resy", type=float, default=1e-6)
    parser.add_argument("--resz", type=float, default=1e-6)
    args = parser.parse_args()
    result = run_replay(args.schedule, args.method, args.policy,
                        args.resx, args.resy, args.resz)
    print(json.dumps({"name": result["name"],
                      "stats": result["stats"]}, indent=2))


if __name__ == "__main__":
    main()
