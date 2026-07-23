"""Find the minimum dwell for complete solidification before track two.

The criterion is exact on the tracked RDF event history: immediately before
the second track begins, no cell may have a liquid interval ``tm < t < tl``.
The search uses the two-track dwell path itself and writes
``results/min_dwell.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

import pandas as pd


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BIN = os.environ.get("THESIS_BIN", os.path.join(ROOT, "build", "bin", "3DThesis"))
PYTHON = sys.executable
SIDE_MM = 10.0
VELOCITY_MM_S = 3000.0


def run_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("I_MPI_FABRICS", env.get("THESIS_MPI_FABRICS", "shm"))
    return env


def probe(dwell_s: float, power_w: float, timestep_s: float,
          resx_m: float, resy_m: float, resz_m: float) -> tuple[bool, dict]:
    case = os.path.join(HERE, "cases", "_dwell_probe")
    shutil.rmtree(case, ignore_errors=True)
    subprocess.run([
        PYTHON, os.path.join(HERE, "make_case.py"),
        "--out", case, "--name", "TwoDwellProbe", "--policy", "dwell",
        "--turn-dwell", str(dwell_s), "--power", str(power_w),
        "--timestep", str(timestep_s), "--resx", str(resx_m),
        "--resy", str(resy_m), "--resz", str(resz_m),
    ], check=True, stdout=subprocess.DEVNULL)
    try:
        subprocess.run([BIN, "./ParamInput.txt"], cwd=case, env=run_env(),
                       check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        rdf = pd.read_csv(os.path.join(
            case, "Data", "TwoDwellProbe.RDF.Final.csv"))
        rdf.columns = [c.strip() for c in rdf.columns]
        second_start_s = SIDE_MM / VELOCITY_MM_S + dwell_s
        live = ((rdf["tm"].values < second_start_s)
                & (rdf["tl"].values > second_start_s))
        overshoot = (float((rdf.loc[live, "tl"] - second_start_s).max())
                     if live.any() else 0.0)
        detail = {
            "dwell_s": dwell_s,
            "second_track_start_s": second_start_s,
            "liquid_cells": int(live.sum()),
            "max_liquid_overshoot_s": overshoot,
        }
        return not live.any(), detail
    finally:
        shutil.rmtree(case, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hi", type=float, default=1.5e-3,
                    help="initial passing-bracket candidate in seconds")
    ap.add_argument("--tol", type=float, default=25e-6,
                    help="dwell bisection tolerance in seconds")
    ap.add_argument("--power", type=float, default=150.0)
    ap.add_argument("--timestep", type=float, default=5e-5)
    ap.add_argument("--resx", type=float, default=50e-6)
    ap.add_argument("--resy", type=float, default=12.5e-6)
    ap.add_argument("--resz", type=float, default=1e-6)
    args = ap.parse_args()

    lo, hi = 0.0, args.hi
    ok, detail = probe(hi, args.power, args.timestep,
                       args.resx, args.resy, args.resz)
    print(f"dwell {hi*1e3:7.3f} ms: {'PASS' if ok else 'FAIL'} "
          f"({detail['liquid_cells']} liquid cells)", flush=True)
    while not ok:
        lo, hi = hi, 2.0 * hi
        ok, detail = probe(hi, args.power, args.timestep,
                           args.resx, args.resy, args.resz)
        print(f"dwell {hi*1e3:7.3f} ms: {'PASS' if ok else 'FAIL'} "
              f"({detail['liquid_cells']} liquid cells)", flush=True)

    history = [detail]
    while hi - lo > args.tol:
        mid = 0.5 * (lo + hi)
        ok, detail = probe(mid, args.power, args.timestep,
                           args.resx, args.resy, args.resz)
        history.append(detail)
        print(f"dwell {mid*1e3:7.3f} ms: {'PASS' if ok else 'FAIL'} "
              f"({detail['liquid_cells']} liquid cells)", flush=True)
        if ok:
            hi = mid
        else:
            lo = mid

    _ok, passing = probe(hi, args.power, args.timestep,
                         args.resx, args.resy, args.resz)
    _ok, binding = probe(lo, args.power, args.timestep,
                         args.resx, args.resy, args.resz)
    payload = {
        "min_dwell_s": hi,
        "bracket_s": [lo, hi],
        "tolerance_s": args.tol,
        "criterion": "no liquid cells immediately before track two (RDF)",
        "power_W": args.power,
        "tracking_step_s": args.timestep,
        "resolution_m": {"x": args.resx, "y": args.resy, "z": args.resz},
        "passing_probe": passing,
        "binding_failed_probe": binding,
        "history": history,
    }
    result_dir = os.path.join(HERE, "results")
    os.makedirs(result_dir, exist_ok=True)
    path = os.path.join(result_dir, "min_dwell.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
