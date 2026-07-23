"""Derive the minimal global turnaround dwell for the square raster.

Criterion (same as the raster demo): at the instant each powered line starts,
the entire field has solidified — no cell's RDF liquid interval [tm, tl)
straddles a line-start time. Evaluated at worst case = the unoptimized
nominal power. Bisects the per-turn dwell to a 25 us tolerance.

Each probe is a full-path 3DThesis run (coarse z, 50 us tracking step); this
brute force is exactly what task 3's Newton-on-dwell derivative replaces.

Usage: python find_min_dwell.py [--hi 2e-3] [--tol 25e-6] [--power 150]
Writes min_dwell.json.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mp_lib as R

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.environ.get("THESIS_BIN", R.BIN_DBL)
PY = sys.executable
V_MM = 3000.0     # mm/s
SIDE = 10.0       # mm
HATCH = 0.1       # mm
TSTEP = 5e-5      # s, tracking step for the probes
NLINES = int(round(SIDE / HATCH)) + 1
LINE_S = SIDE / V_MM


def probe(dwell, power, keep=None):
    """Run the full square at this dwell; return (ok, worst) where worst is
    (liquid cell count, turn index, overshoot seconds) at the worst start."""
    case = keep or os.path.join(HERE, "_dwell_probe")
    subprocess.run([PY, os.path.join(HERE, "make_case.py"), "--out", case,
                    "--name", "Probe", "--power", str(power),
                    "--timestep", str(TSTEP), "--turn-dwell", str(dwell)],
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run([BIN, "./ParamInput.txt"], cwd=case, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rdf = pd.read_csv(os.path.join(case, "Data", "Probe.RDF.Final.csv"))
    rdf.columns = [c.strip() for c in rdf.columns]
    tm, tl = rdf["tm"].values, rdf["tl"].values
    worst = (0, -1, 0.0)
    for k in range(1, NLINES):
        tk = k * (LINE_S + dwell)
        live = (tm < tk) & (tl > tk)
        n = int(live.sum())
        if n > worst[0]:
            worst = (n, k, float((tl[live] - tk).max()))
    if keep is None:
        shutil.rmtree(case, ignore_errors=True)
    return worst[0] == 0, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hi", type=float, default=2e-3,
                    help="initial upper bracket (s), doubled until it passes")
    ap.add_argument("--tol", type=float, default=25e-6)
    ap.add_argument("--power", type=float, default=150.0)
    args = ap.parse_args()

    lo, hi = 0.0, args.hi
    ok, worst = probe(hi, args.power)
    print(f"dwell {hi*1e3:7.3f} ms  {'PASS' if ok else 'FAIL'}  worst {worst}",
          flush=True)
    while not ok:
        lo, hi = hi, 2.0 * hi
        ok, worst = probe(hi, args.power)
        print(f"dwell {hi*1e3:7.3f} ms  {'PASS' if ok else 'FAIL'}  worst {worst}",
              flush=True)
    history = [(hi, True)]
    while hi - lo > args.tol:
        mid = 0.5 * (lo + hi)
        ok, worst = probe(mid, args.power)
        print(f"dwell {mid*1e3:7.3f} ms  {'PASS' if ok else 'FAIL'}  worst {worst}",
              flush=True)
        history.append((mid, ok))
        if ok:
            hi = mid
        else:
            lo = mid

    # binding-turn diagnostic just below the threshold
    _, worst = probe(lo if lo > 0 else hi * 0.5, args.power)
    out = dict(min_dwell_s=hi, bracket_s=[lo, hi], tol_s=args.tol,
               power_W=args.power, tracking_step_s=TSTEP,
               binding_turn=worst[1], binding_liquid_cells=worst[0],
               binding_overshoot_s=worst[2],
               criterion="no liquid cell at any line-start time (RDF)",
               history_ms=[[d * 1e3, bool(o)] for d, o in history])
    with open(R.data_path("min_dwell.json", write=True), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
