"""Replay an optimized schedule through a full tracked simulation (RDF +
Solidification, same settings as the baselines) — the optimizer only samples
segment-end snapshots, so this is the continuous-time validation, including
the line-start zones inside each first segment.

Usage: python verify_optimized.py {zero|dwell}
Builds case_opt_{policy}/ (via make_case.py, then swaps in the optimized
segmented path), runs the double build, and post-processes traces.
"""
import os
import subprocess
import sys

import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mp_lib as R
from common import scan as Scan                     # noqa: E402

policy = sys.argv[1]
resz = float(sys.argv[2]) if len(sys.argv) > 2 else 5e-6
suffix = ""
dwell = 0.0 if policy == "zero" else R.dwell_from_json()
here = os.path.dirname(os.path.abspath(__file__))
case = os.path.join(here, "cases", f"case_opt_{policy}{suffix}")
name = f"SqOpt{policy.capitalize()}"
PY = sys.executable

# base case (writes Beam at 150 W / sigma 200 um, Domain, Mode, Output)
args = [PY, os.path.join(here, "make_case.py"), "--out", case, "--name", name,
        "--resz", str(resz), "--timestep", "5e-5"]
if dwell > 0:
    args += ["--turn-dwell", str(dwell)]
subprocess.run(args, check=True)

# optimized segmented path: per-segment Pmod + width factor sigma/SIGMA;
# zero-dwell hops inherit the controls of the segment they follow
opt = pd.read_csv(R.data_path(f"optimized_{policy}.csv"))
div, powered, hop_after = R.build_path(dwell)
pmods = {int(r.idx): float(r.pmod) for r in opt.itertuples()}
wmods = {int(r.idx): float(r.sigma) / R.SIGMA for r in opt.itertuples()}
for i, h in hop_after.items():
    pmods[h] = pmods[i]
    wmods[h] = wmods[i]
rows = R.set_pmods(div, pmods, wmods)
Scan.ExportScan(rows, outFile=os.path.join(case, "Path.txt"))

subprocess.run([R.BIN_DBL, "./ParamInput.txt"], cwd=case, check=True)
subprocess.run([PY, os.path.join(here, "postprocess.py"), case, name,
                "5e-5", str(dwell)], check=True)
subprocess.run([PY, os.path.join(os.path.dirname(here), "common",
                                 "fullfield_stats.py"), case, name],
               check=True)
