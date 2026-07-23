"""Replay a triangle optimized schedule through a full tracked simulation
(RDF + Solidification, case_pub settings) and post-process traces.

Usage: python verify_optimized.py {zero|dwell} [z-resolution-m]
Builds case_opt_{policy}/ from make_case.py, swaps in the optimized
segmented path, runs the double build, then triangle postprocess.py.
"""
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MP_GEOM", "triangle")
sys.path.insert(0, os.path.dirname(HERE))
from common import mp_lib as R                            # noqa: E402
from common import scan as Scan                           # noqa: E402

policy = sys.argv[1]
resz = float(sys.argv[2]) if len(sys.argv) > 2 else 5e-6
os.chdir(HERE)
dwell = 0.0 if policy == "zero" else R.dwell_from_json()
case = os.path.join(HERE, "cases", f"case_opt_{policy}")
name = f"TriOpt{policy.capitalize()}"
PY = sys.executable

args = [PY, os.path.join(HERE, "make_case.py"), "--out", case, "--name", name,
        "--power", "150", "--resz", str(resz), "--timestep", "5e-5"]
if dwell > 0:
    args += ["--turn-dwell", str(dwell)]
subprocess.run(args, check=True)

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
subprocess.run([PY, os.path.join(HERE, "postprocess.py"), case, name, "5e-5"],
               check=True)
subprocess.run([PY, os.path.join(os.path.dirname(HERE), "common",
                                 "fullfield_stats.py"), case, name],
               check=True)
