"""Replay the optimized per-segment schedule through a full tracked simulation
(the optimizer only samples segment-end snapshots; this is the continuous-time
validation). Builds cases/case_opt_zero/, swaps in the optimized segmented
path, runs the double build, and regenerates the whole-track statistics and the
fig17/fig16 views.

Usage: python verify_optimized.py [resz]
"""
import os
import subprocess
import sys

import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mp_lib as R
from common import scan as Scan                       # noqa: E402

resz = float(sys.argv[1]) if len(sys.argv) > 1 else 5e-6
here = os.path.dirname(os.path.abspath(__file__))
common = os.path.join(os.path.dirname(here), "common")
case = os.path.join(here, "cases", "case_opt_zero")
name = "TrackOpt"
PY = sys.executable

# base case (calibrated Beam / Domain / Mode / Output at 150 W, sigma 200 um)
subprocess.run([PY, os.path.join(here, "make_case.py"), "--out", case,
                "--name", name, "--resz", str(resz), "--timestep", "5e-5"],
               check=True)

# optimized segmented path: per-segment Pmod + width factor sigma/SIGMA + speed
opt = pd.read_csv(R.data_path("optimized_zero.csv"))
div, powered, hop_after = R.build_path(0.0)
pmods = {int(r.idx): float(r.pmod) for r in opt.itertuples()}
wmods = {int(r.idx): float(r.sigma) / R.SIGMA for r in opt.itertuples()}
vels = ({int(r.idx): float(r.vel) for r in opt.itertuples()}
        if "vel" in opt.columns else None)
rows = R.set_pmods(div, pmods, wmods, vels=vels)
Scan.ExportScan(rows, outFile=os.path.join(case, "Path.txt"))

subprocess.run([R.BIN_DBL, "./ParamInput.txt"], cwd=case, check=True)
subprocess.run([PY, os.path.join(common, "fullfield_stats.py"), case, name],
               check=True)
subprocess.run([PY, os.path.join(common, "make_plots.py"), "fig17",
                "case_opt_zero", name], check=True)
subprocess.run([PY, os.path.join(common, "make_plots.py"), "fig16",
                "case_opt_zero", name], check=True)
