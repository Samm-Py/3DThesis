"""Measure the developed single-track melt pool from a --single-track run.

Reads the RDF event list (a cell is liquid on [tm, tl)) and reports the
numbers that size the square/triangle optimization setup:

  - developed pool length / width / depth (steady window, beam mid-line)
  - trailing and leading lengths relative to the beam
  - startup distance (beam travel until depth reaches 95% of steady)
  - solidification tail after beam-off (sets the minimal-dwell scale)

Usage: python calibrate.py <case_dir> <name> [dt] [v_mm_per_s] [side_mm]
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mp_lib as R

case = os.path.abspath(sys.argv[1])
name = sys.argv[2]
DT = float(sys.argv[3]) if len(sys.argv) > 3 else 5e-5
V_MM = float(sys.argv[4]) if len(sys.argv) > 4 else 3000.0   # mm/s
SIDE = float(sys.argv[5]) if len(sys.argv) > 5 else 10.0     # mm

rdf = pd.read_csv(os.path.join(case, "Data", f"{name}.RDF.Final.csv"))
rdf.columns = [c.strip() for c in rdf.columns]
x = rdf["x"].values * 1e3   # mm
y = rdf["y"].values * 1e3
z = rdf["z"].values * 1e3
tm, tl = rdf["tm"].values, rdf["tl"].values

t_end_beam = SIDE / V_MM                     # beam reaches line end (s)
nbin = int(np.ceil(tl.max() / DT)) + 2
b0 = np.minimum((tm / DT).astype(np.int64), nbin - 1)
b1 = np.minimum((tl / DT).astype(np.int64), nbin - 1)

n_act = b1 - b0 + 1
idx = np.repeat(b0, n_act) + (np.arange(n_act.sum()) -
                              np.repeat(np.cumsum(n_act) - n_act, n_act))
xmax = np.full(nbin, -np.inf); xmin = np.full(nbin, np.inf)
ymax = np.full(nbin, -np.inf); ymin = np.full(nbin, np.inf)
zmin = np.full(nbin, np.inf)
np.maximum.at(xmax, idx, np.repeat(x, n_act))
np.minimum.at(xmin, idx, np.repeat(x, n_act))
np.maximum.at(ymax, idx, np.repeat(y, n_act))
np.minimum.at(ymin, idx, np.repeat(y, n_act))
np.minimum.at(zmin, idx, np.repeat(z, n_act))

t = np.arange(nbin) * DT
act = np.isfinite(xmax)
L = np.where(act, xmax - xmin, 0.0)
W = np.where(act, ymax - ymin, 0.0)
D = np.where(act, -zmin, 0.0)
xb = np.minimum(t * V_MM, SIDE)              # beam x (mm)
trail = np.where(act, xb - xmin, 0.0)
lead = np.where(act, xmax - xb, 0.0)

# steady window: beam between 60% and 95% of the line
w = (xb > 0.6 * SIDE) & (xb < 0.95 * SIDE) & act
steady = {k: float(np.median(v[w])) for k, v in
          dict(length=L, width=W, depth=D, trail=trail, lead=lead).items()}

# startup: beam travel until depth first holds >= 95% of steady
d95 = 0.95 * steady["depth"]
i95 = np.argmax((D >= d95) & (t < t_end_beam))
startup_mm = float(xb[i95])
startup_s = float(t[i95])

# solidification tail after the beam reaches the line end
tail_s = float(tl.max() - t_end_beam)

summary = dict(
    steady_pool_mm=steady,
    startup_distance_mm=startup_mm,
    startup_time_s=startup_s,
    solidification_tail_s=tail_s,
    last_liquid_s=float(tl.max()),
    beam_end_s=t_end_beam,
    n_events=int(len(rdf)),
    dt=DT,
)
with open(R.calibration_path(f"{name}_calibration.json", write=True), "w") as f:
    json.dump(summary, f, indent=2)

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix"})
fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
for ax, tr, lab in zip(axes, [L, D, W],
                       ["pool length (mm)", "pool depth (mm)", "pool width (mm)"]):
    ax.plot(t * 1e3, tr, color="#4878A8", lw=1.0)
    ax.set_ylabel(lab)
    ax.axvline(t_end_beam * 1e3, color="0.6", lw=0.7, ls="--")
axes[0].set_title(f"single track, {name}: beam off at {t_end_beam*1e3:.2f} ms")
axes[-1].set_xlabel("time (ms)")
fig.tight_layout()
fig.savefig(R.fig_path(os.path.join("calibration", f"{name}_calibration.png")), dpi=150)

pd.DataFrame(dict(t=t, beam_x=xb, length=L, width=W, depth=D,
                  trail=trail, lead=lead)).to_csv(
    R.calibration_path(f"{name}_trace.csv", write=True), index=False)

print(json.dumps(summary, indent=2))
