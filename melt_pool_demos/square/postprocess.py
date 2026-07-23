"""Post-process a square-raster case: melt-pool traces from the RDF event
list (a cell is liquid on [tm, tl)), summary metrics, and the per-line-start
residual-liquid diagnostic that quantifies the merged-pool regime.

Usage: python postprocess.py <case_dir> <name> [dt_bin] [dwell_s]
Writes compact outputs to ``results/fullfield/`` and its diagnostic plot to
``figures/diagnostics/``.
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
DWELL = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0

V_MM, SIDE, HATCH = 3000.0, 10.0, 0.1
NLINES = int(round(SIDE / HATCH)) + 1
LINE_S = SIDE / V_MM
HOP_S = 0.0 if DWELL > 0 else HATCH / V_MM     # dwell rows replace the hop
XRES, ZRES = 50e-6, None                        # zres read from Domain.txt

with open(os.path.join(case, "Domain.txt")) as f:
    dom = f.read().split()
ZRES = float(dom[dom.index("Z"):][dom[dom.index("Z"):].index("Res") + 1])

rdf = pd.read_csv(os.path.join(case, "Data", f"{name}.RDF.Final.csv"))
rdf.columns = [c.strip() for c in rdf.columns]
x = rdf["x"].values * 1e3
y = rdf["y"].values * 1e3
z = rdf["z"].values * 1e3
tm, tl = rdf["tm"].values, rdf["tl"].values
cellvol = (XRES * 1e3) ** 2 * (ZRES * 1e3)      # mm^3

nbin = int(np.ceil(tl.max() / DT)) + 2
b0 = np.minimum((tm / DT).astype(np.int64), nbin - 1)
b1 = np.minimum((tl / DT).astype(np.int64), nbin - 1)
dv = np.zeros(nbin); np.add.at(dv, b0, 1.0); np.add.at(dv, b1, -1.0)
vol = np.cumsum(dv) * cellvol

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

# residual liquid at each line start (the merged-pool / zero-dwell diagnostic)
starts, residual_mm3 = [], []
for k in range(1, NLINES):
    tk = k * (LINE_S + (DWELL if DWELL > 0 else HOP_S))
    if tk > tl.max():
        break
    starts.append(tk)
    residual_mm3.append(float(((tm < tk) & (tl > tk)).sum() * cellvol))

summary = dict(
    peak_volume_mm3=float(vol.max()),
    peak_length_mm=float(L.max()),
    peak_width_mm=float(W.max()),
    peak_depth_mm=float(D.max()),
    total_build_time_s=float(tl.max()),
    beam_path_time_s=NLINES * LINE_S + (NLINES - 1) *
        (DWELL if DWELL > 0 else HOP_S),
    dwell_per_turn_s=DWELL,
    residual_liquid_at_starts_mm3=dict(
        max=float(max(residual_mm3)) if residual_mm3 else 0.0,
        mean=float(np.mean(residual_mm3)) if residual_mm3 else 0.0,
        n_nonzero=int(sum(r > 0 for r in residual_mm3))),
    n_events=int(len(rdf)),
    n_melted_cells=int(rdf.groupby(["x", "y", "z"]).ngroups),
    dt=DT, zres=ZRES,
)
with open(R.fullfield_path(f"{name}_summary.json", write=True), "w") as f:
    json.dump(summary, f, indent=2)

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix"})
fig, axes = plt.subplots(3, 1, figsize=(7.5, 7.5), sharex=True)
for ax, tr, lab in zip(axes, [L, D, vol],
                       ["melt pool length (mm)", "melt pool depth (mm)",
                        "melt pool volume (mm$^3$)"]):
    ax.plot(t, tr, color="#4878A8", lw=0.6)
    ax.set_ylabel(lab)
axes[0].set_title(f"{name}: 10 mm square raster"
                  + (f", dwell {DWELL*1e3:g} ms/turn" if DWELL > 0 else
                     ", no dwell"))
axes[-1].set_xlabel("time (s)")
fig.tight_layout()
fig.savefig(R.fig_path(os.path.join("diagnostics", f"{name}_traces.png")), dpi=150)

pd.DataFrame(dict(t=t, volume=vol, length=L, width=W, depth=D)).to_csv(
    R.fullfield_path(f"{name}_traces.csv", write=True), index=False)

print(json.dumps(summary, indent=2))
