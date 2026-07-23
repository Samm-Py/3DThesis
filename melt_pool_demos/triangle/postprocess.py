"""Rebuild the paper's Figures 16 and 17 from the triangle-case outputs.

Inputs (from case_full/Data):
  Triangle.Solidification.Final.csv  -- per melted grid point: tSol, G, V,
                                        numMelt, MP_* (last solidification)
  Triangle.RDF.Final.csv             -- per melt EVENT: x,y,z,tm,tl,cr
                                        (melt time, solidification time)

The RDF event list is the workhorse: a cell is liquid on [tm, tl), so any
instantaneous pool property is an aggregation over active events:
  volume(t) = cellvol * #active          (Fig 17a)
  length(t) = max x - min x of active    (Fig 17b, "measured in x-direction")
  depth(t)  = -min z of active           (Fig 17c)
  width(t)  = max y - min y of active    (extra; y-extent across scan lines)

Note MP_depth from MP_Stats multiplies the z-INDEX by the x resolution
(src/Melt.cpp), so with anisotropic grids it must be rescaled by zres/xres;
we use the RDF depth instead, which needs no correction.

Writes traces.csv, fig17_traces.png, fig16_maps.png, extra_maps.png, and
summary.json to ``validation/results`` by default. Set ``MP_VALIDATION_DIR``
to override this location.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.unicode_minus": False,
})

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "cases", "case_full")
NAME = sys.argv[2] if len(sys.argv) > 2 else "Triangle"
DATA = os.path.join(CASE, "Data")
DT = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4   # trace resolution (= tracking timestep)
OUT = os.environ.get("MP_VALIDATION_DIR", os.path.join(HERE, "validation", "results"))
os.makedirs(OUT, exist_ok=True)

sol = pd.read_csv(os.path.join(DATA, f"{NAME}.Solidification.Final.csv"))
rdf = pd.read_csv(os.path.join(DATA, f"{NAME}.RDF.Final.csv"))
print(f"grid rows {len(sol)}, RDF events {len(rdf)}")

xres = np.diff(np.unique(sol.x.values))[:1].item()
zvals = np.unique(sol.z.values)
zres = np.diff(zvals)[:1].item()
cellvol_mm3 = (xres * 1e3) ** 2 * (zres * 1e3)   # mm^3 per cell-event

# ---------------- Fig 17 traces from RDF events ----------------
tm = rdf.tm.values
tl = rdf.tl.values
t_end = tl.max()
nbin = int(np.ceil(t_end / DT)) + 1
times = np.arange(nbin) * DT

# volume: +1 at melt, -1 at solidify, cumulative over time bins
b0 = np.minimum((tm / DT).astype(np.int64), nbin - 1)
b1 = np.minimum((tl / DT).astype(np.int64), nbin - 1)
dv = np.zeros(nbin)
np.add.at(dv, b0, 1.0)
np.add.at(dv, b1, -1.0)
volume = np.cumsum(dv) * cellvol_mm3

# extents: scatter max/min of x, y, z over every bin each event is active in
n_active = (b1 - b0 + 1)
idx = np.repeat(b0, n_active) + (np.arange(n_active.sum()) -
                                 np.repeat(np.cumsum(n_active) - n_active, n_active))
def extent(vals, agg):
    out = np.full(nbin, np.nan)
    fill = -np.inf if agg is np.maximum else np.inf
    acc = np.full(nbin, fill)
    agg.at(acc, idx, np.repeat(vals, n_active))
    ok = np.isfinite(acc)
    out[ok] = acc[ok]
    return out

xmax = extent(rdf.x.values, np.maximum); xmin = extent(rdf.x.values, np.minimum)
ymax = extent(rdf.y.values, np.maximum); ymin = extent(rdf.y.values, np.minimum)
zmin = extent(rdf.z.values, np.minimum)
length = (xmax - xmin) * 1e3
width = (ymax - ymin) * 1e3
depth = -zmin * 1e3

tr = pd.DataFrame(dict(t_ms=times * 1e3, volume_mm3=volume,
                       length_mm=length, width_mm=width, depth_mm=depth))
tr.to_csv(os.path.join(OUT, f"{NAME}_traces.csv"), index=False)

fig, axs = plt.subplots(1, 3, figsize=(10.6, 3.2))
panels = [
    ("length_mm", "Melt Pool Length (mm)", "A", (0.0, 8.0), np.arange(0, 8.1, 2)),
    ("depth_mm", "Melt Pool Depth (mm)", "B", (0.0, 0.20), np.arange(0, 0.201, 0.05)),
    ("volume_mm3", r"Melt Pool Volume ($\mathrm{mm}^3$)", "C",
     (0.0, 0.25), np.arange(0, 0.251, 0.05)),
]
for ax, (col, label, tag, ylim, yticks) in zip(axs, panels):
    ax.plot(tr.t_ms * 1e-3, tr[col], color="#4C72B0", lw=0.7)
    ax.set_xlim(0.0, 0.155)
    ax.set_xticks([0.0, 0.05, 0.10, 0.15])
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    if ylim[1] <= 0.25:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(label)
    ax.text(0.04, 0.96, tag, transform=ax.transAxes, fontsize=12, va="top")
    ax.tick_params(which="major", direction="out", length=3.5, width=0.8,
                   color="#666666")
    ax.tick_params(which="minor", direction="out", length=2.0, width=0.6,
                   color="#666666")
    for spine in ax.spines.values():
        spine.set_color("#666666")
        spine.set_linewidth(0.8)
fig.subplots_adjust(left=0.08, right=0.985, bottom=0.20, top=0.96, wspace=0.43)
fig.savefig(os.path.join(OUT, f"{NAME}_fig17_traces.png"), dpi=300)
plt.close(fig)

# ---------------- Fig 16 maps from the solidification grid ----------------
def grid_field(df, xcol, ycol, val):
    xs, ys = np.unique(df[xcol].values), np.unique(df[ycol].values)
    gi = np.searchsorted(xs, df[xcol].values)
    gj = np.searchsorted(ys, df[ycol].values)
    F = np.full((ys.size, xs.size), np.nan)
    F[gj, gi] = df[val].values
    return xs * 1e3, ys * 1e3, F

surface_z = zvals.max()
on_surface = np.isclose(sol.z.values, surface_z, rtol=0.0, atol=zres / 2)
surf = sol[on_surface & (sol.G > 0)]
xs, ys, Gtop = grid_field(surf, "x", "y", "G")

xc = 5.0e-3
xslice = sol[(np.abs(sol.x - xc) < xres / 2) & (sol.G > 0)]
ysl, zsl, Gslice = grid_field(xslice, "y", "z", "G")

fig = plt.figure(figsize=(11, 8.2))
gs = fig.add_gridspec(2, 1, height_ratios=[5.2, 1.0], hspace=0.28)
ax = fig.add_subplot(gs[0])
norm = LogNorm(vmin=5e4, vmax=1e7)   # the paper's colorbar range
im = ax.pcolormesh(xs, ys, Gtop, norm=norm, cmap="RdBu_r", shading="nearest")
ax.set_aspect("equal")
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.text(0.01, 0.97, "A  top surface", transform=ax.transAxes, va="top")
fig.colorbar(im, ax=ax, label="G (K/m)", shrink=0.85)

ax2 = fig.add_subplot(gs[1])
im2 = ax2.pcolormesh(ysl, zsl * 1e3 / 1e3 * 1000, Gslice, norm=norm,
                     cmap="RdBu_r", shading="nearest")
ax2.set_xlabel("y (mm)  [slice at x = 5 mm]"); ax2.set_ylabel("z (µm)")
ax2.text(0.01, 0.92, "B  centre cross-section", transform=ax2.transAxes, va="top")
fig.suptitle("Solidification thermal gradient (cf. Stump & Plotkowski Fig. 16)")
fig.savefig(os.path.join(OUT, f"{NAME}_fig16_maps.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------- extras: V map, remelt count, solidification time --------
fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
xs, ys, Vtop = grid_field(surf, "x", "y", "V")
im = axs[0].pcolormesh(xs, ys, Vtop, norm=LogNorm(vmin=1e-3, vmax=3),
                       cmap="RdBu_r", shading="nearest")
fig.colorbar(im, ax=axs[0], label="V (m/s)", shrink=0.8)
axs[0].set_title("Solid-liquid interface velocity")

surf_nm = sol[on_surface & (sol.numMelt > 0)]
xs, ys, NM = grid_field(surf_nm, "x", "y", "numMelt")
im = axs[1].pcolormesh(xs, ys, NM, cmap="magma", shading="nearest",
                       vmin=0, vmax=np.nanpercentile(NM, 99))
fig.colorbar(im, ax=axs[1], label="times melted", shrink=0.8)
axs[1].set_title("Remelt count")

xs, ys, TS = grid_field(surf, "x", "y", "tSol")
im = axs[2].pcolormesh(xs, ys, TS * 1e3, cmap="magma", shading="nearest")
fig.colorbar(im, ax=axs[2], label="tSol (ms)", shrink=0.8)
axs[2].set_title("Solidification time")

for ax in axs:
    ax.set_aspect("equal"); ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, f"{NAME}_extra_maps.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------- summary ----------------
summary = dict(
    x_resolution_um=round(float(xres * 1e6), 9),
    z_resolution_um=round(float(zres * 1e6), 9),
    grid_points_melted=int((sol.tSol > 0).sum()),
    rdf_events=int(len(rdf)),
    scan_end_ms=float(tm.max() * 1e3),
    last_solidification_ms=float(t_end * 1e3),
    peak_volume_mm3=float(np.nanmax(volume)),
    peak_length_mm=float(np.nanmax(length)),
    peak_depth_mm=float(np.nanmax(depth)),
    max_remelts=int(sol.numMelt.max()),
)
with open(os.path.join(OUT, f"{NAME}_summary.json"), "w") as f:
    json.dump(summary, f, indent=1)
print(json.dumps(summary, indent=1))
