#!/usr/bin/env python3
"""Visualize an OTI snapshot: temperature and its design-variable sensitivities.

Reads a snapshot CSV produced by the OTI build (columns x,y,z,T,dT_dx,...,
dT_dcps) and renders each field as a 2D heatmap over the (x,y) grid. Useful for
seeing where temperature is most sensitive to each process/material parameter.

Usage:
  python3 visualize.py PATH/TO/snapshot.Snapshot.00.csv [--out fig.png]
"""

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIELDS = ["T", "dT_dx", "dT_dy", "dT_dQ", "dT_dkon", "dT_drho", "dT_dcps"]
LABELS = {
    "T": "T [K]",
    "dT_dx": "dT/dx [K/m]", "dT_dy": "dT/dy [K/m]",
    "dT_dQ": "dT/dQ [K/W]", "dT_dkon": "dT/dk [K/(W/m/K)]",
    "dT_drho": "dT/drho [K/(kg/m^3)]", "dT_dcps": "dT/dcp [K/(J/kg/K)]",
}


def load_grid(path):
    rows = list(csv.DictReader(open(path)))
    cols = rows[0].keys()
    xs = np.array(sorted(set(float(r["x"]) for r in rows)))
    ys = np.array(sorted(set(float(r["y"]) for r in rows)))
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    grids = {f: np.full((len(ys), len(xs)), np.nan) for f in cols}
    for r in rows:
        ix, iy = xi[float(r["x"])], yi[float(r["y"])]
        for f in cols:
            grids[f][iy, ix] = float(r[f])
    return xs, ys, grids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="oti_fields.png")
    args = ap.parse_args()

    xs, ys, grids = load_grid(args.csv)
    fields = [f for f in FIELDS if f in grids]
    extent = [xs[0] * 1e3, xs[-1] * 1e3, ys[0] * 1e3, ys[-1] * 1e3]  # mm

    ncol = 4
    nrow = (len(fields) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    for k, f in enumerate(fields):
        ax = axes.flat[k]
        ax.axis("on")
        data = grids[f]
        # Diverging colormap centred at 0 for derivatives; sequential for T.
        # T has a near-singular peak at the beam centre, so clip to a robust
        # percentile or it washes the whole field to black.
        if f == "T":
            vmax = np.nanpercentile(data, 99.5)
            im = ax.imshow(data, origin="lower", extent=extent, aspect="auto",
                           cmap="inferno", vmin=np.nanmin(data), vmax=vmax)
        else:
            lim = np.nanpercentile(np.abs(data), 99.5)
            im = ax.imshow(data, origin="lower", extent=extent, aspect="auto",
                           cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(LABELS.get(f, f), fontsize=10)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("OTI temperature field and parameter sensitivities", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}  ({len(fields)} fields, grid {len(xs)}x{len(ys)})")


if __name__ == "__main__":
    main()
