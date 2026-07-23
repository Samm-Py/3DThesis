"""Solidification thermal-gradient map panel for the paper (Fig. 16 style).

``fig16_maps(case_dir, name)`` renders the top-surface and centre-section
G maps for one tracked case's raw ``Data/`` into ``<name>_fig16_maps.png``.
The square and triangle greedy figure drivers
(``common/make_greedy_raster_figures.py``) import it to build the baseline
and optimized panels that ``make_paper_maps.py`` then recomposes into the
paper's ``<geometry>_maps_zero`` figure. Geometry follows the demo directory
the caller runs from.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mp_lib as R

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix",
    "font.size": 9.5, "axes.labelsize": 10, "axes.titlesize": 10,
    "legend.fontsize": 8.5, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "axes.linewidth": 0.7,
    "lines.solid_capstyle": "round",
})


def _case_path(directory, filename=None):
    path = os.path.join("cases", directory)
    if not os.path.exists(path):
        path = directory
    return os.path.join(path, filename) if filename else path


def _read_solidification(case_dir, name):
    data = os.path.join(_case_path(case_dir), "Data")
    sol = pd.read_csv(os.path.join(data, f"{name}.Solidification.Final.csv"))
    sol.columns = [c.strip() for c in sol.columns]
    return sol


def fig16_maps(case_dir, name):
    """Fig 16: solidification thermal gradient G — top surface plus a centre
    cross-section (slice at the geometric x-centre)."""
    sol = _read_solidification(case_dir, name)
    zvals = np.unique(sol.z.values)
    zres = np.min(np.diff(zvals))

    def grid_field(df, xcol, ycol, val):
        xs, ys = np.unique(df[xcol].values), np.unique(df[ycol].values)
        gi = np.searchsorted(xs, df[xcol].values)
        gj = np.searchsorted(ys, df[ycol].values)
        F = np.full((ys.size, xs.size), np.nan)
        F[gj, gi] = df[val].values
        return xs * 1e3, ys * 1e3, F

    on_surface = np.isclose(sol.z.values, zvals.max(), rtol=0.0, atol=zres / 2)
    surf = sol[on_surface & (sol.G > 0)]
    xs, ys, Gtop = grid_field(surf, "x", "y", "G")

    # centre slice: snap to the nearest melted grid column to the x-centre
    # (the arithmetic midpoint can fall between columns or in a gap, e.g. on
    # the triangle), so the cross-section is never empty
    gx = np.unique(sol.x.values[sol.G.values > 0])
    xc = gx[np.argmin(np.abs(gx - 0.5 * (gx.min() + gx.max())))]
    xslice = sol[(sol.x == xc) & (sol.G > 0)]
    ysl, zsl, Gslice = grid_field(xslice, "y", "z", "G")

    fig = plt.figure(figsize=(11, 8.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[5.2, 1.0], hspace=0.28)
    norm = LogNorm(vmin=5e4, vmax=1e7)                  # the paper's range
    ax = fig.add_subplot(gs[0])
    im = ax.pcolormesh(xs, ys, Gtop, norm=norm, cmap="RdBu_r", shading="nearest")
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title("A  Top surface", loc="left", pad=7)
    fig.colorbar(im, ax=ax, label="G (K/m)", shrink=0.85)

    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.pcolormesh(ysl, zsl * 1e3, Gslice, norm=norm, cmap="RdBu_r",
                         shading="nearest")
    ax2.set_xlabel(f"y (mm)  [slice at x = {xc*1e3:.1f} mm]")
    ax2.set_ylabel("z (µm)")
    ax2.set_title("B  Centre cross-section", loc="left", pad=7)
    fig.suptitle("Solidification thermal gradient "
                 "(cf. Stump & Plotkowski Fig. 16)")
    fig.savefig(R.fig_path(f"{name}_fig16_maps.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"{name}_fig16_maps written")


if __name__ == "__main__":
    fig16_maps(sys.argv[1], sys.argv[2])
