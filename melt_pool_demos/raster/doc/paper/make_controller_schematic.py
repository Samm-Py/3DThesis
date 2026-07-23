"""Generate the sequential control-block schematic used in the paper."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/controller-schematic-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/controller-schematic-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle


HERE = Path(__file__).resolve().parent
BLUE = "#1f77b4"
LIGHT_BLUE = "#dcecf7"
GRAY = "#8a8a8a"
LIGHT_GRAY = "#ececeb"
DARK = "#222222"


def arrow(ax, start, end, **kwargs):
    defaults = dict(arrowstyle="-|>", mutation_scale=11, lw=1.0,
                    color=DARK)
    defaults.update(kwargs)
    ax.add_patch(FancyArrowPatch(start, end, **defaults))


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 9.5,
        "pdf.fonttype": 42,
    })
    fig, (path_ax, pool_ax, legend_ax) = plt.subplots(
        1, 3, figsize=(9.2, 2.95),
        gridspec_kw={"width_ratios": (1.65, 1.0, 0.68), "wspace": 0.10})

    # (a) Causal, sequential block update along the scan path.
    path_ax.set_xlim(0.0, 10.0)
    path_ax.set_ylim(0.0, 4.2)
    path_ax.axis("off")
    y0, height = 1.25, 0.62
    blocks = (
        (0.65, 2.35, LIGHT_GRAY, r"$\mathbf{u}_{i-2}$"),
        (2.35, 4.15, LIGHT_GRAY, r"$\mathbf{u}_{i-1}$"),
        (4.15, 8.25, LIGHT_BLUE, r"trial $\mathbf{u}_{i}$"),
    )
    for left, right, color, label in blocks:
        path_ax.add_patch(Rectangle(
            (left, y0), right - left, height, facecolor=color,
            edgecolor=GRAY if color == LIGHT_GRAY else BLUE, lw=1.0))
        path_ax.text((left + right) / 2, y0 + height / 2, label,
                     ha="center", va="center")
    arrow(path_ax, (0.45, y0 - 0.25), (9.25, y0 - 0.25), lw=1.15)
    path_ax.text(6.20, y0 - 0.43, "scan direction", va="top", ha="center",
                 fontsize=8.6)
    for x, label in ((4.15, r"$x_i$"), (8.25, r"$x_{i+1}$")):
        path_ax.plot((x, x), (0.72, 2.35), color=DARK, lw=0.8,
                     ls=(0, (2, 2)))
        path_ax.text(x, 0.58, label, ha="center", va="top")
    path_ax.plot((0.65, 4.15), (2.18, 2.18), color=GRAY, lw=0.9)
    path_ax.plot((0.65, 0.65), (2.18, 2.02), color=GRAY, lw=0.9)
    path_ax.plot((4.15, 4.15), (2.18, 2.02), color=GRAY, lw=0.9)
    path_ax.plot((2.40, 2.40), (2.18, 2.72), color=GRAY, lw=0.9)
    path_ax.text(2.30, 3.13, "previous controls and\nthermal history fixed",
                 ha="center", va="center", color=DARK, fontsize=8.8)
    path_ax.annotate(
        "choose trial\n" +
        r"$\mathbf{u}_i=(P_{\mathrm{mod},i},\sigma_i,v_i)$",
        xy=(4.15, y0 + height), xytext=(6.05, 3.12), ha="center",
        arrowprops=dict(arrowstyle="->", lw=0.9, color=BLUE), color=BLUE,
        fontsize=8.8)
    path_ax.annotate(
        "evaluate endpoint", xy=(8.25, y0 + height), xytext=(7.70, 2.62),
        ha="center", arrowprops=dict(arrowstyle="->", lw=0.9, color=DARK))
    path_ax.text(0.10, 4.02, "(a) Sequential block update",
                 ha="left", va="top", fontweight="bold")

    # (b) Width/depth objective at the end of the trial block.
    pool_ax.set_xlim(-2.2, 2.2)
    pool_ax.set_ylim(-2.15, 1.25)
    pool_ax.set_aspect("equal")
    pool_ax.axis("off")
    surface_z = 0.45
    target = Ellipse((0.0, surface_z - 2.15 / 2), 3.35, 2.15,
                     fill=False, edgecolor=DARK, lw=1.25,
                     ls=(0, (5, 3)))
    actual = Ellipse((0.0, surface_z - 1.78 / 2), 3.05, 1.78,
                     facecolor=LIGHT_BLUE, edgecolor=BLUE, lw=1.4,
                     alpha=0.85)
    surface_handle, = pool_ax.plot(
        (-2.05, 2.05), (surface_z, surface_z), color=GRAY, lw=1.0,
        zorder=0, label=r"top surface ($z=0$)")
    pool_ax.add_patch(target)
    pool_ax.add_patch(actual)
    beam_handle, = pool_ax.plot(
        0.0, surface_z, marker="o", ms=4.2, color="#d62728", lw=0,
        label="beam centre")
    # w_i is the half-width: dimension from the scan centreline (x=0) to the
    # liquidus edge, not the full edge-to-edge span.
    arrow(pool_ax, (0.0, -0.42), (1.52, -0.42),
          arrowstyle="<->", mutation_scale=9, color=BLUE, lw=1.0)
    pool_ax.text(0.76, -0.34, r"$w_i$", color=BLUE,
                 ha="center", va="bottom")
    arrow(pool_ax, (0.0, surface_z - 0.02), (0.0, surface_z - 1.76),
          arrowstyle="<->", mutation_scale=9, color=BLUE, lw=1.0)
    pool_ax.text(0.10, -0.90, r"$d_i$", color=BLUE,
                 ha="left", va="center")
    target_handle, = pool_ax.plot(
        [], [], color=DARK, lw=1.25, ls=(0, (5, 3)),
        label=r"target $(w^*,d^*)$")
    trial_handle, = pool_ax.plot(
        [], [], color=BLUE, lw=1.4, label=r"trial $(w_i,d_i)$")
    legend_ax.axis("off")
    legend_ax.legend(
        handles=(beam_handle, surface_handle, target_handle, trial_handle),
        loc="center left", bbox_to_anchor=(0.0, 0.50), frameon=False, ncol=1,
        handlelength=2.0, borderaxespad=0.0)
    pool_ax.text(0.0, -1.92,
                 r"reduce $\widetilde{\mathbf{r}}_i$; then commit $\mathbf{u}_i$",
                 ha="center", va="center")
    pool_ax.text(-2.10, 1.15, r"(b) Endpoint $y$--$z$ section at $x_{i+1}$",
                 ha="left", va="top", fontweight="bold")

    output = HERE / "figures" / "controller_block_schematic.pdf"
    fig.savefig(output, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
