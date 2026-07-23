"""Recompose the square thermal-gradient maps for the paper.

The full-field postprocessor writes the top surface above a centre section.
For the paper, the centre section is rotated so that its scan-coordinate axis
is vertical and it can sit beside the surface map. This preserves the map
data while using the page width more efficiently.
"""

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/square-paper-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/square-paper-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
import numpy as np


HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
FULLFIELD = HERE / "results" / "fullfield"

CASES = {
    "zero": (
        ("SqBaselineX50Y10Z1Zero_fig16_maps.png",
         "SqBaselineX50Y10Z1Zero"),
        ("SqGreedy1X50Y10Z1Zero_fig16_maps.png",
         "SqGreedy1X50Y10Z1Zero"),
    ),
    "dwell": (
        ("SqBaselineX50Y10Z1Dwell_fig16_maps.png",
         "SqBaselineX50Y10Z1Dwell"),
        ("SqGreedy1X50Y10Z1Dwell_fig16_maps.png",
         "SqGreedy1X50Y10Z1Dwell"),
    ),
}

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix",
    "font.size": 9.5, "axes.labelsize": 10,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def _border_rows(mask: np.ndarray) -> list[int]:
    """Return centres of the four long horizontal plot borders."""
    candidates = np.flatnonzero(mask.sum(axis=1) > 0.4 * mask.shape[1])
    groups = np.split(candidates, np.flatnonzero(np.diff(candidates) > 1) + 1)
    return [int(group[np.argmax(mask[group].sum(axis=1))])
            for group in groups if group.size]


def _extract_panels(filename: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract the surface field and data-only centre section from Fig. 16."""
    image = mpimg.imread(FIGURES / filename)[..., :3]
    mask = image.max(axis=2) < 0.30
    rows = _border_rows(mask)
    if len(rows) != 4:
        raise RuntimeError(f"could not identify plot borders in {filename}")
    top0, top1, section0, section1 = rows

    # The surface plot has two nearly full-height vertical spines. The
    # colour-bar spines are shorter and are excluded by this threshold.
    columns = np.flatnonzero(mask.sum(axis=0) > 0.55 * mask.shape[0])
    groups = np.split(columns, np.flatnonzero(np.diff(columns) > 1) + 1)
    borders = [int(group[np.argmax(mask[:, group].sum(axis=0))])
               for group in groups if group.size]
    if len(borders) != 2:
        raise RuntimeError(f"could not identify surface borders in {filename}")
    left, right = borders

    section_columns = np.flatnonzero(mask[section0])
    section_left = int(section_columns[0])
    section_right = int(section_columns[-1])
    surface = image[top0 + 1:top1, left + 1:right]
    section = image[
        section0 + 1:section1, section_left + 1:section_right]
    # Original rows are depth and columns are y. Transposition places depth
    # on the horizontal axis and y on the vertical axis.
    return surface, np.transpose(section, (1, 0, 2))


def _centre_depth_um(stem: str) -> float:
    """Maximum section depth at the x=5 mm centreline."""
    path = FULLFIELD / f"{stem}_fusionmap.csv"
    closest = float("inf")
    maximum = 0.0
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            delta = abs(float(row["x"]) - 0.005)
            depth = float(row["depth"]) * 1e6
            if delta < closest - 1e-12:
                closest, maximum = delta, depth
            elif abs(delta - closest) <= 1e-12:
                maximum = max(maximum, depth)
    return maximum


def make_policy_figure(
        policy: str,
        cases: tuple[tuple[str, str], tuple[str, str]]) -> None:
    fig = plt.figure(figsize=(7.2, 8.8))
    grid = fig.add_gridspec(
        2, 3, width_ratios=(1.0, 0.24, 0.045),
        hspace=0.14, wspace=0.10)
    for row, (filename, stem) in enumerate(cases):
        surface, section = _extract_panels(filename)
        depth = _centre_depth_um(stem)
        map_axis = fig.add_subplot(grid[row, 0])
        section_axis = fig.add_subplot(grid[row, 1], sharey=map_axis)

        map_axis.imshow(
            surface, extent=(0.0, 10.0, 0.0, 10.0),
            origin="upper", interpolation="nearest")
        map_axis.set_aspect("equal")
        map_axis.set_xlim(0.0, 10.0)
        map_axis.set_ylim(0.0, 10.0)
        map_axis.set_xlabel("x (mm)")
        map_axis.set_ylabel("y (mm)")

        section_axis.imshow(
            section, extent=(0.0, depth, 0.0, 10.0),
            origin="lower", interpolation="nearest", aspect="auto")
        section_axis.set_xlim(0.0, 110.0)
        section_axis.set_ylim(0.0, 10.0)
        section_axis.set_xticks((0, 50, 100))
        section_axis.set_xlabel("depth ($\mu$m)")
        section_axis.tick_params(labelleft=False)

    colour_axis = fig.add_subplot(grid[:, 2])
    scalar = ScalarMappable(
        norm=LogNorm(vmin=5e4, vmax=1e7), cmap="RdBu_r")
    fig.colorbar(scalar, cax=colour_axis, label="G (K/m)")
    fig.subplots_adjust(left=0.085, right=0.93, bottom=0.06, top=0.995)
    for extension in ("png", "pdf"):
        fig.savefig(
            FIGURES / f"square_maps_{policy}.{extension}",
            dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote square_maps_{policy}.png / .pdf")


def main() -> None:
    for policy, cases in CASES.items():
        make_policy_figure(policy, cases)


if __name__ == "__main__":
    main()
