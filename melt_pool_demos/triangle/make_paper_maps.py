"""Recompose the triangle thermal-gradient maps for the paper.

The centre section is rotated so that its scan-coordinate axis is vertical
and it can sit beside the top-surface map. Baseline and optimized cases are
placed in rows on common depth and colour scales.
"""

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/triangle-paper-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/triangle-paper-cache")

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
        ("TriBaselineX50Y10Z1Zero_fig16_maps.png",
         "TriBaselineX50Y10Z1Zero"),
        ("TriGreedy1X50Y10Z1Zero_fig16_maps.png",
         "TriGreedy1X50Y10Z1Zero"),
    ),
    "dwell": (
        ("TriBaselineX50Y10Z1Dwell_fig16_maps.png",
         "TriBaselineX50Y10Z1Dwell"),
        ("TriGreedy1X50Y10Z1Dwell_fig16_maps.png",
         "TriGreedy1X50Y10Z1Dwell"),
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
    candidates = np.flatnonzero(mask.sum(axis=1) > 0.4 * mask.shape[1])
    groups = np.split(candidates, np.flatnonzero(np.diff(candidates) > 1) + 1)
    return [int(group[np.argmax(mask[group].sum(axis=1))])
            for group in groups if group.size]


def _extract_panels(filename: str) -> tuple[np.ndarray, np.ndarray]:
    image = mpimg.imread(FIGURES / filename)[..., :3]
    mask = image.max(axis=2) < 0.30
    rows = _border_rows(mask)
    if len(rows) != 4:
        raise RuntimeError(f"could not identify plot borders in {filename}")
    top0, top1, section0, section1 = rows

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
    return surface, np.transpose(section, (1, 0, 2))


def _map_metadata(stem: str) -> tuple[tuple[float, float, float, float], float]:
    """Surface bounds in mm and maximum depth at the x=5 mm centreline."""
    path = FULLFIELD / f"{stem}_fusionmap.csv"
    xmin = ymin = float("inf")
    xmax = ymax = float("-inf")
    closest = float("inf")
    maximum = 0.0
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            x, y = float(row["x"]), float(row["y"])
            depth = float(row["depth"]) * 1e6
            xmin, xmax = min(xmin, x), max(xmax, x)
            ymin, ymax = min(ymin, y), max(ymax, y)
            delta = abs(x - 0.005)
            if delta < closest - 1e-12:
                closest, maximum = delta, depth
            elif abs(delta - closest) <= 1e-12:
                maximum = max(maximum, depth)
    bounds = (xmin * 1e3, xmax * 1e3, ymin * 1e3, ymax * 1e3)
    return bounds, maximum


def make_policy_figure(
        policy: str,
        cases: tuple[tuple[str, str], tuple[str, str]]) -> None:
    prepared = []
    all_depths = []
    for filename, stem in cases:
        surface, section = _extract_panels(filename)
        bounds, depth = _map_metadata(stem)
        prepared.append((surface, section, bounds, depth))
        all_depths.append(depth)
    depth_limit = 25.0 * math.ceil(max(all_depths) / 25.0)

    fig = plt.figure(figsize=(7.2, 8.0))
    grid = fig.add_gridspec(
        2, 3, width_ratios=(1.0, 0.28, 0.045),
        hspace=0.15, wspace=0.10)
    for row, (surface, section, bounds, depth) in enumerate(prepared):
        xmin, xmax, ymin, ymax = bounds
        map_axis = fig.add_subplot(grid[row, 0])
        section_axis = fig.add_subplot(grid[row, 1], sharey=map_axis)

        map_axis.imshow(
            surface, extent=(xmin, xmax, ymin, ymax),
            origin="upper", interpolation="nearest")
        map_axis.set_aspect("equal")
        map_axis.set_xlim(0.0, 10.0)
        map_axis.set_ylim(0.0, 8.7)
        map_axis.set_xlabel("x (mm)")
        map_axis.set_ylabel("y (mm)")

        section_axis.imshow(
            section, extent=(0.0, depth, ymin, ymax),
            origin="lower", interpolation="nearest", aspect="auto")
        section_axis.set_xlim(0.0, depth_limit)
        section_axis.set_ylim(0.0, 8.7)
        section_axis.set_xticks(np.arange(0.0, depth_limit + 1.0, 50.0))
        section_axis.set_xlabel("depth ($\mu$m)")
        section_axis.tick_params(labelleft=False)

    colour_axis = fig.add_subplot(grid[:, 2])
    scalar = ScalarMappable(
        norm=LogNorm(vmin=5e4, vmax=1e7), cmap="RdBu_r")
    fig.colorbar(scalar, cax=colour_axis, label="G (K/m)")
    fig.subplots_adjust(left=0.085, right=0.93, bottom=0.065, top=0.995)
    for extension in ("png", "pdf"):
        fig.savefig(
            FIGURES / f"triangle_maps_{policy}.{extension}",
            dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote triangle_maps_{policy}.png / .pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy", choices=("zero", "dwell", "both"), default="zero",
        help="turnaround policy to compose (default: continuous/zero dwell)")
    args = parser.parse_args()
    policies = (("zero", "dwell") if args.policy == "both"
                else (args.policy,))
    for policy in policies:
        cases = CASES[policy]
        make_policy_figure(policy, cases)


if __name__ == "__main__":
    main()
