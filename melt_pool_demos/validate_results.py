#!/usr/bin/env python3
"""Fast, solver-free checks for the publication result tables.

Two layers are checked:

1. the historical per-segment power/sigma record (raster, square, triangle
   CSV tables) that established the controller;
2. the paper record (raster/doc/paper/main.tex): the greedy continuous
   P/sigma/v beam-on statistics for the square and triangle 50/10/1 um
   replays, the two-track block-length sweep, and the paper figure set.

Every expected number below is exactly a number printed in the paper; the
tolerance is half a unit in the last reported digit.
"""

from __future__ import annotations

import csv
import decimal
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
PAPER = ROOT / "raster/doc/paper"


def rows(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def values(table, column):
    return [float(row[column]) for row in table]


def cv(table, column):
    data = values(table, column)
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    return 100.0 * math.sqrt(variance) / mean


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def matches(actual: float, reported: str) -> bool:
    """True when `actual` rounds to the reported literal."""
    tolerance = 0.5 * 10.0 ** decimal.Decimal(reported).as_tuple().exponent
    return abs(actual - float(reported)) <= tolerance


def require_reported(label, actual, reported):
    require(matches(actual, reported),
            f"{label}: committed value {actual!r} does not round to the "
            f"paper's {reported}")


def check_pair(label, base_path, optimized_path, expected_rows, max_cv):
    base = rows(base_path)
    optimized = rows(optimized_path)
    require(len(base) == expected_rows, f"{label}: baseline has {len(base)} rows")
    require(len(optimized) == expected_rows,
            f"{label}: optimized has {len(optimized)} rows")
    bw, bd = cv(base, "w"), cv(base, "d")
    ow, od = cv(optimized, "w"), cv(optimized, "d")
    require(ow < bw and od < bd, f"{label}: optimization did not reduce both CVs")
    require(ow <= max_cv[0] and od <= max_cv[1],
            f"{label}: optimized CV {ow:.2f}/{od:.2f}% exceeds tolerance")
    print(f"PASS {label:22s} rows={expected_rows:3d}  CV w/d "
          f"{bw:6.2f}/{bd:6.2f}% -> {ow:6.2f}/{od:6.2f}%")


def beam_on(path: Path):
    with path.open() as handle:
        return json.load(handle)["beam_on"]


def check_paper_stats(label, path, depth, width_um, volume):
    stats = beam_on(ROOT / path)
    require_reported(f"{label} depth mean", stats["depth_um"]["mean"], depth[0])
    require_reported(f"{label} depth std", stats["depth_um"]["std"], depth[1])
    require_reported(f"{label} width mean",
                     1e3 * stats["width_mm"]["mean"], width_um[0])
    require_reported(f"{label} width std",
                     1e3 * stats["width_mm"]["std"], width_um[1])
    require_reported(f"{label} volume mean",
                     stats["volume_mm3"]["mean"], volume[0])
    require_reported(f"{label} volume std",
                     stats["volume_mm3"]["std"], volume[1])
    print(f"PASS {label:22s} depth {depth[0]} +/- {depth[1]} um, "
          f"width {width_um[0]} +/- {width_um[1]} um, "
          f"volume {volume[0]} +/- {volume[1]} mm^3")


def check_twotrack_sweep():
    expected = (
        ("unoptimized baseline", "TwoBaselineX50Y1Z1Continuous",
         "69.4", "12.4"),
        ("5 mm blocks", "TwoGreedy5localboxX50Y1Z1Continuous",
         "65.1", "9.4"),
        ("2.5 mm blocks", "TwoGreedy2p5X50Y1Z1Continuous",
         "63.4", "7.8"),
        ("1 mm blocks (selected)", "TwoGreedy1PrevCoolX50Y1Z1Continuous",
         "62.5", "6.6"),
        ("0.5 mm blocks", "TwoGreedy0p5localboxX50Y1Z1Continuous",
         "68.4", "10.3"),
        ("0.1 mm blocks", "TwoGreedy0p1localboxX50Y1Z1Continuous",
         "101.6", "16.3"),
    )
    for label, name, mean, std in expected:
        stats = beam_on(
            ROOT / "two_tracks/results/fullfield" / f"{name}_fullfield.json")
        require_reported(f"two-track {label} depth mean",
                         stats["depth_um"]["mean"], mean)
        require_reported(f"two-track {label} depth std",
                         stats["depth_um"]["std"], std)
    require((ROOT / "two_tracks/results/"
             "greedy_1_prevblock_coolseed_continuous.json").exists(),
            "selected two-track schedule JSON is missing")
    print("PASS two-track block-length sweep (Table 2)")


def check_paper_artifacts():
    figures = [
        "controller_block_schematic.pdf",
        "twotrack_base_maps.pdf",
        "twotrack_base_controls.pdf",
        "twotrack_base_traces.pdf",
        "square_maps_zero.pdf",
        "square_traces_zero.pdf",
        "square_process_parameters.pdf",
        "triangle_maps_zero.pdf",
        "triangle_traces_zero.pdf",
        "triangle_process_parameters.pdf",
    ]
    required = [PAPER / "main.tex", PAPER / "main.pdf",
                PAPER / "references.bib"]
    required += [PAPER / "figures" / name for name in figures]
    required += [
        ROOT / "raster/results/min_dwell.json",
        ROOT / "square/results/min_dwell.json",
        ROOT / "square/results/calibration/CalTrack_calibration.json",
        ROOT / "square/results/greedy_1_zero.json",
        ROOT / "triangle/results/greedy_1_zero.json",
        ROOT / "triangle/results/min_dwell.json",
        ROOT / "raster/doc/optimization_notes.pdf",
        ROOT / "raster/figures/raster_dims.png",
        ROOT / "square/figures/square_traces_zero.png",
        ROOT / "triangle/figures/triangle_traces_zero.png",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required
               if not path.exists()]
    require(not missing, "missing publication artifacts: " + ", ".join(missing))
    print("PASS paper figure set and publication artifacts")


def main():
    # -- historical per-segment power/sigma record -------------------------
    check_pair(
        "raster",
        ROOT / "raster/results/baseline.csv",
        ROOT / "raster/results/optimized.csv",
        14,
        (1.0, 1.0),
    )
    for policy in ("dwell", "zero"):
        check_pair(
            f"square/{policy}",
            ROOT / f"square/results/baseline_{policy}.csv",
            ROOT / f"square/results/optimized_{policy}.csv",
            202,
            (0.5, 0.5),
        )
    check_pair(
        "triangle/dwell",
        ROOT / "triangle/results/baseline_dwell.csv",
        ROOT / "triangle/results/optimized_dwell.csv",
        131,
        (1.0, 4.0),
    )
    check_pair(
        "triangle/zero",
        ROOT / "triangle/results/baseline_zero.csv",
        ROOT / "triangle/results/optimized_zero.csv",
        131,
        (5.0, 3.0),
    )

    with (ROOT / "triangle/results/min_dwell.json").open() as handle:
        triangle_dwell = json.load(handle)
    require(triangle_dwell["binding_turn"] == 79,
            "triangle minimal-dwell binding turn changed")

    # -- paper record: greedy continuous P/sigma/v -------------------------
    check_paper_stats(
        "paper square baseline",
        "square/results/fullfield/SqBaselineX50Y10Z1Zero_fullfield.json",
        depth=("106.0", "12.7"), width_um=("388", "27"),
        volume=("0.0798", "0.0240"))
    check_paper_stats(
        "paper square optimized",
        "square/results/fullfield/SqGreedy1X50Y10Z1Zero_fullfield.json",
        depth=("62.2", "2.5"), width_um=("323", "9.9"),
        volume=("0.01293", "0.00127"))
    check_paper_stats(
        "paper triangle baseline",
        "triangle/results/fullfield/TriBaselineX50Y10Z1Zero_fullfield.json",
        depth=("118.7", "21.2"), width_um=("484", "170"),
        volume=("0.11274", "0.04715"))
    check_paper_stats(
        "paper triangle optimized",
        "triangle/results/fullfield/TriGreedy1X50Y10Z1Zero_fullfield.json",
        depth=("63.7", "4.4"), width_um=("329", "15"),
        volume=("0.01176", "0.00160"))
    check_twotrack_sweep()
    check_paper_artifacts()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, FileNotFoundError, KeyError, ValueError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        raise SystemExit(1)
