#!/usr/bin/env python3
"""Fast, solver-free checks of the committed results against the paper.

Every expected number below is exactly a number printed in the paper
(paper/main.tex); the tolerance is half a unit in the last reported digit.
Checked: the square and triangle beam-on statistics tables, the two-track
block-length sweep, and the existence of every figure the paper includes.
"""

from __future__ import annotations

import decimal
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
PAPER = ROOT / "paper"


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
    print(f"PASS {label:18s} depth {depth[0]} +/- {depth[1]} um, "
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
        ROOT / "square/results/calibration/CalTrack_calibration.json",
        ROOT / "square/results/target_zero.json",
        ROOT / "square/results/greedy_1_zero.json",
        ROOT / "triangle/results/target_zero.json",
        ROOT / "triangle/results/greedy_1_zero.json",
        ROOT / "square/figures/square_traces_zero.png",
        ROOT / "triangle/figures/triangle_traces_zero.png",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required
               if not path.exists()]
    require(not missing, "missing publication artifacts: " + ", ".join(missing))
    print("PASS paper figure set and publication artifacts")


def main():
    check_paper_stats(
        "square baseline",
        "square/results/fullfield/SqBaselineX50Y10Z1Zero_fullfield.json",
        depth=("106.0", "12.7"), width_um=("388", "27"),
        volume=("0.0798", "0.0240"))
    check_paper_stats(
        "square optimized",
        "square/results/fullfield/SqGreedy1X50Y10Z1Zero_fullfield.json",
        depth=("62.2", "2.5"), width_um=("323", "9.9"),
        volume=("0.01293", "0.00127"))
    check_paper_stats(
        "triangle baseline",
        "triangle/results/fullfield/TriBaselineX50Y10Z1Zero_fullfield.json",
        depth=("118.7", "21.2"), width_um=("484", "170"),
        volume=("0.11274", "0.04715"))
    check_paper_stats(
        "triangle optimized",
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
