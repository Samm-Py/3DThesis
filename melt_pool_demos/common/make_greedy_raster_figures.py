"""Publication figures for the new 1 mm greedy square/triangle studies.

Run from ``square/`` or ``triangle/`` after the optimized refined replays.
The path parser honors every scheduled velocity, so folded profiles and
per-line statistics remain correct for variable-speed control.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/greedy-raster-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/greedy-raster-cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COMMON = os.path.dirname(os.path.abspath(__file__))
DEMOS = os.path.dirname(COMMON)
DEMO = os.path.abspath(os.getcwd())
GEOMETRY = os.path.basename(DEMO)
if GEOMETRY not in ("square", "triangle"):
    raise RuntimeError(
        "run make_greedy_raster_figures.py from square/ or triangle/")
sys.path.insert(0, DEMOS)
os.environ.setdefault("MP_GEOM", GEOMETRY)

from common.make_plots import fig16_maps  # noqa: E402


FIGURES = os.path.join(DEMO, "figures")
FULLFIELD = os.path.join(DEMO, "results", "fullfield")
C_BASE = "#9A9A98"
C_OPT = "#3D6B9E"
C_ZERO = "#B3512E"
C_DWELL = "#3D6B9E"
C_TARGET = "#1A1A1A"

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix",
    "font.size": 9.5, "axes.labelsize": 10, "axes.titlesize": 10,
    "legend.fontsize": 8.5, "xtick.direction": "in",
    "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def configuration(policy: str, resolution: str) -> dict:
    if GEOMETRY == "square":
        prefix = "Sq"
        legacy_baseline = {
            "zero": ("case_zero", "SqZero"),
            "dwell": ("case_dwell", "SqDwell"),
        }[policy]
    else:
        prefix = "Tri"
        legacy_baseline = {
            "zero": ("case_pub_z1", "TriPubZ1"),
            "dwell": ("case_mindwell", "TriMinDwell"),
        }[policy]
    if resolution == "X50Y50Z1":
        baseline = legacy_baseline
    else:
        baseline = (
            f"case_baseline_{resolution.lower()}_{policy}",
            f"{prefix}Baseline{resolution}{policy.capitalize()}",
        )
    optimized_name = f"{prefix}Greedy1{resolution}{policy.capitalize()}"
    optimized_case = (
        f"case_greedy1_{resolution.lower()}_{policy}")
    return {
        "schedule": os.path.join(DEMO, "results", f"greedy_1_{policy}.json"),
        "baseline_case": os.path.join(DEMO, "cases", baseline[0]),
        "baseline_name": baseline[1],
        "optimized_case": os.path.join(DEMO, "cases", optimized_case),
        "optimized_name": optimized_name,
    }


def load_policy(policy: str, resolution: str) -> dict:
    cfg = configuration(policy, resolution)
    with open(cfg["schedule"]) as f:
        cfg["saved"] = json.load(f)
    for kind in ("baseline", "optimized"):
        name = cfg[f"{kind}_name"]
        trace = pd.read_csv(
            os.path.join(FULLFIELD, f"{name}_traces.csv"))
        # Compare the same head-local width/depth regulated by the optimizer.
        # Retain global length/volume for the part-level panels.
        if {"local_width", "local_depth"}.issubset(trace.columns):
            trace["width"] = trace["local_width"]
            trace["depth"] = trace["local_depth"]
        cfg[f"{kind}_trace"] = trace
        with open(os.path.join(FULLFIELD, f"{name}_fullfield.json")) as f:
            cfg[f"{kind}_stats"] = json.load(f)
        cfg[f"{kind}_lines"] = path_lines(
            os.path.join(cfg[f"{kind}_case"], "Path.txt"))
        cfg[f"{kind}_samples"] = line_samples(
            cfg[f"{kind}_trace"], cfg[f"{kind}_lines"])
    return cfg


def path_lines(path: str) -> list[dict]:
    """Horizontal scan-line timing and local-distance maps from Path.txt."""
    lines, current = [], None
    t, previous = 0.0, None
    with open(path) as f:
        rows = f.read().splitlines()
    for row in rows:
        values = row.strip().split("\t")
        if values[0] not in ("0", "1"):
            continue
        point = np.array([float(values[1]), float(values[2])])
        if values[0] == "1":
            t += float(values[5])
            previous = point
            continue
        if previous is None:
            previous = point
            continue
        distance = float(np.linalg.norm(point - previous))
        duration = distance * 1e-3 / float(values[5])
        horizontal = abs(point[1] - previous[1]) <= 1e-9 and distance > 0
        if horizontal:
            if current is None or abs(point[1] - current["y"]) > 1e-9:
                current = {
                    "y": float(point[1]), "start": t, "end": t + duration,
                    "length": 0.0, "segments": [],
                }
                lines.append(current)
            start_distance = current["length"]
            current["segments"].append({
                "start": t, "end": t + duration,
                "s0": start_distance, "s1": start_distance + distance,
            })
            current["length"] += distance
            current["end"] = t + duration
        t += duration
        previous = point
    return lines


def line_samples(trace: pd.DataFrame, lines: list[dict]) -> list[pd.DataFrame]:
    """Trace samples for each scan line, with actual local beam distance."""
    out = []
    times = trace["t"].values
    for line in lines:
        indices, positions = [], []
        for segment in line["segments"]:
            mask = ((times >= segment["start"])
                    & (times < segment["end"]))
            ids = np.flatnonzero(mask)
            if not len(ids):
                continue
            fraction = ((times[ids] - segment["start"])
                        / (segment["end"] - segment["start"]))
            positions.extend(
                segment["s0"] + fraction * (segment["s1"] - segment["s0"]))
            indices.extend(ids)
        frame = trace.iloc[indices].copy()
        frame["s_local"] = positions
        out.append(frame.sort_values("s_local"))
    return out


def save(fig, stem: str) -> None:
    os.makedirs(FIGURES, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(os.path.join(FIGURES, f"{stem}.{extension}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.png / .pdf")


def optimization_target(cfg: dict, column: str) -> float:
    """Target in the units used by the compact trace (mm)."""
    if column == "width":
        # The OTI controller stores its radius-like half-span.
        return 2e-3 * float(cfg["saved"]["target"]["width_um"])
    if column == "depth":
        return 1e-3 * float(cfg["saved"]["target"]["depth_um"])
    raise KeyError(column)


def build_coordinate(samples: list[pd.DataFrame]) -> tuple[np.ndarray, dict]:
    x, values = [], {c: [] for c in ("depth", "width")}
    for line_number, frame in enumerate(samples):
        if frame.empty:
            continue
        length = frame["s_local"].max()
        local = (frame["s_local"].values / length
                 if length > 0 else np.zeros(len(frame)))
        x.extend(line_number + local)
        for column in values:
            values[column].extend(frame[column].values)
        x.append(np.nan)
        for column in values:
            values[column].append(np.nan)
    return np.asarray(x), {key: np.asarray(value)
                           for key, value in values.items()}


def comparison_figure(data: dict[str, dict]) -> None:
    policies = [policy for policy in ("zero", "dwell") if policy in data]
    fig, axes = plt.subplots(
        2, len(policies), figsize=(6.2 * len(policies), 6.0),
        sharex="col", gridspec_kw={"hspace": 0.10, "wspace": 0.20},
        squeeze=False)
    for column, policy in enumerate(policies):
        cfg = data[policy]
        for kind, color, label, width in (
                ("baseline", C_BASE, "baseline", 0.8),
                ("optimized", C_OPT, "1 mm optimized", 1.0)):
            x, values = build_coordinate(cfg[f"{kind}_samples"])
            axes[0, column].plot(
                x, values["depth"] * 1e3, color=color, lw=width, label=label)
            axes[1, column].plot(
                x, values["width"] * 1e3, color=color, lw=width, label=label)
        depth_ref = optimization_target(cfg, "depth") * 1e3
        width_ref = optimization_target(cfg, "width") * 1e3
        axes[0, column].axhline(
            depth_ref, color=C_TARGET, lw=0.7, ls=(0, (5, 3)),
            label="optimization target")
        axes[1, column].axhline(
            width_ref, color=C_TARGET, lw=0.7, ls=(0, (5, 3)))
        title = ("continuous serpentine" if policy == "zero"
                 else "solidification dwell")
        axes[0, column].set_title(title)
        axes[1, column].set_xlabel("scan line + fractional line position")
        for axis in axes[:, column]:
            axis.set_ylim(bottom=0)
            axis.grid(axis="y", color="#D8D8D6", lw=0.45, alpha=0.7)
    axes[0, 0].set_ylabel("melt-pool depth ($\\mu$m)")
    axes[1, 0].set_ylabel("melt-pool full width ($\\mu$m)")
    axes[0, 0].legend(frameon=False, ncol=3, loc="upper left",
                      bbox_to_anchor=(0.0, 1.22))
    save(fig, f"{GEOMETRY}_baseline_vs_optimized")


def controls_figure(data: dict[str, dict]) -> None:
    fig, axes = plt.subplots(
        3, 1, figsize=(8.2, 6.5), sharex=True,
        gridspec_kw={"hspace": 0.12})
    panels = (
        ("power_w", "Power (W)"),
        ("sigma_um", "Beam $\\sigma$ ($\\mu$m)"),
        ("velocity_m_per_s", "Scan velocity (m/s)"),
    )
    colors = {"zero": C_ZERO, "dwell": C_DWELL}
    labels = {"zero": "continuous serpentine",
              "dwell": "solidification dwell"}
    for policy, cfg in data.items():
        rows = cfg["saved"]["schedule"]
        positions, counts = [], {}
        totals = {}
        for row in rows:
            totals[row["line"]] = totals.get(row["line"], 0) + 1
        for row in rows:
            line = row["line"]
            counts[line] = counts.get(line, 0) + 1
            positions.append(
                line - 1 + counts[line] / totals[line])
        for axis, (key, _label) in zip(axes, panels):
            axis.step(positions, [row[key] for row in rows], where="post",
                      color=colors[policy], lw=0.9, label=labels[policy])
    nominal = (150.0, 200.0, 3.0)
    for axis, (_, label), reference in zip(axes, panels, nominal):
        axis.axhline(reference, color=C_BASE, lw=0.7, ls=(0, (5, 3)))
        axis.set_ylabel(label)
        axis.grid(axis="y", color="#D8D8D6", lw=0.45, alpha=0.7)
    axes[-1].set_xlabel("scan line + fractional line position")
    axes[0].legend(frameon=False, ncol=2)
    save(fig, f"{GEOMETRY}_process_parameters")


def per_line_statistics(samples: list[pd.DataFrame],
                        column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, low, high = [], [], []
    for frame in samples:
        values = frame[column].values
        if not len(values):
            mean.append(np.nan); low.append(np.nan); high.append(np.nan)
        else:
            mean.append(float(np.mean(values)))
            bounds = (np.percentile(values, [5, 95]) if len(values) >= 4
                      else (np.min(values), np.max(values)))
            low.append(float(bounds[0])); high.append(float(bounds[1]))
    return np.asarray(mean), np.asarray(low), np.asarray(high)


def traces_figure(policy: str, cfg: dict) -> None:
    fig, axes = plt.subplots(
        3, 1, figsize=(8.0, 7.2), sharex=True,
        gridspec_kw={"hspace": 0.10})
    panels = (
        ("depth", 1e3, "Melt-pool depth ($\\mu$m)"),
        ("width", 1e3, "Melt-pool full width ($\\mu$m)"),
        ("volume", 1.0, "Melt-pool volume (mm$^3$)"),
    )
    for axis, (key, scale, label) in zip(axes, panels):
        for kind, color, name in (
                ("baseline", C_BASE, "baseline"),
                ("optimized", C_OPT, "1 mm optimized")):
            mean, low, high = per_line_statistics(
                cfg[f"{kind}_samples"], key)
            line = np.arange(1, len(mean) + 1)
            axis.fill_between(
                line, low * scale, high * scale,
                color=color, alpha=0.25, lw=0)
            axis.plot(line, mean * scale, color=color, lw=1.1, label=name)
        axis.set_ylabel(label)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", color="#D8D8D6", lw=0.45, alpha=0.7)
    axes[0].axhline(
        optimization_target(cfg, "depth") * 1e3,
        color=C_TARGET, lw=0.7, ls=(0, (5, 3)),
        label="optimization target")
    axes[1].axhline(
        optimization_target(cfg, "width") * 1e3,
        color=C_TARGET, lw=0.7, ls=(0, (5, 3)))
    axes[-1].set_xlabel("scan line")
    axes[0].legend(frameon=False, ncol=3)
    save(fig, f"{GEOMETRY}_traces_{policy}")


def folded_curves(samples: list[pd.DataFrame], column: str,
                  grid: np.ndarray) -> np.ndarray:
    curves = []
    for frame in samples:
        if frame.empty:
            continue
        x = frame["s_local"].values
        y = frame[column].values
        curve = np.full_like(grid, np.nan, dtype=float)
        if len(x) == 1:
            curve[np.argmin(np.abs(grid - x[0]))] = y[0]
        else:
            inside = grid <= x.max()
            curve[inside] = np.interp(grid[inside], x, y)
        curves.append(curve)
    return np.asarray(curves)


def profiles_figure(policy: str, cfg: dict) -> None:
    grid = np.linspace(0.0, 10.0, 161)
    fig, axes = plt.subplots(
        2, 2, figsize=(9.4, 6.0), sharex=True,
        gridspec_kw={"hspace": 0.12, "wspace": 0.23})
    panels = (
        (axes[0, 0], "depth", 1e3, "Melt-pool depth ($\\mu$m)"),
        (axes[0, 1], "width", 1e3, "Melt-pool full width ($\\mu$m)"),
        (axes[1, 0], "length", 1.0, "Melt-pool length (mm)"),
        (axes[1, 1], "volume", 1.0, "Melt-pool volume (mm$^3$)"),
    )
    for axis, key, scale, label in panels:
        for kind, color, name in (
                ("baseline", C_BASE, "baseline"),
                ("optimized", C_OPT, "1 mm optimized")):
            curves = folded_curves(cfg[f"{kind}_samples"], key, grid)
            axis.plot(grid, curves.T * scale, color=color, lw=0.35, alpha=0.10)
            count = np.isfinite(curves).sum(axis=0)
            mean = np.divide(
                np.nansum(curves, axis=0), count,
                out=np.full(grid.shape, np.nan), where=count > 0)
            axis.plot(
                grid, mean * scale,
                color=color, lw=1.5, label=name)
        axis.set_ylabel(label)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", color="#D8D8D6", lw=0.45, alpha=0.7)
    axes[0, 0].axhline(
        optimization_target(cfg, "depth") * 1e3,
        color=C_TARGET, lw=0.7, ls=(0, (5, 3)),
        label="optimization target")
    axes[0, 1].axhline(
        optimization_target(cfg, "width") * 1e3,
        color=C_TARGET, lw=0.7, ls=(0, (5, 3)))
    axes[1, 0].set_xlabel("distance from start of scan line (mm)")
    axes[1, 1].set_xlabel("distance from start of scan line (mm)")
    axes[0, 0].legend(frameon=False, ncol=3, loc="upper left",
                      bbox_to_anchor=(0.0, 1.25))
    save(fig, f"{GEOMETRY}_profiles_{policy}")


def maps(data: dict[str, dict]) -> None:
    for cfg in data.values():
        for kind in ("baseline", "optimized"):
            case = cfg[f"{kind}_case"]
            name = cfg[f"{kind}_name"]
            source = os.path.join(
                case, "Data", f"{name}.Solidification.Final.csv")
            if os.path.exists(source):
                fig16_maps(case, name)
            else:
                print(f"map pending: {source}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", default="X50Y10Z1")
    parser.add_argument(
        "--policy", choices=("zero", "dwell", "both"), default="zero",
        help="turnaround policy to plot (default: continuous/zero dwell)")
    parser.add_argument("--skip-maps", action="store_true")
    args = parser.parse_args()
    data = {}
    policies = (("zero", "dwell") if args.policy == "both"
                else (args.policy,))
    for policy in policies:
        cfg = configuration(policy, args.resolution)
        required = (
            cfg["schedule"],
            os.path.join(FULLFIELD, f"{cfg['optimized_name']}_traces.csv"),
        )
        if all(os.path.exists(path) for path in required):
            data[policy] = load_policy(policy, args.resolution)
        else:
            print(f"{policy} figures pending optimized schedule/replay")
    if not data:
        raise RuntimeError("no completed greedy raster replay found")
    # Paper figure set only: the P/sigma/v control history, the per-line
    # beam-on traces, and the Fig. 16 map panels (recomposed by
    # make_paper_maps.py). The comparison and within-line profile figures are
    # not used by the paper.
    controls_figure(data)
    for policy, cfg in data.items():
        traces_figure(policy, cfg)
    if not args.skip_maps:
        maps(data)


if __name__ == "__main__":
    main()
