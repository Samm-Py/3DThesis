"""Generate the three two-track figures used by the paper.

The retained comparison is the nominal continuous serpentine against the
selected 1 mm continuous P/sigma/v schedule. Figures are written as vector
PDFs under ``figures/``.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/two_tracks-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/two_tracks-cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FULLFIELD = os.path.join(HERE, "results", "fullfield")
FIGURES = os.path.join(HERE, "figures")
HOP_MM = 0.1e-3

C_TARGET = "black"
C_TURN = "tab:red"
C_BASE = "tab:gray"
C_OPT = "tab:blue"
POLICY_TITLE = {"continuous": "continuous serpentine",
                "dwell": "solidification dwell"}

# The base case worked forward from: the simplest continuous raster that
# modulates only P, sigma, v (no dwell), cool-seeded at the turnaround with
# previous-block warm start, presented against the nominal serpentine.
BASE_POLICY = "continuous"
BASE_SCHED = "greedy_1_prevblock_coolseed"
BASE_CASE = "TwoGreedy1PrevCool"
BASE_LABEL = "P/$\\sigma$/v optimized"
BASE_SERIES = [
    ("baseline (nominal)", None, "TwoBaseline",
     dict(color=C_BASE, ls="--", lw=1.1)),
    (BASE_LABEL, BASE_SCHED, BASE_CASE, dict(color=C_OPT, ls="-", lw=2.0)),
]


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "stix",
        "font.size": 9.5, "axes.labelsize": 10, "axes.titlesize": 10.5,
        "legend.fontsize": 8.6, "xtick.direction": "in",
        "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def dwell_s(saved=None) -> float:
    if saved is not None and saved.get("turnaround_dwells_s"):
        return float(saved["turnaround_dwells_s"][0])
    with open(os.path.join(HERE, "results", "min_dwell.json")) as f:
        return float(json.load(f)["min_dwell_s"])


def schedule_of(stem, policy):
    if stem is None:
        return None
    with open(os.path.join(HERE, "results", f"{stem}_{policy}.json")) as f:
        return json.load(f)


def case_name(case_stem, policy):
    return f"{case_stem}X50Y1Z1{policy.capitalize()}"


def have_traces(case_stem, policy):
    return os.path.exists(os.path.join(
        FULLFIELD, f"{case_name(case_stem, policy)}_traces.csv"))


def track2_start_ms(saved, policy):
    if saved is None:
        t, turn_v = 0.010 / 3.0, 3.0
    else:
        seg = saved["segment_mm"] * 1e-3
        first = [r for r in saved["schedule"] if r["line"] == 1]
        t = sum(seg / r["velocity_m_per_s"] for r in first)
        turn_v = first[-1]["velocity_m_per_s"]
    t += HOP_MM / turn_v if policy == "continuous" else dwell_s(saved)
    return 1e3 * t


def save(fig, stem):
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(os.path.join(FIGURES, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.pdf")


def traces_figure(policy, series, stem, title, subtitle, baseline_policy=None):
    # ``baseline_policy`` lets the reference/baseline series be drawn from a
    # different turnaround policy than the optimized curve -- e.g. a continuous
    # nominal serpentine baseline against an adaptive-dwell optimized schedule.
    baseline_policy = baseline_policy or policy
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.2), sharex=True,
                             gridspec_kw={"hspace": 0.12})
    fig.subplots_adjust(right=0.71, top=0.84)
    handles, labels = [], []
    for label, sched_stem, case_stem, kw in series:
        pol = baseline_policy if case_stem == "TwoBaseline" else policy
        if not have_traces(case_stem, pol):
            continue
        saved = schedule_of(sched_stem, pol)
        trace = pd.read_csv(os.path.join(
            FULLFIELD, f"{case_name(case_stem, pol)}_traces.csv"))
        t2 = track2_start_ms(saved, pol)
        x = trace["t"] * 1e3 - t2
        # Stats come from the authoritative fullfield_stats output (the same
        # beam-on mean/std reported in the README), not an ad-hoc trace mask.
        with open(os.path.join(
                FULLFIELD, f"{case_name(case_stem, pol)}_fullfield.json")) \
                as f:
            bo = json.load(f)["beam_on"]
        d, w = bo["depth_um"], bo["width_mm"]
        (h,) = axes[0].plot(x, trace["depth"] * 1e3, **kw)
        axes[1].plot(x, trace["width"] * 1e3, **kw)
        handles.append(h)
        labels.append(
            f"{label}\n"
            f"  depth {d['mean']:.1f}$\\pm${d['std']:.1f}\n"
            f"  width {w['mean']*1e3:.0f}$\\pm${w['std']*1e3:.0f} ($\\mu$m)")
    base = pd.read_csv(os.path.join(
        FULLFIELD, f"{case_name('TwoBaseline', baseline_policy)}_traces.csv"))
    dev = base[(base["t"] >= 2.5e-3) & (base["t"] < 3.25e-3)]
    (ref,) = axes[0].plot([], [], color=C_TARGET, lw=0.7, ls=(0, (5, 3)))
    axes[0].axhline(1e3 * dev["depth"].median(), color=C_TARGET, lw=0.7,
                    ls=(0, (5, 3)))
    axes[1].axhline(1e3 * dev["width"].median(), color=C_TARGET, lw=0.7,
                    ls=(0, (5, 3)))
    (turn,) = axes[0].plot([], [], color=C_TURN, lw=0.7, ls=(0, (2, 2)))
    for row in range(2):
        axes[row].axvline(0.0, color=C_TURN, lw=0.7, ls=(0, (2, 2)))
        axes[row].set_ylim(bottom=0.0)
        axes[row].grid(axis="y", color="#D8D8D6", lw=0.45, alpha=0.7)
    axes[0].set_ylabel("melt-pool depth ($\\mu$m)")
    axes[1].set_ylabel("melt-pool full width ($\\mu$m)")
    axes[1].set_xlabel("time from start of track 2 (ms)")
    fig.legend(handles + [ref, turn],
               labels + ["nominal target", "track-2 start"],
               loc="center left", bbox_to_anchor=(0.76, 0.5), frameon=False,
               handlelength=1.6, labelspacing=0.9)
    fig.suptitle(title, y=0.965, fontsize=11.5)
    fig.text(0.40, 0.885, subtitle, ha="center", va="top", fontsize=8.6)
    save(fig, stem)


def intervals_of(saved, policy):
    if saved is None:
        rows = [{"line": ln, "power_w": 150.0, "sigma_um": 200.0,
                 "velocity_m_per_s": 3.0} for ln in (1, 2) for _ in range(4)]
        seg = 2.5e-3
    else:
        rows, seg = saved["schedule"], saved["segment_mm"] * 1e-3
    out, t, shift = [], 0.0, 0.0
    for i, row in enumerate(rows):
        dt = seg / row["velocity_m_per_s"]
        out.append({"start": t, "end": t + dt, "power": row["power_w"],
                    "sigma": row["sigma_um"],
                    "velocity": row["velocity_m_per_s"]})
        t += dt
        if row["line"] == 1 and (i + 1 == len(rows)
                                 or rows[i + 1]["line"] == 2):
            if policy == "continuous":
                dt = HOP_MM / row["velocity_m_per_s"]
                out.append({"start": t, "end": t + dt, "power": row["power_w"],
                            "sigma": row["sigma_um"],
                            "velocity": row["velocity_m_per_s"]})
                t += dt
            else:
                turn_dwell = dwell_s(saved)
                out.append({"start": t, "end": t + turn_dwell, "power": 0.0,
                            "sigma": np.nan, "velocity": np.nan})
                t += turn_dwell
            shift = t
    for iv in out:
        iv["start"] -= shift
        iv["end"] -= shift
    return out


def controls_figure(policy, stem, opt_sched_stem="greedy_1_localbox",
                    opt_label="1 mm optimized", suptitle=None,
                    baseline_policy=None):
    baseline_policy = baseline_policy or policy
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 6.0), sharex=True,
                             gridspec_kw={"hspace": 0.12})
    fig.subplots_adjust(right=0.76, top=0.92)
    panels = (("power", "Power (W)"), ("sigma", "Beam $\\sigma$ ($\\mu$m)"),
              ("velocity", "Scan velocity (m/s)"))
    opt_saved = schedule_of(opt_sched_stem, policy)
    hist = {"baseline": intervals_of(None, baseline_policy),
            opt_label: intervals_of(opt_saved, policy)}
    handles = {}
    for row, (key, ylabel) in enumerate(panels):
        ax = axes[row]
        for kind, kw in (("baseline", dict(color=C_BASE, ls="--", lw=1.0)),
                         (opt_label,
                          dict(color=C_OPT, ls="-", lw=1.6))):
            h = hist[kind]
            edges = [h[0]["start"]] + [iv["end"] for iv in h]
            handles[kind] = ax.stairs([iv[key] for iv in h],
                                      1e3 * np.asarray(edges), baseline=None,
                                      **kw)
        ax.axvline(0.0, color=C_TURN, lw=0.7, ls=(0, (2, 2)))
        if policy == "dwell":
            ax.axvspan(-1e3 * dwell_s(opt_saved), 0.0, color="#D8D8D6",
                       alpha=0.45, lw=0.0)
        ax.grid(axis="y", color="#D8D8D6", lw=0.45, alpha=0.7)
        ax.set_ylabel(ylabel)
    axes[0].set_ylim(bottom=0.0)
    axes[-1].set_xlabel("time from start of track 2 (ms)")
    turn = plt.Line2D([], [], color=C_TURN, lw=0.7, ls=(0, (2, 2)))
    extra = [turn]
    extra_labels = ["track-2 start"]
    if policy == "dwell":
        extra.append(plt.Rectangle((0, 0), 1, 1, facecolor="#D8D8D6",
                                   alpha=0.45, edgecolor="none"))
        extra_labels.append("beam-off dwell")
    fig.legend(list(handles.values()) + extra,
               list(handles.keys()) + extra_labels,
               loc="center left", bbox_to_anchor=(0.78, 0.5), frameon=False,
               labelspacing=0.9)
    fig.suptitle(suptitle or f"1 mm control histories versus baseline "
                 f"({POLICY_TITLE[policy]})", y=0.975, fontsize=11.5)
    save(fig, stem)


def _gfield(df):
    xs, ys = np.unique(df["x"].values), np.unique(df["y"].values)
    gi = np.searchsorted(xs, df["x"].values)
    gj = np.searchsorted(ys, df["y"].values)
    F = np.full((ys.size, xs.size), np.nan)
    F[gj, gi] = df["G"].values
    return xs * 1e3, ys * 1e3, F


def maps_figure(policy, stem, opt_case_stem="TwoGreedy1localbox",
                opt_label="1 mm optimized", suptitle=None,
                baseline_policy=None):
    baseline_policy = baseline_policy or policy
    rows = [("TwoBaseline", "baseline"),
            (opt_case_stem, opt_label)]
    fig, axes = plt.subplots(len(rows), 1, figsize=(8.4, 4.0),
                             gridspec_kw={"hspace": 0.5})
    fig.subplots_adjust(right=0.86, top=0.86)
    norm = LogNorm(vmin=5e4, vmax=1e7)
    im = None
    for ax, (case_stem, ctitle) in zip(axes, rows):
        pol = baseline_policy if case_stem == "TwoBaseline" else policy
        name = case_name(case_stem, pol)
        case = os.path.join(
            HERE, "cases",
            f"case_baseline_hires_{pol}" if case_stem == "TwoBaseline"
            else f"case_{case_stem.lower().replace('two', '')}_hires_{pol}")
        sol_path = os.path.join(case, "Data",
                                f"{name}.Solidification.Final.csv")
        ax.set_title(ctitle, fontsize=9.8, loc="left", pad=3)
        if not os.path.exists(sol_path):
            ax.text(0.5, 0.5, "(pending replay)", ha="center", va="center",
                    transform=ax.transAxes, color="#999")
            ax.set_yticks([])
            continue
        sol = pd.read_csv(sol_path)
        zvals = np.unique(sol.z.values)
        zres = np.min(np.diff(zvals)) if zvals.size > 1 else 1.0
        surf = sol[np.isclose(sol.z.values, zvals.max(), atol=zres / 2)
                   & (sol.G > 0)]
        xs, ys, G = _gfield(surf)
        im = ax.pcolormesh(xs, ys, G, norm=norm, cmap="RdBu_r",
                           shading="gouraud")
        ax.set_ylabel("y (mm)")
        ax.set_ylim(ys.min(), ys.max())
        ax.margins(x=0)
    axes[-1].set_xlabel("x (mm)")
    if im is not None:
        fig.colorbar(im, ax=axes, label="G (K/m)", shrink=0.85, pad=0.02,
                     aspect=28)
    fig.suptitle(suptitle or f"Solidification thermal gradient: baseline vs "
                 f"1 mm-optimized ({POLICY_TITLE[policy]})", y=0.97,
                 fontsize=11.5)
    save(fig, stem)


def main():
    style()
    traces_figure(
        BASE_POLICY, BASE_SERIES, "twotrack_base_traces",
        "Two-track melt-pool regulation — continuous P/$\\sigma$/v",
        "modulating power, beam width, and velocity against the nominal "
        "serpentine (no dwell)")
    controls_figure(
        BASE_POLICY, "twotrack_base_controls",
        opt_sched_stem=BASE_SCHED, opt_label=BASE_LABEL,
        suptitle="Continuous P/$\\sigma$/v control history versus nominal")
    maps_figure(
        BASE_POLICY, "twotrack_base_maps",
        opt_case_stem=BASE_CASE, opt_label=BASE_LABEL,
        suptitle="Solidification thermal gradient: nominal vs "
                 "continuous P/$\\sigma$/v optimized")


if __name__ == "__main__":
    main()
