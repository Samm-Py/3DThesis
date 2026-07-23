"""Publication figures for the uniformity studies (square/triangle:
geometry follows the demo directory the script is run from).

  python make_plots.py dwell             -> square_dims_dwell.png/.pdf
  python make_plots.py zero              -> square_dims_zero.png/.pdf
  python make_plots.py compare           -> square_compare.png/.pdf
  python make_plots.py traces   <policy> -> square_traces_<policy>.png/.pdf
  python make_plots.py profiles <policy> -> square_profiles_<policy>.png/.pdf
  python make_plots.py fusionmap <policy>-> square_fusionmap_<policy>.png/.pdf
  python make_plots.py fig17 <case> <nm> -> <nm>_fig17_traces.png
  python make_plots.py fig16 <case> <nm> -> <nm>_fig16_maps.png

Per-policy figure: width/depth vs controlled path distance (baseline vs
optimized, target line), with the optimized power and beam-sigma schedules
underneath; bound-pinned segments ticked. Compare figure: both policies'
optimized dims on one pair of axes. fig17/fig16 are the single-case paper
views (L/D/V vs time; solidification thermal-gradient maps) for a tracked
case's raw Data, matching the triangle demo's postprocess.py.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, MaxNLocator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mp_lib as R

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix",
    "font.size": 9.5, "axes.labelsize": 10, "axes.titlesize": 10,
    "legend.fontsize": 8.5, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "axes.linewidth": 0.7,
    "lines.solid_capstyle": "round",
})
C_BASE = "#9A9A98"      # baseline: recessive warm gray
C_OPT = "#3D6B9E"       # optimized: muted blue (matches the triangle plots)
C_TGT = "#1A1A1A"
C_PIN = "#B3512E"       # bound-pinned marker


def cv(v):
    return 100.0 * np.std(v) / np.mean(v)


def load(policy):
    base = pd.read_csv(R.data_path(f"baseline_{policy}.csv"))
    opt = pd.read_csv(R.data_path(f"optimized_{policy}.csv"))
    tgt = json.load(open(R.data_path(f"target_{policy}.json")))
    return base, opt, tgt


def _case_path(directory, filename=None):
    path = os.path.join("cases", directory)
    if not os.path.exists(path):
        path = directory
    return os.path.join(path, filename) if filename else path


def _replay_artifact(directory, filename):
    """Prefer centralized compact results; accept legacy case-local files."""
    path = R.fullfield_path(filename)
    return path if os.path.exists(path) else _case_path(directory, filename)


def policy_figure(policy):
    base, opt, tgt = load(policy)
    s_b, s_o = base["s_mm"].values, opt["s_mm"].values
    fig, axes = plt.subplots(
        4, 1, figsize=(7.0, 7.6), sharex=True,
        gridspec_kw=dict(height_ratios=[1.4, 1.4, 1.0, 1.0], hspace=0.10))

    for ax, col, lab, t in (
            (axes[0], "w", "half-width $w$ ($\\mu$m)", tgt["w_star"]),
            (axes[1], "d", "depth $d$ ($\\mu$m)", tgt["d_star"])):
        ax.plot(s_b, base[col] * 1e6, color=C_BASE, lw=1.0,
                label=f"unoptimized (CV {cv(base[col].values):.1f}%)")
        ax.plot(s_o, opt[col] * 1e6, color=C_OPT, lw=1.2,
                label=f"optimized (CV {cv(opt[col].values):.2f}%)")
        ax.axhline(t * 1e6, color=C_TGT, lw=0.7, ls=(0, (5, 3)),
                   label="single-track target")
        ax.set_ylabel(lab)
        ax.legend(frameon=False, loc="upper left",
                  bbox_to_anchor=(1.015, 1.02), borderaxespad=0.0,
                  handlelength=1.6)

    axes[2].plot(s_o, opt["pmod"] * R.P_BASE, color=C_OPT, lw=1.0)
    axes[2].axhline(R.PMOD0 * R.P_BASE, color=C_BASE, lw=0.7, ls=(0, (5, 3)))
    axes[2].set_ylabel("power (W)")
    axes[3].plot(s_o, opt["sigma"] * 1e6, color=C_OPT, lw=1.0)
    axes[3].axhline(R.SIGMA * 1e6, color=C_BASE, lw=0.7, ls=(0, (5, 3)))
    axes[3].set_ylabel("$\\sigma$ ($\\mu$m)")
    axes[3].set_xlabel("controlled path distance (mm)")

    pin = opt[opt["pinned"].fillna("") != ""]
    if len(pin):
        for ax, col, scale in ((axes[2], "pmod", R.P_BASE),
                               (axes[3], "sigma", 1e6)):
            ax.plot(pin["s_mm"], pin[col] * scale, ls="none", marker="|",
                    ms=6, mew=1.1, color=C_PIN)
        axes[2].text(1.015, 0.98, f"{len(pin)}/{len(opt)}\nsegments\nat a bound",
                     transform=axes[2].transAxes, ha="left", va="top",
                     color=C_PIN, fontsize=8)

    if policy == "dwell":
        title = f"minimal dwell ({R.dwell_from_json()*1e3:.3f} ms/turn)"
    else:
        title = "zero dwell (continuous serpentine)"
    axes[0].set_title(f"10 mm {R.GEOM} raster, {title}")
    for ext in ("png", "pdf"):
        fig.savefig(R.fig_path(f"{R.GEOM}_dims_{policy}.{ext}"), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"square_dims_{policy}: baseline CV w/d = "
          f"{cv(base['w'].values):.2f}/{cv(base['d'].values):.2f}% -> "
          f"optimized {cv(opt['w'].values):.2f}/{cv(opt['d'].values):.2f}%  "
          f"pinned {len(pin)}/{len(opt)}")


def compare_figure():
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.8), sharex=True,
                             gridspec_kw=dict(hspace=0.12))
    styles = {"dwell": dict(color=C_OPT, lw=1.1),
              "zero": dict(color="#B3512E", lw=1.1)}
    names = {"dwell": "minimal dwell", "zero": "zero dwell"}
    tgt = None
    for policy in ("dwell", "zero"):
        _, opt, tgt = load(policy)
        for ax, col in ((axes[0], "w"), (axes[1], "d")):
            ax.plot(opt["s_mm"], opt[col] * 1e6, **styles[policy],
                    label=f"{names[policy]} (CV {cv(opt[col].values):.2f}%)")
    for ax, col, key in ((axes[0], "w", "w_star"), (axes[1], "d", "d_star")):
        ax.axhline(tgt[key] * 1e6, color=C_TGT, lw=0.7, ls=(0, (5, 3)),
                   label="target")
        ax.legend(frameon=False, loc="upper left",
                  bbox_to_anchor=(1.015, 1.02), borderaxespad=0.0)
    axes[0].set_ylabel("half-width $w$ ($\\mu$m)")
    axes[1].set_ylabel("depth $d$ ($\\mu$m)")
    axes[1].set_xlabel("controlled path distance (mm)")
    axes[0].set_title("optimized pool dimensions: price of removing the dwell")
    for ext in ("png", "pdf"):
        fig.savefig(R.fig_path(f"{R.GEOM}_compare.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("square_compare written")


def _tags(policy):
    """(baseline case dir, baseline name, optimized name, subtitle)."""
    if R.GEOM == "single_track":
        return "case_base", "Track", "TrackOpt", "single track"
    if R.GEOM == "triangle":
        b = {"zero": ("case_pub_z1", "TriPubZ1"),
             "dwell": ("case_mindwell", "TriMinDwell")}[policy]
        opt = f"TriOpt{policy.capitalize()}"
    else:
        b = {"zero": ("case_zero", "SqZero"),
             "dwell": ("case_dwell", "SqDwell")}[policy]
        opt = f"SqOpt{policy.capitalize()}"
    sub = ("zero dwell (continuous serpentine)" if policy == "zero" else
           f"minimal dwell ({R.dwell_from_json()*1e3:.3f} ms/turn)")
    return b[0], b[1], opt, sub


def traces_figure(policy):
    """Per-line pool statistics, baseline vs optimized replay. The process
    is line-periodic, so the x-axis is the scan line index: per line, the
    mean and the p5-p95 band of the instantaneous beam-on samples for
    depth / volume / length (raw samples against build time render every
    ~3 ms line as a 1-pixel needle — the information is in the envelope).
    A detail strip below resolves three consecutive lines in real time.
    Panels are annotated with the whole-build beam-on mean±sigma from
    <name>_fullfield.json (the headline statistics, which include the
    regrowth ramps the p5 edge clips)."""
    bdir, bname, oname, sub = _tags(policy)
    tag = (bname, oname, sub)
    base = pd.read_csv(_replay_artifact(bdir, f"{tag[0]}_traces.csv"))
    opt = pd.read_csv(_replay_artifact(f"case_opt_{policy}", f"{tag[1]}_traces.csv"))
    tgt = json.load(open(R.data_path(f"target_{policy}.json")))

    def _ff(d, nm):
        try:
            return json.load(open(_replay_artifact(d, f"{nm}_fullfield.json")))["beam_on"]
        except FileNotFoundError:
            return None
    ff = {"unoptimized": _ff(bdir, tag[0]),
          "optimized": _ff(f"case_opt_{policy}", tag[1])}

    # scan-line windows from the geometry (same construction as profiles)
    dwell = 0.0 if policy == "zero" else R.dwell_from_json()
    starts = R.line_start_times(R.build_rows(dwell))
    V_MM = R.V * 1e3                                          # mm/s
    lens = [abs(x1 - x0) for x0, x1, _ in R.scan_lines()]
    lines = [(t0, t0 + Lmm / V_MM) for t0, Lmm in zip(starts, lens)]
    nl = len(lines)

    def per_line(df):
        """(mean, lo, hi) arrays over lines; lo/hi = p5/p95 of the beam-on
        samples in the line's own window (min/max when a short line has too
        few samples for percentiles; NaN when it has none, e.g. the
        triangle's sub-sample apex lines)."""
        t = df["t"].values
        out = {}
        for c in ("length", "depth", "volume"):
            v = df[c].values
            mean = np.full(nl, np.nan)
            lo = np.full(nl, np.nan)
            hi = np.full(nl, np.nan)
            for k, (a, b) in enumerate(lines):
                m = (t >= a) & (t < b)
                n = int(m.sum())
                if n == 0:
                    continue
                s = v[m]
                mean[k] = s.mean()
                lo[k], hi[k] = (np.percentile(s, [5.0, 95.0]) if n >= 4
                                else (s.min(), s.max()))
            out[c] = (mean, lo, hi)
        return out

    pl = {"unoptimized": per_line(base), "optimized": per_line(opt)}
    kx = np.arange(nl)

    fig = plt.figure(figsize=(10.8, 6.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.55],
                          hspace=0.38, wspace=0.30)
    axes = [fig.add_subplot(gs[0, j]) for j in range(3)]
    for ax in axes[1:]:
        ax.sharex(axes[0])
    ax_zoom = fig.add_subplot(gs[1, :])

    panels = ((axes[0], "depth", 1e3, "depth_um", "{:.0f}±{:.0f}",
               "melt pool depth ($\\mu$m)", tgt["d_star"] * 1e6),
              (axes[1], "volume", 1.0, "volume_mm3", "{:.3f}±{:.3f}",
               "melt pool volume (mm$^3$)", None),
              (axes[2], "length", 1.0, "length_mm", "{:.2f}±{:.2f}",
               "melt pool length (mm)", None))
    for ax, col, sc, key, f2, lab, t in panels:
        for series, color in (("unoptimized", C_BASE), ("optimized", C_OPT)):
            mean, lo, hi = pl[series][col]
            ax.fill_between(kx, lo * sc, hi * sc, color=color, alpha=0.30,
                            lw=0)
            ax.plot(kx, mean * sc, color=color, lw=1.3)
        if t is not None:
            ax.axhline(t, color=C_TGT, lw=0.7, ls=(0, (5, 3)))
        ax.set_ylabel(lab)
        ax.set_ylim(bottom=0)
        ax.margins(y=0.26)
        ax.set_xlabel("scan line")
        if ff["optimized"] and ff["unoptimized"]:
            note = (f"beam-on: {f2.format(ff['unoptimized'][key]['mean'], ff['unoptimized'][key]['std'])}"
                    f" $\\rightarrow$ {f2.format(ff['optimized'][key]['mean'], ff['optimized'][key]['std'])}")
            ax.text(0.975, 0.975, note, transform=ax.transAxes, ha="right",
                    va="top", fontsize=7.5, color=C_TGT,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none",
                              pad=1.5))

    # detail strip: three consecutive mid-build lines in real time (the only
    # panel where the 50 us samples themselves are shown)
    k0 = nl // 2
    t0 = lines[k0][0] - 0.4e-3
    t1 = lines[k0 + 2][1] + 0.4e-3
    if dwell > 0:
        # break the trace across beam-off windows: each pool is its own arc
        ivals, _ = R.beam_on_intervals(_case_path(bdir, "Path.txt"))
        gaps = [(a2, b2) for (_, a2), (b2, _) in zip(ivals[:-1], ivals[1:])
                if b2 - a2 > 1e-9]
        for df in (base, opt):
            t = df["t"].values
            for a2, b2 in gaps:
                df.loc[(t > a2) & (t < b2), "depth"] = np.nan
    for df, color in ((base, C_BASE), (opt, C_OPT)):
        m = (df["t"] >= t0) & (df["t"] <= t1)
        ax_zoom.plot(df["t"][m] * 1e3, df["depth"][m] * 1e3, color=color,
                     lw=1.3, marker=".", ms=2.5)
    ax_zoom.axhline(tgt["d_star"] * 1e6, color=C_TGT, lw=0.7, ls=(0, (5, 3)))
    ax_zoom.set_xlim(t0 * 1e3, t1 * 1e3)
    ax_zoom.set_ylim(bottom=0)
    ax_zoom.set_xlabel("time (ms)")
    ax_zoom.set_ylabel("depth ($\\mu$m)")
    ax_zoom.set_title(
        f"detail: scan lines {k0}–{k0 + 2} in real time — " +
        ("each pool grows, holds, and is extinguished by its dwell"
         if dwell > 0 else
         "the pool survives every turnaround (no dwell)"), fontsize=9)
    # mark the zoomed lines in the depth panel
    axes[0].axvspan(k0 - 0.5, k0 + 2.5, color="#B3512E", alpha=0.25, lw=0)

    handles = [plt.Line2D([], [], color=C_BASE, lw=1.4, label="unoptimized"),
               plt.Line2D([], [], color=C_OPT, lw=1.4, label="optimized"),
               mpatches.Patch(facecolor="0.55", alpha=0.35, lw=0,
                              label="per-line p5–p95 (beam-on)"),
               plt.Line2D([], [], color=C_TGT, lw=0.9, ls=(0, (5, 3)),
                          label="single-track target (depth)")]
    fig.legend(handles=handles, frameon=False, ncol=len(handles),
               loc="upper center", bbox_to_anchor=(0.5, 0.97))
    fig.suptitle(f"10 mm {R.GEOM} raster, {sub}: per-line pool statistics",
                 y=1.015)
    for ext in ("png", "pdf"):
        fig.savefig(R.fig_path(f"{R.GEOM}_traces_{policy}.{ext}"), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"{R.GEOM}_traces_{policy} written")


def profiles_figure(policy):
    """Folded per-line view: every line's beam-on trace overlaid against
    position along the line (all sampled lines light, mean bold). Lines with
    two or more samples are interpolated; a one-sample apex line is retained
    as a point rather than excluded or expanded into a fabricated profile.
    The natural presentation of a cyclic process — the regrowth transient
    appears once, where it physically lives, and line-to-line variation is
    the thickness of the bundle."""
    bdir, bname, oname, sub = _tags(policy)
    tag = (bname, oname, sub)
    # scan-line windows from the geometry itself (duration can't separate
    # short apex lines from hops on the triangle)
    dwell = 0.0 if policy == "zero" else R.dwell_from_json()
    starts = R.line_start_times(R.build_rows(dwell))
    V_MM = R.V * 1e3                                          # mm/s
    lens = [abs(x1 - x0) for x0, x1, _ in R.scan_lines()]
    lines = [(t0, t0 + Lmm / V_MM) for t0, Lmm in zip(starts, lens)]
    xg = np.linspace(0.0, R.S, 141)

    def fold(df):
        t = df["t"].values
        out = {c: [] for c in ("length", "depth", "width", "volume")}
        for a, b in lines:
            m = (t >= a) & (t < b)
            n = int(m.sum())
            if n == 0:
                continue
            xi = (t[m] - a) * V_MM
            for c in out:
                if n == 1:
                    cur = np.full_like(xg, np.nan)
                    cur[np.argmin(np.abs(xg - xi[0]))] = df[c].values[m][0]
                else:
                    cur = np.interp(xg, xi, df[c].values[m])
                cur[xg > (b - a) * V_MM] = np.nan
                out[c].append(cur)
        return {c: np.array(v) for c, v in out.items()}

    base = fold(pd.read_csv(_replay_artifact(bdir, f"{tag[0]}_traces.csv")))
    opt = fold(pd.read_csv(_replay_artifact(f"case_opt_{policy}", f"{tag[1]}_traces.csv")))
    tgt = json.load(open(R.data_path(f"target_{policy}.json")))

    def _ff(d, nm):
        try:
            return json.load(open(_replay_artifact(d, f"{nm}_fullfield.json")))["beam_on"]
        except FileNotFoundError:
            return None
    ff = {"unoptimized": _ff(bdir, tag[0]),
          "optimized": _ff(f"case_opt_{policy}", tag[1])}

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.0), sharex=True,
                             gridspec_kw=dict(hspace=0.12, wspace=0.22))
    panels = ((axes[0, 0], "length", 1.0, "length_mm", "{:.2f}±{:.2f}",
               "melt pool length (mm)", None),
              (axes[0, 1], "depth", 1e3, "depth_um", "{:.0f}±{:.0f}",
               "melt pool depth ($\\mu$m)", tgt["d_star"] * 1e6),
              (axes[1, 0], "width", 1.0, "width_mm", "{:.2f}±{:.2f}",
               "liquid lateral extent (mm)", None),
              (axes[1, 1], "volume", 1.0, "volume_mm3", "{:.3f}±{:.3f}",
               "melt pool volume (mm$^3$)", None))
    for ax, col, sc, key, f2, lab, t in panels:
        for curves, color in ((base[col], C_BASE), (opt[col], C_OPT)):
            ax.plot(xg, curves.T * sc, color=color, lw=0.4, alpha=0.10)
            # A line with one 50 us sample has no resolvable profile shape;
            # show the measurement itself rather than silently dropping it.
            one = np.isfinite(curves).sum(axis=1) == 1
            if one.any():
                iy, ix = np.where(np.isfinite(curves[one]))
                ax.scatter(xg[ix], curves[one][iy, ix] * sc, s=5,
                           color=color, alpha=0.30, linewidths=0, zorder=2)
            ax.plot(xg, np.nanmean(curves, axis=0) * sc, color=color, lw=1.6)
        if t is not None:
            ax.axhline(t, color=C_TGT, lw=0.7, ls=(0, (5, 3)))
        ax.set_ylabel(lab)
        ax.set_ylim(bottom=0)
        ax.margins(y=0.24)
        if ff["optimized"] and ff["unoptimized"]:
            note = (f"beam-on: {f2.format(ff['unoptimized'][key]['mean'], ff['unoptimized'][key]['std'])}"
                    f" $\\rightarrow$ {f2.format(ff['optimized'][key]['mean'], ff['optimized'][key]['std'])}")
            ax.text(0.985, 0.965, note, transform=ax.transAxes, ha="right",
                    va="top", fontsize=8, color=C_TGT,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none",
                              pad=1.5))
    for ax in axes[1, :]:
        ax.set_xlabel("position along line (mm)")
    handles = [plt.Line2D([], [], color=C_BASE, lw=1.6,
                          label="unoptimized (all lines + mean)"),
               plt.Line2D([], [], color=C_OPT, lw=1.6,
                          label="optimized (all lines + mean)"),
               plt.Line2D([], [], color=C_TGT, lw=0.9, ls=(0, (5, 3)),
                          label="single-track target (depth)")]
    fig.legend(handles=handles, frameon=False, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 0.965))
    nshown = base["depth"].shape[0]
    shown = f"all {nshown} sampled lines" if nshown == len(lines) else \
            f"{nshown}/{len(lines)} sampled lines"
    fig.suptitle(f"10 mm {R.GEOM} raster, {sub}: per-line pool profiles, "
                 f"{shown} overlaid", y=1.0)
    for ext in ("png", "pdf"):
        fig.savefig(R.fig_path(f"{R.GEOM}_profiles_{policy}.{ext}"), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"{R.GEOM}_profiles_{policy} written")


def _fusion_map(case, name):
    """Deepest melt per (x,y) column over the whole build: from the compact
    map CSV when present, else recomputed from the raw RDF."""
    mp = R.fullfield_path(f"{name}_fusionmap.csv")
    if os.path.exists(mp):
        dm = pd.read_csv(mp).rename(columns={"depth": "d"})
    else:
        rdf = pd.read_csv(os.path.join(case, "Data", f"{name}.RDF.Final.csv"))
        rdf.columns = [c.strip() for c in rdf.columns]
        dm = (-rdf.groupby(["x", "y"])["z"].min()).reset_index(name="d")
    xs, ys = np.unique(dm.x), np.unique(dm.y)
    grid = np.full((ys.size, xs.size), np.nan)
    grid[np.searchsorted(ys, dm.y), np.searchsorted(xs, dm.x)] = dm.d * 1e6
    return xs * 1e3, ys * 1e3, grid


def fusionmap_figure(policy):
    """The part-level view: fusion-depth maps, baseline vs optimized, with
    the interior-column histograms."""
    tag = {"zero": ("SqZero", "SqOptZero"),
           "dwell": ("SqDwell", "SqOptDwell")}[policy]
    def _case(d):
        p = os.path.join("cases", d)
        return p if os.path.exists(p) else d
    maps = [_fusion_map(_case(f"case_{policy}"), tag[0]),
            _fusion_map(_case(f"case_opt_{policy}"), tag[1])]
    tgt = json.load(open(R.data_path(f"target_{policy}.json")))

    vmin = min(np.nanmin(g) for *_, g in maps)
    vmax = max(np.nanmax(g) for *_, g in maps)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6),
                             gridspec_kw=dict(width_ratios=[1.15, 1.15, 1.0],
                                              wspace=0.28))
    for ax, (xs, ys, g), lab in zip(axes[:2], maps,
                                    ("unoptimized", "optimized")):
        im = ax.pcolormesh(xs, ys, g, cmap="viridis", vmin=vmin, vmax=vmax,
                           shading="auto")
        ax.set_aspect("equal")
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    fig.colorbar(im, ax=axes[:2], pad=0.015, shrink=0.9,
                 label="fusion depth ($\\mu$m)")

    H = 0.1
    for (xs, ys, g), col, lab in zip(maps, (C_BASE, C_OPT),
                                     ("unoptimized", "optimized")):
        inx = (xs > xs.min() + H) & (xs < xs.max() - H)
        iny = (ys > ys.min() + H) & (ys < ys.max() - H)
        d = g[np.ix_(iny, inx)].ravel()
        d = d[np.isfinite(d)]
        axes[2].hist(d, bins=np.arange(vmin - 2.5, vmax + 5, 5), density=True,
                     histtype="stepfilled", alpha=0.55, color=col,
                     label=f"{lab}\n(CV {100*d.std()/d.mean():.1f}%, "
                           f"$\\sigma$ {d.std():.1f} $\\mu$m)")
    axes[2].axvline(tgt["d_star"] * 1e6, color=C_TGT, lw=0.8, ls=(0, (5, 3)),
                    label="target")
    axes[2].set_xlabel("fusion depth ($\\mu$m)")
    axes[2].set_ylabel("density (interior columns)")
    axes[2].legend(frameon=False, loc="upper left",
                   bbox_to_anchor=(1.02, 1.02), borderaxespad=0.0, fontsize=8)
    fig.suptitle(f"whole-build fusion depth, {policy} dwell policy",
                 fontsize=11, y=1.02)
    for ext in ("png", "pdf"):
        fig.savefig(R.fig_path(f"{R.GEOM}_fusionmap_{policy}.{ext}"), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"{R.GEOM}_fusionmap_{policy} written")


# --- paper-style single-case figures (cf. Stump & Plotkowski Figs 16, 17) ---
# Reproduce the triangle demo's Fig 17 (L/D/V vs time from the RDF event list)
# and Fig 16 (solidification thermal-gradient maps) for any tracked case, so
# the square raster carries the same publication views. Axis limits are fixed
# per geometry so a policy's baseline and optimized panels are directly
# comparable (the triangle uses its own postprocess.py with the same intent).
_FIG17_LIMITS = {                       # (ylim, yticks) per L/D/V panel
    "triangle": dict(length=((0.0, 8.0), np.arange(0, 8.1, 2)),
                     depth=((0.0, 0.20), np.arange(0, 0.201, 0.05)),
                     volume=((0.0, 0.25), np.arange(0, 0.251, 0.05))),
    "square": dict(length=((0.0, 7.0), np.arange(0, 7.1, 2)),
                   depth=((0.0, 0.15), np.arange(0, 0.151, 0.05)),
                   volume=((0.0, 0.12), np.arange(0, 0.121, 0.04))),
    "single_track": dict(length=((0.0, 2.5), np.arange(0, 2.6, 0.5)),
                         depth=((0.0, 0.10), np.arange(0, 0.101, 0.02)),
                         volume=((0.0, 0.02), np.arange(0, 0.0201, 0.005))),
}


def _read_case(case_dir, name):
    data = os.path.join(_case_path(case_dir), "Data")
    sol = pd.read_csv(os.path.join(data, f"{name}.Solidification.Final.csv"))
    rdf = pd.read_csv(os.path.join(data, f"{name}.RDF.Final.csv"))
    sol.columns = [c.strip() for c in sol.columns]
    rdf.columns = [c.strip() for c in rdf.columns]
    return sol, rdf


def _rdf_traces(rdf, dt):
    """Instantaneous pool volume/length/depth vs time from the melt-event
    list: a cell is liquid on [tm, tl), so each property is an aggregation
    over the events active in each time bin (same method as postprocess.py)."""
    tm, tl = rdf.tm.values, rdf.tl.values
    nbin = int(np.ceil(tl.max() / dt)) + 1
    times = np.arange(nbin) * dt
    b0 = np.minimum((tm / dt).astype(np.int64), nbin - 1)
    b1 = np.minimum((tl / dt).astype(np.int64), nbin - 1)
    dv = np.zeros(nbin)
    np.add.at(dv, b0, 1.0)
    np.add.at(dv, b1, -1.0)
    n_active = b1 - b0 + 1
    idx = np.repeat(b0, n_active) + (np.arange(n_active.sum()) -
                                     np.repeat(np.cumsum(n_active) - n_active,
                                               n_active))

    def extent(vals, agg):
        acc = np.full(nbin, -np.inf if agg is np.maximum else np.inf)
        agg.at(acc, idx, np.repeat(vals, n_active))
        out = np.full(nbin, np.nan)
        ok = np.isfinite(acc)
        out[ok] = acc[ok]
        return out

    xmax = extent(rdf.x.values, np.maximum)
    xmin = extent(rdf.x.values, np.minimum)
    zmin = extent(rdf.z.values, np.minimum)
    return dict(t=times, volume=np.cumsum(dv), length=(xmax - xmin) * 1e3,
                depth=-zmin * 1e3, active=np.isfinite(xmax))


def fig17_traces(case_dir, name, dt=5e-5):
    """Fig 17: melt-pool length / depth / volume vs time from the RDF."""
    _, rdf = _read_case(case_dir, name)
    xres = np.min(np.diff(np.unique(rdf.x.values)))
    yres = np.min(np.diff(np.unique(rdf.y.values)))
    zres = np.min(np.diff(np.unique(rdf.z.values)))
    cellvol = ((xres * 1e3) * (yres * 1e3) *
               (zres * 1e3))                            # mm^3 per cell-event
    tr = _rdf_traces(rdf, dt)
    volume = tr["volume"] * cellvol

    lims = _FIG17_LIMITS.get(R.GEOM, _FIG17_LIMITS["square"])
    t_end = tr["t"].max()
    # short builds (single track ~3 ms) need ms on the time axis, else every
    # tick rounds to 0.00 s
    tscale, tlabel, tfmt = ((1e3, "Time (ms)", "%.1f") if t_end < 0.02
                            else (1.0, "Time (s)", "%.2f"))
    fig, axs = plt.subplots(1, 3, figsize=(10.6, 3.2))
    panels = [(tr["length"], "Melt Pool Length (mm)", "A", lims["length"]),
              (tr["depth"], "Melt Pool Depth (mm)", "B", lims["depth"]),
              (volume, r"Melt Pool Volume ($\mathrm{mm}^3$)", "C",
               lims["volume"])]
    for ax, (series, label, tag, (ylim, yticks)) in zip(axs, panels):
        ax.plot(tr["t"] * tscale, series, color=C_OPT, lw=0.7)
        ax.set_xlim(0.0, t_end * tscale)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(FormatStrFormatter(tfmt))
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(
            FormatStrFormatter("%.3f" if ylim[1] < 0.05 else "%.2f"))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.set_xlabel(tlabel)
        ax.set_ylabel(label)
        ax.text(0.04, 0.96, tag, transform=ax.transAxes, fontsize=12, va="top")
        ax.tick_params(which="major", direction="out", length=3.5, width=0.8,
                       color="#666666")
        ax.tick_params(which="minor", direction="out", length=2.0, width=0.6,
                       color="#666666")
        for spine in ax.spines.values():
            spine.set_color("#666666")
            spine.set_linewidth(0.8)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.20, top=0.96,
                        wspace=0.43)
    fig.savefig(R.fig_path(f"{name}_fig17_traces.png"), dpi=300)
    plt.close(fig)
    print(f"{name}_fig17_traces: peak L/D/V = {np.nanmax(tr['length']):.2f} mm"
          f" / {np.nanmax(tr['depth']):.3f} mm / {np.nanmax(volume):.4f} mm^3")


def fig16_maps(case_dir, name):
    """Fig 16: solidification thermal gradient G — top surface plus a centre
    cross-section (slice at the geometric x-centre)."""
    sol, _ = _read_case(case_dir, name)
    zvals = np.unique(sol.z.values)
    zres = np.min(np.diff(zvals))
    xres = np.min(np.diff(np.unique(sol.x.values)))

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
    which = sys.argv[1]
    if which == "compare":
        compare_figure()
    elif which == "fig17":
        fig17_traces(sys.argv[2], sys.argv[3])
    elif which == "fig16":
        fig16_maps(sys.argv[2], sys.argv[3])
    elif which == "traces":
        traces_figure(sys.argv[2])
    elif which == "fusionmap":
        fusionmap_figure(sys.argv[2])
    elif which == "profiles":
        profiles_figure(sys.argv[2])
    else:
        policy_figure(which)
