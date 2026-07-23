"""Comparison plots for the raster power + beam-width uniformity demo.

  raster_dims.png    - width and depth vs distance along the scan, unoptimised
                       vs optimised (separate panels), plus the optimised
                       power schedule that produced them.
  raster_overlay.png - all four dimension traces on one axes.

Reads baseline.csv / optimized.csv (run run_baseline.py / run_optimized.py
first)."""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import raster_lib as R

PAPER="#FAF9F7"; INK="#232629"; MUTED="#6B6E72"; STEEL="#3E6B8C"; MELT="#C2491C"; DEP="#2E8B57"
plt.rcParams.update({"figure.facecolor":PAPER,"axes.facecolor":PAPER,"axes.edgecolor":MUTED,
    "axes.labelcolor":INK,"xtick.color":MUTED,"ytick.color":MUTED,"text.color":INK,"font.size":10.5})

base = pd.read_csv(R.data_path("baseline.csv"))
opt  = pd.read_csv(R.data_path("optimized.csv"))
dwell = json.load(open(R.data_path("min_dwell.json")))["dwell_s"]
sb, so = base['s_mm'].values, opt['s_mm'].values   # measurement grids differ
tgt = json.load(open(R.data_path("target.json")))
w_star, d_star = tgt["w_star"], tgt["d_star"]
n_lines = int(round(R.S / R.H)) - 0   # lines actually scanned
n_lines = base['line'].max() + 1


def mark_lines(ax):
    for j in range(1, n_lines):
        ax.axvline(j * R.S, color=MUTED, lw=0.6, alpha=0.25)


def cv(v):
    return 100.0 * np.std(v) / np.mean(v)


# ---------- figure 1: width / depth panels + control schedules ----------
fig, (axw, axd, axp) = plt.subplots(3, 1, figsize=(10.6, 7.6), sharex=True,
                                    gridspec_kw=dict(height_ratios=[1, 1, 0.75],
                                                     hspace=0.14))
fig.subplots_adjust(right=0.72)
fig.suptitle("Serpentine raster - melt-pool uniformity, power + beam-width control\n"
             f"(60 W / σ {R.SIGMA*1e6:.0f} µm baseline, minimal turnaround "
             f"dwell {dwell*1e3:.2f} ms)", fontsize=11)

axw.plot(sb, base['w']*1e6, "o-", ms=3.5, lw=1.4, color=MELT, alpha=0.45,
         label=f"unoptimised (coefficient of variation {cv(base['w']):.1f}%)")
axw.plot(so, opt['w']*1e6, "o-", ms=3.5, lw=1.8, color=MELT,
         label=f"optimised (coefficient of variation {cv(opt['w']):.1f}%)")
axw.axhline(w_star*1e6, color=MUTED, lw=0.8, ls="--")
axw.text(1.01, w_star*1e6, f"target {w_star*1e6:.1f} µm",
         transform=axw.get_yaxis_transform(), va="center", ha="left",
         fontsize=8.5, color=MUTED)
axw.set_ylabel("width, half-span (µm)")
axw.legend(fontsize=9, frameon=False, loc="upper left",
           bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
mark_lines(axw)

axd.plot(sb, base['d']*1e6, "o-", ms=3.5, lw=1.4, color=DEP, alpha=0.45,
         label=f"unoptimised (coefficient of variation {cv(base['d']):.1f}%)")
axd.plot(so, opt['d']*1e6, "o-", ms=3.5, lw=1.8, color=DEP,
         label=f"optimised (coefficient of variation {cv(opt['d']):.1f}%)")
axd.axhline(d_star*1e6, color=MUTED, lw=0.8, ls="--")
axd.text(1.01, d_star*1e6, f"target {d_star*1e6:.1f} µm",
         transform=axd.get_yaxis_transform(), va="center", ha="left",
         fontsize=8.5, color=MUTED)
axd.set_ylabel("depth (µm)")
axd.legend(fontsize=9, frameon=False, loc="upper left",
           bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
mark_lines(axd)

power_opt, = axp.step(so, opt['pmod']*R.P_BASE, where="pre", color=STEEL, lw=1.8,
                      label="power (optimised)")
power_base = axp.axhline(R.PMOD0*R.P_BASE, color=STEEL, lw=1.2, alpha=0.4, ls="--",
                         label="power (unoptimised)")
axp.set_ylabel("power (W)", color=STEEL)
axp.tick_params(axis='y', labelcolor=STEEL)
axp.set_xlabel("distance along scan (mm)   [vertical lines = turnarounds]")
axp.set_ylim(0, R.PMOD0*R.P_BASE*1.2)
mark_lines(axp)
SIGC = "#7B4B94"
axs = axp.twinx()
sigma_opt, = axs.step(so, opt['sigma']*1e6, where="pre", color=SIGC, lw=1.6,
                      alpha=0.8, label="σ (optimised)")
sigma_base = axs.axhline(R.SIGMA*1e6, color=SIGC, lw=1.0, alpha=0.35, ls="--",
                         label="σ (unoptimised)")
axs.set_ylim(0, opt['sigma'].max()*1e6*1.6)
axs.set_ylabel("beam σ (µm)", color=SIGC, fontsize=9.5)
axs.tick_params(axis='y', labelcolor=SIGC)
axp.legend([power_opt, power_base, sigma_opt, sigma_base],
           ["power (optimised)", "power (unoptimised)",
            "σ (optimised)", "σ (unoptimised)"],
           fontsize=9, frameon=False, loc="upper left",
           bbox_to_anchor=(1.08, 1.0), borderaxespad=0.0)

for ax in (axw, axd, axp):
    ax.grid(alpha=0.12)
fig.savefig(R.fig_path("raster_dims.png"), dpi=140, bbox_inches="tight", facecolor=PAPER)
plt.close(fig)

# ---------- figure 2: everything overlaid ----------
fig, ax = plt.subplots(figsize=(9.8, 4.4))
fig.subplots_adjust(right=0.74)
ax.plot(sb, base['w']*1e6, "o--", ms=3, lw=1.2, color=MELT, alpha=0.45,
        label="width - unoptimised")
ax.plot(sb, base['d']*1e6, "o--", ms=3, lw=1.2, color=DEP, alpha=0.45,
        label="depth - unoptimised")
ax.plot(so, opt['w']*1e6, "o-", ms=3, lw=1.8, color=MELT, label="width - optimised")
ax.plot(so, opt['d']*1e6, "o-", ms=3, lw=1.8, color=DEP, label="depth - optimised")
ax.axhline(w_star*1e6, color=MUTED, lw=0.8, ls="--")
ax.text(sb[-1], w_star*1e6, " targets", va="bottom", fontsize=8.5, color=MUTED)
ax.axhline(d_star*1e6, color=MUTED, lw=0.8, ls="--")
mark_lines(ax)
ax.set_xlabel("distance along scan (mm)   [vertical lines = turnarounds]")
ax.set_ylabel("pool size (µm)")
ax.set_title("Melt-pool dimensions, unoptimised vs optimised (power + beam width)",
             fontsize=11)
ax.grid(alpha=0.12)
ax.legend(fontsize=9, frameon=False, loc="center left",
          bbox_to_anchor=(1.01, 0.5), borderaxespad=0.0)
fig.savefig(R.fig_path("raster_overlay.png"), dpi=140, bbox_inches="tight", facecolor=PAPER)
plt.close(fig)
print("wrote figures/raster_dims.png, figures/raster_overlay.png")
