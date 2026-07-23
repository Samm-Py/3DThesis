"""Animation of the serpentine raster: `python make_animation.py [unoptimized|optimized]`.

unoptimized (default): constant 60 W everywhere (reads baseline.csv).
optimized:             the power-only schedule from run_optimized.py
                       (reads optimized.csv); the sketched beam Gaussian's
                       height scales with the current power.

Three panels:
  left   - top-down surface temperature over the whole square, with the scan
           plan, the beam position and the melt (1733 K) contour;
  right  - melt-pool cross-section (y-z slice at the beam) like the single-
           track demos, with the beam Gaussian sketched above the surface
           (decaying residue is shown during the turnaround dwells);
  bottom - running width/depth-vs-time trace (the same data as baseline.csv).

Frames are taken at a fixed time step: the path is truncated at time t (the
current powered line cut at the interpolated position, or the current dwell
shortened) and 3DThesis snapshots the field at that instant. Two runs per
frame: a shallow full-square domain for the top-down view and a fine local
y-z domain at the beam for the cross-section.

Writes raster_<case>.gif.
"""
import sys
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import raster_lib as R

PAPER="#FAF9F7"; INK="#232629"; MUTED="#6B6E72"; STEEL="#3E6B8C"; MELT="#C2491C"; DEP="#2E8B57"
T0, TMAX = 1273.0, 2500.0
DT   = 0.18e-3      # frame time step (s)
FPS  = 12
plt.rcParams.update({"figure.facecolor":PAPER,"axes.facecolor":PAPER,"axes.edgecolor":MUTED,
    "axes.labelcolor":INK,"xtick.color":MUTED,"ytick.color":MUTED,"text.color":INK,"font.size":10})


def timeline(div, pmods):
    """Per path line: (t0, t1, mode, x0, y0, x1, y1, pmod). Positions in mm."""
    rows, t, prev = [], 0.0, (0.0, 0.0)
    for i in range(len(div)):
        vals = div[i].strip().split('\t')
        if vals[0] not in ('0', '1'):
            continue
        x, y = float(vals[1]), float(vals[2])
        if vals[0] == '1':
            dt = float(vals[5])
            rows.append((t, t + dt, 1, x, y, x, y, 0.0, i))
        else:
            dt = np.hypot(x - prev[0], y - prev[1]) * 1e-3 / float(vals[5])
            rows.append((t, t + dt, 0, prev[0], prev[1], x, y,
                         pmods.get(i, float(vals[4])), i))
        t += dt
        prev = (x, y)
    return rows, t


def truncate_at(div, pmods, wmods, rows, t):
    """Path lines truncated at absolute time t; returns (lines, beam_xy_mm,
    beam_on, track_y_mm, pmod_now, sigma_now_m)."""
    out = [div[0]]                                   # header
    track_y = 0.0

    def styled(vals, i):
        vals = list(vals)
        vals[4] = str(pmods.get(i, float(vals[4])))
        if i in wmods:
            while len(vals) < 7:
                vals.append('1.0')
            vals[6] = repr(wmods[i])
        return vals

    for (t0, t1, mode, x0, y0, x1, y1, pm, i) in rows:
        vals = div[i].strip().split('\t')
        if mode == 0:
            vals = styled(vals, i)
        if t1 <= t:
            out.append('\t'.join(vals) + '\n')
            if mode == 0:
                track_y = y1
            continue
        # partial line
        f = (t - t0) / (t1 - t0)
        if mode == 1:
            vals[5] = str(float(vals[5]) * f)
            out.append('\t'.join(vals) + '\n')
            return out, (x1, y1), False, track_y, 0.0, R.SIGMA
        bx, by = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
        vals[1], vals[2] = f"{bx:.6f}", f"{by:.6f}"
        out.append('\t'.join(vals) + '\n')
        return (out, (bx, by), True, y1, pmods.get(i, pm),
                wmods.get(i, 1.0) * R.SIGMA)
    return (out, (x1, y1), mode == 0, track_y,
            pm if mode == 0 else 0.0, wmods.get(i, 1.0) * R.SIGMA)


def seg_end_times(rows, info):
    """Absolute time at which each powered segment (baseline.csv rows) ends."""
    ends = {i: t1 for (t0, t1, mode, *_, i) in rows if mode == 0}
    return np.array([ends[e] for e in info])


def top_down(lines):
    R.write_domain(-0.2e-3, (R.S + 0.2) * 1e-3, -0.2e-3, (R.S - R.H + 0.25) * 1e-3,
                   -10e-6, 8e-6, 8e-6, 10e-6)
    R.run_path(lines, R.BIN_DBL)
    xs, ys, zs, T3 = R.load_field()
    return xs * 1e3, ys * 1e3, T3[:, :, -1]          # surface slice, mm axes


def xsec(lines, bx, ty):
    R.write_domain((bx - 0.03) * 1e-3, (bx + 0.03) * 1e-3,
                   (ty - 0.30) * 1e-3, (ty + 0.30) * 1e-3,
                   -0.17e-3, 10e-6, 3e-6, 3e-6)
    R.run_path(lines, R.BIN_DBL)
    xs, ys, zs, T3 = R.load_field()
    ib = int(np.argmin(np.abs(xs - bx * 1e-3)))
    return ys * 1e6, zs * 1e6, T3[ib, :, :], ty      # um axes (y absolute)


if __name__ == "__main__":
    case = sys.argv[1] if len(sys.argv) > 1 else "unoptimized"
    dwell = json.load(open(R.data_path("min_dwell.json")))["dwell_s"]
    base = pd.read_csv(R.data_path({"unoptimized": "baseline.csv",
                                    "optimized": "optimized.csv"}[case]))
    R.write_beam()
    seg_mm = float(np.round(np.median(np.diff(base['s_mm'])), 4))  # ctrl grid
    div, powered = R.build_path(dwell, seg=seg_mm)
    pmods = dict(zip(base['idx'], base['pmod']))
    wmods = (dict(zip(base['idx'], base['sigma'] / R.SIGMA))
             if 'sigma' in base.columns else {})
    rows, t_total = timeline(div, pmods)
    seg_t = seg_end_times(rows, base['idx'].values)

    ts = np.arange(DT, t_total - 1e-6, DT)
    print(f"total scan time {t_total*1e3:.2f} ms -> {ts.size} frames")

    # ---- simulate all frames ----
    F = []
    for k, t in enumerate(ts):
        lines, (bx, by), on, ty, pm, sg = truncate_at(div, pmods, wmods, rows, t)
        xg, yg, Ttop = top_down(lines)
        yy, zz, Tyz, ty = xsec(lines, bx, ty)
        F.append(dict(t=t, bx=bx, by=by, on=on, ty=ty, pm=pm, sig=sg,
                      Ttop=Ttop, yy=yy, zz=zz, Tyz=Tyz))
        if k % 10 == 0:
            print(f"  frame {k}/{ts.size}  t={t*1e3:5.2f} ms  beam {'on' if on else 'OFF'}")
    xg_mm, yg_mm = xg, yg

    # ---- figure ----
    fig = plt.figure(figsize=(11.2, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.25], height_ratios=[2.1, 1.0],
                          hspace=0.33, wspace=0.24)
    axT = fig.add_subplot(gs[:, 0]); axX = fig.add_subplot(gs[0, 1]); axW = fig.add_subplot(gs[1, 1])
    title = {"unoptimized": "Unoptimised serpentine raster - constant 60 W",
             "optimized": "Optimised serpentine raster - power + beam-width schedule"}[case]
    fig.suptitle(f"{title}, minimal dwell {dwell*1e3:.2f} ms",
                 fontsize=11, y=0.985)

    # top-down
    imT = axT.pcolormesh(xg_mm, yg_mm, np.clip(F[0]['Ttop'].T, T0, TMAX),
                         cmap="magma", vmin=T0, vmax=TMAX, shading="auto")
    for j in range(int(round(R.S / R.H)) + 1):       # scan plan
        yl = j * R.H
        if yl <= R.S - R.H + 1e-9:
            axT.plot([0, R.S], [yl, yl], color="w", lw=0.5, alpha=0.25)
    beam_dot, = axT.plot([], [], "o", color="#7fd4ff", ms=7, mec="w", mew=0.8)
    csT = {"h": None}
    axT.set_aspect("equal")
    axT.set_xlabel("x (mm)"); axT.set_ylabel("y (mm)")
    axT.set_title("top-down surface T", fontsize=10)
    cbT = fig.colorbar(imT, ax=axT, pad=0.02, shrink=0.85); cbT.set_label("T (K)")

    # cross-section
    yy0, zz0 = F[0]['yy'], F[0]['zz']
    yrel0 = yy0 - F[0]['ty'] * 1e3
    imX = axX.pcolormesh(yrel0, zz0, np.clip(F[0]['Tyz'].T, T0, TMAX),
                         cmap="magma", vmin=T0, vmax=TMAX, shading="auto")
    csX = {"h": None}
    amp = abs(zz0.min()) * 0.14
    gy = np.linspace(yrel0.min(), yrel0.max(), 200)
    beamline, = axX.plot([], [], color="#7fd4ff", lw=2)
    off_txt = axX.text(0.02, 0.93, "", transform=axX.transAxes, fontsize=9,
                       color="#7fd4ff")
    axX.set_ylim(zz0.min(), amp * 1.35)
    axX.axhline(0, color="w", lw=0.5, alpha=0.4)
    axX.set_xlabel("y across current track (µm)"); axX.set_ylabel("z (µm)")
    axX.set_title("cross-section at the beam", fontsize=10)

    # width/depth trace (+ power schedule); y-limit from the baseline so the
    # unoptimised and optimised gifs share a scale
    tms = seg_t * 1e3
    ymax = pd.read_csv(R.data_path("baseline.csv"))['w'].max() * 1e6 * 1.25
    axW.plot(tms, base['w'] * 1e6, color=MELT, lw=1.2, alpha=0.3)
    axW.plot(tms, base['d'] * 1e6, color=DEP, lw=1.2, alpha=0.3)
    wl, = axW.plot([], [], color=MELT, lw=2.2, label="width (half-span)")
    dl, = axW.plot([], [], color=DEP, lw=2.2, label="depth")
    if case == "optimized":
        tgt = json.load(open(R.data_path("target.json")))
        axW.axhline(tgt["w_star"] * 1e6, color=MUTED, lw=0.7, ls="--")
    cur = axW.axvline(0, color=MUTED, lw=0.8)
    axW.set_xlim(0, t_total * 1e3); axW.set_ylim(0, ymax)
    axW.grid(alpha=0.15)
    axW.set_xlabel("time (ms)"); axW.set_ylabel("pool size (µm)")
    axW.legend(fontsize=8, loc="lower right", ncol=2, frameon=False)
    axP = axW.twinx()
    axP.step(tms, base['pmod'] * R.P_BASE, where="pre", color=STEEL, lw=1.1,
             alpha=0.55)
    axP.set_ylim(0, R.PMOD0 * R.P_BASE * 1.3)
    axP.set_ylabel("power (W)", color=STEEL, fontsize=9)
    axP.tick_params(axis='y', labelcolor=STEEL, labelsize=8)

    def update(k):
        f = F[k]
        imT.set_array(np.clip(f['Ttop'].T, T0, TMAX).ravel())
        if csT["h"] is not None: csT["h"].remove()
        csT["h"] = axT.contour(xg_mm, yg_mm, f['Ttop'].T, levels=[R.T_LIQ],
                               colors="white", linewidths=1.0)
        beam_dot.set_data([f['bx']], [f['by']])
        beam_dot.set_alpha(1.0 if f['on'] else 0.35)

        yrel = f['yy'] - f['ty'] * 1e3
        imX.set_array(np.clip(f['Tyz'].T, T0, TMAX).ravel())
        if csX["h"] is not None: csX["h"].remove()
        csX["h"] = axX.contour(yrel, f['zz'], f['Tyz'].T, levels=[R.T_LIQ],
                               colors="white", linewidths=1.4)
        if f['on']:
            a = amp * f['pm'] / R.PMOD0          # Gaussian height tracks power
            beamline.set_data(gy, a * np.exp(-0.5 * (gy / (f['sig'] * 1e6)) ** 2))
            off_txt.set_text(f"P = {f['pm'] * R.P_BASE:.0f} W   "
                             f"σ = {f['sig'] * 1e6:.0f} µm")
        else:
            beamline.set_data([], [])
            off_txt.set_text("beam off (turnaround dwell)")
        m = tms <= f['t'] * 1e3
        wl.set_data(tms[m], base['w'].values[m] * 1e6)
        dl.set_data(tms[m], base['d'].values[m] * 1e6)
        cur.set_xdata([f['t'] * 1e3])
        return [imT]

    ani = animation.FuncAnimation(fig, update, frames=len(F), blit=False)
    out = R.fig_path(f"raster_{case}.gif")
    ani.save(out, writer=animation.PillowWriter(fps=FPS),
             savefig_kwargs={"facecolor": PAPER}, dpi=88)
    plt.close(fig)
    print(f"wrote {out}")
