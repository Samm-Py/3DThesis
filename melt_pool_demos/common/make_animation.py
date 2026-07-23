"""Animation of the 10 mm square raster at the Sec. 3.5 physics.

  python make_animation.py {zero|dwell} {unoptimized|optimized} [output-tag]

Three panels (port of melt_pool_demos/raster/make_animation.py):
  left   - top-down surface temperature over the square, scan plan, beam
           position, melt (1610 K) contour;
  right  - melt-pool cross-section (y-z slice at the deepest trailing-pool
           column) with the beam Gaussian sketched above the surface (height
           tracks power, width tracks sigma);
  bottom - instantaneous dimensions measured from the displayed section,
           with the 5 mm controller endpoint diagnostics shown as markers.

Frames at a fixed time step: the path is truncated at time t and 3DThesis
snapshots the field (two runs per frame: shallow full-square top-down +
fine local y-z slab). Writes ``figures/animations/square_{policy}_{case}.gif``;
an optional tag is appended before ``.gif`` for workshop variants.
"""
import os
import sys
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mp_lib as R

PAPER="#FAF9F7"; INK="#232629"; MUTED="#6B6E72"; STEEL="#3E6B8C"; MELT="#C2491C"; DEP="#2E8B57"
T0, TMAX = R.T0, 2100.0
DT   = 2.0e-3       # frame time step (s)
FPS  = 12
plt.rcParams.update({"figure.facecolor":PAPER,"axes.facecolor":PAPER,"axes.edgecolor":MUTED,
    "axes.labelcolor":INK,"xtick.color":MUTED,"ytick.color":MUTED,"text.color":INK,"font.size":10})


def timeline(div, pmods):
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
    out = [div[0]]
    track_y = 0.0
    track_dir = 1.0

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
                if abs(x1 - x0) > 1e-12:
                    track_dir = np.sign(x1 - x0)
            continue
        f = (t - t0) / (t1 - t0)
        if mode == 1:
            vals[5] = str(float(vals[5]) * f)
            out.append('\t'.join(vals) + '\n')
            return out, (x1, y1), False, track_y, 0.0, R.SIGMA, track_dir
        bx, by = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
        vals[1], vals[2] = f"{bx:.6f}", f"{by:.6f}"
        out.append('\t'.join(vals) + '\n')
        scan_dir = np.sign(x1 - x0) if abs(x1 - x0) > 1e-12 else track_dir
        return (out, (bx, by), True, y1, pmods.get(i, pm),
                wmods.get(i, 1.0) * R.SIGMA, scan_dir)
    return (out, (x1, y1), mode == 0, track_y,
            pm if mode == 0 else 0.0, wmods.get(i, 1.0) * R.SIGMA,
            track_dir)


def seg_end_times(rows, idxs):
    ends = {i: t1 for (t0, t1, mode, *_, i) in rows if mode == 0}
    return np.array([ends[e] for e in idxs])


def top_down(lines):
    R.write_domain_box(-0.5e-3, (R.S + 0.5) * 1e-3, -0.5e-3, (R.S + 0.5) * 1e-3,
                       -10e-6, 40e-6, 40e-6, 10e-6)
    R.run_path(lines, R.BIN_DBL)
    xs, ys, zs, T3 = R.load_field()
    return xs * 1e3, ys * 1e3, T3[:, :, -1]


def xsec(lines, bx, ty, scan_dir):
    """y-z slice at the DEEPEST column of the trailing pool (the elongated
    pool deepens well behind the beam; the at-beam slice under-represents
    it). Slab spans 0.7 mm behind to 0.1 mm ahead of the beam."""
    x0, x1 = ((bx - 0.7, bx + 0.1) if scan_dir >= 0
              else (bx - 0.1, bx + 0.7))
    R.write_domain_box(x0 * 1e-3, x1 * 1e-3,
                       (ty - 0.9) * 1e-3, (ty + 0.9) * 1e-3,
                       -0.2e-3, 50e-6, 10e-6, 4e-6)
    R.run_path(lines, R.BIN_DBL)
    xs, ys, zs, T3 = R.load_field()
    melt = T3 > R.T_LIQ                       # (nx, ny, nz), z ascending to 0
    first = np.where(melt.any(axis=2), melt.argmax(axis=2), melt.shape[2])
    ib = (int(first.min(axis=1).argmin()) if melt.any()
          else int(np.argmin(np.abs(xs - bx * 1e-3))))
    return ys * 1e3, zs * 1e6, T3[ib, :, :], ty, xs[ib] * 1e3


def section_dims(yy, zz, Tyz):
    """Half-span and depth (um) of the liquid contour shown in a y-z panel."""
    melt = Tyz > R.T_LIQ
    if not melt.any():
        return 0.0, 0.0
    iy, iz = np.nonzero(melt)
    half_span = 0.5 * (yy[iy].max() - yy[iy].min()) * 1e3
    depth = max(0.0, -zz[iz].min())
    return float(half_span), float(depth)


if __name__ == "__main__":
    policy = sys.argv[1]
    case = sys.argv[2] if len(sys.argv) > 2 else "unoptimized"
    output_tag = sys.argv[3] if len(sys.argv) > 3 else ""
    dwell = 0.0 if policy == "zero" else R.dwell_from_json()
    base = pd.read_csv(R.data_path({"unoptimized": f"baseline_{policy}.csv",
                                    "optimized": f"optimized_{policy}.csv"}[case]))
    R.write_beam()
    div, powered, hop_after = R.build_path(dwell)
    pmods = dict(zip(base['idx'], base['pmod']))
    wmods = (dict(zip(base['idx'], base['sigma'] / R.SIGMA))
             if 'sigma' in base.columns else {})
    for i, h in hop_after.items():
        pmods[h] = pmods[i]
        if i in wmods:
            wmods[h] = wmods[i]
    rows, t_total = timeline(div, pmods)
    seg_t = seg_end_times(rows, base['idx'].values)

    ts = np.arange(DT, t_total - 1e-6, DT)
    print(f"total scan time {t_total*1e3:.1f} ms -> {ts.size} frames", flush=True)

    F = []
    for k, t in enumerate(ts):
        lines, (bx, by), on, ty, pm, sg, scan_dir = truncate_at(
            div, pmods, wmods, rows, t)
        xg, yg, Ttop = top_down(lines)
        yy, zz, Tyz, ty, sx = xsec(lines, bx, ty, scan_dir)
        sec_w, sec_d = section_dims(yy, zz, Tyz)
        F.append(dict(t=t, bx=bx, by=by, on=on, ty=ty, pm=pm, sig=sg,
                      Ttop=Ttop, yy=yy, zz=zz, Tyz=Tyz, sx=sx,
                      sec_w=sec_w, sec_d=sec_d,
                      lag=-(sx - bx) * scan_dir * 1e3))
        if k % 20 == 0:
            print(f"  frame {k}/{ts.size}  t={t*1e3:6.1f} ms", flush=True)

    xg_mm, yg_mm = xg, yg

    fig = plt.figure(figsize=(11.6, 5.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25], height_ratios=[2.1, 1.0],
                          hspace=0.33, wspace=0.24)
    axT = fig.add_subplot(gs[:, 0]); axX = fig.add_subplot(gs[0, 1]); axW = fig.add_subplot(gs[1, 1])
    title = {"unoptimized": "Unoptimized square raster - constant 150 W",
             "optimized": "Optimized square raster - power + beam-width schedule"}[case]
    sub = ("no dwell (continuous)" if dwell == 0
           else f"minimal dwell {dwell*1e3:.2f} ms")
    fig.suptitle(f"{title}, {sub}", fontsize=11, y=0.985)

    imT = axT.pcolormesh(xg_mm, yg_mm, np.clip(F[0]['Ttop'].T, T0, TMAX),
                         cmap="magma", vmin=T0, vmax=TMAX, shading="auto")
    beam_dot, = axT.plot([], [], "o", color="#7fd4ff", ms=6, mec="w", mew=0.8)
    csT = {"h": None}
    axT.set_aspect("equal")
    axT.set_xlabel("x (mm)"); axT.set_ylabel("y (mm)")
    axT.set_title("top-down surface T", fontsize=10)
    cbT = fig.colorbar(imT, ax=axT, pad=0.02, shrink=0.85); cbT.set_label("T (K)")

    yy0, zz0 = F[0]['yy'], F[0]['zz']
    yrel0 = (yy0 - F[0]['ty']) * 1e3                  # um relative to track
    imX = axX.pcolormesh(yrel0, zz0, np.clip(F[0]['Tyz'].T, T0, TMAX),
                         cmap="magma", vmin=T0, vmax=TMAX, shading="auto")
    csX = {"h": None}
    amp = abs(zz0.min()) * 0.14
    gy = np.linspace(yrel0.min(), yrel0.max(), 300)
    beamline, = axX.plot([], [], color="#7fd4ff", lw=2)
    off_txt = axX.text(0.02, 0.04, "", transform=axX.transAxes, fontsize=8,
                       color="white", va="bottom",
                       bbox=dict(facecolor="black", edgecolor="none",
                                 alpha=0.32, pad=2.0))
    axX.set_ylim(zz0.min(), amp * 1.35)
    axX.axhline(0, color="w", lw=0.5, alpha=0.4)
    axX.set_xlabel("y across current track (µm)"); axX.set_ylabel("z (µm)")
    axX.set_title("deepest trailing-pool section", fontsize=10)

    tms = seg_t * 1e3
    frame_tms = np.array([f['t'] for f in F]) * 1e3
    sec_w = np.array([f['sec_w'] for f in F])
    sec_d = np.array([f['sec_d'] for f in F])
    # Pale complete curves provide context; the saturated portions grow with
    # animation time. Hollow circles are deliberately a different sampling
    # family: controller measurements at 5 mm segment ends.
    axW.plot(frame_tms, sec_w, color=MELT, lw=1.1, alpha=0.22)
    axW.plot(frame_tms, sec_d, color=DEP, lw=1.1, alpha=0.22)
    axW.scatter(tms, base['w'] * 1e6, s=8, facecolors="none", edgecolors=MELT,
                linewidths=0.55, alpha=0.38)
    axW.scatter(tms, base['d'] * 1e6, s=8, facecolors="none", edgecolors=DEP,
                linewidths=0.55, alpha=0.38)
    wl, = axW.plot([], [], color=MELT, lw=2.0, label="section half-span")
    dl, = axW.plot([], [], color=DEP, lw=2.0, label="section depth")
    if case == "optimized":
        tgt = json.load(open(R.data_path(f"target_{policy}.json")))
        axW.axhline(tgt["w_star"] * 1e6, color=MUTED, lw=0.7, ls="--")
    cur = axW.axvline(0, color=MUTED, lw=0.8)
    ymax = 1.18 * max(sec_w.max(), sec_d.max(),
                      (base['w'] * 1e6).max(), (base['d'] * 1e6).max())
    axW.set_xlim(0, t_total * 1e3); axW.set_ylim(0, ymax)
    axW.grid(alpha=0.15)
    axW.set_xlabel("time (ms)"); axW.set_ylabel("pool size (µm)")
    axW.set_title("instantaneous section (lines) vs 5 mm endpoints (circles)",
                  fontsize=9)
    axW.legend(fontsize=7, loc="upper right", ncol=2, frameon=False,
               handlelength=1.5, columnspacing=0.8)
    for t0, t1, mode, *_ in rows:
        if mode == 1 and t1 > t0:
            axW.axvspan(t0 * 1e3, t1 * 1e3, color=MUTED, alpha=0.12, lw=0)
    axP = axW.twinx()
    axP.step(tms, base['pmod'] * R.P_BASE, where="pre", color=STEEL, lw=1.1,
             alpha=0.32)
    axP.set_ylim(0, R.PMOD0 * R.P_BASE * 1.3)
    axP.set_ylabel("power (W)", color=STEEL, fontsize=9)
    axP.tick_params(axis='y', labelcolor=STEEL, labelsize=8)

    def update(k):
        f = F[k]
        imT.set_array(np.clip(f['Ttop'].T, T0, TMAX).ravel())
        if csT["h"] is not None: csT["h"].remove()
        csT["h"] = axT.contour(xg_mm, yg_mm, f['Ttop'].T, levels=[R.T_LIQ],
                               colors="white", linewidths=0.9)
        beam_dot.set_data([f['bx']], [f['by']])
        beam_dot.set_alpha(1.0 if f['on'] else 0.35)

        yrel = (f['yy'] - f['ty']) * 1e3
        imX.set_array(np.clip(f['Tyz'].T, T0, TMAX).ravel())
        if csX["h"] is not None: csX["h"].remove()
        csX["h"] = axX.contour(yrel, f['zz'], f['Tyz'].T, levels=[R.T_LIQ],
                               colors="white", linewidths=1.3)
        if f['on']:
            a = amp * f['pm'] / R.PMOD0
            beamline.set_data(gy, a * np.exp(-0.5 * (gy / (f['sig'] * 1e6)) ** 2))
            off_txt.set_text(f"P = {f['pm'] * R.P_BASE:.0f} W   "
                             f"σ = {f['sig'] * 1e6:.0f} µm   "
                             f"slice lag = {f['lag']:.0f} µm\n"
                             f"section half-span = {f['sec_w']:.0f} µm   "
                             f"depth = {f['sec_d']:.0f} µm")
        else:
            beamline.set_data([], [])
            off_txt.set_text(f"beam off (turnaround dwell)   "
                             f"slice lag = {f['lag']:.0f} µm\n"
                             f"section half-span = {f['sec_w']:.0f} µm   "
                             f"depth = {f['sec_d']:.0f} µm")
        m = frame_tms <= f['t'] * 1e3
        wl.set_data(frame_tms[m], sec_w[m])
        dl.set_data(frame_tms[m], sec_d[m])
        cur.set_xdata([f['t'] * 1e3])
        return [imT]

    ani = animation.FuncAnimation(fig, update, frames=len(F), blit=False)
    suffix = f"_{output_tag}" if output_tag else ""
    output = R.fig_path(os.path.join(
        "animations", f"{R.GEOM}_{policy}_{case}{suffix}.gif"))
    ani.save(output,
             writer=animation.PillowWriter(fps=FPS),
             savefig_kwargs={"facecolor": PAPER}, dpi=88)
    plt.close(fig)
    print(f"wrote {output}")
