#!/usr/bin/env python3
"""Animate OTI temperature + sensitivity fields across scan time.

Reads a sequence of snapshot CSVs (columns x,y,z,T,dT_dx,...,dT_dcps), each a
moment in the beam scan, and renders an animation of every field as the hot spot
travels across the domain. Color scales are fixed across frames (computed once
over the whole sequence) so the animation shows real change, not autoscaling
flicker. The near-singular beam peak in T is clipped to a robust percentile so
the surrounding field stays visible.

Outputs an MP4 (if ffmpeg is present) and/or an animated GIF.

Usage:
  python3 animate.py 'PATH/Data/snapshot.Snapshot.*.csv' [--out oti_evolution]
                     [--fps 4] [--fields T,dT_dQ,...] [--frames-dir DIR]
"""

import argparse
import csv
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manim

FIELDS = ["T", "dT_dx", "dT_dy", "dT_dQ", "dT_dkon", "dT_drho", "dT_dcps"]
LABELS = {
    "T": "T [K]",
    "dT_dx": "dT/dx [K/m]", "dT_dy": "dT/dy [K/m]", "dT_dz": "dT/dz [K/m]",
    "dT_dQ": "dT/dQ [K/W]", "dT_dkon": "dT/dk [K/(W/m/K)]",
    "dT_drho": "dT/drho [K/(kg/m^3)]", "dT_dcps": "dT/dcp [K/(J/kg/K)]",
}


def load_grid(path):
    rows = list(csv.DictReader(open(path)))
    cols = list(rows[0].keys())
    xs = np.array(sorted(set(float(r["x"]) for r in rows)))
    ys = np.array(sorted(set(float(r["y"]) for r in rows)))
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    grids = {f: np.full((len(ys), len(xs)), np.nan) for f in cols}
    for r in rows:
        ix, iy = xi[float(r["x"])], yi[float(r["y"])]
        for f in cols:
            grids[f][iy, ix] = float(r[f])
    return xs, ys, grids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", help="glob for snapshot CSVs (quote it)")
    ap.add_argument("--out", default="oti_evolution",
                    help="output basename (writes .mp4 and/or .gif)")
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--fields", default=",".join(FIELDS),
                    help="comma-separated fields to show")
    ap.add_argument("--frames-dir", default=None,
                    help="also write each frame as a still PNG into this dir")
    ap.add_argument("--clip-pct", type=float, default=99.5,
                    help="percentile for robust color limits")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise SystemExit(f"no files match {args.pattern!r}")

    # Load every frame up front.
    frames = [load_grid(p) for p in paths]
    xs, ys, _ = frames[0]
    extent = [xs[0] * 1e3, xs[-1] * 1e3, ys[0] * 1e3, ys[-1] * 1e3]  # mm
    fields = [f for f in args.fields.split(",") if f in frames[0][2]]

    # Fixed color limits per field, computed over the whole sequence so the
    # scale doesn't jump frame to frame.
    limits = {}
    for f in fields:
        stack = np.concatenate([fr[2][f][~np.isnan(fr[2][f])].ravel() for fr in frames])
        if f == "T":
            limits[f] = (float(np.nanmin(stack)),
                         float(np.nanpercentile(stack, args.clip_pct)))
        else:
            lim = float(np.nanpercentile(np.abs(stack), args.clip_pct))
            limits[f] = (-lim, lim)

    ncol = 4
    nrow = (len(fields) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    ims = []
    for k, f in enumerate(fields):
        ax = axes.flat[k]
        ax.axis("on")
        vmin, vmax = limits[f]
        cmap = "inferno" if f == "T" else "RdBu_r"
        im = ax.imshow(frames[0][2][f], origin="lower", extent=extent,
                       aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(LABELS.get(f, f), fontsize=10)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ims.append((im, f))

    nframes = len(frames)
    suptitle = fig.suptitle("", fontsize=13)

    def frac(i):
        return 100.0 * (i + 1) / nframes

    def draw(i):
        for im, f in ims:
            im.set_data(frames[i][2][f])
        suptitle.set_text(
            f"OTI temperature & sensitivities — scan {frac(i):.0f}%  "
            f"(frame {i + 1}/{nframes})")
        return [im for im, _ in ims] + [suptitle]

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    anim = manim.FuncAnimation(fig, draw, frames=nframes, blit=False)

    wrote = []
    if "ffmpeg" in manim.writers.list():
        mp4 = args.out + ".mp4"
        anim.save(mp4, writer=manim.FFMpegWriter(fps=args.fps, bitrate=4000),
                  dpi=120)
        wrote.append(mp4)
    gif = args.out + ".gif"
    anim.save(gif, writer=manim.PillowWriter(fps=args.fps), dpi=90)
    wrote.append(gif)

    if args.frames_dir:
        os.makedirs(args.frames_dir, exist_ok=True)
        for i in range(nframes):
            draw(i)
            fig.savefig(os.path.join(args.frames_dir, f"frame_{i:03d}.png"),
                        dpi=120)

    print(f"wrote {', '.join(wrote)}  "
          f"({nframes} frames, {len(fields)} fields, grid {len(xs)}x{len(ys)})")


if __name__ == "__main__":
    main()
