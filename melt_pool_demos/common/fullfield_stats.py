"""Chunked full-field statistics from a tracked case's RDF event list.

The RDF stores the interval during which each melted grid cell is liquid.
This script samples those intervals at 50 us and reports:

* global instantaneous pool length, full width, depth, and volume;
* beam-head-local full width/depth in the controller's 1 mm rear window;
* the maximum attained fusion depth at each surface column; and
* residual liquid at scan-line starts.

RDF coordinates are cell centres.  Reported in-plane extents therefore add
the projected width of one grid cell to ``max(center)-min(center)``.  This
removes the one-cell low bias that is material on coarser y grids.

The event file is read in chunks so refined 5 um y replays of full raster
patterns do not require the complete RDF or expanded time history in RAM.

Usage: python fullfield_stats.py <case_dir> <name>
Writes compact traces and statistics to ``results/fullfield/``.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mp_lib as R


DT = 5e-5
# A row can span many 50 us samples, and the interval expansion below creates
# one temporary entry per active sample. Large input chunks can therefore be
# far larger in memory than their RDF row count suggests. Keep the default
# deliberately conservative for full square/triangle rasters; callers with
# more RAM can still raise it through MP_RDF_CHUNK_ROWS.
CHUNK_ROWS = int(os.environ.get("MP_RDF_CHUNK_ROWS", "20000"))
LOCAL_BEHIND_MM = 1.0
LOCAL_AHEAD_MM = 0.6
LOCAL_HALF_WIDTH_MM = 1.2

case = os.path.abspath(sys.argv[1])
name = sys.argv[2]
rdf_path = os.path.join(case, "Data", f"{name}.RDF.Final.csv")
path_file = os.path.join(case, "Path.txt")

with open(os.path.join(case, "Domain.txt")) as f:
    dom = f.read().split()


def axis_spec(axis: str) -> tuple[float, float, float]:
    section = dom[dom.index(axis):]
    return (
        float(section[section.index("Min") + 1]),
        float(section[section.index("Max") + 1]),
        float(section[section.index("Res") + 1]),
    )


(xmin_m, xmax_m, xres), (ymin_m, ymax_m, yres), (_, _, zres) = (
    axis_spec(axis) for axis in ("X", "Y", "Z"))
xres_mm, yres_mm, zres_mm = 1e3 * np.array([xres, yres, zres])
cellvol = xres_mm * yres_mm * zres_mm
nx = int(round((xmax_m - xmin_m) / xres)) + 1
ny = int(round((ymax_m - ymin_m) / yres)) + 1


def actual_line_starts(path: str) -> list[float]:
    """Start time of every new horizontal scan line in the actual path."""
    t, previous, line_y, starts = 0.0, None, None, []
    with open(path) as f:
        rows = f.read().splitlines()
    for row in rows:
        values = row.strip().split("\t")
        if values[0] not in ("0", "1"):
            continue
        x, y = float(values[1]), float(values[2])
        if values[0] == "1":
            t += float(values[5])
            previous = (x, y)
            continue
        if previous is not None:
            horizontal = (
                abs(y - previous[1]) <= 1e-9
                and abs(x - previous[0]) > 1e-9)
            if horizontal and (line_y is None or abs(y - line_y) > 1e-9):
                starts.append(t)
                line_y = y
            t += np.hypot(x - previous[0], y - previous[1]) * 1e-3 / float(
                values[5])
        previous = (x, y)
    return starts


def beam_head_history(
        path: str, sample_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Beam position and unit travel direction at every powered sample."""
    hx = np.full(sample_times.shape, np.nan)
    hy = np.full(sample_times.shape, np.nan)
    dx = np.zeros(sample_times.shape)
    dy = np.zeros(sample_times.shape)
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
        delta = point - previous
        distance_mm = float(np.linalg.norm(delta))
        duration = distance_mm * 1e-3 / float(values[5])
        if distance_mm > 0.0 and duration > 0.0:
            mask = (sample_times >= t) & (sample_times < t + duration)
            fraction = (sample_times[mask] - t) / duration
            hx[mask] = previous[0] + fraction * delta[0]
            hy[mask] = previous[1] + fraction * delta[1]
            direction = delta / distance_mm
            dx[mask], dy[mask] = direction
        t += duration
        previous = point
    return hx, hy, dx, dy


# A first lightweight pass determines the sampled time axis.
tl_max = 0.0
for chunk in pd.read_csv(rdf_path, usecols=["tl"], chunksize=CHUNK_ROWS):
    if len(chunk):
        tl_max = max(tl_max, float(chunk["tl"].max()))
nbin = int(np.ceil(tl_max / DT)) + 1
# Tracking output is synchronized to solver timestep boundaries.  Sampling
# those boundaries (rather than an artificial half-step shift) reproduces the
# melt-pool state represented by the event intervals.
sample_times = np.arange(nbin) * DT

head_x, head_y, head_dx, head_dy = beam_head_history(
    path_file, sample_times)

dv = np.zeros(nbin + 1, dtype=np.int64)
xmax = np.full(nbin, -np.inf)
xmin = np.full(nbin, np.inf)
ymax = np.full(nbin, -np.inf)
ymin = np.full(nbin, np.inf)
zmin = np.full(nbin, np.inf)
local_tmax = np.full(nbin, -np.inf)
local_tmin = np.full(nbin, np.inf)
local_zmin = np.full(nbin, np.inf)
fusion_zmin = np.full(nx * ny, np.inf)

line_starts = np.asarray(actual_line_starts(path_file)[1:])
residual_diff = np.zeros(len(line_starts) + 1, dtype=np.int64)


def active_bins(tm: np.ndarray, tl: np.ndarray) -> tuple[np.ndarray, ...]:
    """Inclusive solver-timestep interval satisfying tm <= t <= tl."""
    tolerance = 1e-9
    b0 = np.ceil(tm / DT - tolerance).astype(np.int64)
    b1 = np.floor(tl / DT + tolerance).astype(np.int64)
    b0 = np.clip(b0, 0, nbin - 1)
    b1 = np.clip(b1, 0, nbin - 1)
    keep = b1 >= b0
    return b0[keep], b1[keep], keep


for chunk in pd.read_csv(
        rdf_path, usecols=["x", "y", "z", "tm", "tl"],
        chunksize=CHUNK_ROWS):
    x_m = chunk["x"].to_numpy()
    y_m = chunk["y"].to_numpy()
    z_m = chunk["z"].to_numpy()
    tm = chunk["tm"].to_numpy()
    tl = chunk["tl"].to_numpy()

    # Maximum attained fusion depth, accumulated directly on the domain grid.
    ix = np.rint((x_m - xmin_m) / xres).astype(np.int64)
    iy = np.rint((y_m - ymin_m) / yres).astype(np.int64)
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    np.minimum.at(
        fusion_zmin, iy[inside] * nx + ix[inside], z_m[inside])

    # Residual liquid at all line starts using an interval difference array.
    if len(line_starts):
        first = np.searchsorted(line_starts, tm, side="right")
        after_last = np.searchsorted(line_starts, tl, side="left")
        crosses = after_last > first
        np.add.at(residual_diff, first[crosses], 1)
        np.add.at(residual_diff, after_last[crosses], -1)

    b0, b1, keep = active_bins(tm, tl)
    if not len(b0):
        continue
    x_mm = x_m[keep] * 1e3
    y_mm = y_m[keep] * 1e3
    z_mm = z_m[keep] * 1e3

    np.add.at(dv, b0, 1)
    np.add.at(dv, b1 + 1, -1)

    counts = b1 - b0 + 1
    starts = np.cumsum(counts) - counts
    idx = (
        np.repeat(b0, counts)
        + np.arange(int(counts.sum()))
        - np.repeat(starts, counts))
    event_x = np.repeat(x_mm, counts)
    event_y = np.repeat(y_mm, counts)
    event_z = np.repeat(z_mm, counts)

    np.maximum.at(xmax, idx, event_x)
    np.minimum.at(xmin, idx, event_x)
    np.maximum.at(ymax, idx, event_y)
    np.minimum.at(ymin, idx, event_y)
    np.minimum.at(zmin, idx, event_z)

    valid_head = np.isfinite(head_x[idx])
    rel_x = event_x - head_x[idx]
    rel_y = event_y - head_y[idx]
    longitudinal = head_dx[idx] * rel_x + head_dy[idx] * rel_y
    transverse = -head_dy[idx] * rel_x + head_dx[idx] * rel_y
    local = (
        valid_head
        & (longitudinal >= -LOCAL_BEHIND_MM)
        & (longitudinal <= LOCAL_AHEAD_MM)
        & (np.abs(transverse) <= LOCAL_HALF_WIDTH_MM))
    np.maximum.at(local_tmax, idx[local], transverse[local])
    np.minimum.at(local_tmin, idx[local], transverse[local])
    np.minimum.at(local_zmin, idx[local], event_z[local])

vol = np.cumsum(dv[:-1]) * cellvol
active = np.isfinite(xmax)
length = np.where(active, xmax - xmin + xres_mm, 0.0)
width = np.where(active, ymax - ymin + yres_mm, 0.0)
depth_um = np.where(active, -zmin, 0.0) * 1e3

local_active = np.isfinite(local_tmax)
cross_cell = np.abs(head_dy) * xres_mm + np.abs(head_dx) * yres_mm
local_width = np.where(
    local_active, local_tmax - local_tmin + cross_cell, 0.0)
local_depth = np.where(local_active, -local_zmin, 0.0)

# Beam-on mask from the actual path.
intervals, t_path = R.beam_on_intervals(path_file)
starts = np.asarray([a for a, _ in intervals])
ends = np.asarray([b for _, b in intervals])
k = np.searchsorted(starts, sample_times, side="right") - 1
on = (
    (k >= 0)
    & (sample_times <= ends[np.clip(k, 0, len(ends) - 1)]))


def describe(series: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    values = series[mask]
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


beam_on = {
    "depth_um": describe(depth_um, on),
    "length_mm": describe(length, on),
    "width_mm": describe(width, on),
    "volume_mm3": describe(vol, on),
    "n_samples": int(on.sum()),
    "beam_on_s": float(on.sum() * DT),
    "total_path_s": float(t_path),
}
# The controller's width target is defined on horizontal scan blocks.
# Turnaround motion has a different transverse axis and is reported through
# the continuous global trace, not mixed into the controlled-width summary.
horizontal = np.isfinite(head_x) & (np.abs(head_dy) <= 1e-12)
local_on = on & local_active & horizontal
beam_on_local = {
    "depth_um": describe(local_depth * 1e3, local_on),
    "width_mm": describe(local_width, local_on),
    "n_samples": int(local_on.sum()),
}

residual = np.cumsum(residual_diff[:-1]) * cellvol
residual_info = {
    "n_line_starts": int(len(line_starts)),
    "n_nonzero": int(np.count_nonzero(residual > 0.0)),
    "max_mm3": float(residual.max()) if len(residual) else 0.0,
    "mean_mm3": float(residual.mean()) if len(residual) else 0.0,
    "worst_line": (
        int(np.argmax(residual)) + 1 if len(residual) else None),
}

valid_columns = np.isfinite(fusion_zmin)
keys = np.flatnonzero(valid_columns)
iy = keys // nx
ix = keys % nx
map_x = xmin_m + ix * xres
map_y = ymin_m + iy * yres
map_depth = -fusion_zmin[keys]
depth_map = pd.DataFrame({"x": map_x, "y": map_y, "depth": map_depth})
interior = (
    (map_x > map_x.min() + 0.1e-3)
    & (map_x < map_x.max() - 0.1e-3)
    & (map_y > map_y.min() + 0.1e-3)
    & (map_y < map_y.max() - 0.1e-3))
d_int = map_depth[interior] * 1e6
spatial = {
    "n_columns": int(interior.sum()),
    "mean_um": float(d_int.mean()),
    "std_um": float(d_int.std()),
    "min_um": float(d_int.min()),
    "max_um": float(d_int.max()),
    "p05_um": float(np.percentile(d_int, 5)),
    "p95_um": float(np.percentile(d_int, 95)),
}

depth_map.to_csv(
    R.fullfield_path(f"{name}_fusionmap.csv", write=True), index=False)
pd.DataFrame({
    "t": sample_times,
    "volume": vol,
    "length": length,
    "width": width,
    "depth": depth_um / 1e3,
    "local_width": local_width,
    "local_depth": local_depth,
}).to_csv(R.fullfield_path(f"{name}_traces.csv", write=True), index=False)

out = {
    "beam_on": beam_on,
    "beam_on_local": beam_on_local,
    "spatial_fusion_depth": spatial,
    "residual_liquid_at_line_starts": residual_info,
    "dt": DT,
    "sample_time_convention": "solver timestep boundaries",
    "extent_convention": "melted-cell outer extent",
    "xres": xres,
    "yres": yres,
    "zres": zres,
}
with open(R.fullfield_path(f"{name}_fullfield.json", write=True), "w") as f:
    json.dump(out, f, indent=2)

print(
    f"{name}: beam-on D {beam_on['depth_um']['mean']:.1f}±"
    f"{beam_on['depth_um']['std']:.1f} um | L "
    f"{beam_on['length_mm']['mean']:.2f}±"
    f"{beam_on['length_mm']['std']:.2f} mm | V "
    f"{beam_on['volume_mm3']['mean']:.4f}±"
    f"{beam_on['volume_mm3']['std']:.4f} mm3 || fusion map "
    f"{spatial['mean_um']:.1f}±{spatial['std_um']:.1f} um "
    f"(p5-p95 {spatial['p05_um']:.0f}-{spatial['p95_um']:.0f})")
