"""Shared machinery for the melt-pool uniformity optimization demos
(square/ and triangle/) at the Stump & Plotkowski Sec. 3.5 calibrated
physics. The drivers in this directory are run FROM a demo directory
(``cd square && python ../common/run_optimized.py zero``); the geometry,
snapshot case, and measurement box follow the working directory and can be
overridden with MP_GEOM / MP_CASE_DIR / MP_BOX_BEHIND / MP_BOX_HALF_Y.

Mirrors melt_pool_demos/raster/raster_lib.py but self-contained: its own
snapshot case directory (cases/snapcase/), IN718 material written
explicitly, isovalue 1610 K, and a merged-pool-sized measurement box. The
path builder reproduces make_case.py's rows exactly so the optimizer
controls the same process the baselines ran.

Zero-dwell paths keep the beam ON through the 0.1 mm serpentine hops; the
hops are excluded from the control set by geometry (pure-y moves) and are
assigned the controls of the segment that precedes them.
"""
import os, sys, json, subprocess, glob, re, shutil
import numpy as np, pandas as pd
from skimage import measure

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))          # 3DThesis repo root
_DEMOS = os.path.dirname(_HERE)
sys.path.insert(0, _DEMOS)

from common import scan as Scan                           # noqa: E402
from common import measurement as Sim                    # noqa: E402

BIN_DBL = os.environ.get("THESIS_BIN",     os.path.join(_ROOT, "build",     "bin", "3DThesis"))
BIN_OTI = os.environ.get("THESIS_BIN_OTI", os.path.join(_ROOT, "build-oti", "bin", "3DThesis"))

# Geometry is auto-detected from the demo directory the drivers run in, and
# the snapshot case lives under that directory; both are env-overridable.
GEOM    = os.environ.get("MP_GEOM") or {
    "triangle": "triangle", "single_track": "single_track",
    "two_tracks": "two_tracks",
}.get(os.path.basename(os.getcwd()), "square")
CASE    = os.environ.get("MP_CASE_DIR",
                         os.path.abspath(os.path.join("cases", "snapcase")))
CSV     = os.path.join(CASE, "Data", "SqSnap.Snapshot.00.csv")

RANKED_SNAPSHOT = re.compile(r"^(?P<base>.+\.Snapshot\.(?P<idx>\d+))\.(?P<rank>\d+)\.csv$")

# ---- process parameters (identical to make_case.py / the triangle demo) ----
V      = 3.0          # raster speed (m/s)
H      = 0.1          # hatch spacing (mm)
S      = float(os.environ.get("MP_SIDE_MM", "10.0"))
                      # square side (mm) / single-track length (mm); override
                      # with MP_SIDE_MM for a smaller/faster apex test bed

SEG    = 0.25 if GEOM == "single_track" else 5.0
                      # control segment length (mm). Square/triangle: 5 mm, 2
                      # per line, >= 2 pool relaxation lengths (trail passage
                      # ~0.7 ms = 2.1 mm). single_track: 0.25 mm, fine enough to
                      # resolve the startup transient (95% depth by ~0.75 mm,
                      # so ~3 segments) that only scan speed can address.
SIGMA  = 200e-6       # lateral beam sigma (m), nominal
AZ     = 10e-6        # absorption depth sigma_z (m)
P_BASE = 150.0        # Beam.txt power (W); segment power = P_BASE * Pmod
PMOD0  = 1.0          # unoptimized multiplier (150 W everywhere)
T_LIQ  = 1610.0       # IN718 liquidus = melt isovalue
T0     = 1273.0
KCOND, CP, RHO = 26.6, 600.0, 7451.0

# ---- measurement box, rotated frame (segment end at origin, track -> +x) ----
BOX_BEHIND = float(os.environ.get("MP_BOX_BEHIND",
    {"triangle": 8.0e-3, "single_track": 2.5e-3}.get(GEOM, 7.0e-3)))
                      # trailing liquid extent to capture (zero-dwell merged
                      # pools reached 6.4 mm total length in the square
                      # baseline; the triangle baseline reached 7.2 mm; the
                      # single-track developed pool trails ~2 mm)
BOX_AHEAD  = 0.6e-3
BOX_HALF_Y = float(os.environ.get("MP_BOX_HALF_Y",
    {"triangle": 2.5e-3, "single_track": 0.5e-3}.get(GEOM, 1.2e-3)))
                      # triangle apex lakes at nominal power span > 1.2 mm
                      # laterally; single track is ~0.3 mm wide
BOX_DEPTH  = 0.25e-3
RES_XY     = 50e-6
RES_Z      = 5e-6

SQRT3 = np.sqrt(3.0)


def ensure_case():
    os.makedirs(os.path.join(CASE, "Data"), exist_ok=True)
    def w(name, text):
        with open(os.path.join(CASE, name), "w") as f:
            f.write(text)
    w("ParamInput.txt", "Simulation\n{\n\tName\t\tSqSnap\n\tMode\t\tMode.txt\n"
      "\tMaterial\tMaterial.txt\n\tBeam\t\tBeam.txt\n\tPath\t\tPath.txt\n}\n"
      "Options\n{\n\tDomain\t\tDomain.txt\n\tOutput\t\tOutput.txt\n"
      "\tSettings\tSettings.txt\n}\n")
    w("Mode.txt", "Snapshots\n{\n\tScanFracs \t100\n\tTracking\tNone\n}\n")
    w("Material.txt", "Constants\n{\n\tT_0\t%g\n\tT_L\t%g\n\tk\t%g\n\tc\t%g\n"
      "\tp\t%g\n}\n" % (T0, T_LIQ, KCOND, CP, RHO))
    w("Output.txt", "Grid\n{\n\tx\t1\n\ty\t1\n\tz\t1\n}\n"
      "Temperature\n{\n\tT\t1\n\tT_hist\t0\n}\n")
    w("Settings.txt", "Compute\n{\n\tMaxThreads\t14\n}\n")


def write_settings(seed_seg=None):
    """Settings.txt, optionally selecting the OTI seed segment (solver
    path-row index, 0-based, header excluded; None = default = last segment,
    the pre-existing current-segment behavior)."""
    extra = "" if seed_seg is None else f"\tSeedSegment\t{int(seed_seg)}\n"
    with open(os.path.join(CASE, "Settings.txt"), "w") as f:
        f.write("Compute\n{\n\tMaxThreads\t14\n%s}\n" % extra)


def write_snapshot_times(times):
    """Select one or more absolute snapshot times for the next solver run."""
    values = ",".join(f"{float(t):.17g}" for t in times)
    with open(os.path.join(CASE, "Mode.txt"), "w") as f:
        f.write(f"Snapshots\n{{\n\tTimes\t{values}\n\tTracking\tNone\n}}\n")


def write_beam(sigma=SIGMA):
    ensure_case()
    with open(os.path.join(CASE, "Beam.txt"), "w") as f:
        f.write("Shape\n{\n\tWidth_X\t\t%g\n\tWidth_Y\t\t%g\n\tDepth_Z\t\t%g\n}\n"
                "Intensity\n{\n\tPower\t\t%g\n\tEfficiency\t1.0\n}\n"
                % (sigma, sigma, AZ, P_BASE))


def scan_lines(nlines=None):
    """[(x_start, x_end, y)] serpentine lines for the selected geometry —
    identical to the corresponding make_case.py.  ``nlines`` optionally
    truncates the geometry for inexpensive controller validation; the default
    preserves every existing square/triangle workflow."""
    if GEOM == "single_track":
        lines = [(0.0, S, 0.0)]             # one straight bead-on-plate line
        return lines[:nlines] if nlines is not None else lines
    if GEOM == "two_tracks":
        lines = [(0.0, S, 0.0), (S, 0.0, H)]
        return lines[:nlines] if nlines is not None else lines
    if GEOM == "triangle":
        height = S * SQRT3 / 2.0
        lines, j = [], 0
        while True:
            y = j * H
            if y >= height:
                break
            half = y / SQRT3                    # edge inset at this height
            x0, x1 = half, S - half
            if x1 - x0 <= 0.0:
                break
            lines.append((x0, x1, y) if j % 2 == 0 else (x1, x0, y))
            j += 1
        return lines[:nlines] if nlines is not None else lines
    lines = [((0.0, S, j * H) if j % 2 == 0 else (S, 0.0, j * H))
             for j in range(int(round(S / H)) + 1)]
    return lines[:nlines] if nlines is not None else lines


def build_rows(dwell, nlines=None):
    """Path rows exactly as make_case.py writes them (header included)."""
    lines = scan_lines(nlines=nlines)
    if not lines:
        raise ValueError("scan path must contain at least one line")
    rows = ["Mode\tX(mm)\tY(mm)\tZ(mm)\tPmod\tVel/Time\n"]
    x0, x1, y = lines[0]
    rows.append(f"1\t{x0:.6f}\t{y:.6f}\t0\t0\t0\n")
    rows.append(f"0\t{x1:.6f}\t{y:.6f}\t0\t1\t{V}\n")
    for x0, x1, y in lines[1:]:
        if dwell > 0.0:
            rows.append(f"1\t{x0:.6f}\t{y:.6f}\t0\t0\t{dwell:g}\n")
        else:
            rows.append(f"0\t{x0:.6f}\t{y:.6f}\t0\t1\t{V}\n")   # hop, beam on
        rows.append(f"0\t{x1:.6f}\t{y:.6f}\t0\t1\t{V}\n")       # scan line
    return rows


def _balanced_segment_scan(lines, segment_size):
    """Split powered moves into near-equal, approximately ``segment_size``
    pieces.  Unlike ``SegmentScan``, this avoids a tiny final remainder on
    shrinking triangle lines.  Square lines that divide exactly are
    unchanged."""
    out = []
    x0 = y0 = 0.0
    for line in lines:
        vals = line.strip().split("\t")
        if vals[0] == "0" and vals[4] == "1":
            x1, y1 = float(vals[1]), float(vals[2])
            distance = np.hypot(x1 - x0, y1 - y0)
            count = max(1, int(np.floor(distance / segment_size + 0.5)))
            for part in range(1, count + 1):
                split = list(vals)
                frac = part / count
                split[1] = repr(x0 + frac * (x1 - x0))
                split[2] = repr(y0 + frac * (y1 - y0))
                out.append("\t".join(split) + "\n")
        else:
            out.append(line)
        if vals[0] in ("0", "1"):
            x0, y0 = float(vals[1]), float(vals[2])
    return out


def build_path(dwell, seg=SEG, nlines=None, balanced=False):
    """Segmented path. Returns (div, powered, hop_after) where `powered` are
    the CONTROLLED segment indices (scan-line pieces only) and hop_after[i]
    is the index of the powered hop row that immediately follows controlled
    segment i (zero-dwell only; hops inherit that segment's controls)."""
    rows = build_rows(dwell, nlines=nlines)
    div = (_balanced_segment_scan(rows, seg) if balanced
           else Scan.SegmentScan(rows, segmentSize=seg))
    powered, hops = [], []
    prev = None
    for i, line in enumerate(div):
        vals = line.strip().split('\t')
        if vals[0] not in ('0', '1'):
            continue
        x, y = float(vals[1]), float(vals[2])
        if vals[0] == '0' and vals[4] == '1':
            # scan lines are y-constant in both geometries; any powered move
            # in y is a serpentine hop (diagonal along the triangle's edges)
            if prev is not None and abs(y - prev[1]) > 1e-9:
                hops.append(i)
            else:
                powered.append(i)
        prev = (x, y)
    hop_after = {}
    for h in hops:
        before = [i for i in powered if i < h]
        if before:
            hop_after[before[-1]] = h
    return div, powered, hop_after


def seg_info(div, powered):
    """Per controlled segment: end position, line index (y/hatch), cumulative
    controlled path length (mm)."""
    info, dist, prev = [], 0.0, (0.0, 0.0)
    for i in range(len(div)):
        vals = div[i].strip().split('\t')
        if vals[0] not in ('0', '1'):
            continue
        x, y = float(vals[1]), float(vals[2])
        if i in powered:
            dist += np.hypot(x - prev[0], y - prev[1])
            info.append(dict(idx=i, x=x, y=y, line=int(round(y / H)), s_mm=dist))
        prev = (x, y)
    return info


def set_pmods(lines, pmods, wmods=None, vels=None):
    """Overwrite per-row controls: Pmod (col 4), scan speed (col 5, m/s), and
    the beam-width factor (col 6). `vels` carries the per-segment velocity for
    the speed control; the OTI solver seeds the last (current) segment's speed
    for dT_dv, and every segment's speed sets its own deposition timing."""
    out = list(lines)
    wmods = wmods or {}
    vels = vels or {}
    for idx in set(pmods) | set(wmods) | set(vels):
        if idx >= len(out):
            continue
        vals = out[idx].strip().split('\t')
        if idx in pmods:
            vals[4] = str(pmods[idx])
        if idx in vels:
            vals[5] = repr(vels[idx])
        if idx in wmods:
            while len(vals) < 7:
                vals.append('1.0')
            vals[6] = repr(wmods[idx])
        out[idx] = '\t'.join(vals) + '\n'
    return out


def apply_dwells(lines, dwells):
    """Convert selected beam-on inter-line HOP rows into beam-off DWELL rows.

    A zero-dwell path connects serpentine lines with a beam-on hop
    (``0 x y z 1 V``).  For a line whose adaptive dwell Delta>0 this rewrites
    that hop into a beam-off spot (``1 x y z 0 Delta``) parked at the line
    start -- the make_case dwell semantics.  Delta=0 leaves a beam-on hop
    untouched, so the zero-dwell path is byte-identical, but it reduces an
    existing beam-off dwell row to zero duration.  This lets a fixed-dwell
    policy optimize continuously down to zero without changing row indices.
    ``dwells`` maps a turnaround ROW INDEX to its dwell time (s)."""
    if not dwells:
        return lines
    out = list(lines)
    for idx, dt in dwells.items():
        if idx >= len(out) or dt is None or dt < 0.0:
            continue
        vals = out[idx].strip().split('\t')
        if dt == 0.0 and vals[0] == '0':
            continue                       # preserve the continuous hop
        vals[0] = '1'                 # spot mode (beam parked)
        vals[4] = '0'                 # beam off
        vals[5] = repr(float(dt))     # col 5 is the dwell time for a spot row
        out[idx] = '\t'.join(vals) + '\n'
    return out


def _mpi_size():
    return int(os.environ.get("THESIS_MPI_NP", os.environ.get("THESIS_NP", "1")))


def _run_env():
    env = os.environ.copy()
    env.setdefault("I_MPI_FABRICS", env.get("THESIS_MPI_FABRICS", "shm"))
    return env


def _run_command(binary):
    nproc = _mpi_size()
    if nproc <= 1:
        return [binary, "./ParamInput.txt"]
    mpiexec = os.environ.get("THESIS_MPIEXEC") or shutil.which("mpirun") or shutil.which("mpiexec")
    if not mpiexec:
        raise RuntimeError("THESIS_MPI_NP > 1 but no mpirun/mpiexec was found")
    return [mpiexec, "-n", str(nproc), binary, "./ParamInput.txt"]


def _clean_snapshot_outputs():
    data_dir = os.path.join(CASE, "Data")
    os.makedirs(data_dir, exist_ok=True)
    for path in glob.glob(os.path.join(data_dir, "*.Snapshot.*.csv")):
        os.remove(path)


def _merge_ranked_snapshots():
    data_dir = os.path.join(CASE, "Data")
    groups = {}
    for path in glob.glob(os.path.join(data_dir, "*.csv")):
        m = RANKED_SNAPSHOT.match(os.path.basename(path))
        if not m:
            continue
        groups.setdefault(m.group("idx"), []).append((int(m.group("rank")), path, m.group("base")))
    for idx, parts in groups.items():
        parts.sort()
        out = os.path.join(data_dir, parts[0][2] + ".csv")
        frames = [pd.read_csv(path) for _rank, path, _base in parts]
        cols = list(frames[0].columns)
        for (_rank, path, _base), f in zip(parts, frames):
            if list(f.columns) != cols:
                raise RuntimeError(f"header mismatch while merging {path}")
        merged = pd.concat(frames, ignore_index=True)
        # rank decomposition is not 1D: must restore lexicographic grid order
        merged.sort_values(["x", "y", "z"], kind="mergesort", inplace=True)
        merged.to_csv(out, index=False)
    return bool(groups)


def run_thesis(binary):
    _clean_snapshot_outputs()
    subprocess.run(_run_command(binary), cwd=CASE, env=_run_env(),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    _merge_ranked_snapshots()


def write_domain():
    """Measurement box in the rotated frame (segment end at the origin)."""
    with open(os.path.join(CASE, "Domain.txt"), "w") as f:
        f.write("X\n{\n\tMin\t%.7f\n\tMax\t%.7f\n\tRes\t%g\n}\n"
                "Y\n{\n\tMin\t%.7f\n\tMax\t%.7f\n\tRes\t%g\n}\n"
                "Z\n{\n\tMin\t%.7f\n\tMax\t0\n\tRes\t%g\n}\n"
                % (-BOX_BEHIND, BOX_AHEAD, RES_XY,
                   -BOX_HALF_Y, BOX_HALF_Y, RES_XY, -BOX_DEPTH, RES_Z))


def write_domain_box(xmin, xmax, ymin, ymax, zmin, rx, ry, rz):
    """Arbitrary lab-frame domain (animation views)."""
    ensure_case()
    with open(os.path.join(CASE, "Domain.txt"), "w") as f:
        f.write("X\n{\n\tMin\t%.7f\n\tMax\t%.7f\n\tRes\t%g\n}\n"
                "Y\n{\n\tMin\t%.7f\n\tMax\t%.7f\n\tRes\t%g\n}\n"
                "Z\n{\n\tMin\t%.7f\n\tMax\t0\n\tRes\t%g\n}\n"
                % (xmin, xmax, rx, ymin, ymax, ry, zmin, rz))


def run_path(lines, binary):
    ensure_case()
    Scan.ExportScan(lines, outFile=os.path.join(CASE, "Path.txt"))
    run_thesis(binary)


def load_field():
    d = pd.read_csv(CSV)
    x, y, z, T = d['x'].values, d['y'].values, d['z'].values, d['T'].values
    xs, ys, zs = np.unique(x), np.unique(y), np.unique(z)
    return xs, ys, zs, T.reshape(xs.size, ys.size, zs.size)


def run_segment(div, i, pmods, binary, sigma=None, sig_hist=None, vels=None,
                seed_idx=None, snapshot_times=None, dwells=None):
    """Truncate at segment i, rotate its raster onto +x ending at the origin,
    write the measurement box, run. With `sigma`, Beam.txt carries the
    CURRENT candidate and history segments keep their own optimized sigmas
    via the per-segment width factor sig_hist[j]/sigma. With `vels`, each
    segment carries its own scan speed (col 5). `dwells` maps inter-line HOP
    row indices to beam-off dwell durations (s); those hops become dwell rows
    (adaptive turnaround dwell), the rest stay beam-on. `seed_idx` selects
    which segment the OTI build seeds (dT_dQ/dT_dsig/dT_dv, or dT_ddwell when
    seed_idx is a dwell row), as a DIV row index like the pmods/vels keys;
    None = the current segment i. A history seed gives the cross-segment
    influence derivative d(QoI at end of i)/d(u_j)."""
    ensure_case()
    wmods = None
    if sigma is not None:
        write_beam(sigma)
        if sig_hist:
            wmods = {j: sg / sigma for j, sg in sig_hist.items() if j < i}
    else:
        write_beam()
    if snapshot_times is not None:
        if not len(snapshot_times):
            raise ValueError("snapshot_times must contain at least one time")
        write_snapshot_times(snapshot_times)
    if seed_idx is None:
        write_settings()
    else:
        if not (1 <= seed_idx <= i):
            raise ValueError(f"seed_idx {seed_idx} must be a path row in the "
                             f"truncated path (1..{i})")
        # div rows include the header at 0; solver path rows don't -> j-1
        write_settings(seed_seg=seed_idx - 1)
    lines = set_pmods(div[:i + 1], pmods, wmods, vels=vels)
    lines = apply_dwells(lines, dwells)
    rot = Scan.RotateTranslateLastRasterToX0(lines)
    Scan.ExportScan(rot, outFile=os.path.join(CASE, "Path.txt"))
    write_domain()
    run_thesis(binary)


def _grids(data):
    x, y, z, T = (data[c].values for c in ("x", "y", "z", "T"))
    nums = (np.unique(x).size, np.unique(y).size, np.unique(z).size)
    mins = np.array([x.min(), y.min(), z.min()])
    res = (np.array([x.max(), y.max(), z.max()]) - mins) / (np.array(nums) - 1)
    return T.reshape(nums), mins, res


def clip_check(data=None):
    """Liquid must not touch the measurement-box faces (except the z=0 top);
    a clipped support point silently corrupts h AND dh. Returns list of
    violated faces."""
    data = pd.read_csv(CSV) if data is None else data
    T3, mins, res = _grids(data)
    hot = T3 > T_LIQ
    faces = []
    if hot[0, :, :].any():   faces.append("x-")
    if hot[-1, :, :].any():  faces.append("x+")
    if hot[:, 0, :].any():   faces.append("y-")
    if hot[:, -1, :].any():  faces.append("y+")
    if hot[:, :, 0].any():   faces.append("z-")
    return faces


def measure_dims(path=None):
    """(width half-span, depth, asym) of the T_LIQ isosurface — the double-
    build analogue of the sensitivity extraction, same marching-cubes
    interpolation, our isovalue."""
    data = pd.read_csv(path or CSV)
    T3, mins, res = _grids(data)
    if not (T3 > T_LIQ).any():
        return 0.0, 0.0, 0.0, []
    verts, _, _, _ = measure.marching_cubes(T3, T_LIQ)
    y_left = mins[1] + verts[:, 1].min() * res[1]
    y_right = mins[1] + verts[:, 1].max() * res[1]
    depth = -(mins[2] + verts[:, 2].min() * res[2])
    return 0.5 * (y_right - y_left), depth, y_right + y_left, clip_check(data)


def sens(path=None):
    """Melt-pool QoIs + exact control sensitivities from an OTI snapshot at
    our isovalue (thin wrapper over the generic support-function layer)."""
    h, dh, controls = Sim.ExtractIsoSupportSensitivities(
        inFile=path or CSV, isovalue=T_LIQ)
    out = {'width': 0.5 * (h['y+'] + h['y-']),
           'asym': h['y+'] - h['y-'],
           'depth': h['z-']}
    for p in controls:
        out['dwidth_d' + p] = 0.5 * (dh['y+'][p] + dh['y-'][p])
        out['dasym_d' + p] = dh['y+'][p] - dh['y-'][p]
        out['ddepth_d' + p] = dh['z-'][p]
    return out


RESULTS_DIR, FIGURES_DIR = "results", "figures"
FULLFIELD_DIR = os.path.join(RESULTS_DIR, "fullfield")
CALIBRATION_DIR = os.path.join(RESULTS_DIR, "calibration")


def data_path(name, write=False):
    """Result files (CSV/JSON) live in results/ under the demo dir the
    scripts run from. Reads fall back to the top level for files written by
    older runs or by pipelines that were already in flight."""
    p = os.path.join(RESULTS_DIR, name)
    if write:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        return p
    return p if os.path.exists(p) else name


def fig_path(name):
    path = os.path.join(FIGURES_DIR, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def fullfield_path(name, write=False):
    """Compact replay traces and whole-build statistics."""
    path = os.path.join(FULLFIELD_DIR, name)
    if write:
        os.makedirs(FULLFIELD_DIR, exist_ok=True)
    return path


def calibration_path(name, write=False):
    """Compact single-track calibration outputs."""
    path = os.path.join(CALIBRATION_DIR, name)
    if write:
        os.makedirs(CALIBRATION_DIR, exist_ok=True)
    return path


def dwell_from_json():
    # CWD-relative: each demo dir (square/, triangle/) keeps its own
    # min_dwell.json next to its baseline/optimized results
    with open(data_path("min_dwell.json")) as f:
        return float(json.load(f)["min_dwell_s"])


def beam_on_intervals(path_file):
    """[(t0, t1)] of powered moves from a case's Path.txt, walking the rows;
    gaps between intervals are beam-off (dwell) windows."""
    t, prev, ivals = 0.0, None, []
    with open(path_file) as f:
        rows = f.read().splitlines()
    for line in rows:
        vals = line.strip().split('\t')
        if vals[0] == '1':
            t += float(vals[5])
            prev = (float(vals[1]), float(vals[2]))
        elif vals[0] == '0':
            x, y = float(vals[1]), float(vals[2])
            if prev is not None:
                dt = np.hypot(x - prev[0], y - prev[1]) * 1e-3 / float(vals[5])
                if dt > 0:
                    ivals.append((t, t + dt))
                t += dt
            prev = (x, y)
    return ivals, t


def line_start_times(rows, upto=None):
    """Walk UNSEGMENTED path rows (make_case/build_rows style: one row per
    scan line) and return the absolute start time (s) of every scan line,
    honoring hops, dwells, and varying line lengths. On SegmentScan output
    this would count every 5 mm piece as a line start — don't."""
    t, prev, starts = 0.0, None, []
    for line in rows[:upto]:
        vals = line.strip().split('\t')
        if vals[0] not in ('0', '1'):
            continue
        x, y = float(vals[1]), float(vals[2])
        if vals[0] == '1':
            t += float(vals[5])                 # spot rest (dwell)
            prev = (x, y)
            continue
        if prev is not None:
            if abs(y - prev[1]) <= 1e-9 and vals[4] == '1':
                starts.append(t)
            t += np.hypot(x - prev[0], y - prev[1]) * 1e-3 / float(vals[5])
        prev = (x, y)
    return starts
