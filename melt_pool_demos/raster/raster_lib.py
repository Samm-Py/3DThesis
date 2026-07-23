"""Shared machinery for the serpentine-raster uniformity study.

Raster: left-to-right, hop up one hatch, right-to-left, ... (Scan.MakeSquareRaster)
with a beam-off dwell at every turnaround (Scan.AddTurnDwell). The dwell used by
the demo is the MINIMAL one that lets the previous track's melt residue fully
solidify before the next track starts (derived by find_min_dwell in
raster_demo.py) -- below that, the measured pool at track starts merges with
still-liquid neighbour residue and no power value can hit a width target.

Parameters follow the prior optimisation study (optimize_uniformity.py):
V=0.7 m/s, H=0.15 mm, S=1.0 mm, sigma=10 um, base power 40 W, Pmod0=1.5.
"""
import os, sys, json, subprocess, glob, re, shutil
import numpy as np, pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))          # 3DThesis repo root
_DEMOS = os.path.dirname(_HERE)
sys.path.insert(0, _DEMOS)

from common import scan as Scan                           # noqa: E402
from common import measurement as Sim                    # noqa: E402

BIN_DBL = os.environ.get("THESIS_BIN",     os.path.join(_ROOT, "build",     "bin", "3DThesis"))
BIN_OTI = os.environ.get("THESIS_BIN_OTI", os.path.join(_ROOT, "build-oti", "bin", "3DThesis"))
CASE    = os.environ.get("THESIS_CASE",    os.path.join(_HERE, "cases", "snapcase"))
CSV     = os.path.join(CASE, "Data", "TestSim.Snapshot.00.csv")
RESULTS_DIR = os.path.join(_HERE, "results")
FIGURES_DIR = os.path.join(_HERE, "figures")

RANKED_SNAPSHOT = re.compile(r"^(?P<base>.+\.Snapshot\.(?P<idx>\d+))\.(?P<rank>\d+)\.csv$")

# ---- process parameters (match optimize_uniformity.py defaults) ----
V      = 0.7          # raster speed (m/s)
H      = 0.15         # hatch spacing (mm)
S      = 1.0          # square size (mm)
SEG    = 0.25         # control segment length (mm)
SIGMA  = 10e-6        # lateral beam sigma (m), fixed -- power is the only knob
AZ     = 10e-6        # absorption depth sigma (m)
P_BASE = 40.0         # Beam.txt power (W); segment power = P_BASE * Pmod
PMOD0  = 1.5          # unoptimised power multiplier (60 W everywhere)
T_LIQ  = 1733.0       # melt isotherm used throughout the study
RES    = 10e-6        # measurement-domain resolution (m)
BUF    = 0.5e-3       # measurement-domain buffer (m)


def ensure_case():
    """Create the isolated snapshot case used by all raster drivers."""
    os.makedirs(os.path.join(CASE, "Data"), exist_ok=True)

    def write(name, text):
        with open(os.path.join(CASE, name), "w") as f:
            f.write(text)

    write("ParamInput.txt", "Simulation\n{\n\tName\t\tTestSim\n\tMode\t\tMode.txt\n"
          "\tMaterial\tMaterial.txt\n\tBeam\t\tBeam.txt\n\tPath\t\tPath.txt\n}\n"
          "Options\n{\n\tDomain\t\tDomain.txt\n\tOutput\t\tOutput.txt\n"
          "\tSettings\tSettings.txt\n}\n")
    write("Mode.txt", "Snapshots\n{\n\tScanFracs \t100\n\tTracking\tNone\n}\n")
    write("Material.txt", "Constants\n{\n\tT_0\t1273.0\n\tT_L\t1733.0\n"
          "\tk\t26.6\n\tc\t600.0\n\tp\t7451.0\n}\n")
    write("Output.txt", "Grid\n{\n\tx\t1\n\ty\t1\n\tz\t1\n}\n"
          "Temperature\n{\n\tT\t1\n\tT_hist\t0\n}\n")
    write("Settings.txt", "Compute\n{\n\tMaxThreads\t4\n}\n")


def data_path(name, write=False):
    if write:
        os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, name)


def fig_path(name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    return os.path.join(FIGURES_DIR, name)


def write_beam(sigma=SIGMA):
    ensure_case()
    with open(os.path.join(CASE, "Beam.txt"), "w") as f:
        f.write("Shape\n{\n\tWidth_X\t\t%g\n\tWidth_Y\t\t%g\n\tDepth_Z\t\t%g\n}\n"
                "Intensity\n{\n\tPower\t\t%g\n\tEfficiency\t1.0\n}\n"
                % (sigma, sigma, AZ, P_BASE))


def build_path(dwell, seg=SEG):
    """Serpentine path lines with `dwell` seconds of beam-off rest at each
    turnaround, divided into `seg`-mm control segments. Returns (div, powered)
    where powered are the line indices of the powered sub-segments."""
    ensure_case()
    tmp = os.path.join(CASE, "_raster.txt")
    Scan.MakeSquareRaster(V=V, H=H, S=S, outFile=tmp)
    path = Scan.ConvertToSpotRest(Scan.LoadScan(tmp))
    if dwell > 0.0:
        path = Scan.AddTurnDwell(path, dwell)
    div = Scan.SegmentScan(path, segmentSize=seg)
    powered = [i for i in range(len(div))
               if div[i].strip().split('\t')[0] == '0'
               and div[i].strip().split('\t')[4] == '1']
    return div, powered


def set_pmods(lines, pmods, wmods=None):
    """Copy of the path lines with Pmod (col 4) and, optionally, the beam
    width factor Wmod (col 6, appended when absent) overridden per line
    index."""
    out = list(lines)
    wmods = wmods or {}
    for idx in set(pmods) | set(wmods):
        if idx >= len(out):
            continue
        vals = out[idx].strip().split('\t')
        if idx in pmods:
            vals[4] = str(pmods[idx])
        if idx in wmods:
            while len(vals) < 7:
                vals.append('1.0')
            vals[6] = repr(wmods[idx])
        out[idx] = '\t'.join(vals) + '\n'
    return out


def _mpi_size():
    return int(os.environ.get("THESIS_MPI_NP", os.environ.get("THESIS_NP", "1")))


def _run_env():
    env = os.environ.copy()
    # Intel MPI's default OFI fabric fails in WSL/sandboxed local runs here.
    # shm is the intended local-node transport and can be overridden by the
    # caller with I_MPI_FABRICS or THESIS_MPI_FABRICS.
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
        # The rank decomposition is not 1D in x (e.g. np=4 splits x AND y),
        # so concatenation order is NOT grid order. The measurement code
        # reshapes to (x,y,z) grids, which needs lexicographic ordering.
        merged = pd.concat(frames, ignore_index=True)
        merged.sort_values(["x", "y", "z"], kind="mergesort", inplace=True)
        merged.to_csv(out, index=False)
    return bool(groups)


def run_thesis(binary):
    ensure_case()
    _clean_snapshot_outputs()
    subprocess.run(_run_command(binary), cwd=CASE, env=_run_env(),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    _merge_ranked_snapshots()


def seg_info(div, powered):
    """Per powered segment: end position, line index, cumulative powered
    path length (mm) -- the x-axis of the dimension plots."""
    info, dist, line, prev = [], 0.0, -1, (0.0, 0.0)
    for i in range(len(div)):
        vals = div[i].strip().split('\t')
        if vals[0] not in ('0', '1'):
            continue
        x, y = float(vals[1]), float(vals[2])
        if vals[0] == '1':
            line += 1
        elif i in powered:
            dist += np.hypot(x - prev[0], y - prev[1])
            info.append(dict(idx=i, x=x, y=y, line=line, s_mm=dist))
        prev = (x, y)
    return info


def run_segment(div, i, pmods, binary, sigma=None, sig_hist=None):
    """Prior-study measurement pipeline: truncate at segment i, rotate the
    last raster onto +x ending at the origin, local domain, run.

    With `sigma` given, Beam.txt is written with that base sigma (= the
    CURRENT segment's candidate) and history segments keep their own
    optimised sigmas via the per-segment width factor sig_hist[j]/sigma."""
    wmods = None
    if sigma is not None:
        write_beam(sigma)
        if sig_hist:
            wmods = {j: sg / sigma for j, sg in sig_hist.items() if j < i}
    lines = set_pmods(div[:i + 1], pmods, wmods)
    rot = Scan.RotateTranslateLastRasterToX0(lines)
    Scan.ExportScan(rot, outFile=os.path.join(CASE, "Path.txt"))
    Sim.UpdateDomain(x=0.0, y=0.0, res=RES, buf=BUF,
                     outFile=os.path.join(CASE, "Domain.txt"))
    run_thesis(binary)


def write_domain(xmin, xmax, ymin, ymax, zmin, rx, ry, rz):
    """Lab-frame domain with per-axis resolution (for dwell checks and the
    animation views)."""
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
    T3 = T.reshape(xs.size, ys.size, zs.size)
    return xs, ys, zs, T3
