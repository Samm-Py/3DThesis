"""Validate the OTI dwell sensitivity dT_ddwell (DesignVar DV_DWELL) against
central finite differences, at fixed path geometry.

A beam-off turnaround dwell's DURATION is the control. It has no kinematic
channel (beam off => no nodes on the dwell row), so the WHOLE derivative is the
history observation-time shift: lengthening the pause lets every upstream node
diffuse longer before the snapshot. The test therefore MUST carry inherited
heat -- a dwell with no prior deposition has dT_ddwell == 0 and would pass
trivially.

Self-contained: writes a 2-line + dwell snapshot case (line 1 lays heat, a
beam-off dwell, line 2 observed at scan end), seeds the dwell row, and compares:
  * field level  -- (T(Δ+δ) - T(Δ-δ)) / 2δ  vs the dT_ddwell column, on the
                    highest-signal grid points (measurement-free, tightest);
  * one-sided Δ=0 -- forward FD (T(δ) - T(0)) / δ  vs dT_ddwell at Δ=0+, the
                    "should I open a dwell here?" derivative the optimizer uses.

Run:  python melt_pool_demos/common/validate_ddwell.py [eps]   (default 1e-2)
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = "/root/Research/3d_thesis_test/3DThesis"
BIN_DBL = os.path.join(ROOT, "build", "bin", "3DThesis")
BIN_OTI = os.path.join(ROOT, "build-oti", "bin", "3DThesis")
EPS = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-2
DELTA0 = 0.5e-3            # reference dwell (s)
SEED_DWELL_ROW = 2        # 0-based Path.txt DATA row of the dwell -> SeedSegment
CASE = os.path.join(os.environ.get("TMPDIR", "/tmp"), "validate_ddwell_case")
DATA = os.path.join(CASE, "Data")


def w(name, text):
    with open(os.path.join(CASE, name), "w") as f:
        f.write(text)


def write_case(delta, seed=None):
    os.makedirs(DATA, exist_ok=True)
    # line1 (0,0)->(2,0) on; DWELL parked at (2,0.1) beam off for `delta`;
    # line2 (2,0.1)->(0,0.1) on.  Data-row 2 is the dwell.
    w("Path.txt",
      "Mode\tX(mm)\tY(mm)\tZ(mm)\tPmod\tVel/Time\n"
      "1\t0\t0\t0\t0\t0\n"
      "0\t2\t0\t0\t1\t3\n"
      f"1\t2\t0.1\t0\t0\t{delta:.12g}\n"
      "0\t0\t0.1\t0\t1\t3\n")
    w("Beam.txt", "Shape\n{\n\tWidth_X\t\t200e-6\n\tWidth_Y\t\t200e-6\n"
      "\tDepth_Z\t\t10e-6\n}\nIntensity\n{\n\tPower\t\t150\n\tEfficiency\t1.0\n}\n")
    w("Material.txt", "Constants\n{\n\tT_0\t1273\n\tT_L\t1610\n\tk\t26.6\n"
      "\tc\t600\n\tp\t7451\n}\n")
    w("Mode.txt", "Snapshots\n{\n\tScanFracs\t100\n\tTracking\tNone\n}\n")
    w("Output.txt", "Grid\n{\n\tx\t1\n\ty\t1\n\tz\t1\n}\n"
      "Temperature\n{\n\tT\t1\n\tT_hist\t0\n}\n")
    w("Domain.txt", "X\n{\n\tMin\t-0.0005\n\tMax\t0.0025\n\tRes\t50e-6\n}\n"
      "Y\n{\n\tMin\t-0.0005\n\tMax\t0.0006\n\tRes\t50e-6\n}\n"
      "Z\n{\n\tMin\t-0.0001\n\tMax\t0\n\tRes\t10e-6\n}\n")
    extra = "" if seed is None else f"\tSeedSegment\t{seed}\n"
    w("Settings.txt", "Compute\n{\n\tMaxThreads\t4\n%s}\n" % extra)
    w("ParamInput.txt", "Simulation\n{\n\tName\t\tVD\n\tMode\t\tMode.txt\n"
      "\tMaterial\tMaterial.txt\n\tBeam\t\tBeam.txt\n\tPath\t\tPath.txt\n}\n"
      "Options\n{\n\tDomain\t\tDomain.txt\n\tOutput\t\tOutput.txt\n"
      "\tSettings\tSettings.txt\n}\n")


def run(binary, delta, seed=None):
    write_case(delta, seed)
    for f in os.listdir(DATA):
        os.remove(os.path.join(DATA, f))
    subprocess.run([binary, "./ParamInput.txt"], cwd=CASE, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    csv = [f for f in os.listdir(DATA) if f.endswith(".csv")][0]
    return pd.read_csv(os.path.join(DATA, csv))


def _rel_top(ana, fd, n=300):
    top = np.argsort(-np.abs(ana))[:n]
    rel = np.abs(fd[top] - ana[top]) / np.abs(ana[top])
    return float(np.median(rel)), float(np.max(rel))


def field_check(delta0, one_sided, eps_sweep):
    """Analytic dT_ddwell once, FD swept over step size.  T is written at 6
    sig figs, so the T+-T- difference hits a roundoff floor as delta shrinks;
    the true agreement is read at the sweet spot (largest step whose O(delta^2)
    truncation is still below that floor), exactly as validate_dv reads its
    grid-limited width FD."""
    oti = run(BIN_OTI, delta0, seed=SEED_DWELL_ROW)
    ana = oti["dT_ddwell"].values
    xyz = oti[["x", "y", "z"]].values
    hot = int((oti["T"].values > 1610).sum())
    kind = "one-sided Δ=0" if one_sided else f"central Δ={delta0:.3g}s"
    print(f"\n=== field dT_ddwell, {kind} ===")
    print(f"  melt pts (T>T_liq): {hot};  dT_ddwell range "
          f"[{ana.min():.4g}, {ana.max():.4g}] K/s;  "
          f"dominant sign {'<0 (cooling) OK' if ana.min() < 0 else 'FAIL'}")
    print("    step δ (s)   median%   max%")
    best = 1e9
    for eps in eps_sweep:
        if one_sided:
            d = eps * (delta0 if delta0 > 0 else DELTA0)
            hp, hm = run(BIN_DBL, delta0 + d), run(BIN_DBL, delta0)
            fd = (hp["T"].values - hm["T"].values) / d
        else:
            d = eps * delta0
            hp, hm = run(BIN_DBL, delta0 + d), run(BIN_DBL, delta0 - d)
            fd = (hp["T"].values - hm["T"].values) / (2.0 * d)
        assert np.allclose(xyz, hp[["x", "y", "z"]].values), "grid drifted"
        med, mx = _rel_top(ana, fd)
        print(f"    {d:9.3g}   {med*100:7.3f}  {mx*100:6.3f}")
        best = min(best, mx)
    return best


if __name__ == "__main__":
    sweep = [5e-2, 2e-2, 1e-2, 5e-3, 1e-3]
    try:
        # central: sweet spot near the largest step (truncation << roundoff floor)
        x_central = field_check(DELTA0, one_sided=False, eps_sweep=sweep)
        # one-sided forward FD is O(delta): sweet spot at a smaller step
        x_onesided = field_check(0.0, one_sided=True, eps_sweep=sweep)
    finally:
        shutil.rmtree(CASE, ignore_errors=True)
    ok = x_central < 0.01 and x_onesided < 0.02
    print(f"\n{'PASS' if ok else 'FAIL'}: best-step field dT_ddwell central FD "
          f"< 1% (got {x_central*100:.3f}%) and one-sided Δ=0 forward FD < 2% "
          f"(got {x_onesided*100:.3f}%)")
    print("note: T is written at 6 sig figs; the T+-T- difference roundoff-"
          "floors as δ shrinks, so read agreement at the largest step whose "
          "O(δ^2) truncation stays below that floor (cf. validate_dv width FD).")
