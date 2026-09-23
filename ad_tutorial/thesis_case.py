"""Write, run and read small 3DThesis snapshot cases for the AD tutorial.

This module is plumbing only; the checks live in the check_*.py scripts.
Binaries default to build/ and build-oti/ at the repository root and can be
overridden with the THESIS_BIN and THESIS_BIN_OTI environment variables.
Cases are written under ad_tutorial/runs/ (ignored by git).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BIN = os.environ.get("THESIS_BIN", os.path.join(ROOT, "build", "bin", "3DThesis"))
BIN_OTI = os.environ.get("THESIS_BIN_OTI",
                         os.path.join(ROOT, "build-oti", "bin", "3DThesis"))
RUNS = os.path.join(HERE, "runs")


@dataclass
class Material:
    T0: float = 1273.0    # ambient / initial temperature (K)
    TL: float = 1610.0    # liquidus (K)
    k: float = 26.6       # conductivity (W/(m K))
    c: float = 600.0      # specific heat (J/(kg K))
    rho: float = 7451.0   # density (kg/m^3)

    @property
    def alpha(self) -> float:
        return self.k / (self.rho * self.c)


@dataclass
class Beam:
    # Beam.txt widths. The solver's source is exp(-3 r^2 / a^2), so a width a
    # is sqrt(6) times the Gaussian standard deviation. dT_dsig is d/da.
    width: float = 40e-6   # lateral width a (Width_X = Width_Y), m
    depth: float = 20e-6   # Depth_Z, m (not a design variable)
    power: float = 100.0   # absorbed W (Efficiency is written as 1.0)


# The 316H stainless process of the melt-pool study: absorbed power, and a
# lateral width of 80 um (a Gaussian standard deviation of 80/sqrt(6) um).
MATERIAL_316H = Material(T0=298.0, TL=1709.0, k=32.36, c=582.2, rho=7955.0)
BEAM_316H = Beam(width=80e-6, depth=10e-6, power=350.275)
SPEED_316H = 1.35   # m/s


def line(x_mm, v, pmod=1.0, width_factor=None):
    """Path row: move in a straight line to (x_mm, 0, 0) at speed v (m/s)."""
    return (0, x_mm, pmod, v, width_factor)


def spot(x_mm, duration, pmod=0.0):
    """Path row: hold at (x_mm, 0, 0) for `duration` s (beam off by default)."""
    return (1, x_mm, pmod, duration, None)


def grid(xlim, ylim, zlim, res):
    """Domain as {axis: (min, max, res)} in metres."""
    return {"X": (*xlim, res), "Y": (*ylim, res), "Z": (*zlim, res)}


def write_case(name, *, path, domain, material=None, beam=None,
               seed_segment=None, threads=None):
    """Write a snapshot-at-scan-end case and return its directory.

    `path` is a list of line()/spot() rows; the first row should be a spot
    that places the beam at the start. `seed_segment` is the 0-based Path.txt
    data row whose controls carry the OTI seeds (Settings Compute/SeedSegment);
    None seeds the last row, the solver default.
    """
    material = material or Material()
    beam = beam or Beam()
    threads = threads or min(8, os.cpu_count() or 1)
    case = os.path.join(RUNS, name)
    os.makedirs(case, exist_ok=True)

    files = {
        "ParamInput.txt":
            "Simulation\n{\n\tName\t\tTestSim\n\tMode\t\tMode.txt\n"
            "\tMaterial\tMaterial.txt\n\tBeam\t\tBeam.txt\n\tPath\t\tPath.txt\n}\n"
            "Options\n{\n\tDomain\t\tDomain.txt\n\tOutput\t\tOutput.txt\n"
            "\tSettings\tSettings.txt\n}\n",
        "Mode.txt": "Snapshots\n{\n\tScanFracs\t100\n\tTracking\tNone\n}\n",
        "Material.txt":
            f"Constants\n{{\n\tT_0\t{material.T0!r}\n\tT_L\t{material.TL!r}\n"
            f"\tk\t{material.k!r}\n\tc\t{material.c!r}\n\tp\t{material.rho!r}\n}}\n",
        "Beam.txt":
            f"Shape\n{{\n\tWidth_X\t\t{beam.width!r}\n\tWidth_Y\t\t{beam.width!r}\n"
            f"\tDepth_Z\t\t{beam.depth!r}\n}}\n"
            f"Intensity\n{{\n\tPower\t\t{beam.power!r}\n\tEfficiency\t1.0\n}}\n",
        "Domain.txt": "".join(
            f"{ax}\n{{\n\tMin\t{lo!r}\n\tMax\t{hi!r}\n\tRes\t{res!r}\n}}\n"
            for ax, (lo, hi, res) in domain.items()),
        "Output.txt": "Grid\n{\n\tx\t1\n\ty\t1\n\tz\t1\n}\n"
                      "Temperature\n{\n\tT\t1\n\tT_hist\t0\n}\n",
        "Settings.txt":
            f"Compute\n{{\n\tMaxThreads\t{threads}\n"
            + (f"\tSeedSegment\t{seed_segment}\n" if seed_segment is not None else "")
            + "}\n",
    }
    rows = ["Mode\tX(mm)\tY(mm)\tZ(mm)\tPmod\tVel(m/s)/Time(s)"]
    for mode, x_mm, pmod, param, width_factor in path:
        row = f"{mode}\t{x_mm!r}\t0\t0\t{pmod!r}\t{param!r}"
        if width_factor is not None:
            row += f"\t{width_factor!r}"   # optional 7th column
        rows.append(row)
    files["Path.txt"] = "\n".join(rows) + "\n"

    for filename, content in files.items():
        with open(os.path.join(case, filename), "w") as f:
            f.write(content)
    return case


def run(case, oti=True) -> pd.DataFrame:
    """Run the OTI (default) or plain-double build on `case`; return the CSV."""
    binary = BIN_OTI if oti else BIN
    if not os.path.isfile(binary):
        raise SystemExit(
            f"3DThesis binary not found: {binary}\n"
            "Build it first (see ad_tutorial/README.md, step 0) or set "
            + ("THESIS_BIN_OTI." if oti else "THESIS_BIN."))
    shutil.rmtree(os.path.join(case, "Data"), ignore_errors=True)
    subprocess.run([binary, "ParamInput.txt"], cwd=case, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return pd.read_csv(os.path.join(case, "Data", "TestSim.Snapshot.00.csv"))


class Gate:
    """Collects PASS/FAIL lines and turns them into an exit status."""

    def __init__(self):
        self.failed = []

    def _line(self, ok, label, text):
        if not ok:
            self.failed.append(label)
        print(f"  {'PASS' if ok else 'FAIL'}  {label:<40s} {text}")
        return ok

    def check(self, label, error, limit, note=""):
        """Pass when a relative error is at most `limit`."""
        ok = bool(np.isfinite(error) and error <= limit)
        return self._line(ok, label,
                          f"{error * 100:9.4f} %  (limit {limit * 100:g} %){note}")

    def zero(self, label, values):
        """Pass when every value is exactly zero (a design variable not seeded)."""
        biggest = float(np.max(np.abs(values)))
        return self._line(biggest == 0.0, label, f"max |value| = {biggest:g}")

    @staticmethod
    def info(label, error):
        print(f"  info  {label:<40s} {error * 100:9.4f} %")

    def exit(self):
        if self.failed:
            print(f"\n{len(self.failed)} check(s) failed: {', '.join(self.failed)}")
            raise SystemExit(1)
        print("\nall checks passed")


def max_rel_error(ours, reference, floor=0.1):
    """Largest relative error over the points where |reference| is at least
    `floor` of its maximum (relative error is meaningless at zero crossings)."""
    ours, reference = np.asarray(ours, float), np.asarray(reference, float)
    keep = np.abs(reference) >= floor * np.abs(reference).max()
    return float(np.max(np.abs(ours[keep] - reference[keep]) / np.abs(reference[keep])))
