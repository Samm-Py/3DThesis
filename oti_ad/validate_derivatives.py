#!/usr/bin/env python3
"""Finite-difference validation of the OTI automatic-differentiation outputs.

For a given example, this runs the OTI build once to obtain the analytic
derivatives dT/d(x,y,z,Q,kon,rho,cps) at every grid point, then checks each:

  * x, y, z      : central differences across neighbouring grid points
                   (the temperature field itself, from the OTI run).
  * Q,kon,rho,cps: perturb the value in the input file by +/-h, re-run the
                   *double* build, and central-difference the temperature.

It reports the median and max relative error between the analytic derivative
and the finite-difference estimate for each design variable.

Usage:
  python3 validate_derivatives.py [--example DIR] [--oti EXE] [--double EXE]
"""

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# (design-variable column, input file, key in that file, relative step)
PARAMS = [
    ("dT_dQ",   "Beam.txt",     "Power", 1e-4),
    ("dT_dkon", "Material.txt", "k",     1e-4),
    ("dT_drho", "Material.txt", "p",     1e-4),
    ("dT_dcps", "Material.txt", "c",     1e-4),
]


def run(exe, case_dir):
    """Run 3DThesis in case_dir and return the parsed snapshot CSV rows."""
    subprocess.run([exe, "./ParamInput.txt"], cwd=case_dir,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    data_dir = os.path.join(case_dir, "Data")
    csvs = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not csvs:
        raise RuntimeError("no CSV produced in " + data_dir)
    with open(os.path.join(data_dir, csvs[0])) as fh:
        return list(csv.DictReader(fh))


def temperature_by_point(rows):
    """Map (x,y,z) -> T, with coordinates rounded to stabilise matching."""
    out = {}
    for r in rows:
        key = (round(float(r["x"]), 10), round(float(r["y"]), 10), round(float(r["z"]), 10))
        out[key] = float(r["T"])
    return out


def read_value(path, key):
    pat = re.compile(r"^\s*" + re.escape(key) + r"\s+([-+0-9.eE]+)\s*$")
    with open(path) as fh:
        for line in fh:
            m = pat.match(line)
            if m:
                return float(m.group(1))
    raise KeyError("key %r not found in %s" % (key, path))


def write_value(path, key, value):
    pat = re.compile(r"^(\s*" + re.escape(key) + r"\s+)([-+0-9.eE]+)(\s*)$")
    with open(path) as fh:
        lines = fh.readlines()
    with open(path, "w") as fh:
        for line in lines:
            m = pat.match(line)
            fh.write(m.group(1) + repr(value) + (m.group(3) or "\n") if m else line)


def perturbed_T(double_exe, example, filename, key, base, h):
    """Run the double build with input `key` set to base*(1+/-h); return T maps."""
    fields = {}
    for sign in (+1.0, -1.0):
        with tempfile.TemporaryDirectory() as tmp:
            case = os.path.join(tmp, "case")
            shutil.copytree(example, case)
            shutil.rmtree(os.path.join(case, "Data"), ignore_errors=True)
            write_value(os.path.join(case, filename), key, base * (1.0 + sign * h))
            fields[sign] = temperature_by_point(run(double_exe, case))
    return fields[+1.0], fields[-1.0]


def report(name, errors):
    if not errors:
        print(f"  {name:8s}: (no points above threshold)")
        return
    errors.sort()
    median = errors[len(errors) // 2]
    print(f"  {name:8s}: median rel err {median:.2e}   max {max(errors):.2e}   ({len(errors)} pts)")


def validate_params(oti_rows, example, double_exe):
    analytic = {(round(float(r["x"]), 10), round(float(r["y"]), 10),
                 round(float(r["z"]), 10)): r for r in oti_rows}
    print("Parameter derivatives (vs central difference of the double build):")
    for col, filename, key, h in PARAMS:
        base = read_value(os.path.join(example, filename), key)
        Tp, Tm = perturbed_T(double_exe, example, filename, key, base, h)
        step = base * h
        # Only score where the derivative carries real signal: within the top
        # decade of its peak magnitude. Far-field points (no heating) have both
        # FD and analytic ~0 and would otherwise dominate the statistics with a
        # degenerate 0/0 ratio.
        peak = max(abs(float(r[col])) for r in oti_rows)
        thresh = 1e-2 * peak
        errors, sample = [], None
        for key_xyz, r in analytic.items():
            if key_xyz not in Tp or key_xyz not in Tm:
                continue
            an = float(r[col])
            if abs(an) < thresh:
                continue
            fd = (Tp[key_xyz] - Tm[key_xyz]) / (2.0 * step)
            errors.append(abs(fd - an) / abs(an))
            if sample is None or abs(an) >= peak * 0.999:
                sample = (an, fd)
        report(col, errors)
        if sample:
            print(f"            peak point: analytic {sample[0]:.6g}  FD {sample[1]:.6g}")


def validate_position(oti_rows):
    """Central differences of T across structured-grid neighbours."""
    T, dx_an, dy_an = {}, {}, {}
    xs, ys = set(), set()
    for r in oti_rows:
        x, y = round(float(r["x"]), 10), round(float(r["y"]), 10)
        T[(x, y)] = float(r["T"])
        dx_an[(x, y)] = float(r["dT_dx"])
        dy_an[(x, y)] = float(r["dT_dy"])
        xs.add(x); ys.add(y)
    xs, ys = sorted(xs), sorted(ys)
    if len(xs) < 3 or len(ys) < 3:
        print("Position derivatives: grid too small to finite-difference")
        return
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    # Score only where the gradient carries signal (top decade of its peak).
    px = 1e-2 * max(abs(v) for v in dx_an.values())
    py = 1e-2 * max(abs(v) for v in dy_an.values())
    ex, ey = [], []
    for i in range(1, len(xs) - 1):
        for j in range(1, len(ys) - 1):
            c = (xs[i], ys[j])
            l, rr = (xs[i - 1], ys[j]), (xs[i + 1], ys[j])
            d, u = (xs[i], ys[j - 1]), (xs[i], ys[j + 1])
            if l in T and rr in T and abs(dx_an[c]) > px:
                ex.append(abs((T[rr] - T[l]) / (2 * dx) - dx_an[c]) / abs(dx_an[c]))
            if d in T and u in T and abs(dy_an[c]) > py:
                ey.append(abs((T[u] - T[d]) / (2 * dy) - dy_an[c]) / abs(dy_an[c]))
    print("Position derivatives (vs central difference across grid neighbours):")
    report("dT_dx", ex)
    report("dT_dy", ey)
    print("  dT_dz   : (z has a single grid layer in this example; skipped)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", default=os.path.join(ROOT, "examples", "snapshot"))
    ap.add_argument("--oti", default=os.path.join(ROOT, "build-oti", "bin", "3DThesis"))
    ap.add_argument("--double", default=os.path.join(ROOT, "build", "bin", "3DThesis"))
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        case = os.path.join(tmp, "case")
        shutil.copytree(args.example, case)
        shutil.rmtree(os.path.join(case, "Data"), ignore_errors=True)
        oti_rows = run(args.oti, case)

    print(f"OTI run produced {len(oti_rows)} points with columns: "
          f"{', '.join(oti_rows[0].keys())}\n")
    validate_position(oti_rows)
    print()
    validate_params(oti_rows, args.example, args.double)


if __name__ == "__main__":
    main()
