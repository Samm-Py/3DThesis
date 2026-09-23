#!/usr/bin/env python3
"""Runtime comparison: OTI (AD) build vs plain double build.

Runs the same example N times with each executable and reports the wall-clock
time and the OTI/double overhead factor. The OTI build carries DV_COUNT
first-order derivatives through every operation, so some slowdown is expected;
this quantifies it.

Usage:
  python3 benchmark.py [--example DIR] [--reps N] [--oti EXE] [--double EXE]
"""

import argparse
import os
import shutil
import statistics
import subprocess
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def time_runs(exe, example, reps):
    times = []
    for _ in range(reps):
        with tempfile.TemporaryDirectory() as tmp:
            case = os.path.join(tmp, "case")
            shutil.copytree(example, case)
            shutil.rmtree(os.path.join(case, "Data"), ignore_errors=True)
            t0 = time.perf_counter()
            subprocess.run([exe, "./ParamInput.txt"], cwd=case,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            times.append(time.perf_counter() - t0)
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", default=os.path.join(ROOT, "examples", "snapshot"))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--oti", default=os.path.join(ROOT, "build-oti", "bin", "3DThesis"))
    ap.add_argument("--double", default=os.path.join(ROOT, "build", "bin", "3DThesis"))
    args = ap.parse_args()

    print(f"Example: {args.example}   reps: {args.reps}\n")
    dbl = time_runs(args.double, args.example, args.reps)
    oti = time_runs(args.oti, args.example, args.reps)

    md, mo = statistics.median(dbl), statistics.median(oti)
    print(f"  double : median {md:.3f} s   (min {min(dbl):.3f}, max {max(dbl):.3f})")
    print(f"  OTI    : median {mo:.3f} s   (min {min(oti):.3f}, max {max(oti):.3f})")
    print(f"\n  OTI / double overhead: {mo / md:.2f}x  "
          f"(carrying {7} first-order derivatives)")


if __name__ == "__main__":
    main()
