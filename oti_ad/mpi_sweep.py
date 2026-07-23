#!/usr/bin/env python3
"""MPI scaling + OTI/double overhead sweep for 3DThesis.

For each (build, np) it copies the example to a temp dir, runs under mpirun, and
records the median wall-clock of `reps` runs. Reports per-np speedup, parallel
efficiency, and the OTI/double overhead factor at each np — all on the same
Intel mpicxx builds, so the comparison is apples-to-apples.
"""
import argparse, os, shutil, statistics, subprocess, tempfile, time

OTI = "/root/Research/3d_thesis_test/3DThesis/build-mpi-oti/bin/3DThesis"
DBL = "/root/Research/3d_thesis_test/3DThesis/build-mpi-double/bin/3DThesis"


def time_run(exe, example, np, reps):
    ts = []
    for _ in range(reps):
        with tempfile.TemporaryDirectory() as tmp:
            case = os.path.join(tmp, "case")
            shutil.copytree(example, case)
            shutil.rmtree(os.path.join(case, "Data"), ignore_errors=True)
            t0 = time.perf_counter()
            subprocess.run(["mpirun", "-np", str(np), exe, "./ParamInput.txt"],
                           cwd=case, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=True)
            ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", required=True)
    ap.add_argument("--nps", default="1,2,4,8")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    nps = [int(x) for x in args.nps.split(",")]

    rows = []
    for np in nps:
        d = time_run(DBL, args.example, np, args.reps)
        o = time_run(OTI, args.example, np, args.reps)
        rows.append((np, d, o))
        print(f"np={np:>2}  double {d:7.3f}s   OTI {o:7.3f}s   overhead {o/d:5.2f}x",
              flush=True)

    d1 = rows[0][1]; o1 = rows[0][2]
    print("\n np | double s | OTI s | overhead | dbl-speedup | OTI-speedup | OTI-eff")
    for np, d, o in rows:
        print(f"{np:>3} | {d:8.3f} | {o:6.3f} | {o/d:6.2f}x | "
              f"{d1/d:9.2f}x | {o1/o:9.2f}x | {100*o1/o/np:5.0f}%")


if __name__ == "__main__":
    main()
