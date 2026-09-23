#!/usr/bin/env python3
"""Merge per-rank MPI snapshot slices into full-grid snapshot CSVs.

Under MPI, 3DThesis writes one file per rank per snapshot —
`<name>.Snapshot.NN.<rank>.csv` — each holding that rank's spatial sub-block
(the no-overlap decomposition tiles the grid with no duplicates). This stitches
the slices for each snapshot index NN back into a single
`<name>.Snapshot.NN.csv` that visualize.py / animate.py can consume directly.

Usage:
  python3 merge_ranks.py DATA_DIR [--out-dir DIR] [--name snapshot]
"""

import argparse
import glob
import os
import re
from collections import defaultdict

# <name>.Snapshot.<NN>.<rank>.csv   (rank is the trailing numeric token)
RANKED = re.compile(r"^(?P<base>.+\.Snapshot\.(?P<idx>\d+))\.(?P<rank>\d+)\.csv$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--out-dir", default=None,
                    help="where to write merged files (default: data_dir)")
    args = ap.parse_args()
    out_dir = args.out_dir or args.data_dir
    os.makedirs(out_dir, exist_ok=True)

    groups = defaultdict(list)  # idx -> list of (rank, path)
    for path in glob.glob(os.path.join(args.data_dir, "*.csv")):
        m = RANKED.match(os.path.basename(path))
        if m:
            groups[m.group("idx")].append((int(m.group("rank")), path, m.group("base")))

    if not groups:
        raise SystemExit(f"no ranked snapshot files (*.Snapshot.NN.R.csv) in {args.data_dir}")

    for idx in sorted(groups):
        parts = sorted(groups[idx])  # by rank
        base = parts[0][2]
        out = os.path.join(out_dir, base + ".csv")
        header = None
        rows = 0
        with open(out, "w") as fout:
            for rank, path, _ in parts:
                with open(path) as fin:
                    h = fin.readline()
                    if header is None:
                        header = h
                        fout.write(h)
                    elif h != header:
                        raise SystemExit(f"header mismatch in {path}")
                    for line in fin:
                        fout.write(line)
                        rows += 1
        print(f"snapshot {idx}: merged {len(parts)} ranks -> {os.path.basename(out)} ({rows} rows)")


if __name__ == "__main__":
    main()
