#!/bin/bash
# Reproduce the two-track continuous P/sigma/v case used by the paper.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

MP_BOX_BEHIND=0.001 MP_CASE_DIR="$PWD/cases/snapcase_box" \
  "$PY" optimize_greedy.py \
  --policy continuous \
  --segment-mm 1 \
  --max-iter 12 \
  --tolerance 0.01 \
  --within-line-warm-start previous-block \
  --turn-seed cool

"$PY" verify_schedule.py \
  results/greedy_1_prevblock_coolseed_continuous.json \
  Greedy1PrevCool \
  --policy continuous \
  --resx 50e-6 --resy 1e-6 --resz 1e-6

"$PY" make_publication_figures.py
