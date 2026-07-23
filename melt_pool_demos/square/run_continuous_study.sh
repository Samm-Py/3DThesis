#!/usr/bin/env bash
set -euo pipefail

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PAPER="$HERE/../raster/doc/paper"
PYTHON_BIN=${PYTHON_BIN:-python}

cd "$HERE"

echo "[1/5] Optimizing the 101-line continuous square raster"
"$PYTHON_BIN" optimize_greedy.py \
  --policy zero \
  --segment-mm 1 \
  --max-iter 6 \
  --tolerance 0.05

echo "[2/5] Replaying the optimized schedule at 50/10/1 um"
"$PYTHON_BIN" verify_greedy.py \
  results/greedy_1_zero.json Greedy1 \
  --policy zero \
  --resx 50e-6 \
  --resy 10e-6 \
  --resz 1e-6

echo "[3/5] Generating the continuous square figure set"
"$PYTHON_BIN" make_greedy_figures.py --policy zero
"$PYTHON_BIN" make_paper_maps.py --policy zero

echo "[4/5] Copying the three square PDFs into the paper"
cp figures/square_maps_zero.pdf \
  "$PAPER/figures/square_maps_zero.pdf"
cp figures/square_traces_zero.pdf \
  "$PAPER/figures/square_traces_zero.pdf"
cp figures/square_process_parameters.pdf \
  "$PAPER/figures/square_process_parameters.pdf"

echo "[5/5] Compiling the paper"
(
  cd "$PAPER"
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
)

echo "Square continuous study complete"
