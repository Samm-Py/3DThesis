#!/usr/bin/env bash
set -euo pipefail

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PAPER="$HERE/../paper"
PYTHON_BIN=${PYTHON_BIN:-python}

cd "$HERE"

echo "[1/5] Optimizing the 87-line continuous triangle raster"
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

echo "[3/5] Generating the continuous triangle figure set"
"$PYTHON_BIN" make_greedy_figures.py --policy zero
"$PYTHON_BIN" make_paper_maps.py --policy zero

echo "[4/5] Copying the three triangle PDFs into the paper"
cp figures/triangle_maps_zero.pdf \
  "$PAPER/figures/triangle_maps_zero.pdf"
cp figures/triangle_traces_zero.pdf \
  "$PAPER/figures/triangle_traces_zero.pdf"
cp figures/triangle_process_parameters.pdf \
  "$PAPER/figures/triangle_process_parameters.pdf"

echo "[5/5] Compiling the paper"
(
  cd "$PAPER"
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
)

echo "Triangle continuous study complete"
