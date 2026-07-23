# Square simulation cases

Each `case_*` directory contains the small text inputs needed to reproduce one
3DThesis run:

- `case_track`: developed single-track calibration;
- `case_zero` and `case_dwell`: constant-power full-build baselines;
- `case_opt_zero` and `case_opt_dwell`: full-build optimized replays.

The large solver-generated `Data/` directories are deliberately absent and
ignored. Running the workflows recreates them in place. Compact traces and
statistics are written to `../results/`, while final visual products are
written to `../figures/`.

`cases/snapcase/` is transient scratch space for the per-segment optimizer and
animation drivers; it is generated automatically and ignored in full.

