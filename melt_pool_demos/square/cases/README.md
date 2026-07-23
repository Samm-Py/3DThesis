# Square simulation cases

Each `case_*` directory contains the small text inputs needed to reproduce one
3DThesis run:

- `case_track`: developed single-track calibration;
- `case_baseline_x50y10z1_zero`: constant-power full-build baseline replay;
- `case_greedy1_x50y10z1_zero`: full-build optimized P/σ/v replay.

The large solver-generated `Data/` directories are deliberately absent and
ignored. Running the workflows recreates them in place. Compact traces and
statistics are written to `../results/`, while final visual products are
written to `../figures/`.

`snapcase/` and `snapcase_greedy1_localbox/` are transient scratch cases for
the greedy optimizer; they are generated automatically and ignored in full.
