# Square study results

Compact, versionable data record for the 10 mm square study. Raw 3DThesis
grids and RDF event lists live under `cases/*/Data/` and are regenerable.

## Contents

- `greedy_1_zero.json`: the optimized continuous P/σ/v schedule (1 mm blocks).
- `target_zero.json`: the developed-single-track control target.
- `calibration/`: compact single-track calibration trace and summary.
- `fullfield/`: continuous-time replay traces and beam-on summary statistics
  for the nominal baseline (`SqBaselineX50Y10Z1Zero_*`) and the optimized
  schedule (`SqGreedy1X50Y10Z1Zero_*`) on the 50/10/1 µm grid.

## Beam-on statistics (the paper's square table)

| schedule | depth (µm) | full width (µm) | volume (mm³) |
|---|---:|---:|---:|
| baseline | 106.0 ± 12.7 | 388 ± 27 | 0.0798 ± 0.0240 |
| optimized | 62.2 ± 2.5 | 323 ± 9.9 | 0.01293 ± 0.00127 |

Run `make validate` from `melt_pool_demos/` to check these against the paper.
