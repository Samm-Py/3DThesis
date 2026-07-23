# Square study results

This directory is the compact, versionable data record for the 10 mm square
study. Raw 3DThesis grids and RDF event lists remain under `cases/*/Data/` and
are regenerable.

## Contents

- `baseline_{zero,dwell}.csv`: segment-end measurements at constant power.
- `optimized_{zero,dwell}.csv`: optimized controls and segment-end dimensions.
- `target_{zero,dwell}.json`: developed-single-track control targets.
- `min_dwell.json`: minimal worst-case turnaround dwell and bisection metadata.
- `fullfield/`: continuous-time replay traces plus summary statistics.
- `calibration/`: compact single-track calibration trace and summary.

## Controller diagnostics

| policy | width CV, baseline → optimized | depth CV, baseline → optimized | bound-pinned segments |
|---|---:|---:|---:|
| zero dwell | 4.24% → 0.19% | 10.33% → 0.36% | 0/202 |
| minimal dwell | 3.52% → 0.12% | 7.23% → 0.21% | 0/202 |

## Whole-build statistics

| beam-on mean ± standard deviation | depth (µm) | length (mm) | volume (mm³) | fusion depth (µm) |
|---|---:|---:|---:|---:|
| zero, baseline | 104 ± 13.1 | 4.24 ± 1.23 | 0.0793 ± 0.0242 | 104 ± 10.8 |
| zero, optimized | 67 ± 7.5 | 2.22 ± 0.47 | 0.0210 ± 0.0054 | 65 ± 6.8 |
| dwell, baseline | 92 ± 15.5 | 3.40 ± 1.36 | 0.0553 ± 0.0215 | 93 ± 8.7 |
| dwell, optimized | 63 ± 8.8 | 1.99 ± 0.55 | 0.0175 ± 0.0046 | 62 ± 4.6 |

Run the top-level `make validate` target from `melt_pool_demos/` to check the
segment-end tables.
