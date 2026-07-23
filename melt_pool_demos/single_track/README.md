# Single track: the base case and velocity-optimizer testbed

The simplest geometry in the shared pipeline — one straight bead-on-plate line
at the same calibrated Stump & Plotkowski §3.5 physics as the
[square](../square/) and [triangle](../triangle/) demos: IN718 at 1273 K
preheat, 3 m/s, σ_xy = 200 µm, σ_z = 10 µm, 150 W absorbed, melt isotherm
1610 K, 50 µm in-plane / 5 µm z grid, 50 µs tracking step. It runs the same
drivers (`common/`) and produces the same figures, so it is the clean
reference the raster optimization targets, and the base case the per-segment
**velocity** optimizer is being developed on.

## Developed pool (`calibrate.py`)

One 10 mm line at nominal parameters fixes the pool scales — identical to the
square's calibration:

| quantity | value |
|---|---:|
| developed pool length | 2.05 mm (1.95 trailing / 0.10 leading) |
| developed pool width | 0.30 mm (50 µm grid quantization) |
| developed pool depth | 0.060 mm |
| startup distance to 95 % depth | 0.75 mm |
| solidification tail after beam-off | 0.55 ms |

`figures/Track_fig17_traces.png` shows the L/D/V-vs-time trace (startup ramp →
steady state → beam-off decay, on a ms axis); `figures/Track_fig16_maps.png` is
the solidification thermal-gradient map. Whole-track beam-on statistics
(`fullfield_stats.py`): depth 58 ± 8.5 µm, length 1.87 ± 0.50 mm — the spread is
entirely the once-per-track startup, since a constant-power single track is
otherwise uniform (fusion-depth map 59.9 ± 0.6 µm).

## Why this is the velocity base case

The per-segment $2\times2$ Newton controller (power + σ, shared with the raster
demos) has **nothing to do here**: it samples each control segment at its
*end*, where the pool has already developed (the startup is over by 0.75 mm),
so it measures the on-target developed pool at every segment and returns the
baseline unchanged (0/10 segments moved, width/depth CV 0.00/0.14 %). That is
the honest result — and exactly the motivation for velocity control. The single
track's only non-uniformity is the **startup transient**, a *within-segment*,
inherently non-steady feature that power/σ segment-end regulation cannot see but
that scan speed is the natural handle for. With `dT_dv` now implemented and
validated (see `common/validate_dv.py`, `common/verify_analytic_*.py`, and
`raster/doc/optimization_notes.tex`), this base case is the controlled testbed
for that next controller.

## Files

| file | what |
|---|---|
| `make_case.py` | self-contained single-track case at the calibrated parameters (`--length`, `--power`, `--resz`, …); surface tracking → G/V/tSol + RDF |
| `verify_optimized.py` | replays the optimized per-segment schedule through a full tracked sim → `cases/case_opt_zero/`, then whole-track stats + fig17/fig16 |
| `cases/case_base/` | the constant-power baseline (also the calibration source) |
| `results/`, `figures/` | calibration, control tables, `fullfield/` statistics, and the fig17/fig16 views |

The optimization machinery — path building, measurement, the Newton optimizer,
statistics, and figures — is shared in [`../common/`](../common/) and run *from*
this directory (geometry auto-detected as `single_track`; a single line is a
zero-dwell path, so it reuses the `zero` policy). The single-track numerical
path uses 0.25 mm pieces to resolve startup.

## Run

    make single_track          # from melt_pool_demos/

which calibrates, runs the tracked baseline, generates fig17/fig16, then the
per-segment baseline sweep, the `zero` optimization, and the tracked replay.
Individual steps mirror the square demo (`../common/run_baseline.py zero`,
`../common/run_optimized.py zero`, `python verify_optimized.py`).
