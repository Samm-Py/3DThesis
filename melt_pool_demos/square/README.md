# 10 mm square raster: uniformity optimization at the triangle's physics

For the continuous $P/\sigma/v$ paper study, the complete optimization,
$50/10/1~\mu$m replay, figure generation, paper-asset copy, and LaTeX build
can be run with:

```bash
./run_continuous_study.sh
```

(the matching nominal baseline replay is
`python verify_baseline.py --policy zero --resx 50e-6 --resy 10e-6 --resz 1e-6`;
`make square` from the parent directory runs both).

Control geometry for the triangle optimization work
(`melt_pool_demos/triangle/`): the same calibrated Stump & Plotkowski §3.5
physics — IN718 at 1273 K preheat, 3 m/s, σ_xy = 200 µm, σ_z = 10 µm,
150 W absorbed, melt isotherm 1610 K, 0.1 mm hatch, 50 µm in-plane grid —
on a 10 mm **square**, so the line length is constant and the process reaches
a statistically steady state (the triangle's shrinking lines never do).

Two dwell policies are studied, with the same well-formed single-track
target: **zero dwell** (continuous serpentine, pools merge across lines) and
the **minimal worst-case dwell** (field fully solidified at every line
start, as in `melt_pool_demos/raster/`). The gap between their optimized
results is the measured price of removing the dwell.

## Single-track calibration (`case_track`)

One 10 mm bead-on-plate line at nominal parameters fixes the scales
(`calibrate.py`, 5 µm z, 50 µs step):

| quantity | value |
|---|---:|
| developed pool length | 2.05 mm (1.95 trailing / 0.10 leading) |
| developed pool width | 0.30 mm (50 µm grid quantization) |
| developed pool depth | 0.060 mm |
| startup distance to 95 % depth | 0.75 mm |
| solidification tail after beam-off | 0.55 ms |

These set the optimization target (the well-formed traveling steady-state
pool), the control-segment scale (relaxation ≈ trail passage ≈ 0.7 ms →
segments of 2.5–5 mm), and the measurement-box extent (several mm of trail).

## Zero-dwell baseline (`case_zero`, 150 W constant)

101 lines, continuous serpentine, 340 ms path time; 599 s solver / 647 s
wall on 14 threads (1 µm z, 50 µs step). After a ~15-line startup ramp the process is
statistically steady with a strong **within-line sawtooth**:

- length oscillates 2.1 → 6.4 mm and depth 0.09 → 0.125 mm every line
  (2.1× the single-track depth at peak);
- **the field never solidifies between lines**: residual liquid at all 100
  line starts, ~0.06 mm³ on average — about 3× an entire single-track pool —
  so every line begins by merging with its neighbor's still-liquid trail.

## Minimal-dwell baseline (`case_dwell`)

`find_min_dwell.py` bisects the per-turn dwell against the raster demo's
criterion — no liquid cell anywhere at the instant each line starts (checked
exactly from the RDF event list), at worst case = nominal power. Each probe
is a full-path run; this brute-force bisection is exactly the procedure the
planned adaptive-dwell work (OTI Newton on d(criterion)/d(dwell)) replaces.

Result (8 probes, coarse-z 12.5 µm):

    minimal dwell = 0.906 ms/turn  (bracket [0.891, 0.906] ms, tol 25 µs)

Unlike the 1 mm raster (binding turn = the last one, monotone accumulation),
the binding turn here is #36 — mid-square, inside the statistically steady
regime, where which of the ~85 equivalent steady turns binds is essentially
noise. `case_dwell` (0.90625 ms/turn, 1 µm z) confirms the criterion at fine
resolution: **zero residual liquid at all 100 line starts**. The pool now
collapses completely at every turnaround and regrows each line (length
0 → 5.2 mm, depth to 0.111 mm, peak volume 0.083 mm³) — solidified starts,
but the within-line sawtooth remains the uniformity defect.

**Price of the dwell**: total build time 428 ms vs 341 ms zero-dwell —
**+26 %** — for a *global worst-case* per-turn dwell. (The planned adaptive
per-turn dwell attacks exactly this overhead.) Wall clock at 1 µm z / 50 µs
step on 14 threads: case_zero 599 s solver / 647 s wall, case_dwell 468 s
solver / 499 s wall.

## Paper record: greedy continuous P/σ/v on 1 mm blocks

The publication result (`raster/doc/paper/main.tex`, square section) is the
shared greedy controller (`../common/greedy_raster.py`, run through
`optimize_greedy.py`): continuous serpentine scanning, absorbed power,
lateral beam σ, and scan velocity optimized over 1 mm control blocks with
the cool turnaround seed and previous-block warm starts selected in the
two-track study. The committed schedule is `results/greedy_1_zero.json`;
its full tracked replay on the 50/10/1 µm grid
(`verify_greedy.py results/greedy_1_zero.json Greedy1 --policy zero
--resx 50e-6 --resy 10e-6 --resz 1e-6`) is
`results/fullfield/SqGreedy1X50Y10Z1Zero_*`, compared against the matched
baseline replay `SqBaselineX50Y10Z1Zero_*` (`verify_baseline.py`).

Beam-on melt-pool statistics of the tracked replays (mean ± standard
deviation; the numbers in the paper's square table):

| schedule | depth (µm) | full width (µm) | volume (mm³) |
|---|---:|---:|---:|
| baseline | 106.0 ± 12.7 | 388 ± 27 | 0.0798 ± 0.0240 |
| **optimized** | **62.2 ± 2.5** | **323 ± 9.9** | **0.01293 ± 0.00127** |

Figures: `make_greedy_figures.py --policy zero` writes
`figures/square_traces_zero.*` (per-line beam-on statistics) and
`figures/square_process_parameters.*` (P/σ/v histories);
`make_paper_maps.py --policy zero` recomposes the solidification
thermal-gradient maps as `figures/square_maps_zero.pdf`. The three PDFs are
copied into `../raster/doc/paper/figures/` by `run_continuous_study.sh`.

## Historical record: per-segment 2×2 Newton on power + σ

The first square optimization —
before velocity control and 1 mm blocks existed — regulated the same target
with power and σ only, on 5 mm segments, under both dwell policies. Its
committed artifacts (`results/optimized_{zero,dwell}.csv`,
`results/fullfield/SqOpt*`, and the `case_opt_*` cases) are retained as the
development record; the section below describes that study.

202 control segments (2 × 5 mm per line), target = the developed
single-track pool (w* = 163.84 µm half-span, d* = 63.34 µm on the snapshot
grid), previous-block warm starts within each line with a cool reset at each
turnaround. ~220 OTI simulations per policy
(1.1 per segment), ~25 min single-process each.

**Headline — whole-build pool statistics** (`fullfield_stats.py` on the
full tracked replays, `verify_optimized.py` → `cases/case_opt_*`): mean ±
standard deviation of the *instantaneous* pool dimensions over all beam-on
instants (50 µs sampling; beam-off dwell windows excised — a pool
solidifying during a designed dwell is not process variation). The last
column is the fusion-depth map: the deepest melt each interior (x,y) column
ever attained.

| beam-on mean ± σ | depth (µm) | length (mm) | lateral extent (mm) | volume (mm³) | fusion depth (µm) |
|---|---:|---:|---:|---:|---:|
| zero dwell, baseline | 106 ± 12.9 | 4.24 ± 1.23 | 0.34 ± 0.035 | 0.0771 ± 0.0237 | 106 ± 10.7 |
| zero dwell, **optimized** | **69 ± 7.2** | **2.22 ± 0.47** | 0.31 ± 0.016 | **0.0201 ± 0.0053** | 67 ± 6.7 |
| min dwell, baseline | 94 ± 15.4 | 3.40 ± 1.36 | 0.31 ± 0.034 | 0.0536 ± 0.0210 | 95 ± 8.5 |
| min dwell, **optimized** | **64 ± 8.7** | **1.99 ± 0.55** | 0.30 ± 0.026 | **0.0167 ± 0.0044** | 64 ± 4.4 |

The optimizer cuts the absolute pool fluctuation ~2× in depth, ~2.5× in
length, and **~4.5× in volume**, while simultaneously placing the mean on
the requested setpoint (fusion-depth means 67/64 µm attained vs 63.3 µm
requested — the baselines run 50–65 % too deep). The remaining fluctuation is dominated by
the once-per-line startup transient (the pool briefly overshoots to ~85 µm
in the first 1–2 mm of each line, where the fresh track crosses its
neighbor's still-hot end) — a sub-segment feature the per-segment
controller cannot see; fixing it needs a within-segment residual term,
graded segments near turns, or the planned velocity/dwell controls. Note:
"lateral extent" is the y-span of *all* liquid (merged bands at zero
dwell), not the beam-frame track half-width the controller regulates.
All four tracked replays are run at 1 µm z, so the depth statistics are
grid-converged (the triangle depth-resolution study showed only
few-micrometre shifts between 5 µm and 1 µm, consistent with the ~2 µm
shifts observed here when the record moved to 1 µm).

**Price of removing the dwell**: at part level the dwell buys a modestly
tighter build (fusion-depth σ 4.4 vs 6.7 µm; beam-on volume σ 0.0044 vs
0.0053 mm³)
at the cost of **+26 % build time** (428 vs 341 ms) — a genuine trade-off,
not a free lunch in either direction. Schedules are smooth, interior and
line-periodic in both policies: power falls 150 → ~100–115 W (dwelled) /
~96–111 W (zero dwell; lower because inherited background heat does part of
the melting); σ widens slightly to 203–211 µm. The optimized zero-dwell
schedule also leaves 3.5× less inherited liquid at line starts (0.017 vs
0.059 mm³ mean) — running cooler attacks the merged-pool defect indirectly.

**Controller diagnostics** (segment-end view — answers "did the Newton
converge", *not* part quality): width/depth CV 3.52/7.23 % → 0.12/0.21 %
(dwelled) and 4.24/10.33 % → 0.19/0.36 % (zero dwell) at the 202 regulated
segment ends; **0/202 segments pinned at a control bound** in either policy,
so the single-track target is feasible everywhere on this geometry.

Figures (in `figures/`): `square_traces_{dwell,zero}` (per-line pool
statistics — per scan line, the mean and p5–p95 band of the instantaneous
beam-on depth/volume/length, plus a detail strip resolving three mid-build
lines in real time; raw samples against build time render every ~3 ms line
as a one-pixel needle, so the per-line view carries the same information
legibly),
`square_profiles_{dwell,zero}` (all 101 lines folded onto
position-along-line — the clearest view of within-line vs line-to-line
variation), each as PNG + PDF; the paper-style
`Sq{Zero,Dwell,OptZero,OptDwell}_fig1{6,7}` (Fig. 17 L/D/V-vs-time traces
and Fig. 16 solidification-G maps, one per baseline and optimized case, on
fixed axes for before/after comparison) as PNG; plus the four
`figures/animations/square_{policy}_{case}.gif` animations.

## Layout

| dir | contents |
|---|---|
| `.` | the scripts (see below) |
| `results/` | compact data record: control tables, targets, dwell metadata, `fullfield/` replay statistics, and `calibration/` outputs |
| `figures/` | publication PNG/PDF figures plus `animations/`, `calibration/`, and `diagnostics/` |
| `cases/` | self-contained 3DThesis inputs (`case_track`, `case_{zero,dwell}`, `case_opt_*`); raw `Data/` outputs are gitignored and compact analysis products are written to `results/` |
| `logs/` | run logs (gitignored) |

## Files

The machinery shared with the triangle demo — path building, measurement,
the Newton optimizer, statistics, and figures — lives in
[`../common/`](../common/) and is run *from* this directory: the geometry,
snapshot-case location, and measurement box follow the working directory
(override with `MP_GEOM` / `MP_CASE_DIR` / `MP_BOX_BEHIND` /
`MP_BOX_HALF_Y`). Geometry-specific scripts stay here:

| file | what |
|---|---|
| `run_continuous_study.sh` | the paper pipeline: greedy optimization → 50/10/1 µm replay → figures → paper-figure copy → LaTeX build |
| `optimize_greedy.py` | entry point for the shared greedy P/σ/v controller (`../common/greedy_raster.py`) → `results/greedy_1_zero.json` |
| `verify_greedy.py` | full tracked replay of a greedy schedule at a chosen grid → `results/fullfield/SqGreedy1*` |
| `verify_baseline.py` | matched tracked replay of the nominal baseline → `results/fullfield/SqBaseline*` |
| `make_greedy_figures.py` | per-line traces and P/σ/v history figures for a greedy schedule |
| `make_paper_maps.py` | recomposes the surface + centre-section G maps for the paper |
| `make_case.py` | self-contained 3DThesis case: square serpentine (`--turn-dwell`, `--nlines`) or `--single-track` calibration line |
| `find_min_dwell.py` | bisects the minimal worst-case turnaround dwell → `results/min_dwell.json` |
| `verify_optimized.py` | replays a historical per-segment schedule through a full tracked sim → `cases/case_opt_{policy}/`, then writes compact statistics to `results/fullfield/` |
| `postprocess.py` | traces (L/D/V vs t), summary metrics, residual-liquid-at-starts diagnostic → `results/fullfield/` |
| `cases/case_track/`, `cases/case_zero/`, `cases/case_dwell/` | calibration, zero-dwell baseline, minimal-dwell baseline inputs |

Shared drivers in `../common/`:

| file | what |
|---|---|
| `mp_lib.py` | optimization machinery: snapshot case, make_case-identical path builder, hop handling, measurement box + clip check, MPI-aware runner |
| `run_baseline.py` / `run_optimized.py` | per-segment baseline sweep / 2×2 Newton optimizer (`{zero\|dwell}`) |
| `fullfield_stats.py` | whole-build beam-on statistics, compact traces, and the fusion-depth map from a tracked case's RDF |
| `make_plots.py` | publication figures (`dwell`, `zero`, `compare`, `traces {p}`, `profiles {p}`, `fusionmap {p}`) |
| `make_animation.py` | `{zero\|dwell} {unoptimized\|optimized} [output-tag]` → `figures/animations/square_{policy}_{case}[_tag].gif`: top-down surface T + direction-aware deepest trailing-pool section + instantaneous section dimensions, controller endpoint markers, and power trace |
| `calibrate.py` | single-track pool scales from the RDF event list → `results/calibration/` and `figures/calibration/` |

Baseline-case workflow: `python make_case.py --out cases/case_zero --name
SqZero --resz 5e-6 --timestep 5e-5`, then `(cd cases/case_zero &&
<build>/bin/3DThesis ./ParamInput.txt)`, then `python postprocess.py
cases/case_zero SqZero 5e-5`.

Optimization workflow: `python find_min_dwell.py` →
`python ../common/run_baseline.py {zero|dwell}` →
`python ../common/run_optimized.py {zero|dwell}` (needs the OTI build,
`THESIS_BIN_OTI`) → `python verify_optimized.py {zero|dwell}` →
`python ../common/make_plots.py {dwell|zero|compare|traces <policy>}` →
`python ../common/make_animation.py {zero|dwell} {unoptimized|optimized}`.
The same drivers run the triangle demo from its own directory — the
geometry is auto-detected from the working directory.

`make square` from the parent `melt_pool_demos/` directory reproduces the
paper record: the 50/10/1 µm baseline replay followed by
`run_continuous_study.sh`. The calibration record
(`results/calibration/`, `results/target_zero.json`) is committed; regenerate
it first with the `--single-track` + `calibrate.py` workflow above when
starting from a clean checkout.
