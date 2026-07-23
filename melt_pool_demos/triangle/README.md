# Triangle cross-section: reproducing Stump & Plotkowski §3.5 (unoptimized)

For the continuous $P/\sigma/v$ paper study, the complete optimization,
$50/10/1~\mu$m replay, figure generation, paper-asset copy, and LaTeX build
can be run with:

```bash
./run_continuous_study.sh
```

Reproduction of the paper's "Long Length and Time Scales" experiment — the
equilateral-triangle raster (their Figures 16 and 17) — run **unoptimized**
(constant process parameters). It serves two purposes:

1. **Validation** that the OTI modifications to 3DThesis did not change stock
   behavior: a clean-checkout stock binary (git worktree at HEAD, no
   `THESIS_ENABLE_OTI`) and this tree's double build produce **byte-identical**
   `Solidification.Final.csv` and `RDF.Final.csv` on the full case
   (448,951 melted grid points, 2,312,167 melt events). Verified 2026-07-13.
2. A larger, more interesting test geometry for later optimization work
   (continuously varying line length → strong heat accumulation gradients).

## Parameters (from the paper)

| parameter | value | source |
|---|---|---|
| geometry | equilateral triangle, 10 mm sides, raster base→apex | §3.5 |
| hatch | 0.1 mm, serpentine | §3.5 |
| material | IN718: k 26.6, c 600, ρ 7451, T_liq 1610 K | Table 1 |
| beam velocity | 3 m/s | §3.5 |
| beam σ_xy | 200 µm ("spot size"; V* = σv/α = 100.8 matches §3.2) | §3.5 |
| absorbed power | 750 W, efficiency 1.0 | §3.2 (not restated in §3.5) |
| preheat T₀ | 1273 K | §3.2 (not restated in §3.5) |
| grid | 50 µm in-plane, 12.5 µm depth | §3.5 |
| beam σ_z | 10 µm (δ = 0.05) | **not stated in the paper** |

## The dwell-time question (asked to double-check)

Two versions of the paper exist and they answer differently:

- **Draft** (`quadrature and nondimensionalization_v6.pdf`): text never
  mentions a dwell, but its Fig. 17 time axis runs to **~1450 ms** while the
  448 mm serpentine at 3 m/s takes only **149 ms** — implying ~15 ms of
  beam-off time per line. `cases/case_dwell` (15.1 ms/line) reproduces that
  figure's character on that time base.
- **Published** (Appl. Math. Modelling 75 (2019) 787–805,
  `1-s2.0-S0307904X19304093-main.pdf`): Fig. 17 was replaced — its axis is
  **0–0.15 s = exactly the continuous no-dwell path time**. So the published
  run has **no dwell**; the draft figure was evidently from an earlier run
  with per-line dead time. (The published Fig. 16 insets still carry
  timestamps up to 248 ms > 150 ms, and the Fig. 17 caption misnames its own
  panels — the section is genuinely sloppy.)

**Answer: the published experiment uses no dwell.** But the missing knob is
**power**: §3.5 never restates power or preheat. With §3.2's 750 W/1273 K the
continuous raster degenerates into one permanent molten lake (`cases/case_full`:
11.4 mm³ peak melt, no per-line structure — nothing like any version of the
figure). A power sweep at 1273 K preheat against the published envelopes:

| Q (W) | V peak (mm³) | L peak (mm) | D max (mm) | published target |
|---:|---:|---:|---:|---|
| 750 | 11.4 | 10.2 | 0.40 (floor) | — |
| 300 | 1.56 | 9.7 | 0.31 | — |
| 200 | 0.47 | 9.0 | 0.21 | — |
| **150** | **0.197** | **7.2** | **0.150** | **~0.21 / ~7.5 / ~0.16** ✓ |
| 120 | 0.10 | 5.3 | 0.125 | — |

**`cases/case_pub` (150 W absorbed, continuous, no dwell) reproduces the published
Fig. 17** — oscillation structure, envelope shapes, peaks, and the 150 ms
time base — and its G maps carry the published Fig. 16's striped structure
with the low-G pocket below the apex.

## Depth-resolution convergence

The paper explicitly states 12.5 µm z spacing in Section 3.5, but this is
inconsistent with its smooth Fig. 17b. The melt-pool tracking method described
in Section 2.7 stores results on a rectilinear reference grid, so an unsmoothed
depth trace from a 12.5 µm grid would have clearly visible 12.5 µm steps. The
paper documents neither interpolation nor smoothing. A likely explanation is a
decimal-place typo for **1.25 µm**, though the available evidence cannot prove
whether the error is in the stated resolution or the undocumented plotting.

The calibrated case was therefore run at 5, 2.5, and 1 µm z spacing while
holding xy spacing and all physical/process parameters fixed. All three use a
50 µs tracking timestep.

| quantity | 5 µm z | 2.5 µm z | 1 µm z |
|---|---:|---:|---:|
| peak depth | 0.160 mm | 0.160 mm | 0.160 mm |
| depth MAE relative to 1 µm | 2.03 µm | 0.85 µm | — |
| peak volume | 0.19146 mm³ | 0.18911 mm³ | 0.18773 mm³ |
| volume MAE relative to 1 µm | 0.00264 mm³ | 0.00099 mm³ | — |
| distinct depth levels in trace | 26 | 46 | 104 |
| melt-pool length and width traces | identical | identical | identical |
| wall time / peak RSS | 51 s / 0.72 GB | 178 s / 1.40 GB | 208 s / 3.42 GB |

Thus 5 µm is already close to z-converged for the scalar metrics, while 1 µm
is preferred when reproducing the paper's smooth depth trace. The 0.4 mm-deep
domain is not limiting this case; the pool reaches only 0.16 mm.

`postprocess.py` reproduces the published Fig. 17 visual conventions: panels
are ordered length/depth/volume (despite the paper's incorrect caption), time
is in seconds, published axis ranges are fixed, lines use the original muted
blue, and axes use serif STIX/LaTeX-style typography.

## Files

| file | what |
|---|---|
| `make_case.py` | generates a self-contained 3DThesis case (`--side`, `--hatch`, `--turn-dwell`, …) |
| `postprocess.py` | rebuilds Fig 17 (volume/length/depth vs t from the RDF event list), Fig 16 (G maps), extras (V, remelt count, tSol) |
| `cases/case_pub/`, `cases/case_pub_z2p5/`, `cases/case_pub_z1/` | published match at 5, 2.5, and 1 µm z resolution |
| `cases/case_full/`, `cases/case_dwell/` | literal 750 W lake and draft-timing dwell |
| `validation/results/` | reproduction traces, maps, and compact summaries (`TriPub`, `Triangle`, `TriDwell`) |

Run: `python make_case.py --out cases/case_dwell --name TriDwell --turn-dwell 15.1e-3`,
then `(cd cases/case_dwell && <build>/bin/3DThesis ./ParamInput.txt)`, then
`python postprocess.py cases/case_dwell TriDwell`. For the fine published case use
the same workflow with `--power 150 --resz 1e-6 --timestep 5e-5`.

## Implementation notes

- **Surface tracking records the full 3D pool**: depth columns are probed
  every step, so `Solidification.Final.csv` has G/V/tSol at every melted
  (x,y,z) and the RDF has melt/solidify pairs at all z levels (verified
  against the melted-grid z histogram). One Surface run yields both figures;
  no Volume-tracking run needed. (RDF melt times are only recorded in
  Surface mode — `Run.cpp: Solidify_Surface`.)
- **RDF is the workhorse for time series**: a cell is liquid on [tm, tl), so
  volume(t) = cellvol × active count, length(t) = x-extent of active events,
  depth(t) = −min z. Exact, no snapshots needed.
- `MP_Stats`' `MP_depth` multiplies the z-*index* by the **x** resolution
  (`src/Melt.cpp`), so on anisotropic grids rescale by zres/xres; the RDF
  depth needs no correction.
- Wall clock on 14 threads: case_pub 48 s (5 µm z), case_pub_z2p5 170 s
  solver/178 s wall (2.5 µm z), case_pub_z1 190 s solver/208 s wall (1 µm z),
  case_full 112 s, and case_dwell 46 s. All fine cases use a 50 µs step,
  compared with the paper's reported 1973 s on 4 cores (2019).

## Paper record: greedy continuous P/σ/v on 1 mm blocks

The publication result (`raster/doc/paper/main.tex`, triangle section) is
the shared greedy controller (`../common/greedy_raster.py`, run through
`optimize_greedy.py`): continuous serpentine scanning, absorbed power,
lateral beam σ, and scan velocity optimized over nominally 1 mm blocks
(balanced on each shrinking line so no short remainder is left), with the
cool turnaround seed and previous-block warm starts selected in the
two-track study. The committed schedule is `results/greedy_1_zero.json`;
its 50/10/1 µm tracked replay is
`results/fullfield/TriGreedy1X50Y10Z1Zero_*`, compared against the matched
baseline replay `TriBaselineX50Y10Z1Zero_*` (`verify_baseline.py`).

Beam-on melt-pool statistics of the tracked replays (mean ± standard
deviation; the numbers in the paper's triangle table):

| schedule | depth (µm) | full width (µm) | volume (mm³) |
|---|---:|---:|---:|
| baseline | 118.7 ± 21.2 | 484 ± 170 | 0.11274 ± 0.04715 |
| **optimized** | **63.7 ± 4.4** | **329 ± 15** | **0.01176 ± 0.00160** |

`run_continuous_study.sh` runs the whole pipeline and copies
`triangle_maps_zero.pdf`, `triangle_traces_zero.pdf`, and
`triangle_process_parameters.pdf` into `../raster/doc/paper/figures/`.

## Historical record: per-segment 2×2 Newton on power + σ

The first triangle optimization — before velocity control and 1 mm blocks
existed — regulated the same target with power and σ only under both dwell
policies. Its committed artifacts (`results/optimized_{zero,dwell}.csv`,
`results/fullfield/TriOpt*`, `TriMinDwell*`, and the `case_opt_*` /
`case_mindwell` cases) are retained as the development record; the section
below describes that study. Its apex diagnosis is what motivated adding
velocity as a control.

The shared optimizer (`melt_pool_demos/common/`, same drivers as the
square demo, run from this directory — geometry, snapshot case, and
measurement box are auto-detected from the working directory) applied
to the triangle raster:
131 controlled segments over 87 shrinking lines, same single-track target
(w* = 163.8 µm half-span, d* = 63.3 µm), 2×2 Newton on (power, σ) with
exact OTI sensitivities.

**Minimal dwell** (`find_min_dwell.py`): 1.672 ms/turn, **binding turn #79 —
near the apex** (the square's binding turn was mid-part) — which would take
the 149 ms continuous path to 290 ms: **+94 % build time**, vs +26 % for the
square. The apex is why adaptive per-turn dwell is worth building.

**Whole-build results** (beam-on mean ± σ of instantaneous pool dimensions,
dwell windows excised; fusion-depth map over interior columns — the
reporting standard of the square README):

| beam-on mean ± σ | depth (µm) | length (mm) | lateral extent (mm) | volume (mm³) | fusion depth (µm) |
|---|---:|---:|---:|---:|---:|
| zero dwell, baseline | 116 ± 21.3 | 4.06 ± 1.33 | 0.44 ± 0.171 | 0.1128 ± 0.0477 | 111 ± 24.9 |
| zero dwell, **optimized** | **68 ± 8.9** | **2.16 ± 0.60** | 0.30 ± 0.025 | **0.0219 ± 0.0066** | 65 ± 11.6 |
| min dwell, baseline | 87 ± 20.2 | 2.60 ± 1.34 | 0.32 ± 0.041 | 0.0433 ± 0.0230 | 88 ± 18.4 |
| min dwell, **optimized** | **60 ± 11.2** | **1.76 ± 0.66** | 0.29 ± 0.034 | **0.0154 ± 0.0054** | 58 ± 10.1 |

Volume fluctuation drops **7×** (zero dwell) / 4× (dwelled); fusion-depth σ
halves; the mean lands on the setpoint (baselines run up to 80 % too deep at
the apex). Residual σ is larger than the square's — all of it lives at the
apex.

**The apex is where power+σ run out of authority**, and the two policies
fail in opposite directions (segment-end diagnostics; body = lines 0–69,
apex = lines 70+):

- body: w/d CV 0.24/0.34 % (dwelled), 0.30/0.33 % (zero) — square-quality;
- apex, dwelled: CV 1.04/7.83 %, and the 0.07 mm tip line pins at **max
  power (450 W) while still 21 µm short of target depth** — 23 µs of beam
  time cannot develop a pool: the fix is *slowing down* (velocity control);
- apex, zero dwell: CV 11.5/6.2 %, lines 80–83 pin σ at its floor and lines
  84–86 pin at the **minimum power (8 W) with the pool still at/above
  target width** — pure inherited melt: the fix is *waiting* (dwell) or
  slowing down with less power.

Optimization cost: 186 sims / 34 min (dwelled), 223 sims / 67 min (zero),
~1.4–1.7 per segment. Figures (in `figures/`): `triangle_traces_{policy}` (per-line mean +
p5–p95 band of the beam-on pool statistics vs scan line — see the
square README for the design),
`triangle_profiles_{policy}` (the per-line fan collapsing onto target),
and the paper-style `{TriPubZ1,TriMinDwell,TriOptZero,TriOptDwell}_fig1{6,7}`
(Fig. 17 L/D/V-vs-time traces and Fig. 16 solidification-G maps, one per
baseline and optimized case, on fixed axes for before/after comparison).
Results CSVs in `results/`; tracked replays in `cases/case_opt_*` and
`cases/case_mindwell`.
