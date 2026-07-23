# 10 mm equilateral triangle raster: continuous P/σ/v uniformity control

The paper's shrinking-line geometry (`paper/main.tex`, triangle section):
87 serpentine tracks whose lengths decrease towards the apex, at the common
study physics — IN718 at 1273 K preheat, 150 W absorbed, σ_xy = 200 µm,
σ_z = 10 µm, 3 m/s, melt isotherm 1610 K, 0.1 mm hatch. The greedy
controller (`../common/greedy_raster.py`) modulates absorbed power, lateral
beam σ, and scan velocity over nominally 1 mm control blocks (balanced on
each shrinking line so no short remainder is left), with the cool turnaround
seed and previous-block warm starts selected in the two-track study.

The complete study — optimization, 50/10/1 µm replay, figure generation,
paper-asset copy, and LaTeX build — is:

```bash
./run_continuous_study.sh
```

with the matched nominal baseline replay from
`python verify_baseline.py --policy zero --resx 50e-6 --resy 10e-6 --resz 1e-6`
(`make triangle` from the parent directory runs both).

The optimization target is the developed single-track pool calibrated in the
square study (`../square/results/calibration/`); the committed triangle copy
is `results/target_zero.json`.

## Results (the paper's triangle table)

Beam-on melt-pool statistics of the full tracked replays on the 50/10/1 µm
grid (mean ± standard deviation):

| schedule | depth (µm) | full width (µm) | volume (mm³) |
|---|---:|---:|---:|
| baseline | 118.7 ± 21.2 | 484 ± 170 | 0.11274 ± 0.04715 |
| **optimized** | **63.7 ± 4.4** | **329 ± 15** | **0.01176 ± 0.00160** |

The committed schedule is `results/greedy_1_zero.json`; the replay records
are `results/fullfield/TriGreedy1X50Y10Z1Zero_*` and
`TriBaselineX50Y10Z1Zero_*`.

## Files

| file | what |
|---|---|
| `run_continuous_study.sh` | the paper pipeline: greedy optimization → 50/10/1 µm replay → figures → paper-figure copy → LaTeX build |
| `optimize_greedy.py` | entry point for the shared greedy P/σ/v controller (`../common/greedy_raster.py`) → `results/greedy_1_zero.json` |
| `verify_greedy.py` | full tracked replay of a greedy schedule at a chosen grid → `results/fullfield/TriGreedy1*` |
| `verify_baseline.py` | matched tracked replay of the nominal baseline → `results/fullfield/TriBaseline*` |
| `make_greedy_figures.py` | per-line traces (`triangle_traces_zero`) and P/σ/v history (`triangle_process_parameters`) figures |
| `make_paper_maps.py` | recomposes the surface + centre-section solidification-G maps (`triangle_maps_zero`) |
| `make_case.py` | self-contained 3DThesis triangle case (`--side`, `--hatch`, …) |
| `cases/` | committed case inputs for the two replays; raw `Data/` and snapshot cases are gitignored |
| `results/` | compact record: schedule, target, `fullfield/` replay statistics |
| `figures/` | the three paper figures plus the two `*_fig16_maps.png` map panels they are composed from |
