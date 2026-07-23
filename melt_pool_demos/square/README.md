# 10 mm square raster: continuous P/σ/v uniformity control

The paper's first full raster geometry (`paper/main.tex`, square section):
101 equal-length serpentine tracks at the common study physics — IN718 at
1273 K preheat, 150 W absorbed, σ_xy = 200 µm, σ_z = 10 µm, 3 m/s,
melt isotherm 1610 K, 0.1 mm hatch. The greedy controller
(`../common/greedy_raster.py`) modulates absorbed power, lateral beam σ, and
scan velocity over 1 mm control blocks during continuous scanning, with the
cool turnaround seed and previous-block warm starts selected in the
two-track study.

The complete study — optimization, 50/10/1 µm replay, figure generation,
paper-asset copy, and LaTeX build — is:

```bash
./run_continuous_study.sh
```

with the matched nominal baseline replay from
`python verify_baseline.py --policy zero --resx 50e-6 --resy 10e-6 --resz 1e-6`
(`make square` from the parent directory runs both).

## Single-track calibration (`case_track`)

One 10 mm bead-on-plate line at nominal parameters fixes the optimization
target (the developed single-track pool) via `../common/calibrate.py`:

```bash
python make_case.py --out cases/case_track --name CalTrack --single-track --resz 5e-6 --timestep 5e-5
(cd cases/case_track && $THESIS_BIN ./ParamInput.txt)
python ../common/calibrate.py cases/case_track CalTrack
```

The committed record is `results/calibration/` and `results/target_zero.json`
(w* = 163.84 µm half-span, d* = 63.34 µm on the snapshot grid).

## Results (the paper's square table)

Beam-on melt-pool statistics of the full tracked replays on the 50/10/1 µm
grid (mean ± standard deviation):

| schedule | depth (µm) | full width (µm) | volume (mm³) |
|---|---:|---:|---:|
| baseline | 106.0 ± 12.7 | 388 ± 27 | 0.0798 ± 0.0240 |
| **optimized** | **62.2 ± 2.5** | **323 ± 9.9** | **0.01293 ± 0.00127** |

The committed schedule is `results/greedy_1_zero.json`; the replay records
are `results/fullfield/SqGreedy1X50Y10Z1Zero_*` and
`SqBaselineX50Y10Z1Zero_*`.

## Files

| file | what |
|---|---|
| `run_continuous_study.sh` | the paper pipeline: greedy optimization → 50/10/1 µm replay → figures → paper-figure copy → LaTeX build |
| `optimize_greedy.py` | entry point for the shared greedy P/σ/v controller (`../common/greedy_raster.py`) → `results/greedy_1_zero.json` |
| `verify_greedy.py` | full tracked replay of a greedy schedule at a chosen grid → `results/fullfield/SqGreedy1*` |
| `verify_baseline.py` | matched tracked replay of the nominal baseline → `results/fullfield/SqBaseline*` |
| `make_greedy_figures.py` | per-line traces (`square_traces_zero`) and P/σ/v history (`square_process_parameters`) figures |
| `make_paper_maps.py` | recomposes the surface + centre-section solidification-G maps (`square_maps_zero`) |
| `make_case.py` | self-contained 3DThesis case: square serpentine or `--single-track` calibration line |
| `cases/` | committed case inputs (`case_track`, the two replay cases); raw `Data/` and snapshot cases are gitignored |
| `results/` | compact record: schedule, target, calibration, `fullfield/` replay statistics |
| `figures/` | the three paper figures plus the two `*_fig16_maps.png` map panels they are composed from |
