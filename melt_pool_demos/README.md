# Melt-pool uniformity optimization studies

This directory contains the reproducible research workflows accompanying the
OTI-enabled melt-pool control work in 3DThesis. The controller uses exact
first-order temperature sensitivities from one OTI simulation to regulate
melt-pool half-width and depth with per-block absorbed power, lateral beam
width, and scan velocity during continuous serpentine scanning.

The working paper is
[`raster/doc/paper/main.tex`](raster/doc/paper/main.tex); the fuller
mathematical and numerical account is
[`raster/doc/optimization_notes.pdf`](raster/doc/optimization_notes.pdf), with
source in [`optimization_notes.tex`](raster/doc/optimization_notes.tex).
Development plans for controls beyond the paper baseline (adaptive dwell,
trajectory optimization) are kept in [`doc/plans/`](doc/plans/).

## Study progression

| study | purpose | headline result |
|---|---|---|
| [`single_track/`](single_track/) | isolate the effects of power, beam width, and velocity | demonstrates the independent size/shape control authority and calibrates the target pool |
| [`two_tracks/`](two_tracks/) | isolate one serpentine turnaround and neighboring-track interaction | block-length sweep selects 1 mm blocks; tracked depth 69.4 ± 12.4 to 62.5 ± 6.6 µm |
| [`raster/`](raster/) | 1 mm proof of the sequential OTI/Newton controller (power + beam width) | width/depth CV: 9.72/7.14% to 0.71/0.59% |
| [`square/`](square/) | steady 101-line raster at EBM-scale parameters | beam-on depth 106.0 ± 12.7 to 62.2 ± 2.5 µm; pool-volume σ reduced about 19 times |
| [`triangle/`](triangle/) | shrinking-line geometry matching the publication experiment | beam-on depth 118.7 ± 21.2 to 63.7 ± 4.4 µm; pool-volume σ reduced about 29 times |

Machinery shared between the square and triangle studies — scan-path
building, snapshot measurement, the greedy OTI/Newton controller, whole-build
statistics, and the figure suite — lives in [`common/`](common/); its
drivers are run from a study directory and follow the working directory
(`cd square && python optimize_greedy.py --policy zero`). Every study
follows the same layout:

- geometry-specific drivers and study documentation at the study root;
- `cases/` for regenerable 3DThesis inputs and raw solver data;
- `results/` for compact CSV/JSON result tables;
- `figures/` for publication figures;
- `logs/` for regenerable execution logs.

Triangle paper-reproduction artifacts that are supporting validation rather
than controller results live in `triangle/validation/results/`.

## Environment

The workflows require Python 3.9 or newer and the packages in
[`requirements.txt`](requirements.txt):

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The OTI build additionally requires
[`cpp_oti_lib`](https://github.com/Samm-Py/cpp_oti_lib). Its include directory
can be supplied with `-DTHESIS_OTI_INCLUDE_DIR=/path/to/cpp_oti_lib/include`.

## Build and validate

From this directory:

```bash
make build
make build-oti
make validate
```

`make validate` is solver-free. It checks the committed result tables and,
number by number, the beam-on statistics and two-track block-length sweep
reported in the paper against the committed replay records, plus the
existence of every figure the paper includes.

The solver locations can be overridden without editing scripts:

```bash
THESIS_BIN=/path/to/3DThesis \
THESIS_BIN_OTI=/path/to/oti/3DThesis \
make raster
```

MPI is optional and explicit at configure time with
`-DTHESIS_ENABLE_MPI=ON`. Runtime study scripts accept `THESIS_MPI_NP`,
`THESIS_MPIEXEC`, and `THESIS_MPI_FABRICS`.

## Reproduction targets

```bash
make raster       # roughly one minute
make two_tracks   # two-track continuous P/sigma/v case + block-length choice
make square       # baseline replay + full continuous square study + paper build
make triangle     # baseline replay + full continuous triangle study + paper build
make figures      # plots only, using existing compact/full-field results
make paper        # compile the working paper (main.tex)
make notes        # compile the technical notes
```

The square and triangle studies each drive `run_continuous_study.sh` in the
study directory: greedy optimization on 1 mm blocks, the 50/10/1 µm tracked
replay, the figure set, the copy of the three paper PDFs into
`raster/doc/paper/figures/`, and the LaTeX build.

Raw `Data/` directories, build trees, logs, caches, and workflow marker files
are intentionally ignored. Compact result tables, final figures, documentation,
and case-generation code are the publication record.

## Current scientific boundary

Power, beam width, and velocity control the two-track, square, and triangle
geometries well under continuous serpentine scanning; the paper freezes this
as the baseline. The residual error is concentrated at the triangle apex,
where short lines inherit neighbouring heat faster than the continuous
controls can shed it. This motivates the turnaround dwell as an additional
optimized control — sized by the same OTI sensitivity machinery — which is
planned in [`doc/plans/DWELL_DERIVATIVE_PLAN.md`](doc/plans/DWELL_DERIVATIVE_PLAN.md)
and deliberately left out of the publication baseline.
