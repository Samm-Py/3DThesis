# Melt-pool uniformity optimization: paper record

This directory contains the paper accompanying the OTI-enabled melt-pool
control work in 3DThesis, together with exactly the workflows and committed
results that reproduce it. The controller uses exact first-order temperature
sensitivities from one OTI simulation to regulate melt-pool half-width and
depth with per-block absorbed power, lateral beam width, and scan velocity
during continuous serpentine scanning.

The paper is [`paper/main.tex`](paper/main.tex); its figures live in
[`paper/figures/`](paper/figures/).

## Studies

| study | role in the paper |
|---|---|
| [`two_tracks/`](two_tracks/) | smallest geometry with a turnaround and neighbouring-track interaction; block-length sweep selects 1 mm blocks (Table 2, Figures 2–4) |
| [`square/`](square/) | 101-line raster at constant line length (Table 3, Figures 5–7) |
| [`triangle/`](triangle/) | 87-line raster with shrinking lines (Table 4, Figures 8–10) |

Machinery shared between the studies — scan-path building, snapshot
measurement, the greedy OTI/Newton controller, whole-build statistics, and
the figure suite — lives in [`common/`](common/); its drivers run from a
study directory and follow the working directory. `common/` also carries the
OTI derivative validators (`verify_analytic_derivs.py`,
`verify_analytic_moving.py`, `validate_dv.py`, `validate_ddwell.py`)
backing the paper's exact-sensitivity claim. Every study
follows the same layout:

- geometry-specific drivers and study documentation at the study root;
- `cases/` for regenerable 3DThesis inputs and raw solver data;
- `results/` for compact CSV/JSON result tables;
- `figures/` for publication figures.

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

`make validate` is solver-free. It checks, number by number, the beam-on
statistics and two-track block-length sweep reported in the paper against
the committed replay records, plus the existence of every figure the paper
includes.

The solver locations can be overridden without editing scripts:

```bash
THESIS_BIN=/path/to/3DThesis \
THESIS_BIN_OTI=/path/to/oti/3DThesis \
make square
```

MPI is optional and explicit at configure time with
`-DTHESIS_ENABLE_MPI=ON`. Runtime study scripts accept `THESIS_MPI_NP`,
`THESIS_MPIEXEC`, and `THESIS_MPI_FABRICS`.

## Reproduction targets

```bash
make two_tracks   # two-track continuous P/sigma/v case + block-length choice
make square       # baseline replay + full continuous square study + paper build
make triangle     # baseline replay + full continuous triangle study + paper build
make figures      # plots only, using existing compact/full-field results
make paper        # compile the paper (main.tex)
```

The square and triangle studies each drive `run_continuous_study.sh` in the
study directory: greedy optimization on 1 mm blocks, the 50/10/1 µm tracked
replay, the figure set, the copy of the three paper PDFs into
`paper/figures/`, and the LaTeX build.

Raw `Data/` directories, build trees, logs, caches, and snapshot cases are
intentionally ignored. Compact result tables, final figures, the paper, and
case-generation code are the publication record.
