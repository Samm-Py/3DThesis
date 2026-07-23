# Raster Demo TODO

Current state: the raster demo is a reproducible publication baseline. The core loop
works: 3DThesis snapshots plus OTI sensitivities drive a sequential
power/beam-width optimizer, and the regenerated CSV/PNG/GIF/HTML outputs now
tell a consistent story. MPI also works from the Python scripts through
`THESIS_MPI_NP`, with automatic rank-slice merging.

## What Works

- Power + beam-width optimization suppresses the earlier Pmod/sigma ping-pong.
- The optimizer now uses scaled controls, regularization, trust-region limits,
  backtracking, and best-evaluated-state return.
- `results/baseline.csv`, `results/optimized.csv`, `results/target.json`, and
  the plots in `figures/` are coherent on the 0.5 mm segment grid.
- `raster_lib.py` can run MPI-linked 3DThesis binaries from Python:
  `THESIS_MPI_NP=N python3 run_optimized.py`.
- MPI snapshot slices are merged back into `Data/TestSim.Snapshot.00.csv`
  before measurement routines read the data.

## Known Issues

- The historical `OptimizeRasters` and `oti_ad` directories remain useful
  development records, but `melt_pool_demos/` is now the canonical publication
  workflow.
- MPI execution in WSL/local Intel MPI requires `I_MPI_FABRICS=shm`; the helper
  sets this by default, but it is still environment-specific behavior.
- MPI rank slice files are left in the case `Data/` directory after merging.
  This is useful for inspection but can clutter the case.
- The optimization result is demo-specific: it depends on the chosen dwell,
  0.5 mm control segments, the developed-single-track target, and the current
  melt-pool measurement convention.

## Engineering Follow-Ups

Completed in the publication reorganization:

- explicit `THESIS_ENABLE_MPI=ON/OFF` CMake option;
- isolated `cases/snapcase/` rather than the shared legacy case;
- top-level `Makefile`, dependency list, and publication README;
- solver-free `validate_results.py` checks;
- standardized `results/`, `figures/`, `logs/`, and `cases/` layout;
- repository-wide ignore rules for builds and regenerable raw data.

Remaining engineering work:

1. Make MPI cleanup configurable:
   - keep rank slices for debugging by default or behind `THESIS_KEEP_RANKS=1`;
   - otherwise delete slices after successful merge.
2. Normalize any remaining historical documentation in `OptimizeRasters/` and
   `oti_ad/`:
   - update stale "power-only" comments where the workflow is now power + sigma;
   - document `THESIS_BIN`, `THESIS_BIN_OTI`, `THESIS_MPI_NP`,
     `THESIS_MPIEXEC`, and `THESIS_MPI_FABRICS` in one place.

## Scientific Follow-Ups

1. Test sensitivity to control segment length:
   - 0.25 mm, 0.5 mm, 0.75 mm, 1.0 mm.
   - Quantify when thermal memory causes period-2 ping-pong.
2. Test target choice:
   - first segment,
   - developed single track,
   - center-of-square baseline,
   - user-specified width/depth.
3. Sweep dwell time around the minimal dwell:
   - shorter dwell should expose inherited-liquid infeasibility;
   - longer dwell should reduce inter-track coupling.
4. Compare control sets:
   - power only,
   - sigma only,
   - power + sigma,
   - power + sigma + velocity (velocity sensitivities now exist and are used
     by the shared greedy controller in `../common/greedy_raster.py`; this
     raster demo itself still optimizes power + sigma).
5. Add finite-difference spot checks for `dwidth_dQ`, `ddepth_dQ`,
   `dwidth_dsig`, and `ddepth_dsig` on a few representative segments.
6. Investigate whether the current width/depth/asymmetry measurement convention
   remains robust for more asymmetric or multi-component melt pools.

## Current Reproduction Commands

Serial:

```bash
python3 find_min_dwell.py
python3 run_baseline.py
python3 run_optimized.py
python3 make_plots.py
python3 make_animation.py unoptimized
python3 make_animation.py optimized
```

MPI:

```bash
THESIS_MPI_NP=2 python3 run_baseline.py
THESIS_MPI_NP=2 python3 run_optimized.py
THESIS_MPI_NP=2 python3 make_plots.py
THESIS_MPI_NP=2 python3 make_animation.py unoptimized
THESIS_MPI_NP=2 python3 make_animation.py optimized
```

If Intel MPI fails during `MPI_Init()`, force local shared-memory fabric:

```bash
I_MPI_FABRICS=shm THESIS_MPI_NP=2 python3 run_optimized.py
```
