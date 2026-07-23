# Automatic differentiation in 3dThesis via OTI numbers — change report

> **Historical prototype record.** This document describes the original
> seven-direction material-sensitivity prototype at commit `357f006`. The
> current publication workflow uses `otinum<5,1,double>` with directions
> `(x, y, z, Q, sigma)` and seeds `Q` and `sigma` on the current path segment.
> See `melt_pool_demos/raster/doc/optimization_notes.pdf` for the current
> design. The validation and benchmarking history below is retained for
> provenance.

This documents exactly what was changed to add optional first-order automatic
differentiation (AD) to 3dThesis, using order-truncated-imaginary (OTI) numbers
from [cpp_oti_lib](https://github.com/Samm-Py/cpp_oti_lib).

## Summary

- The physics path is written against one type alias, **`Real`**. By default
  `Real = double` and the build is **byte-for-byte identical to upstream**.
- With `-DTHESIS_ENABLE_OTI=ON`, `Real` becomes `oti::otinum<7,1,double>`: a
  value plus first-order partials w.r.t. 7 **design variables** —
  evaluation point `x,y,z`, beam power `Q`, conductivity `kon`, density `rho`,
  specific heat `cps`. (Diffusivity `a = kon/(rho*cps)` is derived and inherits
  their derivatives.)
- Temperature comes back carrying `dT/d(each design variable)`; in OTI builds
  these are written as extra CSV columns `dT_dx … dT_dcps`.
- Total change to existing code: **156 insertions / 88 deletions across 11
  files**, plus one new header and the `oti_ad/` tooling. No physics algorithm
  was modified — the change is essentially a type substitution plus seeding.

## How to build

```bash
# default (double) — unchanged from upstream
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# AD build
cmake -B build-oti -DCMAKE_BUILD_TYPE=Release -DTHESIS_ENABLE_OTI=ON \
      -DTHESIS_OTI_INCLUDE_DIR=/path/to/cpp_oti_lib/include
cmake --build build-oti -j
```

`THESIS_OTI_INCLUDE_DIR` defaults to `../../cpp_oti_lib/include` relative to the
repo. The OTI build requires C++17 (the project standard was bumped from C++11);
the default build is unaffected.

## The design in one idea

A value can only carry a derivative if every operation from the input to that
value runs in the AD type. So we change the *type* of the fields on the
parameter→temperature path and let the existing arithmetic (all operators and
`exp`/`log`/… are overloaded for OTI) propagate the derivatives untouched.
Conversions back to `double` (`thesis::to_double`) appear only at genuine
boundaries: integer indices, file I/O, `double`-typed control/storage. These are
inherent to a targeted swap and explicit by design (an implicit otinum→double
conversion would silently drop derivatives).

## New files

- **`src/oti_scalar.h`** — defines `Real`, the `DesignVar` enum, and the
  `seed()` / `to_double()` / `deriv()` helpers. In the default build these
  compile away to no-ops (`Real = double`).
- **`oti_ad/`** — validation, benchmark, and visualization tooling (below).

## Per-file changes to existing code

| File | +/− | What changed |
|------|-----|--------------|
| `src/CMakeLists.txt` | 15/1 | `THESIS_ENABLE_OTI` option, C++17, cpp_oti_lib include path |
| `src/DataStructs.h` | 30/13 | `int_seg`, `Nodes`, `Material` (kon/rho/cps/a/T_liq/T_init), `Beam::q` fields → `Real`; include `oti_scalar.h` |
| `src/Init.h` / `Init.cpp` | 5+20 | guarded `SetValues(Real&,…)` overload; seed the 7 design variables where parsed |
| `src/Grid.h` | 32/4 | `T` storage → `Real*`; `get_T` returns real part; new `get_T_deriv(p,dv)`; OTI-only `dT_d*` output columns |
| `src/Grid.cpp` | 84/.. | kernel `Calc_T` seeds x/y/z and computes in `Real` (returns the real part, so its ~25 call sites are untouched); solidification kernels stay `double`, reading nodes via `to_double` |
| `src/Calc.cpp` | 14/.. | `beta`/`ct` → `Real` (carry power/diffusivity into nodes); `to_double` where a node value feeds timestep/compression control |
| `src/Util.cpp` / `Util.h` | 14+2 | `to_double` in `Calc_NonD_dt`, `Calc_RMax`, `t0calc`; `InRMax(Real,Real)` so its comparison-only body needs no conversions |
| `src/Melt.cpp` | 24/.. | `to_double` inside grid-index math and perimeter-tracking lambda arguments |
| `src/Run.cpp` | 4/.. | `to_double` when filling a `vector<double>` from `T_init` |

Notably **unchanged**: the RRDF writers' `static_cast<double>(material.T_liq)`
compile as-is, because cpp_oti_lib provides an explicit `operator double`.

## How to read the derivatives

- **CSV (OTI build):** temperature output gains columns
  `dT_dx,dT_dy,dT_dz,dT_dQ,dT_dkon,dT_drho,dT_dcps`.
- **API:** `Grid::get_T_deriv(p, thesis::DV_Q)` etc. returns `dT/d(var)` at point
  `p` (returns 0 in a double build).

## Validation (`oti_ad/validate_derivatives.py`)

Each derivative is checked against finite differences:

- **Parameters Q, kon, rho, cps** — perturb the input value ±0.01%, re-run the
  *double* build, central-difference T. Agreement at the peak point:

  | derivative | analytic | finite diff | rel. err |
  |---|---|---|---|
  | dT/dQ   | 804.481  | 804.167  | 0.04% |
  | dT/dkon | −24474.5 | −24436.1 | 0.16% |
  | dT/drho | −42.19   | −42.28   | 0.20% |
  | dT/dcps | −523.93  | −525.0   | 0.20% |

  (Differences are finite-difference truncation, not AD error — the
  `dT/dQ·Q == T−T_init` identity holds to ~2e-6.)

- **Position x, y** — central differences across grid neighbours; agree to
  finite-difference-truncation level (limited by the 25 µm grid near the sharp
  peak). `z` has a single grid layer in the snapshot example and is skipped.

## Runtime (`oti_ad/benchmark.py`, Release, snapshot example)

| build | median wall time |
|---|---|
| double | 0.46 s |
| OTI (7 derivatives) | 2.18 s |

**Overhead ≈ 4.7×** while producing 7 exact derivatives — cheaper than finite
differencing the same sensitivities, which needs 8–15 re-runs and is inexact.
(Note: optimization matters a lot — an unoptimized build shows ~22×.)

## Visualization (`oti_ad/visualize.py`)

Renders T and all sensitivity fields from one snapshot CSV; see
`oti_ad/oti_fields.png` (regenerated from the final, full-raster snapshot).
T has a near-singular peak at the beam centre, so it is clipped to the 99.5th
percentile or the surrounding field washes out to black.

## Animation across scan time (`oti_ad/animate.py`)

The snapshot mode emits one CSV per `ScanFracs` entry, so a finer list gives a
movie of the fields as the beam travels. Regenerated the snapshot example at
higher fidelity — grid `Res` refined from **25 µm → 12.5 µm** (561×241 nodes,
closer to the 10 µm beam so the melt-pool peak is properly resolved) and
`ScanFracs 2.5,5,…,100` (**40 frames**, one OTI run, ~72 s) — then animated all
7 fields:

```bash
# 12.5 µm grid, 40 frames in a single run
sed -i 's/Res\t2.5e-5/Res\t1.25e-5/' Domain.txt          # X and Y
sed -i 's/^\tScanFracs.*/\tScanFracs\t2.5,5,7.5,...,100/' Mode.txt
build-oti/bin/3DThesis ./ParamInput.txt
python3 oti_ad/animate.py 'examples/snapshot/Data/snapshot.Snapshot.*.csv' \
        --out oti_ad/oti_evolution --fps 8 --frames-dir oti_ad/frames
```

Color limits are computed once over the whole sequence and held fixed across
frames, so the animation shows real evolution rather than per-frame autoscaling
flicker. Outputs:

- **`oti_ad/oti_evolution.mp4`** / **`.gif`** — T and the 7 sensitivities as the
  hot spot rasters across the plate; sensitivity dipoles travel with the beam,
  and by ~95% the full raster of melt tracks is visible in every field.
- **`oti_ad/frames/frame_NNN.png`** — the same panels as individual stills.
- **`oti_ad/anim_data/`** — the 40 source snapshot CSVs (regenerable; ~476 MB,
  so safe to delete once the MP4/GIF/frames are rendered).

## MPI + OTI

3DThesis's MPI mode is a **2-D Cartesian domain decomposition** of the x–y grid:
each rank rewrites its `sim.domain` to a local sub-block (`MpiStructs.h`,
`makeLocalBounds`) and solves it independently. The semi-analytic moving-source
solution computes every point from the beam-path history alone, so **no field
data is communicated** between ranks (there is no `MPI_Send/Recv/Gather/Reduce`
anywhere). Consequently OTI is safe under MPI: no `otinum` is ever serialized
into an `MPI_DOUBLE` buffer (which would silently drop its 7 derivative
components), and derivatives propagate purely locally.

**Verified empirically.** Built an MPI+OTI binary and ran the 12.5 µm snapshot
example on 4 ranks; merging the per-rank slices gives a result that is
**bit-for-bit identical** to the serial OTI run across all 11 columns (T and all
7 derivatives, max abs diff `0.0`). The no-overlap decomposition tiles the grid
exactly — 4 slices sum to the full 135 201 nodes with no duplicates or gaps.

```bash
cmake -B build-mpi-oti -DCMAKE_BUILD_TYPE=Release -DTHESIS_ENABLE_OTI=ON \
      -DTHESIS_OTI_INCLUDE_DIR=/path/to/cpp_oti_lib/include \
      -DCMAKE_CXX_COMPILER=mpicxx        # Thesis_ENABLE_MPI auto-follows MPI_FOUND
cmake --build build-mpi-oti -j
mpirun -n 4 build-mpi-oti/bin/3DThesis ./ParamInput.txt
python3 oti_ad/merge_ranks.py Data            # stitch *.Snapshot.NN.R.csv -> .NN.csv
python3 oti_ad/animate.py 'Data/snapshot.Snapshot.*.csv' --out oti_ad/oti_evolution
```

**One source fix was required.** Snapshot output (unlike Solidification/T_hist/
RDF, which `Main.cpp` already rank-tags) wrote `<name>.Snapshot.NN.csv` with no
rank suffix into a shared `Data/`, so multiple ranks clobbered one file. Added a
`FileNames::rank_suffix` (`""` in serial, `".<rank>"` under MPI; set in
`ThesisMPI::setPrint`) and appended it to the two snapshot `Output` calls in
`Run.cpp`. Serial builds are unaffected — filenames and output are bit-for-bit
unchanged. `oti_ad/merge_ranks.py` stitches the per-rank slices back into
full-grid snapshot CSVs that `visualize.py` / `animate.py` consume directly.

Performance note: the solve is embarrassingly parallel per grid point with zero
communication, so MPI scales near-linearly across nodes (useful for large/3-D
grids); on one node OpenMP (`MaxThreads`) already parallelizes the same loop.
MeltPool Statistics still explicitly refuses MPI (`Run.cpp`).
