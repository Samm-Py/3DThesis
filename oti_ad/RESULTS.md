# OTI automatic differentiation in 3DThesis — results

> **Historical prototype results.** These measurements apply to the original
> seven-direction `(x, y, z, Q, k, rho, cp)` build. The current controller
> build carries five directions `(x, y, z, Q, sigma)` and is documented in
> `melt_pool_demos/raster/doc/optimization_notes.pdf`.

## What was demonstrated

First-order automatic differentiation was added to 3DThesis via OTI numbers
(`oti::otinum<7,1,double>`) as a drop-in substitution for the `Real` type alias.
With `-DTHESIS_ENABLE_OTI=ON`, every temperature value carries exact partial
derivatives w.r.t. **7 design variables** — position `x,y,z`, beam power `Q`,
conductivity `k`, density `ρ`, specific heat `cₚ` — propagated through the
existing physics with no algorithm changes. (See `CHANGES.md` for the full
per-file diff.)

## Key results

- **Correctness (validated against finite differences):** AD sensitivities
  match central differences at the peak point —

  | derivative | analytic | finite diff | rel. err |
  |---|---|---|---|
  | dT/dQ   | 804.481  | 804.167  | 0.04% |
  | dT/dk   | −24474.5 | −24436.1 | 0.16% |
  | dT/dρ   | −42.19   | −42.28   | 0.20% |
  | dT/dcₚ  | −523.93  | −525.0   | 0.20% |

  Residuals are finite-difference truncation, not AD error; the
  `dT/dQ·Q = T − T₀` identity holds to ~2×10⁻⁶.

- **Cost:** ~**4.7× runtime overhead** over the plain `double` build, producing
  **7 exact derivatives in a single pass** — cheaper and more accurate than
  finite-differencing the same sensitivities (8–15 re-runs, inexact).

- **Visualization:** full-field maps of T and all 7 sensitivities
  (`oti_fields.png`), plus a **40-frame time animation**
  (`oti_evolution.mp4` / `.gif`) showing the sensitivity fields track the beam
  as it scans across the plate.

## Conditions / settings

`snapshot` example, refined to a **12.5 µm grid** (561×241 nodes), 1200 W
Gaussian beam (10 µm width), Ti-like material (k 26.6, c 600, ρ 7451,
T₀ 1273 K, T_liq 1610 K); **40 snapshots** evenly spaced over the 10.7 ms scan;
OTI build run time ~72 s.

## Caveats

- The **row of dots** along the top track is the example's scan path
  (spot-mode dwells, 0.5 mm apart — `Path.txt`), not a numerical artifact;
  it is independent of grid resolution.
- T has a **near-singular peak** at the beam centre; field plots use robust
  percentile clipping so the surrounding field stays visible. Fully resolving
  the peak would need a grid at or below the 10 µm beam width.

## Headline

> 3DThesis now produces exact, validated parameter sensitivities of the
> temperature field at ~4.7× the cost of a single forward solve, via a
> non-invasive OTI type swap — enabling gradient-based sensitivity analysis
> and optimization of process parameters.
