# Serpentine raster: melt-pool uniformity with power + beam-width control

The simplest raster case: scan left→right, hop up one hatch, scan right→left,
repeat — and modulate the laser **power** and lateral **beam width (σ)** per
segment to keep the melt-pool dimensions consistent while heat accumulates.
(The demo started power-only; the width knob turned out to be necessary — see
"what didn't work" below.)

Process parameters follow the prior optimisation study
(`OptimizeRasters/optimize_uniformity.py`): V = 0.7 m/s, hatch H = 0.15 mm,
square S = 1.0 mm (7 lines), σ₀ = 10 µm, base power 40 W, unoptimised power
60 W (Pmod = 1.5). Melt isotherm 1733 K throughout. Baseline and optimised
runs are measured on 0.5 mm control segments (see below for why).

## Files

| file | what it is |
|---|---|
| `raster_lib.py` | Shared machinery (path building, domains, measurement via the prior-study pipeline). |
| `find_min_dwell.py` | Derives the **minimal turnaround dwell** by bisection → `results/min_dwell.json`. |
| `run_baseline.py` | Unoptimised sweep (constant 60 W, σ 10 µm) → `results/baseline.csv`. |
| `run_optimized.py` | Sequential per-segment 2×2 Newton on (Pmod, σ) with exact OTI derivatives → `results/optimized.csv`, `results/target.json`. |
| `make_animation.py` | Top-down + cross-section animations in `figures/`. |
| `make_plots.py` | Publication plots in `figures/`. |
| `figures/raster_demo.html` | The assembled page: animations on top, dimension plots below. |

Run order: `find_min_dwell.py` → `run_baseline.py` → `run_optimized.py` →
`make_animation.py` / `make_plots.py`. The scripts create and rewrite an
isolated, regenerable case in `cases/snapcase/` and use the plain build
(`build/bin/3DThesis`); the optimiser needs the OTI build
(`build-oti/bin/3DThesis`, override with `THESIS_BIN_OTI`).

MPI runs are supported by the same scripts. Set `THESIS_MPI_NP=N` to launch
3DThesis through `mpirun -n N` (the optimiser then needs the MPI OTI build,
`build-mpi-oti/bin/3DThesis`, via `THESIS_BIN_OTI`); `THESIS_MPIEXEC` can
override the launcher. For local Intel MPI/WSL runs the helper defaults
`I_MPI_FABRICS=shm` because the default OFI fabric can fail during `MPI_Init`.
Snapshot rank slices are merged back into `Data/TestSim.Snapshot.00.csv` and
**sorted lexicographically by (x, y, z)** before the measurement routines read
them — the rank decomposition is not 1D in x (e.g. np=4 splits x and y), so
plain concatenation is not grid order and silently scrambles the reshaped
field grids (measured pools then span the whole domain). Verified: per-segment
dimensions and OTI sensitivities are digit-identical to serial at np = 2/4/8,
and `run_optimized.py` reproduces the serial `optimized.csv` byte-for-byte.

MPI does **not** speed up *this* demo: the local measurement domains solve in
~1 s, so per-invocation `mpirun` launch + snapshot merge overhead dominates
(optimisation wall time 65 s serial → 99 s at np=4). It pays off on larger
domains/finer grids — on the 12.5 µm refined snapshot the same MPI OTI build
scales 1.56×/1.98×/2.04× at np = 2/4/8 (`oti_ad/refined_sweep.out`).

## The dwell-time wrinkle

At this hatch spacing the return pass starts right next to the previous
track's still-hot end. If the turnaround is instantaneous, the new track's
pool **merges with still-liquid residue** from its neighbour — the measured
width then contains inherited melt that no control value can remove, so a
fixed width target is infeasible at track starts (the failure mode documented
in `Scan.AddTurnDwell`). Too much dwell and the tracks decouple: nothing left
to optimise. The demo therefore uses the **minimal dwell** — just enough that
the whole field has solidified at the instant each new track starts.

`find_min_dwell.py` bisects on the criterion *max field T < 1733 K at the end
of every turnaround dwell*, evaluated at the worst case (the unoptimised 60 W
power). Result:

    minimal dwell = 1.234 ms  (± 0.025 ms)

The binding turn is the **last** one — the residual peak at the end of the
dwell climbs monotonically with line number (1591, 1661, 1697, 1714, 1724,
1730 K for turns 1…6) as background heat accumulates.

## Unoptimised baseline (constant 60 W)

The pool balloons as heat accumulates, on two scales:

- **line to line:** width (half-span) grows 97 → 135 µm and depth 97 → 124 µm
  from the first line to the last;
- **within each line:** each track starts smaller (fresh, post-dwell ground)
  and swells toward the line end — a sawtooth on top of the drift.

Width CV 9.7 %, depth CV 7.1 %; ranges 38 µm and 27 µm. (With σx = σy = σz
and the mirrored half-space source, isotherms are circular in the
cross-section, so on the first line width-half-span = depth exactly — the
pool is a hemisphere.)

## Choosing the target: it must be a sustainable fixed point

The target is the **developed single-track pool** — the last segment of
line 0, where the first track has reached its steady size on cold ground:
w\* = d\* = 97.5 µm.

A uniformity target has to be a state the controlled process can *sustain*:
reachable from a fresh track start at moderate power AND holdable mid-line
inside the control bounds. We first tried the centre-of-square baseline pool
(124 µm) as "most representative" — it is not: it is a snapshot of the
*unoptimised* heat-soaked process, i.e. representative of the defect, not the
goal. Chasing it forces an energy overshoot at every (fully solidified) line
start — a ~175 W spike whose surplus heat persists as a molten lake that the
next two segments cannot control at all (measured width moved ~1 µm across
the entire power range there) — and the schedule degenerates into a
spike-and-coast limit cycle for any bounds and any knob set.

## Control-segment length: longer than the pool's relaxation time

With 0.25 mm control segments (0.36 ms at 0.7 m/s) even the feasible target
produced a period-2 "ping-pong": alternate segments hit the target with
interior controls, and the segments in between overshot at the power floor.
Cause: to hit a w = d (hemispherical) target the Newton pins σ small, which
makes a strongly **superheated** core; that stored heat keeps melting outward
for ~τ = w²/4α ≈ 0.4 ms *after* the beam moves on, so the next segment's
end-of-segment snapshot catches its predecessor's afterglow no matter what its
own controls do. Pool size has thermal inertia — a causal per-segment
controller cannot regulate on a grid finer than the relaxation time.

With **0.5 mm segments (~2 relaxation times)** each segment settles into its
own quasi-steady pool before it is measured, and the ping-pong vanishes.
(The prior study's default segment size was 0.5 mm.)

## Optimised result (power + beam width)

Sequential per-segment 2×2 Newton, solved in scaled control coordinates with
Tikhonov regularisation, a per-component trust region, and a backtracking
merit check. A trial step is accepted only if it reduces the normalised
width/depth residual, and the segment returns the best evaluated state rather
than the last state. Warm starts come from the same within-line position one
line below (the raster is line-periodic; a line-END control is a poor prior
for the next line's fresh-ground start). Exact residual AND Jacobian come from
OTI-build runs; ~3.5 sims per segment. Bounds: Pmod ∈ [0.05, 3.0],
σ ∈ [5, 60] µm.

| | width CV % | depth CV % | w range (µm) | d range (µm) |
|---|---:|---:|---:|---:|
| unoptimised | 9.72 | 7.14 | 38.3 | 26.7 |
| optimised | **0.71** | **0.59** | **2.3** | **1.9** |

The schedule is smooth and interior: line 0 runs at the nominal (60 W,
σ 10 µm); once neighbour preheat exists the power settles to ~41–48 W with σ
near 9.6–10 µm. The previous spike/coast behaviour disappeared once the
Newton step was scaled, regularised, and forced to pass the merit decrease
check before acceptance.

## Outputs

- `raster_unoptimized.gif` — top-down surface T over the square + y–z
  cross-section at the beam (beam Gaussian sketched; residue decay visible
  during dwells) + running width/depth trace.
- `raster_optimized.gif` — the same views for the optimised run; the power
  schedule is overlaid on the trace panel (right axis), the beam Gaussian
  tracks the current power and σ (live readout), and the pool-size axis
  matches the unoptimised gif so the flattening is directly comparable.
- `raster_dims.png` — width and depth vs distance along the scan, unoptimised
  vs optimised, with the optimised power and σ schedules underneath.
- `raster_overlay.png` — all four dimension traces on a single axes.
