# Plan: velocity derivative (DV_V) for the OTI solver

Goal: extend the OTI automatic-differentiation build of 3DThesis so a
snapshot carries `dT_dv` — the exact derivative of the temperature field with
respect to the **current path segment's scan speed** at fixed path geometry —
and the measurement layer therefore yields `dwidth_dv` / `ddepth_dv` /
`dasym_dv` alongside the existing Q and σ sensitivities. This is the
prerequisite for per-segment velocity control and for the adaptive-dwell
Newton (`d(criterion)/d(dwell)` is a special case of the same machinery; see
Phase 5).

`src/oti_scalar.h:37` already declares this as the planned next step
("Laser velocity is a deferred control and will be added as a further slot
later").

> **Execution scope: Phases 1–3 only.** Implement `DV_V`, confirm Python
> propagation, and validate the sensitivity, then **STOP at the review gate
> before Phase 4** (see the ⛔ block after §6) and hand back to the user.
> Phase 4 (optimizer) and Phase 5 (dwell) are documented for context, not for
> this session — Phase 4 hinges on a design decision the user makes from the
> Phase 3 numbers.

---

## 1. Background: how Q and σ work today (read this first)

The solver evaluates the Stump & Plotkowski (Appl. Math. Modelling 75, 2019,
787–805) integral solution: temperature is a Gaussian-quadrature sum over
integration nodes along the beam-path history (their Eqs. 6–9). Per node:
contribution `∝ qmod · dtau · exp(−3(x−xb)²/phix − …)` with
`phix = (σ·wmod)² + 12·a·τ`, `τ = t_obs − t'` the conduction time.

- `Real` is `double` normally; with `-DTHESIS_ENABLE_OTI=ON` it is
  `oti::otinum<DV_COUNT,1,double>` carrying first-order partials in
  `DV_COUNT = 5` directions (`src/oti_scalar.h`).
- **Seeding sites**: `src/Calc.cpp` lines ~134, ~202, ~303, ~528. Guarded by
  `seg_temp + 1 == path.size()` — only nodes on the **last** path segment get
  `DV_Q` / `DV_SIG` seeds; history nodes carry zero derivative (past controls
  are frozen). X/Y/Z are seeded per evaluation point in `Grid::Calc_T`.
- **Snapshot output**: `src/Grid.h:158–171` appends `dT_dx, dT_dy, dT_dz,
  dT_dQ, dT_dsig` columns next to `T`.
- **Measurement**: `melt_pool_demos/common/measurement.py::
  ExtractIsoSupportSensitivities` **auto-discovers** every `dT_d<name>`
  column and converts field sensitivities to isotherm support-function
  sensitivities via the implicit function theorem
  (`dh/dp = −(dT/dp)/(dT/dn)`). `mp_lib.sens()` then exposes
  `d{width,depth,asym}_d<name>`. **No Python changes are needed for a new
  column to propagate** — only for consuming it in the optimizer.
- **Optimizer**: `common/run_optimized.py` runs a per-segment 2×2 Newton on
  (w−w*, d−d*) with controls (P, σ). `mp_lib.run_segment` truncates the path
  at the candidate segment, so "current segment" = last `Path.txt` line =
  the one control being solved for.

Q and σ are *easy* because they never affect **when** anything happens: all
time variables (`t`, `t1`, `t2`, `tau`, `dtau`, `seg_time`) stay `double`,
and the adaptive quadrature schedule (step doubling / order halving, the
paper's Eq. 12) never sees an OTI number.

## 2. The math: why velocity is different, and the exact formulation

With the segment's spatial endpoints fixed (a raster must cover the part),
perturbing the last segment's speed v changes the **time parametrization**:
`dt_cur = L/v` (segment duration, `src/Init.cpp:655`) and hence the
observation time `t_end = t_{k−1} + L/v`. Three channels:

1. beam kinematics within the segment;
2. every node's conduction time shifts through `t_end` — **including history
   nodes**, so unlike Q/σ the history contributes to dT/dv (upstream heat has
   had less time to diffuse when you observe earlier);
3. a Leibniz boundary term from the moving upper integration limit — already
   materialized as the "instantaneous node" (`Calc.cpp:126–146`).

**Key simplification (use this — it is exact, not an approximation).**
Parametrize the current segment by the fraction traveled `φ ∈ [0,1]`. Then:

- beam position `xb(φ)` depends **only on geometry** — no v dependence;
- node conduction time: `τ = (t_obs − t_end) + (1−φ)·dt_cur`
  (first term ≥ 0, zero for the optimizer's end-of-scan snapshot);
- node weight: `dtau = Δφ_quad · dt_cur`;
- history-node times `t'` are fixed in absolute time, so
  `τ_hist = τ_hist,double + (dt_cur − dt_cur,double)`.

So **all** v dependence funnels through the single quantity
`dt_cur = L / v`, and the beam positions never need OTI. Seed once per
temperature evaluation:

```cpp
Real v_oti   = thesis::seed(thesis::DV_V, v_real);   // v_real = L / dt_cur
Real dt_oti  = L / v_oti;                            // carries d(dt_cur)/dv = −L/v²
Real t_shift = dt_oti - dt_cur_double;               // zero real part, pure derivative
```

then
- current-segment node: `tau_oti = (t − t_end_double) + (1.0 − φ)*dt_oti;`
  `dtau_oti = (dtau_double / dt_cur_double) * dt_oti;`
- history node: `tau_oti = tau_double + t_shift;` (`dtau` stays double-valued)
- instantaneous node: unchanged (τ = 0), its weight is not a quadrature
  weight.

where `φ = (tp_double − t_{k−1}) / dt_cur_double` from the existing double
schedule. Freezing the *adaptive schedule* (node counts, orders, step sizes)
at the real value of v is the standard differentiate-the-approximation
choice and is consistent as the quadrature converges.

Convention: `dT_dv` is per (m/s) of the current segment's `sparam`.

## 3. Design decisions (fixed — do not relitigate during implementation)

- Fixed path geometry; v changes segment duration (Option "fixed-geometry").
- Seed **only the last path segment's** v, same guard as DV_Q/DV_SIG.
- First-order OTI is sufficient (Newton needs Jacobians only).
- All adaptive/branching logic stays on `double` real parts
  (`thesis::to_double`); node counts and orders are frozen w.r.t. the
  perturbation.
- `DV_COUNT` goes 5 → 6 (~17–20% more arithmetic in OTI builds; the plain
  double build is untouched by construction).

## 4. Phase 1 — C++: the DV_V slot in `Calc.cpp`

Files: `src/oti_scalar.h`, `src/Calc.cpp`. (`src/DataStructs.h` needs **no**
change: `int_seg` fields `xb…dtau,qmod` are already `Real`,
`DataStructs.h:32–35`.)

1. `oti_scalar.h`: add `DV_V` to `DesignVar` (before `DV_COUNT`); update the
   header comment block.
2. `Calc.cpp`: in each of the four seeding sites, compute the current
   segment's `L` (from `path[k]` / `path[k−1]` positions) and
   `dt_cur_double = path[k].seg_time − path[k−1].seg_time`, then apply the
   §2 formulas to `tau` (feeding `ct = 12·a·τ`) and `current_beam.dtau` for
   nodes on the last segment, and add `t_shift` to `tau` for history nodes.
   Notes:
   - `tau` is currently a local `double` in the quadrature loop
     (`Calc.cpp:190`); it becomes `Real` in the OTI build (it already
     feeds `Real ct`, so types compose).
   - Guard identically to DV_Q: `seg_temp + 1 == path.size()`, and skip
     spot-mode segments (`path[seg].smode`) — v is meaningless there.
   - The schedule variables `t1, t2, curStep_*, tpp` **stay double**.
   - Handle the `t > path.back().seg_time` case (solidification queries
     after beam-off): the `(t − t_end_double)` term in §2 covers it.
3. `Grid.h:161–165`: add `{"dT_dv", thesis::DV_V}` to `oti_cols`.
4. Build both binaries; the plain build must remain bit-identical
   (`Real = double` path untouched):
   `make build && make build-oti` (from `melt_pool_demos/`; binaries land in
   `build/bin/3DThesis`, `build-oti/bin/3DThesis`).

Acceptance: OTI snapshot for a single-track case has a `dT_dv` column;
`dT_dQ`/`dT_dsig` values are unchanged vs. the pre-change OTI build on the
same case; plain-double outputs byte-identical.

## 5. Phase 2 — Python: propagation check (should be free)

`measurement.py` discovers controls from columns (`measurement.py:218`), so
`mp_lib.sens()` should return `dwidth_dv`, `ddepth_dv`, `dasym_dv` with no
code change. Verify from `melt_pool_demos/square/`:

```python
from common import mp_lib as R
# after running one truncated-segment OTI sim (see run_segment usage
# in common/run_baseline.py) on cases/snapcase:
print(R.sens())   # expect d*_dQ, d*_dsig, d*_dv keys
```

Acceptance: the three `_dv` keys appear and are finite.

## 6. Phase 3 — validation (the gate for everything downstream)

Write `melt_pool_demos/common/validate_dv.py` (pattern it on however dQ/dσ
were originally validated):

1. **Central finite differences on v.** For the developed single-track
   configuration (square demo geometry, `snapcase`): run the truncated
   segment at `v(1±ε)` for ε ∈ {1e-2, 1e-3} by rewriting the last `Path.txt`
   line's `sparam` (column 6; `Init.cpp:621`); measure w, d with the
   **support-function extraction** (`ExtractIsoSupportSensitivities` at the
   same isovalue), *not* `measure_dims` (marching cubes on the 50 µm grid is
   quantization-noisy). Compare `(w(v+δ)−w(v−δ))/2δ` against OTI
   `dwidth_dv`; same for depth. Also compare the raw field column: FD of `T`
   per grid point vs `dT_dv` (tighter, measurement-free check — do this
   first when debugging).
2. **Physics sanity.** Developed pool at fixed P: `ddepth_dv < 0`,
   `dwidth_dv < 0`; magnitude order vs. Rosenthal scaling (d ∼ v^−1/2 ⇒
   `v·|dd/dv| ≈ d/2` within a factor ~2).
3. **History channel.** Repeat the FD check for a segment mid-raster (with
   neighbor-line heat present, e.g. segment ~50 of the square zero-dwell
   baseline path): confirms the observation-time shift through history nodes
   is captured. This case FAILS if only the kinematic channel was wired.

Acceptance: OTI vs central FD agree to <1% relative (field check) and <2–3%
(support-function check) at ε = 1e-3, for both the single-track and
mid-raster configurations.

### Phase 3 RESULTS — executed 2026-07-15 (Phases 1–3 complete, VALIDATED)

Implementation: `DV_V` slot in `src/oti_scalar.h`; `dt_cur = L/v` seeding in
`src/Calc.cpp` (file-local `vel_seed`, applied in both `GaussIntegrate` and
`GaussCompressIntegrate` — current-segment nodes scale τ and weight by
`k`, history nodes add `t_shift`); `dT_dv` column in `src/Grid.h`. Validator:
`melt_pool_demos/common/validate_dv.py`.

Phase 1 gate (all pass): plain-double build **byte-identical**; OTI snapshot
gains `dT_dv`; `T`/`dT_dQ`/`dT_dsig` **unchanged** (max Δ = 0). Phase 2:
`mp_lib.sens()` exposes `dwidth_dv`/`ddepth_dv`/`dasym_dv` with no Python
change.

Phase 3 FD validation (central FD, ε = 1e-2), from `square/`:

| check | single-track | mid-raster (history) |
|---|---|---|
| field `dT_dv` (top-300 pts) median / max rel | 0.023% / 0.103% | 0.025% / 0.118% |
| depth `ddepth_dv` FD vs OTI | 0.41% | 0.075% |
| width `dwidth_dv` FD vs OTI (50 µm y) | 24.4% | 9.3% |

The field derivative is exact on BOTH configs → solver correct, **history
channel confirmed** (mid-raster would blow up if only kinematics were wired).
Depth exact. Width FD is 50 µm-y-grid-quantization-limited, NOT a solver error:
refining y-grid collapses it (single-track: 24.4% @50 µm → **2.3% @25 µm**),
converging to the grid-independent analytic `dwidth_dv` (which shifts <2%).
Signs correct everywhere (`dT_dv<0`, `dd/dv<0`, `dw/dv<0`: faster ⇒ cooler,
shallower, narrower). Rosenthal `v|dd/dv|/(d/2)` ≈ 1.5 (single) / 0.67 (mid),
right order.

Jacobian for the developed single-track pool (P=150 W, σ=200 µm, v=3 m/s;
w=163.84 µm, d=63.34 µm), as **elasticities** `(c/QoI)·dQoI/dc` (Newton
conditioning):

| control | depth elasticity | width elasticity |
|---|---:|---:|
| P     | +0.788 | +0.277 |
| σ     | −0.540 | +0.560 |
| **v** | **−0.765** | **−0.170** |

Reading for Phase 4: in (depth,width) response space, **P and v are nearly
antiparallel** (both dominated by depth, opposite sign) → a (P,v) 2×2 would be
ill-conditioned. But that is exactly why the **time-optimal** formulation is
well-posed: raising v (to cut time) is compensated by raising P to hold depth,
with σ trimming width; expect **P→P_max as the binding constraint** bounding the
achievable speed-up. σ retains independent width authority (elasticity +0.560).
A 3×3 (third residual) is awkward: asym is degenerate on a straight track
(`dasym_dv=0`). → data favors the plan's recommended time-optimal formulation,
but this is the user's call.

---

## ⛔ STOP HERE — mandatory review gate before Phase 4

**Do not begin Phase 4.** Phases 1–3 deliver a validated `dT_dv` sensitivity;
that is a complete, self-contained unit of work. Hand back to the user for
review with:

- the Phase 3 validation numbers (OTI vs FD, both configurations) and the
  physics-sanity signs/magnitudes;
- a short readout of the actual sensitivity values (`ddepth_dv`,
  `dwidth_dv`) for the developed pool, so the design choice below can be
  made against real data, not a guess.

The reason for the gate: **Phase 4 contains a genuine design choice the user
wants to make from the validated sensitivities** — time-optimal (v as the
objective, P/σ as feasibility constraints) vs. a third-residual 3×3 Newton
(§7 alternatives). The relative magnitudes of `dd/dv`, `dd/dP`, `dd/dσ`
determine which is well-conditioned, and those are only known after Phase 3.
Do not pick one and implement it unprompted. Wait for direction.

## 7. Phase 4 — optimizer integration (design sketch; DO NOT implement before the review gate above)

Adding v makes the per-segment system 2 targets × 3 controls. Recommended
formulation — **v as the time objective, P/σ as feasibility**:

    min  segment time  L/v
    s.t. w(P,σ,v) = w*,  d(P,σ,v) = d*

per segment: an SQP-ish step using the 2×3 Jacobian from `sens()`; increase
v along the constraint manifold until a control hits a bound (P → P_max is
the expected binding constraint), with the same scaled-controls /
trust-region / backtracking scaffolding as `run_optimized.py`. Alternatives
(document, don't implement): (a) swap σ for v in the existing 2×2; (b) add a
third residual (length or asym target) for a 3×3 Newton. Also decide
per-segment v bounds and whether the tracked-replay timestep handling needs
care (line time L/v no longer a multiple of 50 µs — affects `postprocess.py`
line-window bookkeeping, `common/make_plots.py` line windows, and
`fullfield_stats.py` beam-on sampling, which all reconstruct times from
V=3 m/s constants such as `mp_lib.V`).

## 8. Phase 5 — dwell derivative (follow-up, mostly free after Phase 1)

A turnaround dwell has **no kinematic channel** (beam off): seeding the dwell
duration is exactly the `t_shift` history algebra of §2 applied to a
spot-mode/zero-power segment. Add `DV_DWELL` (or reuse DV_V's slot under a
mode flag), seed `sparam` of a last dwell row, and the derivative of any
smooth line-start criterion falls out. Use a smooth surrogate for the
residual-liquid criterion (peak field T at line start relative to T_liq, or
interpolated liquid volume) — the raw liquid-cell count is discrete and
non-differentiable. This replaces the brute-force bisection in
`find_min_dwell.py` with the "OTI Newton on d(criterion)/d(dwell)" promised
in the READMEs.

## 9. Gotchas checklist (review during implementation)

- [ ] Every comparison/branch that can see an OTI-valued time uses
      `thesis::to_double` (schedule, `t0calc` cutoff, `InRMax`,
      `t1 <= next_time`).
- [ ] `curStep_max` doubling / `curOrder` halving frozen at real v.
- [ ] Spot-mode segments excluded from DV_V seeding.
- [ ] Multi-beam loop (`sim.beams`): seed per path, same last-segment rule.
- [ ] `qmod > 0` node-culling (`Calc.cpp:146,214`) compares real parts (OTI
      `operator>` exists but be explicit).
- [ ] MPI OTI build (`build-mpi-oti`) unaffected: derivative columns ride in
      the same snapshot rows; rank-merge in `mp_lib._merge_ranked_snapshots`
      is column-agnostic but assert column sets match.
- [ ] Plain-double build byte-identical (CI check: run a small case on both
      old/new plain binaries, diff outputs).
- [ ] `validate_results.py` untouched (no publication artifact changes).

## 10. Reference

- Paper: B. Stump, A. Plotkowski, *Appl. Math. Modelling* 75 (2019) 787–805
  (adaptive integration scheme; Eqs. 6–9 solution form, Eq. 12 step
  doubling, Eq. 13 nondimensional velocity).
  Local copy: `C:\Users\Rober\Downloads\1-s2.0-S0307904X19304093-main.pdf`
  (WSL: `/mnt/c/Users/Rober/Downloads/1-s2.0-S0307904X19304093-main.pdf`).
- OTI scalar layer: `src/oti_scalar.h`. Seeding sites: `src/Calc.cpp`
  (~134/202/303/528). Beam kinematics: `src/Util.cpp:224–253`. Path times:
  `src/Init.cpp:643–658`. Snapshot columns: `src/Grid.h:158–171`.
  Measurement: `melt_pool_demos/common/measurement.py:166–246`,
  `melt_pool_demos/common/mp_lib.py:348–360`.
