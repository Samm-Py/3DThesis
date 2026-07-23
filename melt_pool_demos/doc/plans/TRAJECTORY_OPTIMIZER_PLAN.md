# Plan: coupled velocity-trajectory optimizer (single track)

Goal: replace the (diverging) greedy per-segment controller on the single
track with a **coupled trajectory optimization** over the whole velocity
profile `v = (v_1 … v_N)`, using a banded lower-triangular influence matrix
`J_ij = ∂d_i/∂v_j` built from generalized OTI seeding ("seed segment j,
observe at segment i"). Deliverable: an optimized lead-in velocity schedule
that flattens the single track's startup depth transient, verified by tracked
replay (fused-depth CV), plus the write-up of the posedness finding.

Prerequisite state (all DONE, 2026-07-15/16): validated `dT_dv`
(`VELOCITY_DERIVATIVE_PLAN.md` Phases 1–3); per-segment speed plumbing in
`common/mp_lib.py` (`set_pmods`/`run_segment` `vels`, path col 5);
`run_optimized.py` 2×3 gated by `VEL_CONTROL`; `SEG = 0.25 mm` for
single_track; the **divergent 2×3 run** and its Jacobian-collapse diagnosis
(see §1).

> **Execution scope: Phases 1–3 only.** Generalize the seeding, validate it,
> and measure the influence table, then **STOP at the review gate before
> Phase 4** (see the ⛔ block after §7). Phase 4 (the optimizer itself) has
> tuning choices (regularizer weights, band width, null-space anchoring) the
> user wants to make from the measured influence table.

---

## 1. Background: what the divergence proved (read this first)

The greedy per-segment Newton solves, at each segment i, "choose u_i so the
pool measured at the END of segment i hits target." Measured facts
(2026-07-16, single track, 0.25 mm segments):

| segment measured | history behind it | ∂d/∂Pmod (µm) | ∂d/∂σ elast. | ∂d/∂v elast. |
|---|---|---:|---:|---:|
| seg 2 (cold start) | none   | +15.4 | −0.59 | −0.66 |
| seg 4 (0.5 mm in)  | 0.5 mm | +0.01 | ~0.00 | −0.065 |
| seg 10 (developed) | 2 mm   | ~0.00 | ~0.00 | +0.004 |

The endpoint depth is the **trailing-pool max** — heat deposited by the
previous ~8 segments (~2 mm pool footprint). Once history exists, EVERY
control's authority over the endpoint collapses (not a velocity artifact:
P and σ collapse harder). Consequences:

- Greedy per-segment endpoint control is well-posed **only when segment
  length ≫ pool footprint** (the 5 mm rasters — hence their clean
  convergence; the code comment in `mp_lib.py` encodes this as "≥ 2 pool
  relaxation lengths"). At 0.25 mm it is structurally ill-posed: the 2×3 run
  railed (v→8 m/s, 39/40 pinned, depth CV 6.4%→49.3%). That run is kept as
  the honest diagnostic — do not "fix" it by tuning.
- The collapse IS the band structure: `∂d_i/∂v_j ≈ 0` for `i − j ≳ 8`
  segments. The coupled problem is banded lower-triangular → tractable.
- Only a **whole-profile** formulation can regulate a within-footprint
  feature like the 0.75 mm startup transient.

Current OTI seeding (all of DV_Q, DV_SIG, DV_V) is hard-wired to the **last**
path segment (`vel_seed`: `ks = path.size()−1`; Q/σ guards:
`seg_temp + 1 == path.size()`), so only the (near-zero) diagonal of J is
measurable today. That is the gap Phase 1 closes.

## 2. The math: seeding an arbitrary segment j

### 2.1 Velocity (the only control with timing algebra)

Observation at the end of truncated segment i (fixed path *fraction*, the
same convention as the validated current-segment rule). Seed `v_j` of
segment j ≤ i: `Δt_j = t_j − t_{j−1}`, `v_j = L_j/Δt_j`,

```cpp
Real dt_oti  = L_j / thesis::seed(thesis::DV_V, v_j);  // ∂Δt/∂v = −L/v²
Real k       = dt_oti / Δt_j;      // real part 1
Real t_shift = dt_oti − Δt_j;      // real part 0
```

Three zones for a quadrature node deposited at `t'` (τ_d = double-valued
conduction time, `τ_after = t_obs − t_j` = fixed double path duration after
segment j):

| node deposited…      | conduction time τ                        | weight dtau |
|----------------------|------------------------------------------|-------------|
| **before** segment j | `τ_d + t_shift`                          | unchanged   |
| **inside** segment j | `τ_after + (1−φ)·dt_oti`  (φ = fraction through j) | `× k` |
| **after** segment j  | `τ_d` (unchanged)                        | unchanged   |

Physics: before-j heat is deposited at fixed time but *observed* later
(observation shifts with Δt_j); inside-j heat is stretched with the segment;
after-j heat shifts together with the observation → cancels exactly.
**Reduction check**: j = last ⇒ τ_after = 0 ⇒ inside-rule = `τ_d·k`,
before-rule = current history rule — i.e. the existing validated code is the
special case. Equivalent blended form (matches the current code shape):
`τ = τ_d + ((τ_d − τ_after)/Δt_j)·t_shift` for inside-j nodes.

### 2.2 Q and σ (no timing algebra — guard move only)

`∂T/∂Q_j`, `∂T/∂σ_j` at fixed everything else: seed the parameter **only for
nodes of segment j** instead of "only for nodes of the last segment." One
guard change (`seg_temp == seed_seg` instead of `seg_temp + 1 ==
path.size()`), shared with velocity. This is the sensitivity-side half of the
"history for the other parameters" requirement (§3, decision 2) and makes
future P/σ trajectory columns free.

### 2.3 The coupled problem (Phase 4 preview)

```
min over v ∈ [v_lo, v_hi]^N :
    Σ_i (d_i(v) − d*)²  +  λ_s ‖D₁ v‖²  +  λ_m ‖v − v_nom‖²
```

- residual: **depth only** — width is grid-pinned at 163.84 µm across the
  entire baseline; adding it pollutes conditioning for zero information.
- `J_ij = ∂d_i/∂v_j`, lower-triangular (causality), banded (footprint),
  approximately **Toeplitz** in the developed region (translation
  invariance) — exact columns only needed for the startup rows.
- `λ_s‖D₁v‖²` (first differences): smooth schedule, machine-realizable.
- `λ_m‖v − v_nom‖²` (small): anchors the null space — late-track velocities
  that no measurement constrains stay at nominal 3 m/s instead of drifting.
- Gauss–Newton with bound clipping, trust region, backtracking on a merit
  re-evaluated by actual simulation — same scaffolding style as
  `run_optimized.py`.

## 3. Design decisions (fixed — do not relitigate during implementation)

1. **Controls: v only; P and σ frozen at nominal** (Pmod 1, σ = 200 µm).
   The startup defect is a depth deficit that velocity closes alone
   (cold-start elasticity −0.66). P/σ trajectories are future work.
2. **Full control history recorded on EVERY run — all three parameters.**
   Any snapshot (residual or Jacobian) truncated at segment i writes the
   complete current profile into the path: per-segment Pmod (col 4), speed
   (col 5), σ width-factor (col 6), via
   `set_pmods(lines, pmods, wmods, vels)`. P/σ are nominal in this phase but
   are passed explicitly, never implied — the thermal history feeding `d_i`
   must be exactly the profile being optimized, for every parameter, for the
   same reason the velocity history is recorded. (Simulation-side half of
   the requirement; §2.2 is the seeding-side half.)
3. **Seed-segment selection is control-agnostic**: one `seed_seg` input
   steers DV_Q, DV_SIG, and DV_V alike. Default = last segment ⇒ existing
   behavior; **backward compatibility is a hard regression gate** (default
   OTI outputs bit-identical, plain-double build untouched).
4. Observation convention: fixed path fraction (end of truncated segment i),
   identical to the validated current-segment rule.
5. Segment grid: uniform 0.25 mm (N = 40). The regularizer handles the
   underdetermined developed tail; no nonuniform control grid.
6. Target `d*` = developed single-track depth from the 0.25 mm baseline
   (63.34 µm). Width monitored, not a residual.
7. First-order OTI, one seeded direction per run (no otinum template
   changes; the band + Toeplitz structure makes runs cheap enough).
8. Final verdict = tracked replay (`verify_optimized.py`, fused-depth CV),
   never the snapshot residuals — same discipline as the rasters.

## 4. Phase 1 — C++: generalized seed segment

Files: `src/Calc.cpp`, `src/oti_scalar.h` (comment), possibly
`src/Init.cpp`/`src/DataStructs.h` for input plumbing.

1. **Input**: optional integer `SeedSegment` (0-based path-row index of the
   powered segment to seed; default −1 = last segment). Prefer a Settings.txt
   entry (pattern on the existing `Compute/MaxThreads` parsing in Init.cpp);
   an env-var fallback (`THESIS_SEED_SEG`) is acceptable if input plumbing
   is invasive — decide when reading Init.cpp, keep it to one mechanism.
2. **`vel_seed` generalization** (`Calc.cpp`): take the seed index j; compute
   `L_j`, `Δt_j` from `path[j]`/`path[j−1]`; return `k`, `t_shift`, and
   `τ_after = t_obs − path[j].seg_time` (needs the observation time — pass
   `t` in). Skip spot-mode segments as today.
3. **Quadrature loops** (`GaussIntegrate`, `GaussCompressIntegrate`): the
   loops already know each node's segment (`seg_temp`); classify against j
   into the three §2.1 zones. Zone rules as in the table; inside-j weight
   `dtau × k`. The existing `cur_seg` special-casing becomes the j = last
   instance of the general rule.
4. **Q/σ guard move** (§2.2): `seg_temp == seed_seg` at all four seeding
   sites.
5. Edge cases: observation time inside segment j (t_obs < t_j) — cannot
   happen for the snapshot pipeline (truncation at i ≥ j) but guard anyway:
   fall back to the current-segment rule with `τ_after = 0` clamped. Beam-off
   / solidification queries (t beyond path end): `τ_after` formula already
   covers it.
6. Build both binaries.

Acceptance (hard gates):
- plain-double build **byte-identical** on a test case;
- OTI build at default seed: snapshot **bit-identical** to the current
  validated build (same case, same columns, `diff`);
- OTI with `seed_seg = j` (a middle segment): runs, emits finite `dT_dv`,
  `dT_dQ`, `dT_dsig` columns.

## 5. Phase 2 — validation of off-diagonal sensitivities

1. **Analytic (strongest)**: extend `common/verify_analytic_moving.py` to a
   middle-segment seed. The H/W/C channels generalize exactly:
   `H = 12α ∫₀^{t_{j−1}} K·S dt'` (before-j, observation shift),
   `W = (1/Δt_j) ∫_{t_{j−1}}^{t_j} K dt'`,
   `C = (12α/Δt_j) ∫_{t_{j−1}}^{t_j} (t_j^{…}) K·S dt'` (inside-j), and
   **zero** over (t_j, t_obs]. `dT/dv_j = −(L_j/v_j²)·A·(H+W+C)`. Expect
   ~1e-6 median away from zero-crossings, as before. Verify j = last
   reproduces the existing check.
2. **FD on the real path** (extend `common/validate_dv.py`): on the
   single-track 0.25 mm path, for pairs (i, j) with lags i−j ∈
   {0, 1, 2, 4, 8, 12}: central-difference `v_j(1±ε)` (rewrite path col 5,
   double solver, support-function depth at ε = 1e-2) vs OTI
   `ddepth_dv` with `seed_seg = j`. Field-level `dT_dv` FD check first when
   debugging. Also one FD cross-check each for `Q_j` and `σ_j` at lag 2 (the
   moved guard).
3. **Structure checks**: (a) band decay — one full column j (all i ≥ j)
   confirms `|∂d_i/∂v_j| → 0` beyond ~8 segments and fixes the band width B;
   (b) Toeplitz — two developed-region stencils (different i) agree to a few
   %, licensing the Phase 3 compression.

Acceptance: analytic ~1e-6; FD field <1%, FD depth <2–3% at lags through the
band; Q/σ cross-checks pass; B and the Toeplitz error quantified.

## 6. Phase 3 — influence table (Jacobian) assembly

`single_track/traj_jacobian.py`:

1. Exact startup block: for observation rows i in the startup + one footprint
   (first ~12–16 segments), all j with i−j ≤ B: one OTI run each
   (`seed_seg = j`, truncate at i, full profile recorded per §3 decision 2)
   → `ddepth_dv` (store `dwidth_dv` too, as a monitor).
2. Developed region: one B-lag Toeplitz stencil measured at a developed i,
   reused for all later rows; cross-check against one exact developed column
   (Phase 2.3b).
3. Optional `--full` brute-force mode (all banded pairs) as the audit path.
4. Output: `results/traj_jacobian.npz` (J, band, s-grid, run metadata) + a
   readable CSV; print the table (µm per m/s) and the startup-column decay.

Cost: ~100–150 OTI runs ≈ 2 min (exact block) + B runs (stencil); brute-force
banded ≈ 330 runs ≈ 4–5 min. Residual sweeps (Phase 4) are N = 40 double
runs ≈ 17 s each.

---

### Phases 1–3 RESULTS — executed 2026-07-16 (VALIDATED, table assembled)

**Phase 1** (`src/Calc.cpp` `SeedCtx`/`seed_ctx`/`dv_tau`; `src/DataStructs.h`
`settings.seed_seg`; `src/Init.cpp` `Compute/SeedSegment`, default −1, refused
with compression; `mp_lib.write_settings` + `run_segment(seed_idx=…)`).
`SeedSegment` is a 0-based Path.txt DATA row; the solver's internal path
prepends an origin spot (`FileRead_Path`), handled by a +1 in `seed_ctx` —
an off-by-one here was caught by the explicit-last ≡ default gate and fixed.
Gates: plain-double **byte-identical** (tracked case_base); default-seed OTI
**bit-identical**; explicit last-segment seed **bit-identical to default**.

**Phase 2** (`common/verify_analytic_moving.py` extended to middle seeds;
`common/validate_traj.py` new). Analytic: seed-last regressions unchanged
(~1e-6); j=6/8 ~1e-5; j=4,2 show 1–9% vs the CONTINUUM — quadrature
coarseness over old history (step doubling), not wiring: field FD vs OTI on
the discrete solver is 0.04–0.08% median at ALL lags (the solver
differentiates its own approximation exactly). QoI level: naive FD of the
MAX depth shows 15–60% apparent errors — the argmax relocates on the flat
depth plateau (envelope-theorem second-order term); at the FIXED support
column (pure IFT) FD agrees to **0.06–0.29%** for v/Q/σ (v by lag:
2.2% at lag 0 [smallest signal], 0.10–0.29% lags 1–4; Q field 0.0068%
[linear], σ field 0.057%). Band: **peak at lag 3** (the deepest point trails
the beam by ~0.8 mm), super-exponentially dead by lag 5–6 → **B = 6**.
Toeplitz: **0.00%** at two developed observations.

**Phase 3** (`single_track/traj_jacobian.py` → `results/traj_jacobian.npz`
+ `.csv`): 70 OTI runs, 43 s. Exact rows 0–11 + stencil at row 14; exact
row 11 ≡ stencil to 0.00%. Startup block (µm of depth per m/s; d* = 63.34):

| obs (s) | d0 µm | lag0 | lag1 | lag2 | lag3 | lag4 |
|---|---:|---:|---:|---:|---:|---:|
| seg0 (0.25mm) | 39.21 | −8.56 | | | | |
| seg1 (0.50mm) | 54.05 | −3.82 | −8.32 | | | |
| seg2 (0.75mm) | 60.64 | −1.31 | −4.28 | −8.69 | | |
| seg3 (1.00mm) | 63.05 | −0.43 | −0.61 | −9.87 | −4.64 | |
| seg4+ (dev.)  | 63.34 | +0.09 | +0.09 | −4.69 | −11.18 | −0.46 |

Readings: (i) the greedy pathology is now one table — each row's dominant
entry is OFF-diagonal (lag 2–3) except the very first; (ii) only rows 0–3
carry residual (deficits −24.1, −9.3, −2.7, −0.3 µm) — the developed region
is exactly on target, so the solve is ~4 residuals against ~7 effective
startup controls + regularization; (iii) first-order size: closing seg0's
−24 µm via v_0 alone needs Δv ≈ −2.8 m/s, whose lag-1..3 spillover
(−8.3, −8.7, −4.6 µm per m/s) then over-deepens seg1–3 — the coupled solve
must speed those up; expect a strong slow-start → overshoot-correction
profile and real GN iteations (nonlinear at this step size).

**Proposed Phase 4 starting point** (normalized units: r̃ = r/d*,
ṽ = v/3 m/s, J̃ = J·(3/d*), giving J̃ᵀJ̃ diag ≈ 0.33): λ_s = 1e-2 (permits
the physical per-step ramp Δṽ ≈ 0.2–0.3, suppresses oscillation),
λ_m = 1e-3 (≈ 2.5 decades below J̃ᵀJ̃ — anchors the unconstrained developed
tail at nominal without biasing the startup). Trust region |δv| ≤ 1 m/s per
iteration, J refresh if the merit stalls. User's call.

## ⛔ STOP HERE — mandatory review gate before Phase 4

**Do not begin Phase 4.** Phases 1–3 deliver a validated, measured influence
table — a complete unit of work. Hand back to the user with:

- the Phase 2 validation numbers (analytic, FD-by-lag, Q/σ cross-checks);
- the measured band width B and Toeplitz error;
- the influence table itself: startup columns vs developed stencil, in
  µm per (m/s), with the decay profile plotted or tabulated;
- a proposed (λ_s, λ_m) starting point argued FROM those numbers.

Reason: the optimizer's tuning (regularizer weights, band width, null-space
anchor, possibly a shaped `v_nom`) should be chosen from the real table, and
the user decides them. Do not pick values and run unprompted.

## 7. Phase 4 — trajectory Gauss–Newton (design sketch; after the gate)

`single_track/run_trajectory.py`:

1. Unknowns `v ∈ [0.5, 8]^40`; residuals `r_i = d_i(v) − d*` from N truncated
   double-solver snapshots (full profile written per §3 decision 2).
2. GN step: solve `(JᵀJ + λ_s D₁ᵀD₁ + λ_m I) δv = −(Jᵀr + reg. gradients)`,
   clip to bounds + per-iteration trust region (e.g. |δv| ≤ 1 m/s),
   backtracking on the simulated merit.
3. J from Phase 3; refresh (cheap, Toeplitz) only if the merit stalls —
   expect 2–4 iterations total.
4. Outputs: `results/trajectory_zero.csv` (idx, s_mm, v, d, residual) AND an
   `optimized_zero.csv`-schema file (pmod = 1, sigma = nominal, vel = v_i) so
   `verify_optimized.py` replays it unchanged.
5. Expected physics (falsifiable): lead-in ≈ 1.9 m/s relaxing to 3 m/s over
   ~1 mm, smooth, with the coupled solve trimming the overshoot the greedy
   controller couldn't see; snapshot depth spread collapses toward grid
   resolution.

### Phase 4 RESULTS — executed 2026-07-16 (converged; verdict at the fused level is the finding)

`single_track/run_trajectory.py` ran as designed (λ_s=1e-2, λ_m=1e-3, trust
1 m/s, J refresh on stall): merit 0.0842 → 0.0475 over 8 iterations,
480 double + 490 OTI runs, 505 s. Final profile v = (2.29, 3.97, 3.58,
3.00, 2.96, 3.0 …); snapshot residuals max|r| 38.1% → 26.2%, snapshot depth
CV 6.42% → 4.78%. The greedy divergent run is preserved as
`results/optimized_zero_greedy_divergent.csv`.

Three validated findings:

1. **The coupled GN is sound and near-optimal for its objective**: the
   UNREGULARIZED linear floor from the measured J is max|r| = 17.1% (bounds
   inactive) — the nonlinear, regularized solve reached 26%, same regime.
2. **v-only reachability is physics-limited**: the velocity influence
   kernel is flat over lags 0–3 (−8.6, −8.3, −8.7, −4.6 µm per m/s), so a
   sharp startup deficit (−24, −9, −2.7, −0.3 µm) cannot be deconvolved —
   the optimizer must split the error (−26% at seg0, +9% pushed to seg3).
   Adding P does NOT lift the floor: its influence is the same deposited
   heat through the same diffusion kernel (measured developed Q column:
   ≈0 at lags 0–1, peak ~15 µm/Pmod at lags 2–3 — same delayed shape).
3. **The snapshot objective is the wrong objective for fused quality on a
   single track.** Tracked replay: fused-depth map σ 0.61 → **1.89 µm**
   (min/max 55/60 → 50/65 µm) — the "optimized" schedule DEGRADED the real
   outcome. Mechanism: the fused map **self-heals the startup by
   remelting** (a point at x = 0.25 mm stays inside the trailing pool until
   the beam is ~2 mm past; its fused depth records the developed maximum).
   The baseline never had a fused-depth problem; correcting the moving-
   frame diagnostic introduced genuine overshoot (65 µm) and an overspeed
   shallow spot (50 µm).

Consequences for the program: (a) single-track velocity control should
target a quantity that does NOT self-heal — fused-map edges, solidification
conditions (G/V, microstructure), or the track ends; (b) the trajectory
machinery (generalized seeding, banded J, GN) is validated and portable —
the right next users are the raster geometries, where fused-map variance is
real (line starts/turns), and the Phase 5 dwell derivative, whose
influence shape (energy deposited BEFORE the start) actually matches a
startup deficit; (c) any future objective should be defined on the fused/
solidified field, not on segment-end snapshots.

## 8. Phase 5 — verification, figures, write-up (after Phase 4)

1. Tracked replay (`verify_optimized.py`) → fused-depth map CV baseline vs
   optimized; fig17/fig16 for `case_opt_zero`.
2. New figure: v(s) profile over the depth-vs-s trace, baseline vs optimized
   (the money plot).
3. `raster/doc/optimization_notes.tex`: (a) the posedness finding — greedy
   endpoint control requires segment ≫ footprint, with the collapse table
   and the divergent run as evidence; (b) the coupled formulation, three-zone
   generalized seeding (extends the §3.1/Appendix A velocity story);
   (c) results. Update `single_track/README.md` and the Makefile target.
4. Timing bookkeeping audit: non-uniform v breaks any reconstruction that
   assumes `mp_lib.V` constant — `beam_on_intervals` already walks actual
   path speeds (OK), but check `fullfield_stats.py`, `make_plots.py` fig17
   windows, and `calibrate.py` before trusting replay statistics.

## 9. Gotchas checklist

- [ ] Default-seed OTI regression is **bit-identical**, not just "close".
- [ ] Plain-double build byte-identical (OTI code fully compiled out).
- [ ] `seed_seg` indexes path rows, not controlled-segment ordinals — the
      drivers translate via `seg_info` (`idx` column); off-by-one here
      produces plausible-looking wrong Jacobians. Validate with the lag-0 FD.
- [ ] Spot-mode / beam-off rows never seeded (skip as today).
- [ ] `τ_after` uses the actual observation time `t` (solidification queries
      have t > path end).
- [ ] All schedule/branching still on `to_double` real parts; node counts
      frozen at real v (differentiate-the-approximation, as validated).
- [ ] Snapshot truncation: `run_segment(div[:i+1], …)` — the seeded j must
      survive truncation (j ≤ i), assert in the Python driver.
- [ ] Width stays a monitor: if the optimizer ever moves width off its
      grid-pinned 163.84 µm plateau, surface it, don't chase it.
- [ ] MPI OTI build: seed input must reach all ranks identically.
- [ ] `validate_results.py` and all square/triangle artifacts untouched
      (VEL_CONTROL gate keeps rasters at the published 2×2).

## 10. Reference

- Posedness diagnosis + collapse table: this session (2026-07-16), memory
  `velocity-derivative-work.md`; divergent run `single_track/results/
  optimized_zero.csv` (depth CV 49.3%, 39/40 pinned — keep as diagnostic).
- Validated current-segment velocity rule: `VELOCITY_DERIVATIVE_PLAN.md`
  (§2 math, Phase 3 results), `src/Calc.cpp` `vel_seed`,
  `common/verify_analytic_moving.py` (H/W/C channels),
  `raster/doc/optimization_notes.tex` §3.1 + Appendix A.
- Seeding sites / guards: `src/Calc.cpp` (~134/202/303/528 + `vel_seed`);
  path times `src/Init.cpp:643–658`; snapshot columns `src/Grid.h:158–171`.
- Python plumbing: `common/mp_lib.py` (`set_pmods` cols 4/5/6,
  `run_segment`, `seg_info`), `common/measurement.py`
  (`ExtractIsoSupportSensitivities` auto-discovers `dT_d*`),
  `single_track/verify_optimized.py` (reads `vel` column).
- Paper: Stump & Plotkowski, *Appl. Math. Modelling* 75 (2019) 787–805.
  Local: `/mnt/c/Users/Rober/Downloads/1-s2.0-S0307904X19304093-main.pdf`.
