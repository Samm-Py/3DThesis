# Plan: dwell derivative (DV_DWELL) for the OTI solver

Goal: extend the OTI automatic-differentiation build so a snapshot carries
`dT_ddwell` — the exact derivative of the temperature field with respect to
the **duration of a beam-off turnaround dwell**, at fixed path geometry — and
the measurement layer therefore yields `dwidth_ddwell` / `ddepth_ddwell` /
`dasym_ddwell` alongside Q, σ, and v. This promotes dwell from a fixed
*policy* (a single global value found by bisection in `find_min_dwell.py`) to
a **per-line control** the optimizer sets with an exact Newton step, and it
does so by reusing — almost verbatim — the time-parametrization machinery
already built and validated for `DV_V`.

This is the follow-up flagged as Phase 5 of `VELOCITY_DERIVATIVE_PLAN.md`
("A turnaround dwell has no kinematic channel … exactly the `t_shift` history
algebra of §2 applied to a spot-mode/zero-power segment"). Read that plan
first — this document assumes its §1–2 background and §6 validated result.

## Why this is the right lever (motivation, one paragraph)

The three existing controls all act on **energy**: power adds it, σ spreads
it, velocity — even slowing down keeps the beam on — adds it. *None can remove
accumulated heat.* On the shrinking triangle apex the melt pool is dominated
by inherited heat, so (P, σ, v) lose authority together: the endpoint depth
stops responding, and the zero-dwell apex optimization stalls (`greedy_1_zero`
last unit: quit at 26.5 % depth error with all controls interior, pool locked
shallow-and-wide). Dwell is the only actuator that *drains* heat, and — the
useful part — its authority is **strongest exactly where the others go dead**,
because the quantity it differentiates against (the diffusion time of upstream
heat) is precisely what governs the apex pool. See the side-by-side: the
dwell-policy apex recovers depth (line 87 converges to 0.4 %) purely because
the beam-off pause restores cold substrate. `DV_DWELL` makes that recovery a
gradient the optimizer can use, instead of a policy chosen in advance.

> **Execution scope: Phases 1–3 only.** Implement `DV_DWELL`, confirm Python
> propagation, validate against finite differences, then **STOP at the review
> gate before Phase 4**. Phase 4 (optimizer integration) contains a genuine
> design choice — how dwell enters the per-segment Newton and how its
> build-time cost is weighted — that the user makes from the Phase 3 numbers.
> Do not implement it unprompted.

---

## 1. The math: DV_DWELL is DV_V with two of three channels deleted

A turnaround dwell is a **spot-mode, beam-off** path row `js` (written by
`build_rows` / `make_case.py` as `1  x  y  0  0  Δ` — Mode 1, Pmod 0,
Vel/Time = Δ). Its duration is `Δ = path[js].seg_time − path[js−1].seg_time`.
The control is **Δ itself** (a time), not `L/v`.

Recall the DV_V three-zone conduction-time rule (`Calc.cpp::dv_tau`,
lines 84–100) for a node deposited at `tp` on segment `seg`, observed at `t`,
when segment `ctx.seg` is perturbed:

| zone | DV_V | why it survives for dwell |
|---|---|---|
| `seg == ctx.seg` (inside, **stretch**) | `(t − tp)·k` + weight channel | **gone** — beam is OFF during a dwell, so there are *no quadrature nodes* on `js`; the stretch (`k`) and weight channels are never evaluated |
| `seg < ctx.seg` (**before**, shift) | `(t − tp) + t_shift` | **kept, verbatim** — upstream heat is deposited before the dwell; lengthening Δ pushes the observation later, so it diffuses longer |
| `seg > ctx.seg` (**after**) | `t − tp` (unchanged) | **kept** — the line scanned *after* the dwell has its deposition and the observation shifted together, so τ is unchanged |

So the exact dwell rule is the **history `t_shift` branch alone**. Seed the
duration directly (no `L/v`):

```cpp
Real   dt_oti  = thesis::seed(thesis::DV_DWELL, Delta_real);  // d(Δ)/dΔ = 1
Real   t_shift = dt_oti - Delta_double;                       // real part 0, pure derivative
// ctx.k unused (no nodes inside js); ctx.seg = js
```

then, for the observed line's endpoint snapshot:
- node on `seg < js` (upstream heat): `tau_oti = tau_double + t_shift;`
- node on `seg >= js` (dwell has none; scanned line unchanged): `tau_oti = tau_double;`

`dtau` stays double-valued everywhere (the shift moves conduction times, not
quadrature weights). Convention: `dT_ddwell` is per second of the seeded
dwell row's duration. Sign expectation: more dwell ⇒ longer τ upstream ⇒ more
diffusion ⇒ cooler ⇒ `ddepth_ddwell < 0`, `dwidth_ddwell < 0`.

**One-sidedness.** In a strict-zero path there is no dwell row to seed. To
evaluate `d/dΔ` at Δ = 0⁺ (the "should I introduce a dwell here?" gradient),
insert a **zero-duration** spot row before the line and seed it; the
derivative is well-defined (cooling is smooth) but one-sided — the optimizer
may only *add* dwell (`Δ ≥ 0` bound / projected step). A zero-duration dwell
must be a no-op in the plain-double build (see Phase 1 acceptance).

## 2. Design decisions (fixed — do not relitigate during implementation)

- Fixed path geometry; Δ changes only the beam-off gap before one line.
- Seed the derivative through `seed_ctx` (the time-parametrization path),
  exactly like DV_V — **not** at node construction (there are no nodes to
  seed on a beam-off row).
- **Separate slot `DV_DWELL`** (DV_COUNT 6 → 7), *not* a mode-flag reuse of
  DV_V. Rationale: the optimizer needs `dv` for the line *and* `ddwell` for
  the dwell that precedes it **from one OTI run**; distinct slots give both
  simultaneously. (Reuse-under-a-flag is the cheaper alternative if that
  simultaneity is ever dropped — document, don't build.)
- First-order OTI is sufficient.
- All adaptive/branching logic stays on `double` real parts; the frozen-
  schedule argument is identical to DV_V §2.
- ~15 % more OTI arithmetic (6 → 7 directions); the plain build is untouched
  by construction and must stay byte-identical.

## 3. Phase 1 — C++: the DV_DWELL slot

Files: `src/oti_scalar.h`, `src/Calc.cpp`, `src/Grid.h`.

1. `oti_scalar.h`: add `DV_DWELL` to `DesignVar` before `DV_COUNT`; update the
   header comment block (§ around lines 31–49).
2. `Calc.cpp::seed_ctx` (lines 49–82): add a branch **mutually exclusive with
   the DV_V branch** by row type. DV_V guards on `!path[js].smode` (a powered
   line move, line 62); the dwell branch fires on the opposite —
   `path[js].smode && path[js].seg_time > path[js−1].seg_time` (a spot row
   with positive duration) **and** observation past the dwell
   (`t >= path[js].seg_time`):

   ```cpp
   else if (js >= 1 && path[js].smode && t > path[js].seg_time) {
       const double Delta = path[js].seg_time - path[js - 1].seg_time;
       if (Delta >= 0.0) {                    // >= to admit the Δ=0+ probe
           const Real dt_oti = thesis::seed(thesis::DV_DWELL, Delta);
           ctx.t_shift = dt_oti - Delta;      // real part 0.0
           ctx.k       = 1.0;                 // unused (no nodes on js)
           ctx.t_end   = path[js].seg_time;
           ctx.vel     = true;                // reuse the "time seeding active" flag
       }
   }
   ```

   `dv_tau` (lines 86–100) then needs **no change**: `seg < ctx.seg` returns
   `(t − tp) + ctx.t_shift` (the one live channel); `seg > ctx.seg` falls to
   `t − tp`. The `seg == ctx.seg` stretch branch is **provably dead** for a
   dwell, not merely benign — see the traced facts below — so `ctx.k = 1` and
   `ctx.tau_after = 0` are safe by construction.

   **Traced through `GaussIntegrate` (`Calc.cpp:130–328`) — three findings:**

   a. *No node ever survives on the dwell segment.* When the backward walk
      reaches `seg_temp == js`, `GetBeamLoc` returns `qmod = path[js].sqmod = 0`
      (`Util.cpp:249`; the dwell row's Pmod is 0), so the `qmod > 0.0` cull
      (`Calc.cpp:233, 306`, a real-part compare) discards it. The stretch
      branch and the `dtau *= ctx.k` weight channel (`Calc.cpp:304`) are
      computed and thrown away. That is why the stretch channel can be deleted.

   b. **Real hazard — guard the Q/σ node seeding.** The DV_Q/DV_SIG
      node-construction seeds (`Calc.cpp:220–226, 290–296`) are gated on
      `seg_temp == ctx.seg` **alone**, not on the segment being a powered line.
      With `ctx.seg` pointing at a dwell they run `seed(DV_Q, 0)` /
      `seed(DV_SIG, …)` on the beam-off nodes, planting spurious DV_Q/DV_SIG
      partials on zero-value nodes — correct today *only* because the same
      `qmod > 0` cull drops them. Do not rely on cull order: guard both blocks
      to fire only when the seeded segment is a powered line (skip on
      `path[ctx.seg].smode`, e.g. via a `ctx.dwell` flag distinct from
      `ctx.vel`).

   c. **The `t > path[js].seg_time` guard is load-bearing, not defensive.**
      For a solidification query at a time *inside* the dwell
      (`path[js−1].seg_time < t ≤ path[js].seg_time`) the correct
      `dT_ddwell` is **0** — lengthening the total pause cannot change an
      observation taken mid-pause. The strict guard yields exactly 0 there. A
      DV_V-style guard (`t > path[js−1].seg_time`) would instead fire and smear
      a nonzero `t_shift` across every upstream node — a genuine bug. Boundary
      `t == path[js].seg_time`: left-derivative 0 (measure-zero, fine).
3. `Grid.h` (lines 161–165): add `{"dT_ddwell", thesis::DV_DWELL}` to
   `oti_cols`.
4. Build both binaries; plain build must stay bit-identical:
   `make build && make build-oti` from `melt_pool_demos/`.

Acceptance:
- OTI snapshot for a two-line case with a dwell row gains `dT_ddwell`;
  `dT_dQ/dT_dsig/dT_dv` unchanged vs the pre-change OTI build.
- Plain-double outputs byte-identical, **including a path that contains a
  zero-duration dwell row** (the Δ=0⁺ probe must not perturb the double
  solution).

## 4. Phase 2 — Python: propagation check (free)

`measurement.py::ExtractIsoSupportSensitivities` auto-discovers every
`dT_d<name>` column (line 218), so `mp_lib.sens()` returns `dwidth_ddwell`,
`ddepth_ddwell`, `dasym_ddwell` with no code change — **provided the seeded
segment is the dwell row**. The `seed_idx` plumbing already exists
(`mp_lib.run_segment(..., seed_idx=...)` → `write_settings(seed_seg=...)`,
`Calc.cpp` `seed_ctx` resolves file row → path row). New requirement: seed the
*dwell row that precedes the observed line*, not the line itself, and truncate
the path so the observed line's endpoint is the snapshot (the dwell row and
its predecessor must survive truncation — assert this, as the DV_V drivers do
for the seeded line).

Acceptance: from a case whose last two controlled rows are `[dwell js, line
js+1]`, seeding `js` yields finite `d{width,depth,asym}_ddwell` at the line
endpoint.

## 5. Phase 3 — validation (the gate for everything downstream)

Write `melt_pool_demos/common/validate_ddwell.py`, patterned on
`validate_dv.py`:

1. **Central FD on Δ.** Take a mid-raster configuration **with upstream heat**
   (a triangle or square line preceded by several scanned lines, dwell row
   inserted before the last line). Rewrite the dwell row's duration to Δ ± ε
   (ε ∈ {1e-2, 1e-3} of a reference dwell, e.g. 0.5 ms) and compare
   `(w(Δ+δ) − w(Δ−δ))/2δ` to OTI `dwidth_ddwell`; same for depth; and the
   field column `dT_ddwell` vs FD of `T` per grid point (the tightest,
   measurement-free check — do this first when debugging).
2. **The history channel is the *only* channel — so it is the whole test.** A
   dwell with no prior heat has `dT_ddwell ≈ 0` (nothing upstream to keep
   diffusing); the meaningful check *must* carry inherited heat. A version
   that wired nothing would read ≈ 0 and silently "pass" a no-heat case — do
   not use one.
3. **One-sided check at Δ = 0.** Insert a zero-duration dwell, take a
   **forward** FD `(w(δ) − w(0))/δ`, compare to OTI `dwidth_ddwell` at Δ = 0⁺.
   This is the derivative the optimizer actually consumes to decide whether to
   open a dwell.
4. **Physics sanity.** `ddepth_ddwell < 0`, `dwidth_ddwell < 0` (waiting cools
   the inherited lake); magnitude largest for the apex-like lines (most
   inherited heat) and →0 for an isolated first line.

Acceptance: OTI vs central FD < 1 % (field) / < 2–3 % (support function, grid-
limited as in DV_V) at ε = 1e-3, on the mid-raster and Δ=0⁺ configurations.

### Phase 1–3 RESULTS — executed 2026-07-21 (Phases 1–3 complete, VALIDATED)

Implementation: `DV_DWELL` slot in `src/oti_scalar.h` (DV_COUNT 6→7); a dwell
branch in `Calc.cpp::seed_ctx` (fires on `smode && sqmod==0 && t>seg_time[js]`,
seeds the duration Δ into `t_shift`, sets `ctx.dwell`); the DV_Q/DV_SIG node
seeds and the DV_V weight channel guarded with `!ctx.dwell` (both integrators);
`dT_ddwell` column in `src/Grid.h`. `dv_tau` **unchanged** — the dwell rides its
existing `seg < ctx.seg` history branch. Validator:
`melt_pool_demos/common/validate_ddwell.py` (self-contained 2-line + dwell case).

**Phase 1 gate (all pass):** plain-double build **byte-identical** (454347 B,
exact; includes a path carrying the dwell row); OTI snapshot gains `dT_ddwell`;
`T`/`dT_dQ`/`dT_dsig`/`dT_dv` **unchanged** (max Δ = 0). Default seeding →
`dT_ddwell` all zero (dwell not seeded); seeding the dwell row → all-finite,
populated, dominant sign negative (cooling), with a positive near-field tail
(heat spreading outward).

**Phase 2:** `mp_lib.sens()` exposes `dwidth_ddwell`/`ddepth_ddwell`/
`dasym_ddwell` with **no Python change**, finite, correct signs
(`ddepth_ddwell = −0.018`, `dwidth_ddwell = −0.004`: waiting cools ⇒ shallower
and narrower).

**Phase 3 FD validation** (self-contained mid-heat case; the whole derivative
is the history channel, so this carries upstream heat by construction):

| check | best-step agreement | note |
|---|---|---|
| field `dT_ddwell` central FD (top-300 pts) | **0.06 % median / 0.2 % max** | @ δ=2.5e-5 s |
| field `dT_ddwell` one-sided Δ=0 forward FD | 0.5 % median / 1.4 % max | first-order, @ δ=5e-6 s |
| support `ddepth_ddwell` central FD | **0.01–0.07 %** | 10 µm z-grid |
| support `dwidth_ddwell` central FD | 3–4 % | 50 µm y-grid quantization (cf. DV_V width 24 %); **not** a solver error |

T is written at 6 sig figs, so the `T+−T−` difference roundoff-floors as δ
shrinks (max rel 0.2 %→8.8 % from δ=2.5e-5→5e-7 s); read agreement at the
largest step whose O(δ²) truncation stays below that floor — the same
FD-limited reading as DV_V's grid-limited width. Field derivative is exact on a
config with inherited heat → **history channel confirmed** (a no-heat case would
read ≈0 and pass trivially; this one does not). Signs correct everywhere
(`dT_ddwell < 0`, `dd/ddwell < 0`, `dw/ddwell < 0`: waiting ⇒ cooler, shallower,
narrower).

---

## ⛔ STOP HERE — mandatory review gate before Phase 4

Phases 1–3 deliver a validated `dT_ddwell`; that is a complete unit of work.
Hand back with the FD-vs-OTI numbers, the signs/magnitudes, and a readout of
`ddepth_ddwell` for an apex line vs a body line (to show the authority is
concentrated at the apex, as motivated). **Phase 4 hinges on a design choice
made from those numbers** — do not pick one and build it.

## 6. Phase 4 — optimizer integration (DESIGN FOR SIGN-OFF; not implemented)

This is the design to approve before writing code. It targets
`melt_pool_demos/common/greedy_raster.py` (the sequential greedy controller);
nothing here is built yet.

### 6.1 What dwell is, as a control

Dwell is structurally unlike (P, σ, v):

- **Per line-start (per turn), not per control segment.** One `Δ_ℓ ≥ 0` per
  raster line, inserted as a beam-off row before the line's first powered
  segment. Every block on the line inherits its thermal benefit, but the
  control is *owned by the line's first block*.
- **One-sided:** `Δ_ℓ ∈ [0, Δ_max]`, differentiated at `Δ = 0⁺` (Phase 1–3).
  Projected Newton; the optimizer can only *add* dwell.
- **It carries a build-time cost** — dead time. Every other control is free to
  the schedule; dwell trades throughput for uniformity and must be priced.

### 6.2 What the apex Jacobian shows (MEASURED — grounds the architecture)

Measured `∂(w,d)/∂(P,σ,v,Δ)` at a body line (30) and a binding apex line (84),
reconstructing the `greedy_1_zero` stalled thermal state (apex pool 167 µm ×
**42 µm** — the shallow-and-wide stall; body pool on target). Elasticities
`(u/QoI)·∂QoI/∂u`:

| control | body W / D | apex W / D | body∠v | apex∠v |
|---|---|---|---|---|
| P | +0.28 / +0.68 | +0.13 / +0.45 | — | — |
| σ | +0.55 / −0.59 | +0.07 / −0.26 | — | — |
| v | −0.17 / −0.67 | +0.21 / −0.45 | — | — |
| **Δ (dwell)** | **−0.0005 / −0.0012** | **−0.101 / −0.031** | 9° | **98°** |

Two facts set the architecture:

1. **Authority concentration (~200×).** Dwell's width-elasticity goes
   −0.0005 (body) → −0.101 (apex); depth −0.0012 → −0.031. Dwell is dead weight
   in the body and a real lever only at the apex — where P/σ/v stall. This, not
   conditioning, is why dwell activates on stall and stays shut elsewhere.
2. **At the apex dwell is ORTHOGONAL, not collinear.** In the body dwell is
   collinear with v (9°) but negligible. At the apex it is nearly orthogonal to
   v and σ (98°, 88°) and **width-dominated** (W-elast −0.101 ≫ D-elast −0.031).
   So the apex 2×4 is *well*-conditioned (singular-value ratio 2.6) precisely
   because dwell supplies the missing orthogonal direction. **A joint apex
   (P,σ,v,Δ) step is therefore viable** — an earlier draft of this section
   wrongly called the 2×4 ill-conditioned from a collinearity argument; the
   data refutes that at the apex.

**Mechanism (corrected by the data).** The stall is depth-low while P cannot
raise depth without pushing width over tolerance (dD/dP +0.45 but dW/dP +0.14,
width already at target). Dwell's role is **not** to add depth — it is to
**trim width** (its dominant, orthogonal authority: dW/dΔ ≈ −0.34 m/s, so
~50 µs trims width ~16 µm ≈ 10 %), which frees power to restore depth. **Dwell +
power together** is the fix — exactly why the fixed-dwell tip ran 206 W. The
levers are still *underdetermined* (4 controls, 2 residuals) and dwell still
*costs* build time, so the problem is resolved by **pricing + activation**, not
by a square Newton — which points at the recommended architecture.

### 6.3 Re-scoped architecture (Option 3, MEASURED): un-merge + velocity is the
### primary apex fix; dwell is a narrow tip add-on

Running the machinery on the real stalled apex (each line re-solved from the
`greedy_1_zero` state, dwell suppressed, pure (P,σ,v)) settled the architecture
empirically:

| apex line | stall → (P,σ,v)-only re-solve | how |
|---|---|---|
| 80 | 1.5 % → 1.5 % (already fine) | — |
| 83 | 19.2 % → **0.8 %** | slow v 2.06→1.31, depth 51→64 µm |
| 85 | 25.4 % → **1.4 %** | slow **hard** v 2.06→0.61, depth 47→63 µm |
| 86 (tip) | 26.5 % → 25.3 % ✗ | tried to *speed up*, pinned P floor, depth stuck |

**Finding 1 — the apex stall was mostly a MERGE artifact.** `greedy_1_zero`
forced lines 83–87 to share one control (26.5 % residual). Un-merged, each line
solved independently, **velocity (slowing down) clears the entire apex through
line 85** — depth restored 47→63 µm by dropping v toward its floor (0.61 m/s at
line 85). So the primary fix is: **do not merge the apex, and let the existing
(P,σ,v) Newton slow the beam down.** No dwell involved.

**Finding 2 — the failure mode was velocity going the *wrong way*.** The old
un-merged (`PREMERGE`) run pinned lines 84–87 at v = 8 m/s (max) and P = 8 W
(floor) — it *sped up*, shrinking the pool to nothing, instead of slowing to
develop depth. The fix is warm-start / step robustness so the apex re-solve
finds the slow-down basin (a fresh re-solve from the stalled point does).

**Finding 3 — only the geometric tip (line 86, line < pool) truly resists**, and
there even velocity is exhausted. This is regime 3; it needs the target-change,
and dwell is only a partial tool (it trims width but cannot create depth from
absent material — a joint dwell step on line 86 did **not** clear it).

**So the architecture is:**

1. **Primary: un-merge + velocity.** Retire apex merging (it *causes* the stall);
   run the standard per-line (P,σ,v) Newton, whose velocity control slows the
   apex down. Robustify the apex warm-start so it finds the slow-down basin, not
   the speed-up-to-floor one. This clears lines 0–85 to ≤ ~1.5 %.
2. **Add-on: priced dwell on the residual tip only** (`adaptive_dwell_solve`,
   `MP_ADAPTIVE_DWELL`, §6.4). It opens on a stalled single-line apex unit with
   the heat-excess signature (width-high) and trims width via its orthogonal
   authority; priced by `λ_dwell`, projected `Δ ≥ 0`, so it stays minimal and
   opens only where velocity is spent. At the pure geometric tip it is a
   partial help, not a cure — line 86 needs the crossover target-change (§6.8).

`λ_dwell` still sets the throughput/uniformity trade for the add-on; strict-zero
is `λ_dwell → ∞`. But the headline is that **velocity, not dwell, does the bulk
of the apex work** — dwell's scope narrowed to the last line or two once the
data showed slowing down clears the rest.

**Implementation status (2026-07-22).**

- **PRIMARY FIX DONE + VALIDATED.** The apex warm-start is the lever: seeding a
  short apex line (`line_len < APEX_LINE_MM = 2 mm`) from `COOL_SEED`
  (30 W, nominal σ, 1 m/s) instead of hot nominal makes the (P,σ,v) Newton find
  the slow-down basin (`greedy_raster.py`, apex branch in `greedy_initialize`).
  Sequential apex re-solve with committed coupling:

  | line | before → after | solution |
  |---|---|---|
  | 82 | 6.9 % → **3.3 %** | 30 W, v 1.0, d 65 µm |
  | 83 | 11.0 % → **1.4 %** | 22 W, v 1.1, d 62 µm |
  | 84 | 15.2 % → **2.7 %** | 20 W, v 0.8, d 62 µm |
  | 85 | 16.0 % → **4.3 %** | 10 W, v 2.0, d 61 µm |
  | 86 (tip) | 37.3 % → 20.8 % ✗ | regime-3 residual |

  Lines 0–85 clear to ≤ tolerance with depth restored to target; only the
  geometric tip (line 86) remains. Diagnosis: hot-nominal reseed → the
  speed-up-to-floor basin (P=82 W, v=2.08, 14.8 % stuck); COOL_SEED → slow-down
  basin (~1–3 %). Change is apex-scoped (`line_len < 2 mm`), so the body is
  untouched.
- **Dwell add-on built (behind `MP_ADAPTIVE_DWELL`, default off):** per-line
  dwell infra (`mp_lib.apply_dwells`, `run_segment(dwells=…)` — Δ=0
  byte-identical), the dwell Jacobian column (`dwell_column`), and the priced
  projected solver (`adaptive_dwell_solve`).
- **`MP_APEX_MERGE` marked superseded** (it causes the stall).

**Pending:** (a) wire the dwell add-on into `greedy_initialize` to auto-open on
the residual tip (line 86); (b) that add-on should use the alternating
**dwell-step-then-resolve** form (the joint 2×4 backtracking couples all four
controls); (c) a full end-to-end un-merged run to confirm the sequential body→
apex hand-off; (d) the pure tip (86) ultimately needs the crossover
target-change (§6.8) — dwell only trims its width, cannot create depth.

### 6.4 The dwell Jacobian column — one extra OTI run per dwell step

`seed_ctx` seeds a **single** segment, and DV_Q/DV_SIG/DV_V ride the *line*
segment while DV_DWELL rides the *dwell* row — different segments. So the dwell
column needs its **own** OTI run seeding the dwell row (`run_segment(..., seed_idx
= dwell_row_div_index)`; the div-index → path-index identity was verified in
Phase 2). This is cheap because it fires only on stalled apex lines. If that
cost ever matters, the future optimization is to extend `seed_ctx` to seed two
segments at once (line + its dwell) so a single run yields the full 2×4 — the
separate DV_DWELL slot was chosen (M=7) precisely to keep that door open.

### 6.5 Dynamic control units — dwell dissolves the merge

`_control_units`/`apex_merge` groups short apex lines into one shared control
*because a continuous (zero-dwell) path fuses them thermally*. Opening a dwell
on a line is exactly what **breaks** that fusion. So the unit structure becomes
dynamic: a line begins in whatever merged unit the zero-dwell pass built, and
**the moment dwell activates it splits into its own unit** (its members can no
longer share one control across a beam-off reset). Implementation: activation
in step 2 must re-segment the affected unit into per-line units before the
dwell step. This retires the static `MP_APEX_MERGE` flag — merging becomes the
`Δ_ℓ = 0` case of the unified controller.

### 6.6 Policy unification & retiring `find_min_dwell.py`

The `policy ∈ {zero, dwell}` argument and `dwell_for()` disappear: one run
produces `Δ_ℓ ≈ 0` through the body, growing toward the apex. `find_min_dwell.py`
(brute-force bisection on the discrete "no liquid at line start" criterion) is
superseded — its minimal dwell is now the `λ_dwell`-priced optimum, and it
regulates the *pool* (w,d) directly rather than a proxy reset criterion. Keep it
only as an independent cross-check during bring-up.

### 6.7 Downstream bookkeeping (audit before any full replay)

Per-line `Δ_ℓ` makes line times non-uniform, breaking anything that
reconstructs time from `mp_lib.V` and fixed hops:

- `common/fullfield_stats.py` (beam-on sampling windows),
- `common/make_plots.py` (per-line windows),
- `triangle/postprocess.py` (line-window bookkeeping).

`beam_on_intervals` / `line_start_times` (`mp_lib.py:497–539`) already walk the
path honoring dwells — route all three through them. `schedule_from` gains a
per-line `dwell_s`; the tracked-replay case writer must emit the per-line dwell
rows (make_case `--turn-dwell` generalized to a per-line vector).

### 6.8 Scope boundary — dwell does not fix regime 3

Dwell solves the **heat-coupling** failure (shallow-and-wide lock from inherited
heat). It does **not** fix the **measurement-granularity** failure at the very
tip: where the line is shorter than one pool, a single endpoint doesn't
represent the line (the dwell tip swept 100→38 µm within one traverse), and
where the remaining triangle is smaller than one pool, cooling makes no
material. Those need the separate target-change at the crossover and are **out
of scope for Phase 4** — Phase 4 regulates lines that still support an endpoint
measurement.

### 6.9 Open decisions for you (needed before implementation)

1. **`λ_dwell` calibration:** the apex Jacobian is measured (§6.2). Width authority
   `dW/dΔ ≈ −0.34 m/s` at the binding apex line, so a ~50 µs dwell buys ~10 %
   width headroom at a ~50 µs/line throughput cost. `λ_dwell` should price a
   µs of dwell against the width µm it buys; set it so dwell opens only once
   the (P,σ,v) width-headroom is exhausted (i.e. it reproduces the ~50 µs-scale
   apex dwell, not the 1.67 ms global `find_min_dwell` value, which enforced a
   stricter full-reset criterion). Confirm linearity beyond the 0.05 ms
   reference before trusting the elasticity out to the working dwell.
2. **`Δ_max`** per line — a hard cap, and whether it scales with local line
   length / inherited heat.
3. **Trigger precision:** confirm the heat-excess signature (§6.3 step 2) cleanly
   separates the two apex stall modes on the calibrated geometry, so dwell never
   opens on the too-little-heat (velocity/geometry) side.
4. **Coupling to velocity:** dwell (cools) and slow-v (heats) are opposite in
   energy — decide whether they co-solve on the apex line or dwell strictly
   precedes the (P,σ,v) re-solve (the §6.3 sequencing assumes the latter).

## 7. Gotchas checklist (review during implementation)

- [ ] Zero-duration dwell row is a byte-identical no-op in the plain build.
- [ ] DV_DWELL and DV_V seeding branches are mutually exclusive by row type
      (`smode` vs `!smode`); a path row never seeds both.
- [ ] `dv_tau` `seg == ctx.seg` (stretch) is unreachable for a dwell row —
      confirmed dead via the `qmod > 0` cull (`Calc.cpp:233, 306`;
      `Util.cpp:249`). Assert/comment.
- [ ] **Guard the DV_Q/DV_SIG node seeds** (`Calc.cpp:220–226, 290–296`) so
      they do not fire when the seeded segment is a dwell (`smode`) — today
      only cull order keeps the spurious partials out of the sum.
- [ ] Observation-time guard is strict `t > path[js].seg_time` (a query inside
      the dwell must give `dT_ddwell = 0`); do **not** copy DV_V's
      `t > path[js−1].seg_time`.
- [ ] Every comparison/branch that can see an OTI-valued time uses
      `thesis::to_double` (schedule, `t0calc`, `InRMax`, `t1 <= next_time`).
- [ ] Spot rows that are genuine point *exposures* (beam on, Mode 1 with
      power) vs beam-off *dwells* — seed only the beam-off kind. Check `qmod`
      of the row.
- [ ] Multi-beam loop: seed per path, one dwell row per turn.
- [ ] MPI OTI build: derivative column rides existing snapshot rows;
      `_merge_ranked_snapshots` is column-agnostic but assert column sets
      match.
- [ ] `DV_COUNT` 6 → 7 propagated everywhere it is used as the OTI width.
- [ ] `validate_results.py` untouched (no publication-artifact change).

## 8. Reference

- Follows `VELOCITY_DERIVATIVE_PLAN.md` (validated 2026-07-15) — §2 math, §6
  results, §9 checklist are the parent this specializes.
- Seed context / three-zone rule: `src/Calc.cpp:40–100`. Enum:
  `src/oti_scalar.h:50–57`. Snapshot columns: `src/Grid.h:158–171`.
  Path times / `seg_time` / `smode`: `src/Init.cpp:643–658`.
- Dwell row emission: `melt_pool_demos/common/mp_lib.py::build_rows`
  (dwell branch) and `triangle/make_case.py` `--turn-dwell`.
- Current (to-be-retired) global-dwell search and its discrete criterion:
  `triangle/find_min_dwell.py`. Measurement auto-discovery:
  `common/measurement.py:166–246`, `common/mp_lib.py::sens`.
- Diagnosis that motivates this (apex authority loss, zero vs dwell tip):
  the `greedy_1_{zero,dwell}` results and `results/fullfield/*_traces.csv`.
