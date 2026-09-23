# Automatic differentiation in 3DThesis: a guided tour

With `-DTHESIS_ENABLE_OTI=ON`, one 3DThesis run returns the temperature field
and, at every output point, its exact first derivatives with respect to seven
quantities: the point's position (x, y, z) and four process controls of one
path row (power Q, beam width σ, scan speed v, dwell τ). The physics code was
not rewritten to do this. The scalar type was swapped for a number that
carries derivatives, and those derivatives are started ("seeded") in a handful
of places.

This tutorial follows that mechanism through the code, then takes the last
step the melt-pool work needs: turning dT/dp into sensitivities of melt-pool
half-width and depth. Every claim is backed by a script in this directory
that runs in seconds and prints PASS or FAIL.

| step | topic | code | script |
|---|---|---|---|
| 0 | build both solvers | `CMakeLists.txt`, `src/CMakeLists.txt` | `check_stationary.py` as smoke test |
| 1 | what the OTI build outputs | `src/Grid.h` | `first_look.py` |
| 2 | OTI numbers and the `Real` alias | `src/oti_scalar.h` | `check_stationary.py` |
| 3 | seed → propagate → cut → output | `src/Calc.cpp`, `src/Grid.cpp`, `src/Util.cpp` | |
| 4 | the timing controls v and τ | `src/Calc.cpp` (`SeedCtx`, `dv_tau`) | `check_moving.py` A, B |
| 5 | what "exact" means here | `src/Calc.cpp` (adaptive quadrature) | `check_moving.py` C |
| 6 | temperature → melt-pool geometry | `derivations/` | `check_geometry.py` |
| 7 | exercise: add a design variable | three files | your own |

Plan on half a day. Steps 2 to 4 are the core.

## 0. Build both solvers

The OTI numbers come from the header-only library
[Sparrow](https://github.com/ORNL-MDF/Sparrow) (formerly cpp_oti_lib; the
C++ namespace and headers are still `oti` / `otinum`). Clone it next to this
repository; CMake looks for it there.

```bash
# in the directory that contains 3DThesis/
git clone https://github.com/ORNL-MDF/Sparrow.git   # tested at da6bd28

cd 3DThesis
cmake -S . -B build     -DCMAKE_BUILD_TYPE=Release -DTHESIS_ENABLE_MPI=OFF
cmake -S . -B build-oti -DCMAKE_BUILD_TYPE=Release -DTHESIS_ENABLE_MPI=OFF -DTHESIS_ENABLE_OTI=ON
cmake --build build -j && cmake --build build-oti -j
```

If Sparrow lives elsewhere, add `-DTHESIS_OTI_INCLUDE_DIR=/path/to/Sparrow/include`.
Use a Release build: an unoptimized OTI build is several times slower again.

The scripts need Python 3.8+ and `pip install -r ad_tutorial/requirements.txt`
(numpy, pandas, scipy, scikit-image). They look for `build/bin/3DThesis` and
`build-oti/bin/3DThesis`; set `THESIS_BIN` / `THESIS_BIN_OTI` to use others.
Run them from `ad_tutorial/`; each writes its cases to `ad_tutorial/runs/`.

```bash
cd ad_tutorial
python check_stationary.py      # should end with "all checks passed"
```

## 1. First look

```bash
python first_look.py
```

The same 316H case goes through both binaries. The plain build writes
`x, y, z, T`; the OTI build adds seven columns. Each column is a partial
derivative of T at that output point:

| column | derivative of T with respect to | unit |
|---|---|---|
| `dT_dx`, `dT_dy`, `dT_dz` | the output point's coordinates (the temperature gradient) | K/m |
| `dT_dQ` | the seeded row's power, per watt of `Power` × `Pmod` | K/W |
| `dT_dsig` | the seeded row's lateral beam width a (`Width_X` = `Width_Y`, times the row's width factor) | K/m |
| `dT_dv` | the seeded row's scan speed, with the path geometry held fixed | K/(m/s) |
| `dT_dtau` | the seeded row's duration, when that row is a beam-off dwell | K/s |

The **seeded row** is the last row of `Path.txt` unless `Settings.txt` names
another under `Compute` → `SeedSegment` (a 0-based index into the data rows,
header excluded). The script runs the OTI build twice: once seeding the last
line, once seeding the dwell. Q, σ and v belong to one row at a time, and a
beam-off dwell has only τ. So in the second run `dT_dQ`, `dT_dsig` and `dT_dv`
are zero, and in the first `dT_dtau` is zero.

Two notes on units:

- The width a is **not** the Gaussian standard deviation. The source is
  `exp(-3 r²/a²)`, so a = √6 × the standard deviation (the top-level README
  says the same about `Width_X`). `dT_dsig` is ∂T/∂a.
- `Power` is multiplied by `Efficiency` when read (`src/Init.cpp:910` stores
  `q = 2·eff·P`). `dT_dQ` is per watt of the `Power` input. The scripts write
  `Efficiency 1.0` and put absorbed watts in `Power`, so the distinction
  never arises here.

The OTI run costs about 3× the plain run for seven derivatives at once. A
central finite difference would need 14 extra runs and would not be exact.

## 2. The idea: OTI numbers and one type alias

**Order-truncated imaginary (OTI) numbers.** An `otinum<M, 1>` holds a value
and M first-order coefficients, written `a + b₁ε₁ + … + b_Mε_M`, with the
rule εᵢεⱼ = 0. Multiply two of them and the product rule falls out:

```
(a + bε)(c + dε) = ac + (ad + bc)ε
f(a + bε)        = f(a) + f'(a)·b·ε          for exp, log, sqrt, pow, ...
```

Put x = x₀ + 1·ε₁ into a program built from `+ − × ÷` and elementary
functions, and every intermediate value carries its derivative with respect
to x in the ε₁ slot. That is forward-mode AD. With `N = 1`, OTI numbers are
multivariate dual numbers; Sparrow also supports higher orders (N = 2 gives
Hessians), which 3DThesis does not use.

The Sparrow API surface used here is small:

- `otinum<M, N, double>::variable(i, v)` makes `v + 1·εᵢ`;
- `.real()` reads the value;
- `.partial(alpha)` reads the derivative selected by the multi-index `alpha`;
- comparisons (`<`, `>`, `==`, …) look at the real part only, so
  `if (qmod > 0.0)` takes the same branch as before and the arithmetic on
  that branch is differentiated;
- conversion to `double` is `explicit`, so a derivative can never be dropped
  silently.

**One alias.** [`src/oti_scalar.h`](../src/oti_scalar.h) is the whole
abstraction. Read it first; it is short.

```cpp
enum DesignVar { DV_X = 0, DV_Y, DV_Z, DV_Q, DV_SIG, DV_V, DV_TAU, DV_COUNT };

#ifdef THESIS_ENABLE_OTI
using Real = oti::otinum<DV_COUNT, 1, double>;
inline Real   seed(int dv, double v)   { return Real::variable(dv, v); }
inline double to_double(const Real& s) { return s.real(); }
inline double deriv(const Real& s, int dv) { /* s.partial(unit alpha) */ }
#else
using Real = double;                     // seed/to_double/deriv are no-ops
#endif
```

The physics code says `Real` wherever a value may need to carry a derivative.
In the default build that is `double`, the helpers do nothing, and the solver
computes exactly what the plain arithmetic computes.

**The whole idea in one diff.** The first commit on this branch is the
original prototype, and its diff is a good second read:

```bash
git show 5697d70          # "Add optional OTI automatic differentiation"
```

It swaps `double` for `Real` along the path from inputs to temperature, seeds
seven inputs (x, y, z, beam power, conductivity, density, specific heat)
where they are read, and prints seven columns. That is ~250 lines across 13
files, and none of them changes the physics. The next commit (`24e87e3`)
replaced the material properties with the per-row process controls used
today, and most of the subtlety below comes from that change.

`check_stationary.py` is the simplest possible test of the machinery. One
stationary spot is on for 1 ms, so the seeded row is the entire history.
Every OTI column is compared with the closed form in
[`analytic.py`](analytic.py), which does the same integrals with scipy
instead of quadrature nodes and takes every derivative by hand. They agree
to better than 1e-4, and `Q·dT_dQ = T − T0` holds to every printed digit because T is
linear in power.

## 3. Following a derivative through the code

3DThesis is semi-analytic. The temperature at point **x** and time t is a
sum over **quadrature nodes** placed along the beam's past path:

```
T(x, t) = T0 + Σₙ wₙ · exp(−3[(x−xₙ)²/φx + (y−yₙ)²/φy + (z−zₙ)²/φz] + expmodₙ)
φᵢ = aᵢ² + 12 α (t − tₙ)          (beam width, spread by diffusion since deposition)
```

`Calc::GaussIntegrate` ([`src/Calc.cpp`](../src/Calc.cpp)) walks backwards in
time along the path and builds the nodes (position, φ's, weight, strength).
`Grid::Calc_T` ([`src/Grid.cpp`](../src/Grid.cpp)) sums them at each output
point. AD happens in four stages.

### 3a. Seed: where derivatives start

**Position**, per output point, `src/Grid.cpp:283`:

```cpp
const Real xp = thesis::seed(thesis::DV_X, get_x(p));
const Real yp = thesis::seed(thesis::DV_Y, get_y(p));
const Real zp = thesis::seed(thesis::DV_Z, get_z(p));
...
const Real dx = xp - nodes.xb[iter];          // the kernel loop is otherwise unchanged
```

**Power and width**, per node, only on the seeded row (`src/Calc.cpp:236`,
and again at 306, 414 and 644; the compressed-path integrator repeats both
loops, so there are four copies to keep in sync):

```cpp
if (seg_temp == ctx.seg) {                      // this node lies on the seeded row
    axw = thesis::seed(thesis::DV_SIG, thesis::to_double(axw));
    ayw = thesis::seed(thesis::DV_SIG, thesis::to_double(ayw));
    current_beam.qmod = thesis::seed(thesis::DV_Q,
        thesis::to_double(beam.q * current_beam.qmod) / (2.0 * beam.eff))
        * (2.0 * beam.eff) / beam.q;            // dT_dQ is per INPUT watt
}
current_beam.phix = (axw * axw + ct);           // the derivative flows into the node
```

Three details:

- **Why per node, not on `beam.q`?** `beam.q` is shared by every row.
  Seeding it would give the derivative with respect to the power of the
  whole scan (which is what the first prototype did). Seeding only the nodes
  of one row isolates that row, and every other row carries zero derivative.
- **The power expression.** It rebuilds `qmod` as
  `seed(P·Pmod) · 2·eff / q`. The value is unchanged (`q = 2·eff·P`), but the
  seeded variable is now the row's input power P·Pmod, in watts.
- **`seed(dv, to_double(v))`** starts a fresh variable from the value
  alone. Anything v carried before is discarded on purpose.

### 3b. Propagate: which values carry derivatives

Every value computed from a seeded value is `Real`. On the node side that
means `int_seg` and `Nodes` ([`src/DataStructs.h`](../src/DataStructs.h)),
`beta`, `ct = 12·a·τ`, the φ's, and `expmod = log(q) − ½ log(φxφyφz)`
(`Util::AddToNodes`). On the output side it is the stored field `Grid::T`
(`new Real[pnum]`). `Beam::q/ax/ay` and the `Material` properties are also
`Real`, but they are never seeded, so they contribute zero derivative (step 7
changes that).

Nothing else was needed. `exp`, `log`, `pow`, `sqrt` and the operators are
overloaded, so the kernel sum in `Calc_T` produces T and all seven
derivatives in one pass.

### 3c. Cut: where derivatives are deliberately dropped

`thesis::to_double` appears where a value decides something discrete, or
where the derivative is not wanted:

| where | what | why |
|---|---|---|
| `Util::Calc_NonD_dt`, `Calc_RMax`, `t0calc`, `GetRefTime` | integration step sizes, cut-off radius and time | these place the quadrature nodes |
| `Melt.cpp` | search radii and grid indices for melt-pool tracking | discrete |
| `Grid::Calc_Solidification_time`, `Calc_Solidficiaton_Primary/Secondary` | G, V, cooling rate, … | solidification outputs are not differentiated |
| `Grid::Calc_T` return value, `add_T_hist`, `Run::Stork` | T handed to control flow and `double` storage | only `Grid::T` keeps the full number |

The first row has a consequence that shapes the rest of this tutorial:
**node placement is not differentiated.** AD gives the derivative of the
solver's sum with every node held at its time. Where the quadrature has
converged, the sum no longer depends on exactly where the nodes sit, so this
is the derivative of the true integral. (The step sizes do depend on beam
width and speed, through `GetRefTime` and `Calc_NonD_dt`, which is harmless
for the same reason; step 5 shows what happens where the quadrature has not
converged.) Power and width enter only the node *values*, so seeding the
values is enough. Speed and dwell are different: they change *when* heat was
deposited relative to the snapshot, and the node times are exactly what this
row holds fixed. Step 4 deals with that.

### 3d. Output

In OTI builds, `src/Grid.h:161` adds one output column per design variable,
reading `Grid::get_T_deriv(p, dv)` = `thesis::deriv(T[p], dv)`. From C++, call
`get_T_deriv` directly.

Two quick sanity checks from `first_look.py`: `dT_dz` is 0 on the top
surface (the solution mirrors the source there, which makes the surface
adiabatic), and `dT_dy` is 0 on the track centreline, by symmetry.

## 4. The timing controls: v and τ

Speed can't be seeded like power, and here is why. The path geometry is fixed
(`Path.txt` coordinates), so changing row j's speed changes only its duration
Δt = L/v. Nodes are placed in time by double-precision code (3c). The
derivative therefore has to be written into the node quantities by hand, and
it has three zones. The snapshot is at scan end, so the observation time
moves with Δt:

```
deposition time t'  ───────────────────────────────────────────────────────────▶ snapshot t
      rows before j                   row j (seeded)                rows after j
 ├─────────────────────────────┼──────────────────────────────┼──────────────────────┤
 heat is older by dΔt:          nodes spread over a longer     deposited later AND
   u = (t − t') + t_shift        Δt: u stretches and the         observed later by the
                                 weights grow by k = Δt'/Δt      same amount: no change
```

In the code ([`src/Calc.cpp:40–115`](../src/Calc.cpp)):

- `seed_ctx` builds, once per evaluation time,
  `dt_oti = L / seed(DV_V, v)`, then `k = dt_oti / Δt` and
  `t_shift = dt_oti − Δt`. **The real part of k is exactly 1 and the real part
  of `t_shift` is exactly 0.** Multiplying by k or adding `t_shift` leaves
  every value bit-for-bit unchanged and only injects a derivative. That is
  the central trick.
- `dv_tau(ctx, seg, t, tp)` returns each node's conduction time u:
  `(t − tp) + t_shift` for nodes before row j, `τ_after + (t_end − tp)·k` on
  row j, and plain `t − tp` after it.
- Line 320: `dtau *= k` on row j, because the quadrature weights scale with
  the row's duration (the "weight channel").

**The dwell τ** is the same construction with only the first zone. A beam-off
row has no nodes (they are culled by `qmod > 0`), so nothing stretches;
lengthening the pause just ages everything deposited before it. `seed_ctx`
sets `dwell` and `t_shift = seed(DV_TAU, Δt) − Δt`, and `dv_tau` shifts the
earlier nodes. An observation strictly inside the pause gets zero, because it
cannot depend on how long the pause will last.

`analytic.py` writes out the same three zones as integrals: `dv` has the
aging term, the weight term and the stretch term, and `dtau` has only the
aging term.

```bash
python check_moving.py     # cases A and B
```

Case A seeds the last row of an 8-row track and checks all seven columns.
Case B seeds a dwell between two lines. Both agree with the closed form to
better than 1e-5.

Two constraints follow from this design, and the solver enforces or
documents both:

- `SeedSegment` and path compression are incompatible. Compression merges
  old nodes from several rows, so a node can no longer be classified as
  before, on or after row j. `Init` refuses the combination
  (`src/Init.cpp:854`).
- One seeded row per run. A Jacobian across rows (how row i's controls affect
  the pool at row j) needs one run per seeded row.

## 5. What "exact" means here

Case C of `check_moving.py` seeds row 2 of 8, whose heat is ~3 ms old at the
snapshot. Now OTI and the closed form disagree by ~0.4% (`dT_dQ`) to ~2%
(`dT_dsig`). That is not an AD error. The solver integrates old heat
coarsely: as heat ages, the maximum step keeps doubling and the Gauss order
halves, from 16 down to 2 (`src/Calc.cpp:262` and `:336`). AD differentiates the
solver's sum, not the exact integral.

The script proves this with linearity. It switches off every row except row 2
and runs the plain-double solver: that temperature rise equals `Q·dT_dQ` from
the full OTI run to every printed digit, so both miss the closed form by the
same ~0.4%.

For derivatives of the last row, which is what a sequential controller uses,
the quadrature is fine and the agreement is ~1e-5 (cases A and B). Treat
cross-row derivatives of old rows as accurate to the solver's own accuracy
for old heat, which is a few tenths of a percent to a few percent.

## 6. From temperature to melt-pool geometry

The melt pool is the region T ≥ T_L. Its half-width is the largest y on the
liquidus surface and its depth the largest −z. Each is a **support value**
h(e) = max over the surface of e·x, for a direction e = +y or −z. At the
maximizing point x\* the surface is tangent to the plane normal to e, so ∇T is
parallel to e. Differentiating T(x\*(p), p) = T_L with respect to any control
p, the motion of x\* along the surface drops out (the envelope theorem):

```
dh/dp = −(∂T/∂p) / (∇T · e)          at x*
```

Both factors come from one OTI snapshot: ∂T/∂p is a control column and ∇T is
`(dT_dx, dT_dy, dT_dz)`. So the gradient columns are part of the design, not
decoration. Nothing is differentiated through the discrete geometry
extraction: marching cubes only *locates* x\* on the real temperature field.
Why that is better than pushing OTI numbers through marching cubes is
explained in [`derivations/marching_cubes_ad_explanation.pdf`](derivations/marching_cubes_ad_explanation.pdf),
and the edge-by-edge derivation is in
[`derivations/melt_pool_geometry_sensitivity_derivation.pdf`](derivations/melt_pool_geometry_sensitivity_derivation.pdf).

```bash
python check_geometry.py
```

The script uses the 316H process and two passes. A plain run on a 5 µm grid
locates x\*, then a 0.5 µm box around x\* is solved with both builds. It checks
all eight sensitivities (two dimensions × four controls) two ways:

- against the same formula evaluated with the closed-form T at x\*
  (a few parts per million), which checks the OTI columns where the formula
  samples them;
- against a finite difference of the measured pool (≤0.5%), which checks the
  formula itself.

One lesson from building this check is worth knowing before you use these
sensitivities. The widest point of a melt pool is very flat along the track:
here the edge moves 1.4 nm over ±5 µm in x. Meanwhile d(half-width)/dτ
changes by ~2% per µm along the same stretch, because the preheat that τ
controls comes from behind. With the CSV's six significant digits (0.01 K
near 1709 K), marching cubes put x\* one micron off and d(half-width)/dτ came
out 2% off. The script shifts T0 and T_L by the same constant, which changes
nothing the solver computes, to spend those digits where they matter. In
general, a support-point sensitivity is only as good as the location of x\*
along a flat extremum.

## 7. Exercise: make conductivity a design variable

The best way to check your understanding is to add a design variable. Make
the thermal conductivity k one: it needs three small edits in three files.
Then verify it against the closed form. Hint: k enters only through
α = k/(ρc) in φ = a² + 12αu, so ∂K/∂k = (u/k)·∂K/∂u, and `analytic.py`
already has `_dK_du`.

Things to notice once it works:

- Unlike Q, σ, v and τ, k belongs to the whole history, so there is no
  per-row logic. That is how the first prototype (`git show 5697d70`)
  treated power and the material properties.
- The step sizes depend on α (`Util::Calc_NonD_dt`) but are cut (3c). As for
  power and width, holding the nodes fixed gives the derivative of the true
  integral wherever the quadrature has converged.
- The geometry formula of step 6 applies unchanged: dh/dk = −T_k/(∇T·e).

<details>
<summary>Solution</summary>

```diff
--- src/oti_scalar.h
     DV_TAU,     // beam-off turnaround dwell duration (s)
+    DV_K,       // thermal conductivity (W/(m K))
     DV_COUNT

--- src/Init.cpp   (Init::FileRead_Material)
 	Init::SetValues(material.kon, values[0][2], 26.6, "Thermal Conductivity", 1, print);
+	material.kon = thesis::seed(thesis::DV_K, thesis::to_double(material.kon));

--- src/Grid.h   (oti_cols)
-				{"dT_dtau", thesis::DV_TAU},
+				{"dT_dtau", thesis::DV_TAU}, {"dT_dk", thesis::DV_K},
```

`Init::SetDiffusivity` then computes `a = kon/(rho·cps)` in `Real`, and the
derivative reaches every node through `ct = 12·a·τ`. For the check, subclass
`analytic.Reference`:

```python
def dk(self, x, y, z):
    return self.A * sum(p.pmod * self._int(
        p, lambda t, p=p: self._dK_du(p, t, x, y, z) * (self.t_obs - t) / self.m.k)
        for p in self._powered())
```

On the stationary spot and the moving track, `dT_dk` agrees with this to 1e-4
or better.
</details>

## Limits and gotchas

- **First order only.** `N = 1`. Second derivatives would need `N = 2` and a
  larger number per value.
- **One seeded row per run.** A cross-row Jacobian needs one run per row
  (`SeedSegment`), and `SeedSegment` needs path compression off.
- **Only T carries derivatives.** Solidification outputs (G, V, …) are
  computed from `to_double`'d nodes.
- **Node placement is held fixed** (3c). Where the quadrature has converged
  that is the true derivative for Q, σ and material properties; v and τ move
  node times, which is why they are handled by hand (step 4).
- **Six significant digits in the CSV.** Finite-difference checks against
  the output need care; see step 6.
- **σ is the `Width` parameter**, √6 × the Gaussian standard deviation.
- **Cost.** The OTI build makes each value 8 doubles instead of 1 and runs
  ~3× longer.
- **MPI.** Each rank computes its own sub-grid and no `Real` value is ever
  communicated, so OTI works unchanged under MPI (see
  [`../oti_ad/CHANGES.md`](../oti_ad/CHANGES.md)).

## File map

| file | AD content |
|---|---|
| `src/oti_scalar.h` | `Real`, `DesignVar`, `seed` / `to_double` / `deriv` |
| `src/CMakeLists.txt` | `THESIS_ENABLE_OTI`, locating Sparrow |
| `src/DataStructs.h` | which structs hold `Real`; `Settings::seed_seg`; `path_seg::swidth` |
| `src/Calc.cpp` | node construction: Q/σ seeds, `SeedCtx`, `seed_ctx`, `dv_tau` |
| `src/Grid.cpp` | x/y/z seeds in `Calc_T`; solidification kernels read nodes via `to_double` |
| `src/Grid.h` | `Real* T`, `get_T_deriv`, the `dT_d*` output columns |
| `src/Init.cpp`, `Init.h` | `SetValues(Real&, …)`, `SeedSegment`, the optional width-factor path column |
| `src/Util.cpp`, `Melt.cpp`, `Run.cpp` | the cuts in 3c |
| `ad_tutorial/analytic.py` | closed-form T and every derivative, for checking |
| `oti_ad/` | the first prototype's validation and plots (historical) |

The branch history tells the same story in order:

```bash
git log --oneline master..      # oldest at the bottom
```
