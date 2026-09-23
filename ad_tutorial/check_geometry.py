"""Step 5 of the tutorial: from temperature sensitivities to melt-pool geometry.

The melt pool is the region T >= T_L. Its half-width is the largest y on the
liquidus surface and its depth the largest -z: each is a support value
h(e) = max over the surface of e . x, for e = +y and e = -z. At the support
point x* the surface is tangent to the plane normal to e, so grad T is
parallel to e there, and differentiating T(x*(p), p) = T_L gives

    dh/dp = -(dT/dp) / (grad T . e)        evaluated at x*

(implicit-function / envelope theorem; derivations/ has the full argument).
Both factors come from ONE OTI snapshot: dT/dp is a control column and grad T
is (dT_dx, dT_dy, dT_dz). Nothing is differentiated through marching cubes;
it only locates x* on the real temperature field.

Two passes, as in the optimization pipeline: a plain-double run on a coarse
grid finds x*, then everything else is evaluated on a small fine grid around
it. dh/dp is compared with two references:

  exact  the same formula with dT/dp and grad T from analytic.py at x*, which
         checks the OTI columns where the formula samples them;
  FD     a central finite difference of the measured geometry, re-running the
         plain-double solver at p (1 +- STEP), which checks the formula itself.

Case: 316H stainless; a line, a beam-off dwell, and a second, shorter line so
the first line's heat still matters. The pool is measured at scan end.

Run:  python check_geometry.py
"""
import dataclasses

import numpy as np
from scipy import ndimage
from skimage import measure

import analytic
import thesis_case as tc

COARSE, FINE = 5e-6, 0.5e-6   # grid spacing (m) for locating x*, then measuring
STEP = 1e-2                   # relative finite-difference step
LIMIT_EXACT = 5e-4            # 0.05 %, as in the field checks
LIMIT_FD = 1e-2               # 1 %: x* is only known to one fine-grid column

# The CSV keeps six significant digits: 0.01 K near a 1709 K liquidus. That is
# enough to put the flat widest point of the pool a micron off, and there
# d(half-width)/dtau changes by about 2 % per micron (README, step 5). Shifting
# T0 and T_L by the same constant changes nothing the solver computes, since
# only T - T0 and T_L - T0 enter it, and spends the digits where they matter.
SHIFT = 1700.0
material = dataclasses.replace(tc.MATERIAL_316H, T0=tc.MATERIAL_316H.T0 - SHIFT,
                               TL=tc.MATERIAL_316H.TL - SHIFT)
beam = tc.BEAM_316H
V, TAU = tc.SPEED_316H, 0.2e-3
DWELL_ROW = 2
DIRECTIONS = {"half-width": np.array([0.0, 1.0, 0.0]),
              "depth": np.array([0.0, 0.0, -1.0])}
# Half-extent (m) of the fine box around x*: generous along the surface, where
# x* slides as p changes, tight along e, where it moves by about a micron.
BOX = {"half-width": (20e-6, 4e-6, 6e-6), "depth": (20e-6, 6e-6, 4e-6)}


def path(pmod=1.0, width_factor=None, v=V, tau=TAU):
    """Line, beam-off dwell, line. The controls act on the last line and the dwell."""
    return [tc.spot(-2.5, 0.0), tc.line(-1.0, V), tc.spot(-1.0, tau),
            tc.line(-0.4, v, pmod=pmod, width_factor=width_factor)]


def as_grid(df, column):
    """Snapshot column as an (nx, ny, nz) array, x slowest."""
    order = np.lexsort((df.z.values, df.y.values, df.x.values))
    shape = tuple(np.unique(df[c]).size for c in "xyz")
    return df[column].values[order].reshape(shape)


def support(df, name):
    """Support value h(e), the fractional grid index of x*, and x* in metres.

    Marching cubes runs on the real temperature only; its vertices are linear
    interpolations along grid edges, returned as fractional indices.
    """
    lo = np.array([df[c].min() for c in "xyz"])
    T = as_grid(df, "T")
    step = (np.array([df[c].max() for c in "xyz"]) - lo) / (np.array(T.shape) - 1)
    verts = measure.marching_cubes(T, material.TL)[0]
    values = (lo + verts * step) @ DIRECTIONS[name]
    best = int(np.argmax(values))
    return float(values[best]), verts[best], lo + verts[best] * step


def sensitivity(df, name, control):
    """dh/dp = -(dT/dp) / (grad T . e), with the fields sampled trilinearly at x*."""
    idx = support(df, name)[1].reshape(3, 1)
    at = lambda col: float(ndimage.map_coordinates(as_grid(df, col), idx, order=1)[0])
    grad = np.array([at("dT_dx"), at("dT_dy"), at("dT_dz")])
    return -at(control) / (grad @ DIRECTIONS[name])


def run(name, domain, oti=False, seed=None, **controls):
    return tc.run(tc.write_case(name, path=path(**controls), domain=domain, material=material,
                                beam=beam, seed_segment=seed), oti=oti)


coarse = tc.grid((-1.6e-3, -0.3e-3), (0.0, 0.2e-3), (-0.2e-3, 0.0), COARSE)
located = run("geometry_coarse", coarse)
print("316H pool at scan end, located on a "
      f"{COARSE * 1e6:g} um grid, measured on a {FINE * 1e6:g} um grid:")

ref = analytic.Reference(material, beam, path())
# (label, OTI column, seeded row, exact dT/dp, FD perturbation, scale, unit, display)
controls = [
    ("Q", "dT_dQ", None, ref.dQ, lambda s: dict(pmod=1 + s), beam.power, "um/W", 1e6),
    ("sig", "dT_dsig", None, ref.dsig, lambda s: dict(width_factor=1 + s), beam.width,
     "um/um", 1.0),
    ("v", "dT_dv", None, ref.dv, lambda s: dict(v=V * (1 + s)), V, "um/(m/s)", 1e6),
    ("tau", "dT_dtau", DWELL_ROW, lambda *x: ref.dtau(*x, DWELL_ROW),
     lambda s: dict(tau=TAU * (1 + s)), TAU, "um/ms", 1e3),
]

gate = tc.Gate()
for name in DIRECTIONS:
    center = support(located, name)[2]
    half = np.array(BOX[name])
    lo, hi = center - half, center + half
    lo[1], hi[2] = max(lo[1], 0.0), min(hi[2], 0.0)   # y >= 0 side, below the top surface
    box = tc.grid((lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2]), FINE)
    h, _, xstar = support(run("geometry_box", box), name)
    e = DIRECTIONS[name]
    normal_gradient = np.array(ref.grad(*xstar)) @ e
    print(f"\n{name} = {h * 1e6:.2f} um at x* = "
          f"({', '.join(f'{c * 1e6:.1f}' for c in xstar)}) um")
    print(f"  {'':4s}  {'d/d':<5s} {'OTI':>11s} {'exact':>11s} {'FD':>11s}")
    snapshots = {}
    for label, column, seed, dT_dp, perturb, scale, unit, show in controls:
        if seed not in snapshots:
            snapshots[seed] = run("geometry_box_oti", box, oti=True, seed=seed)
        ad = sensitivity(snapshots[seed], name, column)
        exact = -dT_dp(*xstar) / normal_gradient
        plus = support(run("geometry_box_fd", box, **perturb(STEP)), name)[0]
        minus = support(run("geometry_box_fd", box, **perturb(-STEP)), name)[0]
        fd = (plus - minus) / (2 * STEP * scale)
        print(f"  {'':4s}  {label:<5s} {ad * show:11.5g} {exact * show:11.5g} "
              f"{fd * show:11.5g}  {unit}")
        gate.check(f"d({name})/d{label}  vs exact", abs(ad - exact) / abs(exact),
                   LIMIT_EXACT)
        gate.check(f"d({name})/d{label}  vs FD", abs(ad - fd) / abs(fd), LIMIT_FD)
gate.exit()
