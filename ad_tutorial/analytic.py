"""Closed-form reference for the 3DThesis temperature and its derivatives.

For a Gaussian source on the x axis the solver evaluates (Stump & Plotkowski,
absorption efficiency 1)

    T(x) - T0 = A * sum over powered pieces of  Pmod * int K(u, x - x_b(t')) dt'

    K     = (phi_x phi_y phi_z)^(-1/2) exp(-3 [dx^2/phi_x + y^2/phi_y + z^2/phi_z])
    phi_i = a_i^2 + 12 alpha u,      u = t_obs - t'   (conduction time)
    A     = 2 P / (rho c (pi/3)^(3/2))

where a_x = a_y = a is the lateral Beam.txt width (times the row's width
factor) and a_z the depth. The solver approximates each time integral with
Gauss-Legendre quadrature on adaptive steps; here scipy.integrate.quad
evaluates them to ~1e-10, and every derivative is taken by hand, under the
integral sign. The derivative formulas are the point of this file:

    dT/dx    = A sum Pmod int K (-6 dx / phi_x)                 (all pieces)
    dT/dQ_j  = A/P      int_j K                                 (piece j only)
    dT/da_j  = A Pmod_j int_j K a [6 (dx^2 + y^2)/phi^2 - 2/phi]
    dK/du    = K 12 alpha sum_i [3 r_i^2/phi_i^2 - 1/(2 phi_i)]

    dT/dDt_j = A [ sum_{i<j} Pmod_i int_i dK/du                 (older heat ages)
                 + Pmod_j/Dt_j int_j K                          (longer piece, more heat)
                 + Pmod_j/Dt_j int_j dK/du (t_j - t') ]         (piece j stretches)
    dT/dv_j  = -(L_j / v_j^2) dT/dDt_j                          (Dt_j = L_j / v_j)
    dT/dtau  = A sum_{i before the dwell} Pmod_i int_i dK/du    (only aging survives)

Pieces deposited after the seeded row contribute nothing to dT/dv or dT/dtau:
their deposition and the scan-end observation move by the same amount.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad


@dataclass
class Piece:
    row: int        # 0-based Path.txt data row (what Compute/SeedSegment names)
    t0: float       # start time (s)
    t1: float       # end time (s)
    x0: float       # beam x at t0 (m)
    vx: float       # signed beam speed along x (m/s); 0 for a spot
    pmod: float
    width: float    # effective lateral width a (m)


class Reference:
    """Exact T and derivatives for a path built from thesis_case rows."""

    def __init__(self, material, beam, path):
        self.m, self.b = material, beam
        self.A = 2.0 * beam.power / (material.rho * material.c * (np.pi / 3.0) ** 1.5)
        self.pieces, t, x_prev = [], 0.0, None
        for row, (mode, x_mm, pmod, param, wf) in enumerate(path):
            x = x_mm * 1e-3
            width = beam.width * (wf if wf is not None else 1.0)
            if mode == 1:                     # spot: hold at x for `param` s
                dt, vx, x0 = param, 0.0, x
            else:                             # line: from x_prev to x at `param` m/s
                dt = abs(x - x_prev) / param
                vx, x0 = np.sign(x - x_prev) * param, x_prev
            self.pieces.append(Piece(row, t, t + dt, x0, vx, pmod, width))
            t, x_prev = t + dt, x
        self.t_obs = t

    # -- integrand pieces -----------------------------------------------------
    def _geom(self, p, tp, x, y, z):
        u = self.t_obs - tp
        ph = p.width ** 2 + 12.0 * self.m.alpha * u
        pz = self.b.depth ** 2 + 12.0 * self.m.alpha * u
        dx = x - (p.x0 + p.vx * (tp - p.t0))
        K = (ph * ph * pz) ** -0.5 * np.exp(-3.0 * ((dx * dx + y * y) / ph + z * z / pz))
        return K, dx, ph, pz

    def _K(self, p, tp, x, y, z):
        return self._geom(p, tp, x, y, z)[0]

    def _dK_du(self, p, tp, x, y, z):
        K, dx, ph, pz = self._geom(p, tp, x, y, z)
        s = (3 * dx * dx / ph ** 2 - 0.5 / ph + 3 * y * y / ph ** 2 - 0.5 / ph
             + 3 * z * z / pz ** 2 - 0.5 / pz)
        return K * 12.0 * self.m.alpha * s

    def _int(self, p, f):
        if p.t1 <= p.t0:
            return 0.0
        return quad(f, p.t0, p.t1, limit=200)[0]

    def _powered(self):
        return [p for p in self.pieces if p.pmod > 0.0 and p.t1 > p.t0]

    def _seeded(self, seed_row):
        return self.pieces[-1] if seed_row is None else self.pieces[seed_row]

    # -- quantities -----------------------------------------------------------
    def T(self, x, y, z):
        return self.m.T0 + self.A * sum(
            p.pmod * self._int(p, lambda t: self._K(p, t, x, y, z))
            for p in self._powered())

    def grad(self, x, y, z):
        out = []
        for axis in range(3):
            def f(t, p, axis=axis):
                K, dx, ph, pz = self._geom(p, t, x, y, z)
                r, phi = ((dx, ph), (y, ph), (z, pz))[axis]
                return K * (-6.0 * r / phi)
            out.append(self.A * sum(p.pmod * self._int(p, lambda t, p=p: f(t, p))
                                    for p in self._powered()))
        return out

    def dQ(self, x, y, z, seed_row=None):
        p = self._seeded(seed_row)
        if p.pmod <= 0.0:
            return 0.0
        return self.A / self.b.power * self._int(p, lambda t: self._K(p, t, x, y, z))

    def dsig(self, x, y, z, seed_row=None):
        p = self._seeded(seed_row)
        if p.pmod <= 0.0:
            return 0.0

        def f(t):
            K, dx, ph, _ = self._geom(p, t, x, y, z)
            return K * p.width * (6.0 * (dx * dx + y * y) / ph ** 2 - 2.0 / ph)
        return self.A * p.pmod * self._int(p, f)

    def _aging(self, before, x, y, z):
        """A sum_{i < before} Pmod_i int_i dK/du: heat that gets older."""
        return self.A * sum(p.pmod * self._int(p, lambda t, p=p: self._dK_du(p, t, x, y, z))
                            for p in self._powered() if p.row < before)

    def dv(self, x, y, z, seed_row=None):
        p = self._seeded(seed_row)
        if p.vx == 0.0:
            return 0.0                        # spots have no speed
        dt = p.t1 - p.t0
        own = 0.0
        if p.pmod > 0.0:
            own = self.A * p.pmod / dt * (
                self._int(p, lambda t: self._K(p, t, x, y, z))
                + self._int(p, lambda t: self._dK_du(p, t, x, y, z) * (p.t1 - t)))
        dT_dDt = self._aging(p.row, x, y, z) + own
        speed = abs(p.vx)
        return -(speed * dt) / speed ** 2 * dT_dDt

    def dtau(self, x, y, z, seed_row):
        p = self.pieces[seed_row]
        if p.vx != 0.0 or p.pmod > 0.0:
            return 0.0                        # only beam-off spots are dwells
        return self._aging(p.row, x, y, z)
