/****************************************************************************
 * Scalar abstraction for optional automatic differentiation (AD).
 *
 * 3dThesis is written against a single type alias, `Real`, instead of bare
 * `double` on the physics path. By default `Real` IS `double`, so an ordinary
 * build is byte-for-byte identical to upstream.
 *
 * With -DTHESIS_ENABLE_OTI=ON, `Real` becomes an order-truncated-imaginary
 * (OTI) number from cpp_oti_lib: a value that also carries the first-order
 * partial derivatives of that value with respect to a fixed set of "design
 * variables" (see DesignVar). Every arithmetic operator and elementary
 * function is overloaded for OTI numbers, so the existing kernels propagate
 * those derivatives automatically -- e.g. temperature comes back with
 * dT/dx, dT/dq, dT/dkon, ... at no change to the physics expressions.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ****************************************************************************/

#pragma once

namespace thesis {

// Independent quantities we differentiate with respect to; each maps to one OTI
// infinitesimal direction. The set is deliberately minimal -- only the process
// CONTROLS we optimise plus the spatial gradient used to turn a field
// sensitivity into a melt-pool edge/depth sensitivity:
//   X,Y,Z -> spatial gradient dT/dn at the isotherm (implicit-function edge sens)
//   Q     -> SEEDED segment's effective beam power (W)   (control)
//   SIG   -> SEEDED segment's lateral beam width (m)     (control; ax and ay together)
//   V     -> SEEDED segment's scan speed (m/s)           (control)
//   DWELL -> SEEDED beam-off dwell row's duration (s)    (control)
// The seeded segment defaults to the LAST line of the path file (per path) --
// the optimizer's "current" segment -- and can be pointed at any HISTORY
// segment via Settings.txt Compute/SeedSegment (0-based path-row index) for
// cross-segment influence Jacobians ("seed segment j, observe at scan end").
// Q and SIG are seeded at quadrature-node construction in Calc.cpp, only for
// nodes on the seeded segment, so dT_dQ/dT_dsig are exact per-segment control
// derivatives and every other segment deliberately carries zero derivative.
// X/Y/Z are seeded per evaluation point in Grid::Calc_T.
// V is structurally different: at FIXED path geometry it changes only the
// seeded segment's DURATION dt = L/v, which (a) stretches that segment's
// conduction times and quadrature weights and (b) -- because the snapshot is
// taken at scan end -- shifts the observation time relative to every EARLIER
// node, while later nodes shift together with the observation and cancel. So
// unlike Q/SIG, dT_dv has a history channel; see the three-zone rule
// (SeedCtx / dv_tau) in Calc.cpp. Beam POSITIONS are v-independent
// (geometry fixed), so no spatial term.
// DWELL is V's simpler sibling: a beam-off dwell row carries no nodes (culled
// by qmod>0), so ONLY the earlier-node history shift survives -- no stretch,
// no weight channel. Seeding the dwell duration reuses dv_tau's history branch
// verbatim; see the dwell branch of SeedCtx in Calc.cpp.
// Material properties (kon/rho/cps) are NOT differentiated -- they are fixed,
// not controls -- which keeps the algebra small (M=7).
// DV_COUNT (kept last) is the number of directions, i.e. the OTI parameter M.
enum DesignVar {
    DV_X = 0,   // evaluation point x
    DV_Y,       // evaluation point y
    DV_Z,       // evaluation point z
    DV_Q,       // current segment's beam power (W)
    DV_SIG,     // current segment's lateral beam width sigma (m)
    DV_V,       // current segment's scan speed (m/s)
    DV_DWELL,   // SEEDED beam-off dwell row's duration (s)     (control)
    DV_COUNT
};

} // namespace thesis

#ifdef THESIS_ENABLE_OTI

#include "otinum/otinum.hpp"

namespace thesis {

using Real = oti::otinum<DV_COUNT, 1, double>;

// Lift a plain value into design-variable slot `dv` (seeds d/d(dv) = 1).
inline Real seed(int dv, double v) { return Real::variable(dv, v); }

// Real (value) part -- for boundaries that must stay in plain double:
// integer indices, file I/O, and adaptive control flow.
inline double to_double(const Real& s) { return s.real(); }

// Partial derivative d s / d (design variable dv).
inline double deriv(const Real& s, int dv)
{
    Real::alpha_type alpha{};
    alpha[static_cast<std::size_t>(dv)] = 1;
    return s.partial(alpha);
}

} // namespace thesis

#else // default: plain double build

namespace thesis {

using Real = double;

inline Real seed(int /*dv*/, double v) { return v; }
inline double to_double(Real s) { return s; }
inline double deriv(Real /*s*/, int /*dv*/) { return 0.0; }

} // namespace thesis

#endif // THESIS_ENABLE_OTI

// Match how 3dThesis already pulls std::vector/std::string into the global
// namespace, so the rest of the code can just say `Real`.
using thesis::Real;
