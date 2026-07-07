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
// infinitesimal direction. Diffusivity a = kon/(rho*cps) is derived, so it is
// NOT a slot -- its derivative is inherited from kon/rho/cps automatically.
// DV_COUNT (kept last) is the number of directions, i.e. the OTI parameter M.
enum DesignVar {
    DV_X = 0,   // evaluation point x
    DV_Y,       // evaluation point y
    DV_Z,       // evaluation point z
    DV_Q,       // beam power
    DV_KON,     // thermal conductivity
    DV_RHO,     // density
    DV_CPS,     // specific heat
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
