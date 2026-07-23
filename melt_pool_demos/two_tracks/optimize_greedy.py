"""Greedy-only P/sigma/velocity optimization for the two-track experiment.

The scan lines are divided into uniform control blocks (2.5 mm by default).
Each block normally uses one OTI endpoint solve per Newton iteration to obtain
all three control derivatives.  An optional continuous-only experiment adds a
second observation shortly before each track-2 block endpoint to penalize a
pool that is still changing as it crosses the endpoint.  No coupled or
future-segment objective is used.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
DEMOS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, DEMOS)

os.environ.setdefault("MP_GEOM", "two_tracks")
os.environ.setdefault("MP_CASE_DIR", os.path.join(HERE, "cases", "snapcase"))

from common import mp_lib as R  # noqa: E402


U_LO = np.array([0.05, 80e-6, 0.5])
U_HI = np.array([3.0, 400e-6, 8.0])
U_SCALE = np.array([1.0, 200e-6, 3.0])
DU_MAX = np.array([0.50, 80e-6, 1.5])
BACKTRACK = (1.0, 0.5, 0.25, 0.125, 0.0625)
STEP_REG = 2.5e-2
VELOCITY_REG = 1.0e-2
MIN_IMPROVEMENT = 1e-3
DWELL_SCALE = 2.5e-4
DWELL_DU_MAX = 2.0e-4
DWELL_REG = 1.0e-3
# Cool-and-slow seed for the first block after the turnaround (a heat-excess
# transient that pins the Newton in a high-velocity basin); see the measured
# turn/apex finding in common/greedy_raster.py.
COOL_SEED = np.array([0.2, R.SIGMA, 1.0])   # 30 W, nominal sigma, 1 m/s


def segment_tag(segment_mm: float) -> str:
    return f"{segment_mm:g}".replace(".", "p")


def result_path(name: str) -> str:
    path = os.path.join(HERE, "results", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def dwell_for(policy: str) -> float:
    if policy == "continuous":
        return 0.0
    with open(result_path("min_dwell.json")) as f:
        return float(json.load(f)["min_dwell_s"])


def target_value(segment_mm: float = 2.5) -> np.ndarray:
    """Developed nominal pool at the end of the first 10 mm track."""
    div, powered, _ = R.build_path(0.0, seg=segment_mm)
    info = R.seg_info(div, powered)
    pmods = {i: R.PMOD0 for i in powered}
    endpoint = [segment for segment in info if segment["line"] == 0][-1]
    R.run_segment(div, endpoint["idx"], pmods, R.BIN_DBL)
    width, depth, _asym, _clip = R.measure_dims()
    return np.array([width, depth])


def endpoint_observation(div: list[str], idx: int, pmods: dict,
                         sigmas: dict, velocities: dict, u: np.ndarray,
                         binary: str, dwells: dict | None = None,
                         seed_idx: int | None = None
                         ) -> tuple[np.ndarray, np.ndarray | None]:
    trial_pmods, trial_sigmas, trial_velocities = (
        dict(pmods), dict(sigmas), dict(velocities))
    trial_pmods[idx], trial_sigmas[idx], trial_velocities[idx] = map(float, u)
    R.run_segment(div, idx, trial_pmods, binary, sigma=float(u[1]),
                  sig_hist=trial_sigmas, vels=trial_velocities,
                  seed_idx=((idx if seed_idx is None else seed_idx)
                            if binary == R.BIN_OTI else None),
                  dwells=dwells)
    if binary == R.BIN_OTI:
        s = R.sens()
        q = np.array([s["width"], s["depth"]])
        J = np.array([
            [s["dwidth_dQ"] * R.P_BASE, s["dwidth_dsig"], s["dwidth_dv"]],
            [s["ddepth_dQ"] * R.P_BASE, s["ddepth_dsig"], s["ddepth_dv"]],
        ])
        return q, J
    width, depth, _asym, _clip = R.measure_dims()
    return np.array([width, depth]), None


def shortened_segment(div: list[str], idx: int,
                      distance_before_end_mm: float) -> list[str]:
    """Return a path whose current segment ends a fixed distance upstream.

    The shortened row retains the current segment's controls.  Running the
    same endpoint observation on this path therefore measures the pool shortly
    before the true block endpoint without introducing another control block.
    """
    out = list(div)
    end = out[idx].strip().split("\t")
    if end[0] != "0" or float(end[4]) == 0.0:
        raise ValueError(f"row {idx} is not a powered raster segment")
    previous = None
    for row in reversed(out[:idx]):
        fields = row.strip().split("\t")
        if fields[0] in ("0", "1"):
            previous = np.array([float(fields[1]), float(fields[2])])
            break
    if previous is None:
        raise ValueError(f"row {idx} has no preceding path point")
    endpoint = np.array([float(end[1]), float(end[2])])
    delta = endpoint - previous
    length = float(np.linalg.norm(delta))
    # Path coordinates are stored in millimetres (see Path.txt headers), so
    # the requested millimetre window is already in the row-coordinate units.
    window = distance_before_end_mm
    if not 0.0 < window < length:
        raise ValueError(
            f"settling window {distance_before_end_mm:g} mm must be shorter "
            f"than segment {idx} ({length:g} mm)")
    probe = endpoint - (window / length) * delta
    end[1], end[2] = repr(float(probe[0])), repr(float(probe[1]))
    out[idx] = "\t".join(end) + "\n"
    return out


def settling_observation(div: list[str], idx: int, pmods: dict,
                         sigmas: dict, velocities: dict, u: np.ndarray,
                         binary: str, window_mm: float,
                         dwells: dict | None = None
                         ) -> tuple[np.ndarray, np.ndarray | None,
                                    np.ndarray, np.ndarray | None]:
    """Dimensions and their change over the final ``window_mm`` of a block."""
    q_end, jac_end = endpoint_observation(
        div, idx, pmods, sigmas, velocities, u, binary, dwells=dwells)
    probe_div = shortened_segment(div, idx, window_mm)
    q_probe, jac_probe = endpoint_observation(
        probe_div, idx, pmods, sigmas, velocities, u, binary, dwells=dwells)
    delta_q = q_end - q_probe
    delta_jac = (None if jac_end is None
                 else jac_end - jac_probe)
    return q_end, jac_end, delta_q, delta_jac


def start_observation(div: list[str], idx: int, pmods: dict, sigmas: dict,
                      velocities: dict, u: np.ndarray, window_mm: float,
                      dwells: dict | None, dwell_idx: int | None
                      ) -> tuple[np.ndarray, np.ndarray | None]:
    """Pool near the block START -- the point ``window_mm`` upstream of the
    endpoint, i.e. a distance ``length - window_mm`` after the block begins.

    This is the observation that sees the post-dwell cold start: a long dwell
    fully solidifies the substrate, so the pool re-nucleates shallow at the
    block start even when the block endpoint is driven back to target.  The
    OTI Jacobian includes the preceding-dwell column (via ``dwell_column`` on
    the same shortened path) so the penalty is differentiable in the dwell.
    """
    probe_div = shortened_segment(div, idx, window_mm)
    q, jac = endpoint_observation(
        probe_div, idx, pmods, sigmas, velocities, u[:3], R.BIN_OTI,
        dwells=dwells, seed_idx=idx)
    if dwell_idx is not None:
        jac_dwell = dwell_column(
            probe_div, idx, dwell_idx, pmods, sigmas, velocities, u, dwells)
        jac = np.column_stack((jac, jac_dwell))
    return q, jac


def start_depth(div: list[str], idx: int, pmods: dict, sigmas: dict,
                velocities: dict, u: np.ndarray, window_mm: float,
                dwells: dict | None) -> np.ndarray:
    """Double-precision block-start dimensions for line-search trials."""
    probe_div = shortened_segment(div, idx, window_mm)
    q, _ = endpoint_observation(
        probe_div, idx, pmods, sigmas, velocities, u[:3], R.BIN_DBL,
        dwells=dwells)
    return q


def refine_depth_in_endpoint_nullspace(
        div: list[str], idx: int, pmods: dict, sigmas: dict,
        velocities: dict, u: np.ndarray, target: np.ndarray,
        endpoint_tolerance: float, window_mm: float,
        max_iter: int = 6) -> tuple[np.ndarray, list[dict], int, int]:
    """Reduce terminal depth change without leaving the endpoint tolerance.

    Width and depth give a 2x3 endpoint Jacobian.  Its one-dimensional null
    space is the local control direction that leaves both dimensions unchanged
    to first order.  This refinement uses that direction only, and accepts a
    trial only when its endpoint remains inside the existing tolerance band.
    """
    u = np.asarray(u, dtype=float).copy()
    history: list[dict] = []
    oti_runs = double_runs = 0
    for iteration in range(max_iter):
        q, jac, delta_q, delta_jac = settling_observation(
            div, idx, pmods, sigmas, velocities, u, R.BIN_OTI, window_mm)
        oti_runs += 2
        endpoint_residual = (q - target) / target
        endpoint_worst = float(np.max(np.abs(endpoint_residual)))
        depth_change = float(delta_q[1] / target[1])
        print(
            f"    null-depth it{iteration}: endpoint={endpoint_worst*100:.2f}% "
            f"depth-change={depth_change*100:.2f}% "
            f"P={u[0]*R.P_BASE:.1f} W sigma={u[1]*1e6:.1f} um "
            f"v={u[2]:.2f} m/s", flush=True)
        if endpoint_worst > endpoint_tolerance:
            print("    endpoint outside tolerance; skipping null refinement",
                  flush=True)
            break
        if abs(depth_change) <= endpoint_tolerance:
            break

        # Work in scaled control coordinates so SVD and the slope projection
        # are not dominated by the disparate physical units of Pmod, sigma,
        # and velocity.
        endpoint_jac_scaled = (jac / target[:, None]) * U_SCALE[None, :]
        _left, singular, right_t = np.linalg.svd(
            endpoint_jac_scaled, full_matrices=True)
        if singular[-1] <= 1e-12:
            print("    rank-deficient endpoint Jacobian; stopping refinement",
                  flush=True)
            break
        null_direction = right_t[-1]
        depth_gradient_scaled = (
            delta_jac[1] / target[1] * U_SCALE)
        authority = float(depth_gradient_scaled @ null_direction)
        if abs(authority) <= 1e-10:
            print("    null direction has no depth-slope authority; stopping",
                  flush=True)
            break

        scaled_step = (-depth_change / authority) * null_direction
        physical_step = U_SCALE * scaled_step
        ratios = np.divide(
            DU_MAX, np.abs(physical_step),
            out=np.full_like(DU_MAX, np.inf),
            where=np.abs(physical_step) > 0.0)
        physical_step *= min(1.0, float(np.min(ratios)))

        accepted = False
        for alpha in BACKTRACK:
            trial_u = np.clip(u + alpha * physical_step, U_LO, U_HI)
            q_trial, _, delta_trial, _ = settling_observation(
                div, idx, pmods, sigmas, velocities, trial_u,
                R.BIN_DBL, window_mm)
            double_runs += 2
            trial_endpoint = (q_trial - target) / target
            trial_worst = float(np.max(np.abs(trial_endpoint)))
            trial_depth_change = float(delta_trial[1] / target[1])
            if (trial_worst <= endpoint_tolerance
                    and abs(trial_depth_change)
                    < abs(depth_change) * (1.0 - MIN_IMPROVEMENT)):
                u = trial_u
                history.append({
                    "iteration": iteration + 1,
                    "alpha": alpha,
                    "worst_relative_error": trial_worst,
                    "depth_change_relative": trial_depth_change,
                    "pmod": float(u[0]),
                    "sigma": float(u[1]),
                    "velocity": float(u[2]),
                })
                accepted = True
                break
        if not accepted:
            print("    no endpoint-feasible null step; stopping refinement",
                  flush=True)
            break
    return u, history, oti_runs, double_runs


def dwell_column(div: list[str], idx: int, dwell_idx: int, pmods: dict,
                 sigmas: dict, velocities: dict, u: np.ndarray,
                 dwells: dict) -> np.ndarray:
    """Width/depth derivative with respect to the preceding dwell duration."""
    _q, _ = endpoint_observation(
        div, idx, pmods, sigmas, velocities, u[:3], R.BIN_OTI,
        dwells=dwells, seed_idx=dwell_idx)
    s = R.sens()
    return np.array([s["dwidth_ddwell"], s["ddepth_ddwell"]])


def greedy_initialize(div: list[str], powered: list[int], hop_after: dict,
                      info: list[dict], target: np.ndarray, max_iter: int,
                      tolerance: float,
                      block_rate_limit: np.ndarray | None = None,
                      within_line_warm_start: str = "same-position",
                      turn_seed: np.ndarray | None = None,
                      optimize_turn_dwell: bool = False,
                      initial_dwell_s: float = 0.0,
                      max_turn_dwell_s: float = 0.0,
                      dwell_reg: float = DWELL_REG,
                      settling_window_mm: float = 0.0,
                      settling_weight: float = 1.0,
                      settling_tolerance: float = 0.01,
                      start_window_mm: float = 0.0,
                      start_weight: float = 1.0,
                      start_tolerance: float = 0.01) -> dict:
    """Sequential endpoint regulation with P, sigma, and velocity.

    ``block_rate_limit`` optionally bounds each block's committed controls to
    within +-(dPmod, dsigma, dv) of the previous block's, spreading large
    corrections (e.g. the turnaround) over several blocks instead of allowing
    one block to starve the ground it crosses while matching its endpoint.

    ``within_line_warm_start`` controls blocks after the first block of a scan
    line. ``same-position`` reuses the corresponding block from the preceding
    line; ``previous-block`` continues from the immediately upstream optimized
    block. The first block after a turnaround is deliberately reseeded below.
    """
    pmods = {i: R.PMOD0 for i in powered}
    sigmas = {i: R.SIGMA for i in powered}
    velocities = {i: R.V for i in powered}
    history, oti_runs, double_runs = [], 0, 0
    by_position = {}
    u_prev = np.array([R.PMOD0, R.SIGMA, R.V])
    turn_seed = COOL_SEED if turn_seed is None else turn_seed
    committed_dwells: dict[int, float] = {}
    current_line, position = -1, -1
    for segment in info:
        position = 0 if segment["line"] != current_line else position + 1
        current_line = segment["line"]
        idx = segment["idx"]
        label = f"line{segment['line'] + 1}_block{position + 1}"
        if block_rate_limit is None:
            u_lo, u_hi = U_LO, U_HI
        else:
            u_lo = np.maximum(U_LO, u_prev - block_rate_limit)
            u_hi = np.minimum(U_HI, u_prev + block_rate_limit)
        # First block after the turnaround (position 0, line > 0): heat-excess
        # transient -- seed cool-and-slow so the Newton finds the slow-down
        # basin instead of pinning at high velocity (see common/greedy_raster).
        if position == 0 and segment["line"] > 0:
            warm = turn_seed
        elif within_line_warm_start == "previous-block" and position > 0:
            warm = u_prev
        else:
            warm = by_position.get(position, np.array([
                pmods[idx], sigmas[idx], velocities[idx]]))
        u = np.clip(warm, u_lo, u_hi)
        dwell_idx = None
        if optimize_turn_dwell and position == 0 and segment["line"] > 0:
            dwell_idx = idx - 1
            fields = div[dwell_idx].strip().split("\t")
            if fields[0] != "1" or float(fields[4]) != 0.0:
                raise RuntimeError(
                    f"row {dwell_idx} before {label} is not a beam-off dwell")
            committed_dwells[dwell_idx] = initial_dwell_s
            u = np.append(u, initial_dwell_s)
        print(f"  greedy {label}:", flush=True)
        for iteration in range(max_iter):
            if dwell_idx is not None:
                committed_dwells[dwell_idx] = float(u[3])
            active_dwells = committed_dwells or None
            settling_active = (settling_window_mm > 0.0
                               and segment["line"] > 0)
            start_active = (start_window_mm > 0.0
                            and segment["line"] > 0)
            if settling_active:
                q, J, delta_q, delta_J = settling_observation(
                    div, idx, pmods, sigmas, velocities, u[:3], R.BIN_OTI,
                    settling_window_mm, dwells=active_dwells)
                oti_runs += 2
            else:
                q, J = endpoint_observation(
                    div, idx, pmods, sigmas, velocities, u[:3], R.BIN_OTI,
                    dwells=active_dwells)
                delta_q = delta_J = None
                oti_runs += 1
            if dwell_idx is not None:
                Jd = dwell_column(
                    div, idx, dwell_idx, pmods, sigmas, velocities, u,
                    committed_dwells)
                J = np.column_stack((J, Jd))
                oti_runs += 1
            if start_active:
                # Depth at the block start after the (frozen or trial) dwell.
                # The start Jacobian already carries the dwell column when the
                # dwell is still free, so the penalty steers Delta directly.
                q_start, J_start = start_observation(
                    div, idx, pmods, sigmas, velocities, u, start_window_mm,
                    active_dwells, dwell_idx)
                oti_runs += 2 if dwell_idx is not None else 1
                start_residual = (q_start[1] - target[1]) / target[1]
                start_worst = abs(float(start_residual))
                start_jac_row = J_start[1] / target[1]
            else:
                start_residual = 0.0
                start_worst = 0.0
                start_jac_row = None
            residual = (q - target) / target
            endpoint_merit = 0.5 * float(residual @ residual)
            physical_merit = endpoint_merit
            if settling_active:
                settling_residual = delta_q / target
                physical_merit += (0.5 * settling_weight
                                   * float(settling_residual
                                           @ settling_residual))
                settling_worst = float(
                    np.max(np.abs(settling_residual)))
            else:
                settling_residual = None
                settling_worst = 0.0
            if start_active:
                physical_merit += 0.5 * start_weight * start_residual ** 2
            vrel = (u[2] - R.V) / R.V
            merit = (physical_merit
                     + 0.5 * VELOCITY_REG * vrel * vrel)
            if dwell_idx is not None:
                merit += 0.5 * dwell_reg * (u[3] / DWELL_SCALE) ** 2
            worst = float(np.max(np.abs(residual)))
            print(f"    it{iteration}: merit={merit:.6f} "
                  f"worst={worst*100:.2f}% P={u[0]*R.P_BASE:.1f} W "
                  f"sigma={u[1]*1e6:.1f} um v={u[2]:.2f} m/s"
                  + (f" settle={settling_worst*100:.2f}%"
                     if settling_active else "")
                  + (f" start={start_worst*100:.2f}%"
                     if start_active else "")
                  + (f" dwell={u[3]*1e3:.3f} ms"
                     if dwell_idx is not None else ""), flush=True)
            if (worst <= tolerance
                    and settling_worst <= settling_tolerance
                    and start_worst <= start_tolerance):
                break

            u_scale = (np.append(U_SCALE, DWELL_SCALE)
                       if dwell_idx is not None else U_SCALE)
            du_max = (np.append(DU_MAX, DWELL_DU_MAX)
                      if dwell_idx is not None else DU_MAX)
            Jn = J / target[:, None]
            solve_residual = residual
            if settling_active:
                Jn = np.vstack((
                    Jn,
                    np.sqrt(settling_weight)
                    * delta_J / target[:, None],
                ))
                solve_residual = np.concatenate((
                    solve_residual,
                    np.sqrt(settling_weight) * settling_residual,
                ))
            if start_active:
                Jn = np.vstack((
                    Jn,
                    np.sqrt(start_weight) * start_jac_row[None, :],
                ))
                solve_residual = np.concatenate((
                    solve_residual,
                    [np.sqrt(start_weight) * start_residual],
                ))
            Js = Jn * u_scale[None, :]
            lhs = Js.T @ Js + STEP_REG * np.eye(u.size)
            rhs = -(Js.T @ solve_residual)
            vscale = u_scale[2] / R.V
            lhs[2, 2] += VELOCITY_REG * vscale * vscale
            rhs[2] -= VELOCITY_REG * vscale * vrel
            if dwell_idx is not None:
                lhs[3, 3] += dwell_reg
                rhs[3] -= dwell_reg * u[3] / DWELL_SCALE
            du = np.clip(u_scale * np.linalg.solve(lhs, rhs),
                         -du_max, du_max)
            accepted = False
            for alpha in BACKTRACK:
                trial_lo = (np.append(u_lo, 0.0)
                            if dwell_idx is not None else u_lo)
                trial_hi = (np.append(u_hi, max_turn_dwell_s)
                            if dwell_idx is not None else u_hi)
                trial_u = np.clip(u + alpha * du, trial_lo, trial_hi)
                trial_dwells = dict(committed_dwells)
                if dwell_idx is not None:
                    trial_dwells[dwell_idx] = float(trial_u[3])
                trial_active = trial_dwells or None
                if settling_active:
                    q_trial, _, trial_delta_q, _ = settling_observation(
                        div, idx, pmods, sigmas, velocities, trial_u[:3],
                        R.BIN_DBL, settling_window_mm, dwells=trial_active)
                    double_runs += 2
                else:
                    q_trial, _ = endpoint_observation(
                        div, idx, pmods, sigmas, velocities, trial_u[:3],
                        R.BIN_DBL, dwells=trial_active)
                    trial_delta_q = None
                    double_runs += 1
                r_trial = (q_trial - target) / target
                trial_endpoint_merit = 0.5 * float(r_trial @ r_trial)
                trial_physical_merit = trial_endpoint_merit
                if settling_active:
                    trial_settling_residual = trial_delta_q / target
                    trial_physical_merit += (
                        0.5 * settling_weight
                        * float(trial_settling_residual
                                @ trial_settling_residual))
                else:
                    trial_settling_residual = None
                # When an augmented term is active the priced direction must
                # not purchase a worse width/depth endpoint to improve it.
                endpoint_guard = (
                    float(np.max(np.abs(r_trial))) <= max(worst, tolerance)
                    if (settling_active or start_active) else True)
                if start_active:
                    q_start_trial = start_depth(
                        div, idx, pmods, sigmas, velocities, trial_u,
                        start_window_mm, trial_active)
                    double_runs += 1
                    trial_start_residual = (
                        (q_start_trial[1] - target[1]) / target[1])
                    trial_physical_merit += (
                        0.5 * start_weight * trial_start_residual ** 2)
                vrel_trial = (trial_u[2] - R.V) / R.V
                trial_merit = (trial_physical_merit
                               + 0.5 * VELOCITY_REG
                               * vrel_trial * vrel_trial)
                if dwell_idx is not None:
                    trial_merit += (0.5 * dwell_reg
                                    * (trial_u[3] / DWELL_SCALE) ** 2)
                # Width/depth regulation remains a hard priority.  The weak
                # velocity and dwell terms shape the underdetermined Newton
                # direction but cannot purchase a worse physical endpoint.
                comparison_merit = (physical_merit
                                    if (settling_active or start_active)
                                    else endpoint_merit)
                if (trial_physical_merit
                        < comparison_merit * (1.0 - MIN_IMPROVEMENT)
                        and endpoint_guard):
                    u = trial_u
                    history.append({
                        "block": label, "iteration": iteration + 1,
                        "alpha": alpha, "merit": trial_merit,
                        "worst_relative_error": float(
                            np.max(np.abs(r_trial))),
                        "pmod": float(u[0]), "sigma": float(u[1]),
                        "velocity": float(u[2]),
                    })
                    if dwell_idx is not None:
                        history[-1]["dwell_s"] = float(u[3])
                    if settling_active:
                        history[-1]["settling_worst_relative"] = float(
                            np.max(np.abs(trial_settling_residual)))
                    if start_active:
                        history[-1]["start_worst_relative"] = float(
                            abs(trial_start_residual))
                    accepted = True
                    break
            if not accepted:
                if dwell_idx is not None:
                    # The priced joint direction has reached its useful dwell
                    # tradeoff.  Freeze Delta, then use the remaining Newton
                    # iterations to recover the hard width/depth tolerance
                    # with P/sigma/v only (alternating dwell-step-then-resolve).
                    committed_dwells[dwell_idx] = float(u[3])
                    print("    fixing dwell and re-solving process controls",
                          flush=True)
                    u = u[:3]
                    dwell_idx = None
                    continue
                print("    no improving greedy step; stopping block",
                      flush=True)
                break
        pmods[idx], sigmas[idx], velocities[idx] = map(float, u[:3])
        if dwell_idx is not None:
            committed_dwells[dwell_idx] = float(u[3])
        by_position[position] = u[:3].copy()
        u_prev = u[:3].copy()
        if idx in hop_after:
            hop = hop_after[idx]
            pmods[hop], sigmas[hop], velocities[hop] = map(float, u[:3])
    return {"oti_runs": oti_runs, "double_runs": double_runs,
            "history": history,
            "turnaround_dwells_s": list(committed_dwells.values()),
            "base_controls": (pmods, sigmas, velocities)}


def schedule_from(info: list[dict], base: tuple[dict, dict, dict]) -> list[dict]:
    pmods, sigmas, velocities = base
    return [{
        "segment": number,
        "line": segment["line"] + 1,
        "end_x_mm": segment["x"],
        "end_y_mm": segment["y"],
        "power_w": R.P_BASE * pmods[segment["idx"]],
        "sigma_um": 1e6 * sigmas[segment["idx"]],
        "velocity_m_per_s": velocities[segment["idx"]],
    } for number, segment in enumerate(info, start=1)]


def optimize(policy: str, segment_mm: float, max_iter: int,
             tolerance: float, target,
             within_line_warm_start: str = "same-position",
             optimize_turn_dwell: bool = False,
             max_turn_dwell_s: float | None = None,
             dwell_reg: float = DWELL_REG,
             turn_seed_mode: str = "auto",
             settling_window_mm: float = 0.0,
             settling_weight: float = 1.0,
             settling_tolerance: float = 0.01,
             start_window_mm: float = 0.0,
             start_weight: float = 1.0,
             start_tolerance: float = 0.01) -> dict:
    dwell = dwell_for(policy)
    if optimize_turn_dwell and policy != "dwell":
        raise ValueError("turnaround dwell optimization requires dwell policy")
    if settling_window_mm > 0.0 and policy != "continuous":
        raise ValueError("terminal settling is currently a continuous-only test")
    if start_window_mm > 0.0 and not optimize_turn_dwell:
        raise ValueError("start-of-block penalty targets the post-dwell cold "
                         "start; use it with --optimize-turn-dwell")
    if max_turn_dwell_s is None:
        max_turn_dwell_s = dwell
    if turn_seed_mode == "cool":
        turn_seed = COOL_SEED
    elif turn_seed_mode == "nominal":
        turn_seed = np.array([R.PMOD0, R.SIGMA, R.V])
    else:
        turn_seed = (np.array([R.PMOD0, R.SIGMA, R.V])
                     if policy == "dwell" else COOL_SEED)
    div, powered, hop_after = R.build_path(dwell, seg=segment_mm)
    info = R.seg_info(div, powered)
    if len(info) != 2 * round(R.S / segment_mm):
        raise RuntimeError(f"unexpected {len(info)}-block path for "
                           f"segment length {segment_mm:g} mm")
    greedy = greedy_initialize(
        div, powered, hop_after, info, target, max_iter, tolerance,
        within_line_warm_start=within_line_warm_start,
        turn_seed=turn_seed,
        optimize_turn_dwell=optimize_turn_dwell,
        initial_dwell_s=dwell,
        max_turn_dwell_s=max_turn_dwell_s,
        dwell_reg=dwell_reg,
        settling_window_mm=settling_window_mm,
        settling_weight=settling_weight,
        settling_tolerance=settling_tolerance,
        start_window_mm=start_window_mm,
        start_weight=start_weight,
        start_tolerance=start_tolerance)
    base = greedy["base_controls"]
    schedule = schedule_from(info, base)
    turn_dwells = greedy["turnaround_dwells_s"]
    if turn_dwells:
        first_after_turn = next(
            row for row in schedule if row["line"] == 2)
        first_after_turn["dwell_before_s"] = turn_dwells[0]
    payload = {
        "policy": policy,
        "method": "sequential greedy endpoint regulation only",
        "within_line_warm_start": within_line_warm_start,
        "segment_mm": segment_mm,
        "max_iter": max_iter,
        "tolerance": tolerance,
        "optimize_turn_dwell": optimize_turn_dwell,
        "turn_seed": turn_seed_mode,
        "settling_window_mm": settling_window_mm,
        "settling_weight": settling_weight,
        "settling_tolerance": settling_tolerance,
        "start_window_mm": start_window_mm,
        "start_weight": start_weight,
        "start_tolerance": start_tolerance,
        "dwell_regularization": dwell_reg,
        "max_turn_dwell_s": max_turn_dwell_s,
        "turnaround_dwells_s": turn_dwells,
        "target": {"width_um": float(target[0] * 1e6),
                   "depth_um": float(target[1] * 1e6)},
        "oti_runs": greedy["oti_runs"],
        "double_runs": greedy["double_runs"],
        "history": greedy["history"],
        "schedule": schedule,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("continuous", "dwell", "both"),
                        default="both")
    parser.add_argument("--segment-mm", type=float, default=2.5)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument(
        "--within-line-warm-start",
        choices=("same-position", "previous-block"),
        default="same-position",
        help="warm start for block 2+ within each scan line")
    parser.add_argument(
        "--optimize-turn-dwell", action="store_true",
        help="optimize one bounded dwell at the track-1/track-2 turnaround")
    parser.add_argument(
        "--max-turn-dwell-ms", type=float, default=None,
        help="upper dwell bound in ms (default: fixed minimum dwell)")
    parser.add_argument(
        "--dwell-regularization", type=float, default=DWELL_REG,
        help="quadratic throughput penalty for adaptive turnaround dwell")
    parser.add_argument(
        "--turn-seed", choices=("auto", "nominal", "cool"), default="auto",
        help="turnaround initialization (auto: cool continuous, nominal dwell)")
    parser.add_argument(
        "--settling-window-mm", type=float, default=0.0,
        help="penalize width/depth change over this distance before each "
             "track-2 block endpoint (0 disables)")
    parser.add_argument(
        "--settling-weight", type=float, default=1.0,
        help="least-squares weight on terminal width/depth change")
    parser.add_argument(
        "--settling-tolerance", type=float, default=0.01,
        help="maximum normalized terminal dimension change for convergence")
    parser.add_argument(
        "--start-window-mm", type=float, default=0.0,
        help="penalize the block-START depth deficit measured this distance "
             "before the endpoint (0 disables); pairs with --optimize-turn-"
             "dwell to select the dwell by physics instead of --dwell-"
             "regularization")
    parser.add_argument(
        "--start-weight", type=float, default=1.0,
        help="least-squares weight on the block-start depth deficit")
    parser.add_argument(
        "--start-tolerance", type=float, default=0.02,
        help="maximum normalized block-start depth deficit for convergence")
    args = parser.parse_args()
    if args.segment_mm <= 0.0 or R.S / args.segment_mm % 1.0 > 1e-9:
        parser.error("--segment-mm must divide the 10 mm track length")
    if args.optimize_turn_dwell and args.policy != "dwell":
        parser.error("--optimize-turn-dwell requires --policy dwell")
    if args.max_turn_dwell_ms is not None and args.max_turn_dwell_ms < 0.0:
        parser.error("--max-turn-dwell-ms must be nonnegative")
    if args.settling_window_mm < 0.0:
        parser.error("--settling-window-mm must be nonnegative")
    if args.settling_window_mm >= args.segment_mm:
        parser.error("--settling-window-mm must be shorter than a block")
    if args.settling_weight <= 0.0:
        parser.error("--settling-weight must be positive")
    if args.settling_tolerance <= 0.0:
        parser.error("--settling-tolerance must be positive")
    if args.settling_window_mm > 0.0 and args.policy != "continuous":
        parser.error("--settling-window-mm currently requires continuous")
    if args.start_window_mm < 0.0:
        parser.error("--start-window-mm must be nonnegative")
    if args.start_window_mm >= args.segment_mm:
        parser.error("--start-window-mm must be shorter than a block")
    if args.start_weight <= 0.0:
        parser.error("--start-weight must be positive")
    if args.start_tolerance <= 0.0:
        parser.error("--start-tolerance must be positive")
    if args.start_window_mm > 0.0 and not args.optimize_turn_dwell:
        parser.error("--start-window-mm requires --optimize-turn-dwell")

    tag = segment_tag(args.segment_mm)
    target = target_value(args.segment_mm)
    policies = (("continuous", "dwell") if args.policy == "both"
                else (args.policy,))
    start = time.time()
    for policy in policies:
        print(f"\n{policy}: greedy-only {args.segment_mm:g} mm blocks",
              flush=True)
        payload = optimize(policy, args.segment_mm, args.max_iter,
                           args.tolerance, target,
                           args.within_line_warm_start,
                           optimize_turn_dwell=args.optimize_turn_dwell,
                           max_turn_dwell_s=(None
                               if args.max_turn_dwell_ms is None
                               else args.max_turn_dwell_ms * 1e-3),
                           dwell_reg=args.dwell_regularization,
                           turn_seed_mode=args.turn_seed,
                           settling_window_mm=args.settling_window_mm,
                           settling_weight=args.settling_weight,
                           settling_tolerance=args.settling_tolerance,
                           start_window_mm=args.start_window_mm,
                           start_weight=args.start_weight,
                           start_tolerance=args.start_tolerance)
        variant = ("_prevblock"
                   if args.within_line_warm_start == "previous-block" else "")
        if args.optimize_turn_dwell:
            variant += "_turndwell"
        if args.turn_seed != "auto":
            variant += f"_{args.turn_seed}seed"
        if args.settling_window_mm > 0.0:
            variant += ("_settling"
                        + segment_tag(args.settling_window_mm))
        if args.start_window_mm > 0.0:
            variant += "_startpen"
        with open(result_path(f"greedy_{tag}{variant}_{policy}.json"), "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  solves: {payload['oti_runs']} OTI + "
              f"{payload['double_runs']} double", flush=True)
    print(f"wall time: {time.time() - start:.1f} s")


if __name__ == "__main__":
    main()
