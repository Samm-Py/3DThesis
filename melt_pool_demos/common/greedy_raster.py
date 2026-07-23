"""Shared greedy P/sigma/velocity controller for square and triangle rasters.

This is the raster extension of the finalized two-track prototype:

* 1 mm control blocks by default;
* endpoint width/depth measured in a 1 mm head-local box;
* absorbed power, lateral beam sigma, and scan velocity as controls;
* sequential commitment with previous-block warm starts within each line;
* balanced near-1 mm segmentation on shrinking triangle lines.

Use ``--nlines`` for an inexpensive truncated-raster validation before the
full 101-line optimization.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import numpy as np


COMMON = os.path.dirname(os.path.abspath(__file__))
DEMOS = os.path.dirname(COMMON)
DEMO = os.path.abspath(os.getcwd())
GEOMETRY = os.path.basename(DEMO)
if GEOMETRY not in ("square", "triangle"):
    raise RuntimeError("run greedy_raster.py from square/ or triangle/")
sys.path.insert(0, COMMON)
sys.path.insert(0, DEMOS)

os.environ.setdefault("MP_GEOM", GEOMETRY)
os.environ.setdefault("MP_BOX_BEHIND", "0.001")
os.environ.setdefault("MP_BOX_HALF_Y", "0.0012")
os.environ.setdefault(
    "MP_CASE_DIR", os.path.join(DEMO, "cases", "snapcase_greedy1_localbox"))

from common import mp_lib as R  # noqa: E402


U_LO = np.array([0.05, 80e-6, 0.5])
U_HI = np.array([3.0, 400e-6, 8.0])
U_SCALE = np.array([1.0, 200e-6, 3.0])
DU_MAX = np.array([0.50, 80e-6, 1.5])
BACKTRACK = (1.0, 0.5, 0.25, 0.125, 0.0625)
STEP_REG = 2.5e-2
VELOCITY_REG = 1.0e-2
MIN_IMPROVEMENT = 1e-3

# ---- adaptive per-line dwell (Phase 4; MP_ADAPTIVE_DWELL=1) ----
# A stalled apex line opens a beam-off dwell before it: the joint (P,sigma,v,
# Delta) step trims width (dwell's orthogonal apex authority, measured dW/dDelta
# ~= -0.3 m/s) so power can restore depth. Delta is priced linearly by
# LAMBDA_DWELL (build-time cost) and projected to Delta >= 0, so it opens only
# where (P,sigma,v) provably cannot reach target and stays minimal. Default off
# -> committed_dwells stays empty -> every path row is the zero-dwell one.
ADAPTIVE_DWELL = os.environ.get("MP_ADAPTIVE_DWELL", "0") == "1"
DWELL_MAX = float(os.environ.get("MP_DWELL_MAX", "2e-3"))    # s, per-line cap
DWELL_SCALE = 1.0e-4        # s, characteristic dwell (Newton column scaling)
DWELL_DU_MAX = 3.0e-4       # s, max dwell change per step
LAMBDA_DWELL = float(os.environ.get("MP_LAMBDA_DWELL", "5e-3"))  # larger -> less dwell
DWELL_MAX_ITER = 6
U_LO4 = np.array([U_LO[0], U_LO[1], U_LO[2], 0.0])
U_HI4 = np.array([U_HI[0], U_HI[1], U_HI[2], DWELL_MAX])
U_SCALE4 = np.array([U_SCALE[0], U_SCALE[1], U_SCALE[2], DWELL_SCALE])
DU_MAX4 = np.array([DU_MAX[0], DU_MAX[1], DU_MAX[2], DWELL_DU_MAX])

# ---- turn/apex warm-start (measured 2026-07-22) ----
# The FIRST block after every serpentine turn (and every short apex line, which
# is a single such block) is a heat-EXCESS transient. Seed that block COOL and
# slow to avoid the high-velocity local basin. Within a line, seed every later
# block from the immediately preceding accepted control so that the process
# parameters evolve continuously along the track.
COOL_SEED = np.array([0.2, R.SIGMA, 1.0])   # 30 W, nominal sigma, 1 m/s


# SUPERSEDED (DWELL_DERIVATIVE_PLAN §6.3, measured 2026-07-22): merging short
# apex lines into one shared control is what CAUSES the apex stall -- un-merged,
# each line's own velocity (slowing down) clears the apex through the tip line.
# Kept only to reproduce the older merged results; leave MP_APEX_MERGE unset.
APEX_MERGE = os.environ.get("MP_APEX_MERGE", "0") == "1"


def _control_units(info: list[dict], segment_mm: float,
                   apex_merge: bool) -> list[list[dict]]:
    """Group the controlled segments into control units.

    A segment on a body line (whose whole scan length >= ``segment_mm``, so it
    can form proper blocks) is its own unit -- unchanged behaviour, whatever the
    balanced splitter made of that line. An apex line shorter than the control
    length is one sub-block; consecutive such short lines accumulate, across the
    serpentine hops, into a unit of >= ``segment_mm`` scan length, and a
    trailing stub merges into the last apex unit so no unit is left far below
    ``segment_mm``. Hop length is excluded from the budget (hops inherit their
    unit's control, as in the square). With ``apex_merge`` false every segment
    is its own unit. In particular, apex lines separated by a beam-off dwell
    must remain independent transient exposures rather than sharing a control.
    """
    prev_s = 0.0
    enriched = []
    for e in info:
        enriched.append(dict(e, seg_len=e["s_mm"] - prev_s))
        prev_s = e["s_mm"]
    if not apex_merge:
        return [[e] for e in enriched]
    line_len: dict[int, float] = {}
    for e in enriched:
        line_len[e["line"]] = line_len.get(e["line"], 0.0) + e["seg_len"]
    short = lambda e: line_len[e["line"]] < segment_mm
    units: list[list[dict]] = []
    cur: list[dict] = []
    acc = 0.0
    for e in enriched:
        if not short(e):                    # body line: leave its blocks alone
            if cur:
                units.append(cur)
                cur, acc = [], 0.0
            units.append([e])
        else:                               # apex line: accumulate to ~1 mm
            cur.append(e)
            acc += e["seg_len"]
            if acc >= segment_mm:
                units.append(cur)
                cur, acc = [], 0.0
    if cur:
        if units and short(units[-1][-1]):
            units[-1].extend(cur)           # trailing stub joins last apex unit
        else:
            units.append(cur)
    return units


def segment_tag(segment_mm: float) -> str:
    return f"{segment_mm:g}".replace(".", "p")


def result_path(name: str) -> str:
    path = os.path.join(DEMO, "results", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def dwell_for(policy: str) -> float:
    if policy == "zero":
        return 0.0
    with open(result_path("min_dwell.json")) as f:
        return float(json.load(f)["min_dwell_s"])


def target_value(segment_mm: float) -> np.ndarray:
    """Developed nominal pool at the end of one 10 mm track."""
    div, powered, _ = R.build_path(
        0.0, seg=segment_mm, nlines=1, balanced=True)
    info = R.seg_info(div, powered)
    pmods = {idx: R.PMOD0 for idx in powered}
    endpoint = info[-1]
    R.run_segment(div, endpoint["idx"], pmods, R.BIN_DBL)
    width, depth, _asym, clip = R.measure_dims()
    # The head-local box intentionally truncates the roughly 2 mm trailing
    # pool at x-.  Width and depth extrema remain local to the beam head; all
    # other clipped faces would corrupt those measurements.
    unexpected = [face for face in clip if face != "x-"]
    if unexpected:
        raise RuntimeError(
            f"developed target is clipped at box faces: {unexpected}")
    return np.array([width, depth])


def endpoint_observation(div: list[str], idx: int, pmods: dict,
                         sigmas: dict, velocities: dict, u: np.ndarray,
                         binary: str, group_idxs: Optional[list[int]] = None,
                         dwells: Optional[dict] = None
                         ) -> tuple[np.ndarray, np.ndarray | None]:
    trial_pmods = dict(pmods)
    trial_sigmas = dict(sigmas)
    trial_velocities = dict(velocities)
    # A merged apex unit applies one control across several short lines (and
    # the beam-on hops between them); set every member to the candidate u, and
    # measure at idx = the unit's last segment. Singletons pass group_idxs=None.
    # Only the first three components of u (P, sigma, v) are per-segment; a
    # fourth (dwell) rides `dwells`, not the segment control.
    for j in (group_idxs or [idx]):
        trial_pmods[j], trial_sigmas[j], trial_velocities[j] = map(
            float, u[:3])
    R.run_segment(
        div, idx, trial_pmods, binary, sigma=float(u[1]),
        sig_hist=trial_sigmas, vels=trial_velocities,
        seed_idx=(idx if binary == R.BIN_OTI else None), dwells=dwells)
    clip = R.clip_check()
    unexpected = [face for face in clip if face != "x-"]
    if unexpected:
        print(f"    WARNING: liquid clipped at box faces {unexpected}",
              flush=True)
    if binary == R.BIN_OTI:
        s = R.sens()
        q = np.array([s["width"], s["depth"]])
        jac = np.array([
            [s["dwidth_dQ"] * R.P_BASE, s["dwidth_dsig"], s["dwidth_dv"]],
            [s["ddepth_dQ"] * R.P_BASE, s["ddepth_dsig"], s["ddepth_dv"]],
        ])
        return q, jac
    width, depth, _asym, _clip = R.measure_dims()
    return np.array([width, depth]), None


def dwell_column(div: list[str], idx: int, hop: int, pmods: dict, sigmas: dict,
                 velocities: dict, u: np.ndarray, dwells: dict) -> np.ndarray:
    """[dwidth/dDelta, ddepth/dDelta] at the current (P,sigma,v,dwells) state,
    from one OTI run seeding the dwell HOP row (dT_ddwell). The observed line's
    controls are set to u[:3]; `dwells` (including this line's current Delta)
    fixes the thermal state."""
    trial_pmods = {**pmods, idx: float(u[0])}
    trial_sigmas = {**sigmas, idx: float(u[1])}
    trial_velocities = {**velocities, idx: float(u[2])}
    R.run_segment(div, idx, trial_pmods, R.BIN_OTI, sigma=float(u[1]),
                  sig_hist=trial_sigmas, vels=trial_velocities,
                  seed_idx=hop, dwells=dwells)
    s = R.sens()
    return np.array([s["dwidth_ddwell"], s["ddepth_ddwell"]])


def adaptive_dwell_solve(div, idx, hop, pmods, sigmas, velocities, u3, target,
                         tolerance, committed_dwells, label):
    """Open a beam-off dwell before a stalled apex line and jointly regulate
    (P, sigma, v, Delta). The 2x4 Newton is well-conditioned at the apex
    because dwell is the orthogonal width lever (measured); Delta is priced
    linearly (LAMBDA_DWELL) and projected to Delta >= 0, so it opens only when
    it improves the physical residual and stays minimal. Returns (u4, history).
    Prior lines' committed dwells stay in the thermal state throughout."""
    u = np.array([u3[0], u3[1], u3[2], 0.0])
    history = []
    oti, dbl = 0, 0
    for iteration in range(DWELL_MAX_ITER):
        dwells = {**committed_dwells, hop: float(u[3])}
        try:
            q, jac3 = endpoint_observation(
                div, idx, pmods, sigmas, velocities, u[:3], R.BIN_OTI,
                dwells=dwells)
            col = dwell_column(
                div, idx, hop, pmods, sigmas, velocities, u, dwells)
        except ValueError:
            break                                   # no melt: give up on dwell
        oti += 2
        jac = np.hstack([jac3, col[:, None]])        # 2x4
        residual = (q - target) / target
        endpoint_merit = 0.5 * float(residual @ residual)
        worst = float(np.max(np.abs(residual)))
        print(f"    dwell it{iteration}: worst={worst*100:.2f}% "
              f"P={u[0]*R.P_BASE:.1f} W sigma={u[1]*1e6:.1f} um "
              f"v={u[2]:.2f} m/s dwell={u[3]*1e3:.3f} ms", flush=True)
        if worst <= tolerance:
            break

        jac_scaled = (jac / target[:, None]) * U_SCALE4[None, :]
        lhs = jac_scaled.T @ jac_scaled + STEP_REG * np.eye(4)
        rhs = -(jac_scaled.T @ residual)
        vscale = U_SCALE4[2] / R.V
        vrel = (u[2] - R.V) / R.V
        lhs[2, 2] += VELOCITY_REG * vscale * vscale
        rhs[2] -= VELOCITY_REG * vscale * vrel
        # linear build-time price on dwell: bias the step toward less dwell
        # (acceptance below still requires a physical residual improvement).
        rhs[3] -= LAMBDA_DWELL * U_SCALE4[3]
        du = np.clip(U_SCALE4 * np.linalg.solve(lhs, rhs), -DU_MAX4, DU_MAX4)

        accepted = False
        for alpha in BACKTRACK:
            trial = np.clip(u + alpha * du, U_LO4, U_HI4)
            td = {**committed_dwells, hop: float(trial[3])}
            q_t, _ = endpoint_observation(
                div, idx, pmods, sigmas, velocities, trial[:3], R.BIN_DBL,
                dwells=td)
            dbl += 1
            tr = (q_t - target) / target
            tm = 0.5 * float(tr @ tr)
            if tm < endpoint_merit * (1.0 - MIN_IMPROVEMENT):
                u = trial
                history.append({
                    "block": label, "iteration": iteration + 1, "alpha": alpha,
                    "worst_relative_error": float(np.max(np.abs(tr))),
                    "pmod": float(u[0]), "sigma": float(u[1]),
                    "velocity": float(u[2]), "dwell_s": float(u[3])})
                accepted = True
                break
        if not accepted:
            print("    no improving dwell step; stopping", flush=True)
            break
    return u, history, oti, dbl


def greedy_initialize(div: list[str], powered: list[int], hop_after: dict,
                      info: list[dict], target: np.ndarray, max_iter: int,
                      tolerance: float, segment_mm: float,
                      apex_merge: bool,
                      checkpoint_path: Optional[str] = None,
                      resume: bool = False) -> dict:
    pmods = {idx: R.PMOD0 for idx in powered}
    sigmas = {idx: R.SIGMA for idx in powered}
    velocities = {idx: R.V for idx in powered}
    history, oti_runs, double_runs = [], 0, 0
    current_line, position = -1, -1
    committed = []

    if resume:
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            raise RuntimeError("--resume requested but no checkpoint exists")
        with open(checkpoint_path) as f:
            saved = json.load(f)
        committed = saved["committed"]
        history = saved["history"]
        oti_runs = int(saved["oti_runs"])
        double_runs = int(saved["double_runs"])
        for order, row in enumerate(committed):
            segment = info[order]
            if int(row["idx"]) != segment["idx"]:
                raise RuntimeError("checkpoint path does not match geometry")
            idx = segment["idx"]
            u_saved = np.array([
                row["pmod"], row["sigma"], row["velocity"]], dtype=float)
            pmods[idx], sigmas[idx], velocities[idx] = map(float, u_saved)
            if idx in hop_after:
                hop = hop_after[idx]
                pmods[hop], sigmas[hop], velocities[hop] = map(
                    float, u_saved)
        if committed:
            current_line = int(committed[-1]["line"])
            position = int(committed[-1]["position"])
        print(f"  resumed {len(committed)}/{len(info)} committed blocks",
              flush=True)

    nominal = np.array([R.PMOD0, R.SIGMA, R.V])
    units = _control_units(info, segment_mm, apex_merge)
    last_u = nominal.copy()
    seen = 0
    for unit in units:
        if seen + len(unit) <= len(committed):     # already committed (resume)
            seen += len(unit)
            rep0 = unit[-1]["idx"]
            last_u = np.array([pmods[rep0], sigmas[rep0], velocities[rep0]])
            continue
        seen += len(unit)
        rep = unit[-1]
        idx = rep["idx"]                            # measure at the unit's end
        merged = len(unit) > 1
        member_idxs = [e["idx"] for e in unit]
        # inter-member beam-on hops carry the unit's control during the trial
        member_hops = [hop_after[e["idx"]] for e in unit[:-1]
                       if e["idx"] in hop_after]
        group_idxs = member_idxs + member_hops

        if merged:
            length = sum(e["seg_len"] for e in unit)
            label = (f"apexL{unit[0]['line'] + 1}-{rep['line'] + 1}"
                     f"_merge{len(unit)}x({length:.2f}mm)")
            warm = last_u                          # continue from the last unit
        else:
            new_line = rep["line"] != current_line
            position = 0 if new_line else position + 1
            current_line = rep["line"]
            label = f"line{rep['line'] + 1}_block{position + 1}"
            if new_line:
                # The first line has no inherited turnaround and starts from
                # nominal. Every later line starts in the measured cool/slow
                # turnaround basin.
                warm = nominal if rep["line"] == 0 else COOL_SEED
            else:
                # Continue from the block just accepted on this same track.
                warm = last_u
        u = np.clip(warm, U_LO, U_HI)
        print(f"  greedy {label}:", flush=True)

        # Most warm-started blocks in a developed raster already satisfy the
        # endpoint tolerance.  A cheap double precheck avoids paying for OTI
        # derivatives when no Newton update will be taken.
        q_pre, _ = endpoint_observation(
            div, idx, pmods, sigmas, velocities, u, R.BIN_DBL,
            group_idxs=group_idxs)
        double_runs += 1
        pre_residual = (q_pre - target) / target
        pre_worst = float(np.max(np.abs(pre_residual)))
        if pre_worst <= tolerance:
            print(
                f"    precheck: worst={pre_worst*100:.2f}% "
                f"P={u[0]*R.P_BASE:.1f} W sigma={u[1]*1e6:.1f} um "
                f"v={u[2]:.2f} m/s",
                flush=True)
        else:
            for iteration in range(max_iter):
                try:
                    q, jac = endpoint_observation(
                        div, idx, pmods, sigmas, velocities, u, R.BIN_OTI,
                        group_idxs=group_idxs)
                except ValueError:
                    # No melt at the current iterate -- typically a warm start
                    # ratcheted onto the power/sigma floor by upstream heat, so
                    # the OTI isosurface extraction has nothing to measure and
                    # raises. Recover the way run_optimized.py does: add energy
                    # (raise power, tighten the beam toward nominal, slow down)
                    # and retry, instead of letting the ValueError kill the run.
                    oti_runs += 1
                    u = np.clip(np.array([
                        max(2.0 * u[0], 1.0),
                        max(0.5 * u[1], R.SIGMA),
                        0.5 * u[2],
                    ]), U_LO, U_HI)
                    print(
                        f"    it{iteration}: no melt; raising power, tightening "
                        f"beam, slowing -> P={u[0]*R.P_BASE:.1f} W "
                        f"sigma={u[1]*1e6:.1f} um v={u[2]:.2f} m/s",
                        flush=True)
                    continue
                oti_runs += 1
                residual = (q - target) / target
                endpoint_merit = 0.5 * float(residual @ residual)
                vrel = (u[2] - R.V) / R.V
                merit = (endpoint_merit
                         + 0.5 * VELOCITY_REG * vrel * vrel)
                worst = float(np.max(np.abs(residual)))
                print(
                    f"    it{iteration}: merit={merit:.6f} "
                    f"worst={worst*100:.2f}% P={u[0]*R.P_BASE:.1f} W "
                    f"sigma={u[1]*1e6:.1f} um v={u[2]:.2f} m/s",
                    flush=True)
                if worst <= tolerance:
                    break

                jac_normalized = jac / target[:, None]
                jac_scaled = jac_normalized * U_SCALE[None, :]
                lhs = jac_scaled.T @ jac_scaled + STEP_REG * np.eye(3)
                rhs = -(jac_scaled.T @ residual)
                vscale = U_SCALE[2] / R.V
                lhs[2, 2] += VELOCITY_REG * vscale * vscale
                rhs[2] -= VELOCITY_REG * vscale * vrel
                du = np.clip(
                    U_SCALE * np.linalg.solve(lhs, rhs), -DU_MAX, DU_MAX)

                accepted = False
                for alpha in BACKTRACK:
                    trial_u = np.clip(u + alpha * du, U_LO, U_HI)
                    q_trial, _ = endpoint_observation(
                        div, idx, pmods, sigmas, velocities, trial_u,
                        R.BIN_DBL, group_idxs=group_idxs)
                    double_runs += 1
                    trial_residual = (q_trial - target) / target
                    trial_endpoint_merit = (
                        0.5 * float(trial_residual @ trial_residual))
                    trial_vrel = (trial_u[2] - R.V) / R.V
                    trial_merit = (
                        trial_endpoint_merit
                        + 0.5 * VELOCITY_REG * trial_vrel * trial_vrel)
                    # Width/depth regulation is the primary objective.  The
                    # weak nominal-velocity term shapes the underdetermined
                    # update but may never justify accepting a worse physical
                    # endpoint.
                    if (trial_endpoint_merit
                            < endpoint_merit * (1.0 - MIN_IMPROVEMENT)):
                        u = trial_u
                        history.append({
                            "block": label,
                            "iteration": iteration + 1,
                            "alpha": alpha,
                            "merit": trial_merit,
                            "endpoint_merit": trial_endpoint_merit,
                            "worst_relative_error": float(
                                np.max(np.abs(trial_residual))),
                            "pmod": float(u[0]),
                            "sigma": float(u[1]),
                            "velocity": float(u[2]),
                        })
                        accepted = True
                        break
                if not accepted:
                    print("    no improving greedy step; stopping block",
                          flush=True)
                    break

        # commit the unit's single control to every member segment (and its
        # inheriting hop); record one committed row per member, in info order
        for e in unit:
            j = e["idx"]
            pmods[j], sigmas[j], velocities[j] = map(float, u)
            if j in hop_after:
                hop = hop_after[j]
                pmods[hop], sigmas[hop], velocities[hop] = map(float, u)
            committed.append({
                "idx": j,
                "line": int(e["line"]),
                "position": (-1 if merged else position),
                "pmod": float(u[0]),
                "sigma": float(u[1]),
                "velocity": float(u[2]),
            })
        last_u = u.copy()
        if checkpoint_path is not None:
            with open(checkpoint_path, "w") as f:
                json.dump({
                    "committed": committed,
                    "history": history,
                    "oti_runs": oti_runs,
                    "double_runs": double_runs,
                }, f, indent=2)

    return {
        "oti_runs": oti_runs,
        "double_runs": double_runs,
        "history": history,
        "base_controls": (pmods, sigmas, velocities),
    }


def schedule_from(info: list[dict],
                  base: tuple[dict, dict, dict]) -> list[dict]:
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


def optimize(policy: str, segment_mm: float, nlines: int, max_iter: int,
             tolerance: float, target: np.ndarray, checkpoint_path: str,
             resume: bool) -> dict:
    dwell = dwell_for(policy)
    # A beam-off dwell deliberately resets the thermal state between scan
    # lines. Sharing one control across several such lines and evaluating only
    # the last endpoint can overheat the longer members in order to develop a
    # pool on the shortest member. Apex merging is therefore reserved for the
    # thermally coupled continuous path.
    apex_merge = APEX_MERGE and dwell == 0.0
    if APEX_MERGE and not apex_merge:
        print(
            "  apex merge disabled: beam-off dwell separates the short lines",
            flush=True)
    div, powered, hop_after = R.build_path(
        dwell, seg=segment_mm, nlines=nlines, balanced=True)
    info = R.seg_info(div, powered)
    if not info:
        raise RuntimeError("segmented raster contains no controlled blocks")
    greedy = greedy_initialize(
        div, powered, hop_after, info, target, max_iter, tolerance,
        segment_mm, apex_merge,
        checkpoint_path=checkpoint_path, resume=resume)
    payload = {
        "geometry": GEOMETRY,
        "policy": policy,
        "method": "sequential greedy endpoint regulation only",
        "controls": ["absorbed_power", "lateral_sigma", "scan_velocity"],
        "apex_merge": apex_merge,
        "segment_mm": segment_mm,
        "nlines": nlines,
        "measurement_box_behind_m": R.BOX_BEHIND,
        "measurement_box_half_width_m": R.BOX_HALF_Y,
        "target": {
            "width_um": float(target[0] * 1e6),
            "depth_um": float(target[1] * 1e6),
        },
        "oti_runs": greedy["oti_runs"],
        "double_runs": greedy["double_runs"],
        "history": greedy["history"],
        "schedule": schedule_from(info, greedy["base_controls"]),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("zero", "dwell", "both"),
                        default="both")
    parser.add_argument("--segment-mm", type=float, default=1.0)
    parser.add_argument(
        "--nlines", type=int, default=0,
        help="number of raster lines (0 = complete geometry)")
    parser.add_argument("--max-iter", type=int, default=6)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument(
        "--resume", action="store_true",
        help="resume from the block-level checkpoint for this configuration")
    args = parser.parse_args()

    if args.segment_mm <= 0.0 or R.S / args.segment_mm % 1.0 > 1e-9:
        parser.error("--segment-mm must divide the 10 mm track length")
    full_nlines = len(R.scan_lines())
    nlines = full_nlines if args.nlines == 0 else args.nlines
    if not 1 <= nlines <= full_nlines:
        parser.error(f"--nlines must be between 1 and {full_nlines}")

    target = target_value(args.segment_mm)
    policies = ("zero", "dwell") if args.policy == "both" else (args.policy,)
    tag = segment_tag(args.segment_mm)
    suffix = "" if nlines == full_nlines else f"_{nlines}lines"
    start = time.time()
    for policy in policies:
        print(
            f"\n{policy}: {nlines}-line {GEOMETRY}, "
            f"{args.segment_mm:g} mm greedy blocks, "
            f"local box={R.BOX_BEHIND*1e3:g} mm",
            flush=True)
        output = result_path(f"greedy_{tag}_{policy}{suffix}.json")
        checkpoint = output + ".checkpoint"
        payload = optimize(
            policy, args.segment_mm, nlines, args.max_iter,
            args.tolerance, target, checkpoint, args.resume)
        with open(output, "w") as f:
            json.dump(payload, f, indent=2)
        print(
            f"  solves: {payload['oti_runs']} OTI + "
            f"{payload['double_runs']} double; wrote {output}",
            flush=True)
    print(f"wall time: {time.time() - start:.1f} s")


if __name__ == "__main__":
    main()
