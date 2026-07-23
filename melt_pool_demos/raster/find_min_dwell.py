"""Derive the MINIMAL turnaround dwell for the serpentine raster.

Criterion: at the end of every turnaround dwell (the instant before the next
track starts), the entire temperature field is below the melt isotherm
(1733 K) -- i.e. the previous track's melt residue has fully solidified.
Below this dwell, the pool measured at track starts merges with still-liquid
neighbour residue and a fixed width target is infeasible for ANY power
(the failure mode documented in Scan.AddTurnDwell).

The check runs at the UNOPTIMISED power (Pmod0=1.5, 60 W) -- the worst case --
so the derived dwell is valid for both the baseline and optimised runs.
Bisection over dwell; per candidate, every turnaround is checked (last turn
first: it is the hottest, so failures exit early).

Writes min_dwell.json = {"dwell_s": ..., "tol_s": ..., "turn_Tmax_K": [...]}
"""
import json
import numpy as np
import raster_lib as R


def turn_indices(div):
    """Line indices of the turnaround rests (mode-1 lines after the first
    powered raster)."""
    turns, seen = [], False
    for i in range(len(div)):
        vals = div[i].strip().split('\t')
        if vals[0] == '0':
            seen = True
        elif vals[0] == '1' and seen:
            turns.append(i)
    return turns


def turn_tmax(div, i_turn, pmods):
    """Max field T at the end of turnaround dwell line i_turn. Domain covers
    the just-finished track and the two below it (the hot residue)."""
    vals = div[i_turn].strip().split('\t')
    y_next = float(vals[2])                      # rest sits at the NEXT track's y
    y_done = y_next - R.H                        # the just-finished track
    R.write_domain(-0.2e-3, (R.S + 0.2) * 1e-3,
                   (y_done - 2.2 * R.H) * 1e-3, (y_next + 0.6 * R.H) * 1e-3,
                   -0.15e-3, 10e-6, 8e-6, 8e-6)
    R.run_path(R.set_pmods(div[:i_turn + 1], pmods), R.BIN_DBL)
    _, _, _, T3 = R.load_field()
    return float(T3.max())


def all_solid(dwell, verbose=True):
    div, powered = R.build_path(dwell)
    pmods = {j: R.PMOD0 for j in powered}
    tmaxes = []
    for i_turn in reversed(turn_indices(div)):
        tm = turn_tmax(div, i_turn, pmods)
        tmaxes.append(tm)
        if tm >= R.T_LIQ:
            if verbose:
                print(f"  dwell {dwell*1e3:6.3f} ms: LIQUID at turn line {i_turn} "
                      f"(Tmax {tm:.0f} K)")
            return False, tmaxes
    if verbose:
        print(f"  dwell {dwell*1e3:6.3f} ms: all {len(tmaxes)} turns solid "
              f"(hottest {max(tmaxes):.0f} K)")
    return True, tmaxes


if __name__ == "__main__":
    R.write_beam()
    lo, hi, tol = 0.0, 2.0e-3, 25e-6

    ok, _ = all_solid(hi)
    while not ok:                                # widen until feasible
        hi *= 2.0
        ok, _ = all_solid(hi)
    ok0, _ = all_solid(lo)
    if ok0:
        print("even zero extra dwell is solid; minimal dwell = 0")
        hi = lo

    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        ok, _ = all_solid(mid)
        if ok:
            hi = mid
        else:
            lo = mid

    _, tmaxes = all_solid(hi, verbose=False)
    print(f"\nminimal dwell = {hi*1e3:.3f} ms  (+/- {tol*1e3:.3f} ms)")
    print(f"per-turn Tmax at that dwell (last->first): "
          + ", ".join(f"{t:.0f}" for t in tmaxes))
    out = R.data_path("min_dwell.json", write=True)
    with open(out, "w") as f:
        json.dump({"dwell_s": hi, "tol_s": tol, "turn_Tmax_K": tmaxes}, f, indent=1)
