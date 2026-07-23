"""
Melt-pool uniformity optimiser -- exact-derivative (OTI) version.

Goal
----
Hold the melt pool geometrically uniform along a scan path as heat
accumulates: keep both

    width = (y_right - y_left)/2   (half-span, ExtractMPDims_Span convention)
    depth =  downward isosurface extent

at a fixed target for every powered segment, instead of letting the pool
balloon. Two controls per segment:

    Pmod   -- power multiplier on the current segment (path column 4)
    sigma  -- lateral Gaussian beam width Width_X=Width_Y (Beam.txt)

Method
------
Per powered segment, a 2x2 Newton solve on u=(Pmod, sigma):

    r(u) = (w(u) - w*, d(u) - d*),      u <- u - J^{-1} r

where BOTH the residual and the Jacobian come from a single run of the OTI
3DThesis build: the snapshot carries the analytic field derivatives
(dT_dQ, dT_dsig, dT_dy, dT_dz) and simulate.ExtractMPSensitivities assembles
them into dw/dQ, dw/dsig, dd/dQ, dd/dsig at the extremal isotherm points
(implicit-function relation; FD-verified to <1%). No finite differencing
anywhere; 1 simulation per Newton iteration, warm-started per segment.

Exactness notes
---------------
Both Jacobian entries are exact per-segment control derivatives: the OTI
build seeds DV_Q/DV_SIG on the CURRENT (last) path segment's quadrature
nodes only, so dT_dQ = dT/d(this segment's input power, W) and dT_dsig =
dT/d(this segment's lateral width, m); history nodes carry zero derivative
(FD-verified, incl. that history-power FD does NOT match the column).

History replay: segment i's history replays the already-optimised Pmods AND
sigmas of segments j<i (sequential causality, like the real process). Sigma
is per-segment via the 7th path column ("Wmod" = sigma_j / base sigma in
Beam.txt). History speed stays at the raster speed; velocity as a control is
deferred.
"""

import os
import sys
import copy
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.chdir(_HERE)

# The optimiser NEEDS the derivative columns -> default to the OTI build.
os.environ.setdefault(
    "THESIS_BIN",
    os.path.abspath(os.path.join(_HERE, "..", "build-oti", "bin", "3DThesis")))

import scan as Scan
import simulate as Sim

CSV = "3DThesis/TestInputs/Data/TestSim.Snapshot.00.csv"

# Control bounds: power multiplier and lateral beam sigma (m).
PMOD_LO, PMOD_HI = 0.3, 3.0
SIG_LO,  SIG_HI  = 5.0e-6, 60.0e-6
AZ_FIXED     = 10.0e-6   # depth/absorption sigma: material property, not a knob
BEAM_POWER_W = 40.0      # Beam.txt base power; segment power = BEAM_POWER_W*Pmod


def write_beam(sigma):
    with open("3DThesis/TestInputs/Beam.txt", "w") as f:
        f.write("Shape\n{\n\tWidth_X\t\t%g\n\tWidth_Y\t\t%g\n\tDepth_Z\t\t%g\n}\n"
                "Intensity\n{\n\tPower\t\t%g\n\tEfficiency\t1.0\n}\n"
                % (sigma, sigma, AZ_FIXED, BEAM_POWER_W))


def set_controls(lines, pmods, wmods=None):
    """Return a copy of the path lines with Pmod (col 4) and/or the beam
    width factor (optional col 6, appended when absent) overridden for the
    line indices in `pmods` / `wmods`."""
    out = copy.deepcopy(lines)
    wmods = wmods or {}
    for idx in set(pmods) | set(wmods):
        vals = out[idx].strip().split('\t')
        if idx in pmods:
            vals[4] = str(pmods[idx])
        if idx in wmods:
            while len(vals) < 7:
                vals.append('1.0')
            vals[6] = repr(wmods[idx])
        out[idx] = '\t'.join(vals) + '\n'
    return out


def run_segment(div, i, pmods, sigma, res, buf, sig_hist=None):
    """Simulate up to segment i with the given per-segment Pmods (must include
    i itself) and the candidate sigma on the CURRENT segment. History segments
    keep their own optimised sigmas via the per-segment width column
    (factor sigma_j / sigma, relative to Beam.txt's base = candidate sigma)."""
    write_beam(sigma)
    wmods = {}
    if sig_hist:
        wmods = {j: s / sigma for j, s in sig_hist.items() if j < i}
    lines = set_controls(div[:i + 1],
                         {k: v for k, v in pmods.items() if k <= i}, wmods)
    rot = Scan.RotateTranslateLastRasterToX0(lines)
    Scan.ExportScan(rot, outFile="3DThesis/TestInputs/Path.txt")
    x, y = Scan.GetEndPosition()
    Sim.UpdateDomain(x=x / 1000.0, y=y / 1000.0, res=res, buf=buf)
    Sim.Run()


def newton_segment(div, i, pmods_hist, sig_hist, u0, target, res, buf,
                   tol=0.01, max_iter=5, verbose=True):
    """Solve (w,d)=(w*,d*) for u=(Pmod, sigma) at segment i.
    Returns (u, (w,d,asym), n_runs)."""
    u = np.array([np.clip(u0[0], PMOD_LO, PMOD_HI),
                  np.clip(u0[1], SIG_LO, SIG_HI)])
    tgt = np.asarray(target, float)
    n_runs = 0
    for it in range(max_iter):
        pmods = dict(pmods_hist); pmods[i] = u[0]
        run_segment(div, i, pmods, u[1], res, buf, sig_hist=sig_hist)
        n_runs += 1
        s = Sim.ExtractMPSensitivities(inFile=CSV)
        q = np.array([s['width'], s['depth']])
        r = q - tgt
        rel = np.max(np.abs(r) / tgt)
        if verbose:
            print(f"      it{it}: Pmod={u[0]:.3f} sig={u[1]*1e6:5.2f}um -> "
                  f"w={q[0]*1e6:6.2f} d={q[1]*1e6:6.2f} "
                  f"asym={s['asym']*1e6:6.2f}um  relres={rel:.4f}")
        if rel < tol:
            return u, (s['width'], s['depth'], s['asym']), n_runs
        # Exact Jacobian from the same run (power column rescaled to Pmod
        # units; see approximation note 2 in the module docstring).
        J = np.array([[s['dwidth_dQ'] * BEAM_POWER_W, s['dwidth_dsig']],
                      [s['ddepth_dQ'] * BEAM_POWER_W, s['ddepth_dsig']]])
        try:
            step = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            print("      singular Jacobian; keeping current controls")
            return u, (s['width'], s['depth'], s['asym']), n_runs
        u_new = np.array([np.clip(u[0] + step[0], PMOD_LO, PMOD_HI),
                          np.clip(u[1] + step[1], SIG_LO, SIG_HI)])
        # Stall detection: when a control bound is active the tolerance may be
        # unreachable (e.g. sigma pinned at its floor -> aspect ratio d/w<1
        # cannot be restored without the velocity DOF). If the clamped update
        # no longer moves the controls, further iterations are wasted sims.
        if np.max(np.abs(u_new - u) / np.maximum(np.abs(u), 1e-12)) < 1e-3:
            if verbose:
                print("      stalled at control bounds; accepting best residual")
            return u, (s['width'], s['depth'], s['asym']), n_runs
        u = u_new
    # max_iter reached: report the last evaluated state
    return u, (s['width'], s['depth'], s['asym']), n_runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--V", type=float, default=0.7, help="raster speed (m/s)")
    ap.add_argument("--H", type=float, default=0.15, help="hatch spacing (mm)")
    ap.add_argument("--S", type=float, default=1.0, help="raster size (mm)")
    ap.add_argument("--seg", type=float, default=0.5, help="segment size (mm)")
    ap.add_argument("--res", type=float, default=10e-6)
    ap.add_argument("--buf", type=float, default=0.5e-3)
    ap.add_argument("--Pmod0", type=float, default=1.5, help="baseline power mult")
    ap.add_argument("--sigma0", type=float, default=10e-6, help="baseline sigma (m)")
    ap.add_argument("--w-target", type=float, default=None, help="width target (m)")
    ap.add_argument("--d-target", type=float, default=None, help="depth target (m)")
    ap.add_argument("--max-seg", type=int, default=0, help="cap # powered segments")
    ap.add_argument("--dwell", type=float, default=0.0,
                    help="extra beam-off dwell at each turnaround (s); ~0.7e-3 "
                         "lets the neighbour residue solidify so the width "
                         "target stays feasible at track starts")
    args = ap.parse_args()

    Scan.MakeSquareRaster(V=args.V, H=args.H, S=args.S,
                          outFile="3DThesis/TestInputs/_uni.txt")
    path = Scan.ConvertToSpotRest(Scan.LoadScan("3DThesis/TestInputs/_uni.txt"))
    if args.dwell > 0.0:
        path = Scan.AddTurnDwell(path, args.dwell)
    div = Scan.SegmentScan(path, segmentSize=args.seg)
    powered = [i for i in range(len(div))
               if div[i].strip().split('\t')[0] == '0'
               and div[i].strip().split('\t')[4] == '1']
    if args.max_seg > 0:
        powered = powered[:args.max_seg]

    # ---- Baseline: every powered segment at (Pmod0, sigma0) ----
    print(f"=== BASELINE (fixed Pmod={args.Pmod0}, sigma={args.sigma0*1e6:.1f}um) ===")
    print(f"{'seg':>4} {'width(um)':>10} {'depth(um)':>10} {'asym(um)':>9} {'d/w':>6}")
    base_pmods = {j: args.Pmod0 for j in powered}
    base = []
    for i in powered:
        run_segment(div, i, base_pmods, args.sigma0, args.res, args.buf)
        w, d, a = Sim.ExtractMPDims_Span(inFile=CSV)
        base.append((i, w, d, a))
        print(f"{i:>4} {w*1e6:10.2f} {d*1e6:10.2f} {a*1e6:9.2f} {d/w:6.3f}")

    w_star = args.w_target if args.w_target else base[0][1]
    d_star = args.d_target if args.d_target else base[0][2]
    print(f"\nTarget: w*={w_star*1e6:.2f}um  d*={d_star*1e6:.2f}um"
          f"{'' if args.w_target else '  (from first baseline segment)'}")

    # ---- Optimised: sequential per-segment Newton, warm-started ----
    print("\n=== OPTIMISED (per-segment Newton on Pmod, sigma; exact OTI Jacobian) ===")
    opt_pmods = {}
    opt_sigmas = {}
    opt = []
    u = np.array([args.Pmod0, args.sigma0])
    total_runs = 0
    for i in powered:
        print(f"   seg {i}:")
        u, (w, d, a), n = newton_segment(div, i, opt_pmods, opt_sigmas, u,
                                         [w_star, d_star], args.res, args.buf)
        opt_pmods[i] = u[0]
        opt_sigmas[i] = u[1]
        opt.append((i, u[0], u[1], w, d, a))
        total_runs += n
    print(f"\n{'seg':>4} {'Pmod':>6} {'sig(um)':>8} {'width(um)':>10} "
          f"{'depth(um)':>10} {'asym(um)':>9}")
    for i, P, sg, w, d, a in opt:
        print(f"{i:>4} {P:6.3f} {sg*1e6:8.2f} {w*1e6:10.2f} {d*1e6:10.2f} {a*1e6:9.2f}")
    print(f"total simulations: {total_runs} "
          f"({total_runs/len(powered):.1f} per segment)")

    # ---- Uniformity summary ----
    bw = np.array([b[1] for b in base]); bd = np.array([b[2] for b in base])
    ow = np.array([o[3] for o in opt]);  od = np.array([o[4] for o in opt])
    cv = lambda v: 100.0 * np.std(v) / np.mean(v)
    print("\n=== UNIFORMITY ===")
    print(f"{'':>10} {'width CV%':>10} {'depth CV%':>10} "
          f"{'w range(um)':>12} {'d range(um)':>12}")
    print(f"{'baseline':>10} {cv(bw):10.2f} {cv(bd):10.2f} "
          f"{(bw.max()-bw.min())*1e6:12.2f} {(bd.max()-bd.min())*1e6:12.2f}")
    print(f"{'optimised':>10} {cv(ow):10.2f} {cv(od):10.2f} "
          f"{(ow.max()-ow.min())*1e6:12.2f} {(od.max()-od.min())*1e6:12.2f}")


if __name__ == "__main__":
    main()
