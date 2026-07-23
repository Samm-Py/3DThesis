"""Unoptimised baseline: every powered segment at Pmod0 (60 W). Measures
melt-pool width (half-span), depth and asymmetry at the end of each powered
segment via the prior-study pipeline. Writes baseline.csv."""
import json
import pandas as pd
import raster_lib as R
from common import measurement as Sim

if __name__ == "__main__":
    dwell = json.load(open(R.data_path("min_dwell.json")))["dwell_s"]
    R.write_beam()
    # Match run_optimized.py's control/measurement cadence. The 0.25 mm grid is
    # useful diagnostically, but it is not the grid the stabilized optimizer
    # controls against.
    div, powered = R.build_path(dwell, seg=0.5)
    info = R.seg_info(div, powered)
    pmods = {j: R.PMOD0 for j in powered}

    rows = []
    print(f"dwell {dwell*1e3:.3f} ms, Pmod0={R.PMOD0} ({R.PMOD0*R.P_BASE:.0f} W)")
    print(f"{'seg':>4} {'line':>4} {'s(mm)':>6} {'w(um)':>7} {'d(um)':>7} {'asym(um)':>8}")
    for e in info:
        R.run_segment(div, e['idx'], pmods, R.BIN_DBL)
        w, d, a = Sim.ExtractMPDims_Span(inFile=R.CSV)
        rows.append(dict(**e, pmod=R.PMOD0, w=w, d=d, asym=a))
        print(f"{e['idx']:>4} {e['line']:>4} {e['s_mm']:6.2f} "
              f"{w*1e6:7.2f} {d*1e6:7.2f} {a*1e6:8.2f}")
    out = R.data_path("baseline.csv", write=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")
