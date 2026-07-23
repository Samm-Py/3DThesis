"""Unoptimized baseline for one dwell policy: every controlled segment at
Pmod0 (150 W), measured at end-of-segment via the snapshot pipeline
(double build, marching-cubes dims at 1610 K).

Usage: python run_baseline.py {zero|dwell}   -> baseline_{policy}.csv
"""
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import mp_lib as R

policy = sys.argv[1]
dwell = 0.0 if policy == "zero" else R.dwell_from_json()

div, powered, hop_after = R.build_path(dwell)
info = R.seg_info(div, powered)
pmods = {j: R.PMOD0 for j in powered}
pmods.update({h: R.PMOD0 for h in hop_after.values()})

rows, t0 = [], time.time()
print(f"{policy}: dwell {dwell*1e3:.3f} ms, Pmod0={R.PMOD0} "
      f"({R.PMOD0*R.P_BASE:.0f} W), {len(info)} segments")
print(f"{'seg':>4} {'line':>4} {'s(mm)':>7} {'w(um)':>7} {'d(um)':>7} "
      f"{'asym(um)':>8}  clip")
for e in info:
    R.run_segment(div, e['idx'], pmods, R.BIN_DBL)
    w, d, a, clip = R.measure_dims()
    rows.append(dict(**e, pmod=R.PMOD0, sigma=R.SIGMA, w=w, d=d, asym=a,
                     clipped=";".join(clip)))
    print(f"{e['idx']:>4} {e['line']:>4} {e['s_mm']:7.1f} "
          f"{w*1e6:7.2f} {d*1e6:7.2f} {a*1e6:8.2f}  {clip or ''}", flush=True)
out = R.data_path(f"baseline_{policy}.csv", write=True)
pd.DataFrame(rows).to_csv(out, index=False)
print(f"wrote {out}  ({time.time()-t0:.0f} s)")
