"""Square entry point for the shared refined baseline replay."""

import os
import runpy

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
os.environ.setdefault("MP_GEOM", "square")
runpy.run_path(
    os.path.join(os.path.dirname(HERE), "common",
                 "verify_baseline_raster.py"),
    run_name="__main__")
