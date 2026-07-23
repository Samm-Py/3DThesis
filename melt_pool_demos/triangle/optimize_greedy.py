"""Triangle entry point for the shared greedy raster controller."""

import os
import runpy

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
os.environ.setdefault("MP_GEOM", "triangle")
runpy.run_path(
    os.path.join(os.path.dirname(HERE), "common", "greedy_raster.py"),
    run_name="__main__")
