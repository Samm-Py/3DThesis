"""Square entry point for the shared 1 mm greedy figure set."""

import os
import runpy

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
os.environ.setdefault("MP_GEOM", "square")
runpy.run_path(
    os.path.join(os.path.dirname(HERE), "common",
                 "make_greedy_raster_figures.py"),
    run_name="__main__")
