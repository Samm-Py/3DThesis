# Square study figures

The three paper figures, each as PNG and PDF:

- `square_traces_zero`: per-line whole-build pool statistics — per scan line,
  the mean and p5–p95 band of the instantaneous beam-on depth/volume/length,
  plus a real-time detail strip (`make_greedy_figures.py`);
- `square_process_parameters`: the optimized power, beam-σ, and scan-velocity
  histories (`make_greedy_figures.py`);
- `square_maps_zero`: solidification thermal-gradient G maps — top surface
  plus centre cross-section — for baseline and optimized
  (`make_paper_maps.py`).

`SqBaselineX50Y10Z1Zero_fig16_maps.png` and
`SqGreedy1X50Y10Z1Zero_fig16_maps.png` are the individual map panels that
`make_paper_maps.py` recomposes into `square_maps_zero`.

All files are regenerable from `make square`, which also copies the three
PDFs into `../../paper/figures/`.
