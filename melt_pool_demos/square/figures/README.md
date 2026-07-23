# Square study figures

The top-level figures are the curated publication set:

- `square_traces_*` (PNG/PDF): per-line whole-build pool statistics — per
  scan line, the mean and p5–p95 band of the instantaneous beam-on
  depth/volume/length, plus a real-time detail strip;
- `square_profiles_*` (PNG/PDF): the same traces folded by position within
  each scan line — the clearest view of within-line vs line-to-line variation;
- `Sq{Zero,Dwell,OptZero,OptDwell}_fig17_traces.png`: melt-pool length /
  depth / volume vs build time from the RDF event list (paper Fig. 17 style),
  one per baseline/optimized case, on fixed axes for direct before/after
  comparison;
- `Sq{Zero,Dwell,OptZero,OptDwell}_fig16_maps.png`: solidification thermal
  gradient G — top surface plus centre cross-section (paper Fig. 16 style).

Supporting visual products are separated into:

- `animations/`: baseline/optimized and zero/dwell GIFs;
- `calibration/`: the developed single-track calibration plot;
- `diagnostics/`: individual replay plots emitted during post-processing.

All files are regenerable from `make square` (or the individual
`../common/make_plots.py` commands); the `fig17`/`fig16` views and animations
additionally require regenerated raw case `Data/`.
