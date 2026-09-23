# oti_ad: the first AD prototype (historical)

This directory records the first version of automatic differentiation in
3DThesis (commit `5697d70`), which differentiated the temperature with respect
to position, beam power and the material properties k, rho and c_p. It is kept
for provenance, not maintained.

- [`CHANGES.md`](CHANGES.md): what the prototype changed, file by file, and
  the MPI + OTI verification (still valid: no `Real` value is communicated).
- [`RESULTS.md`](RESULTS.md): its finite-difference validation and a runtime
  benchmark.
- `oti_fields.png`, `oti_evolution.gif` / `.mp4`: the prototype's temperature
  and sensitivity fields over a raster scan.
- The Python scripts and `case_refined/` expect the prototype's output
  columns (`dT_dkon`, `dT_drho`, `dT_dcps`) and do not run against the
  current build.

The current design and runnable checks are in
[`../ad_tutorial/`](../ad_tutorial/README.md).
