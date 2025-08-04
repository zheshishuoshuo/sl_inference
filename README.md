sl_inference now uses a two-stage workflow for fast likelihood evaluation.

## 1. Grid building

Use `grid_builder.build_grid` to pre-compute lensing observables on a two-dimensional grid of dark-matter parameters (`gamma_dm` and `logM_dm`). The results are stored in an HDF5 table.

## 2. Likelihood evaluation

`likelihood.GridLikelihood` loads the HDF5 table and evaluates the likelihood for hyper-parameters `eta` by spline-integrating the weighted grid.

## 3. Minimal example

Run

```
python -m sl_inference.minimal_example
```

to generate mock data, build a grid, and execute a single MCMC step. All generated files are written under `data/` with separate sub-directories for mock data, interpolation tables, and inference outputs.
