# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `annulus` pattern on both axes. As a geometry: 200 points sampled uniformly by area on a ring around `(0.5, 0.5)` between radii 0.20 and 0.45, no background (parameters `annulus_inner_radius`, `annulus_outer_radius`). As a mingling pattern: a radial split of the colours, giving a colour band on `uniform` geometry, a core and a shell per cluster on `cluster` geometry, and nested rings on `annulus` geometry.
- `Dataset.groups`: shared-geometry IDs. Each geometry realization is drawn once and all three mingling patterns colour that same point set.
- `simulate()` returns the `geometry` and `mingling` of the sample alongside `coords` and `colours`.
- A 3x3 overview figure (`assets/grid.png`), and tests for the annulus patterns and for the evaluation.

### Changed
- **Breaking:** the benchmark is a 3x3 grid, spatial geometry (`uniform`, `cluster`, `annulus`) crossed with mingling (`uniform`, `cluster`, `annulus`), with class names `<geometry>_<mingling>`. `load_dataset()` now returns 900 point clouds (9 classes, 100 samples each).
- **Breaking:** the colouring axis is renamed to mingling: `mixed` is now `uniform` and `separated` is now `cluster`, so `uniform_mixed` becomes `uniform_uniform` and `cluster_separated` becomes `cluster_cluster`.
- `evaluate()` and `PHBaseline` use stratified group 10-fold cross-validation, grouped by shared geometry, so a point set never appears in both a training and a test fold. Chance is 1/9; a colour-blind method cannot exceed 1/3.
- `PHBaseline` orders coordinates canonically before computing persistent homology, so colour-dependent array order cannot affect the result.
- `simulate()`, `load_dataset()` and `generate()` raise `ValueError` on unknown `params` keys instead of silently ignoring them.

### Removed
- `hclust_order`, `hclust_linkage` and `hclust_blue_indices` from the metadata of `simulate()`. They held point positions from before the points are sorted by colour, so they pointed at the wrong points.
- Class figures use the nine new class names, and the baseline confusion matrix is 9x9.

## [0.1.0] - 2026-06-23

### Added
- `load_dataset()`: the 2x2 benchmark of two-colour point clouds in the unit square. Spatial layout (`uniform`, `cluster`) crossed with colouring (`mixed`, `separated`); 100 samples per class, 200 points each, balanced 100/100 between the colours, seed 42. The mixed and separated versions of a layout reuse the same points, so only the colouring tells them apart.
- `simulate()` to draw a single point cloud from the generator.
- `evaluate(method, dataset)`: balanced accuracy under stratified 10-fold cross-validation (chance 0.25) for any method implementing `fit(samples, y)` and `predict(samples)`.
- `PHBaseline` and `run_baseline()`: the colour-blind reference. Alpha-complex persistent homology in dimensions 0 and 1, persistence images, and logistic regression with the regularization strength chosen by inner cross-validation.
- Examples (`quickstart.py`, `validate_method.py`), the figure script, a reproducibility test, and `CITATION.cff`.

[Unreleased]: https://github.com/subthaumic/chromabench/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/subthaumic/chromabench/releases/tag/v0.1.0
