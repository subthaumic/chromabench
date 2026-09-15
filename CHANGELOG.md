# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- No changes yet.

## [0.1.0] - 2026-06-23

### Added
- `load_dataset()`: the 2x2 benchmark of two-colour point clouds in the unit square. Spatial layout (`uniform`, `cluster`) crossed with colouring (`mixed`, `separated`); 100 samples per class, 200 points each, balanced 100/100 between the colours, seed 42. The mixed and separated versions of a layout reuse the same points, so only the colouring tells them apart.
- `simulate()` to draw a single point cloud from the generator.
- `evaluate(method, dataset)`: balanced accuracy under stratified 10-fold cross-validation (chance 0.25) for any method implementing `fit(samples, y)` and `predict(samples)`.
- `PHBaseline` and `run_baseline()`: the colour-blind reference. Alpha-complex persistent homology in dimensions 0 and 1, persistence images, and logistic regression with the regularization strength chosen by inner cross-validation.
- Examples (`quickstart.py`, `validate_method.py`), the figure script, a reproducibility test, and `CITATION.cff`.

[Unreleased]: https://github.com/subthaumic/chromabench/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/subthaumic/chromabench/releases/tag/v0.1.0
