# chromabench

[![PyPI](https://img.shields.io/pypi/v/chromabench)](https://pypi.org/project/chromabench/)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/subthaumic/chromabench/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22772348.svg)](https://doi.org/10.5281/zenodo.22772348)

A minimal validation for **chromatic topological data analysis**.

If your method is meant to see the shape and colour distribution of a point cloud together, `chromabench` is a simple check that it does.
The benchmark is a **3x3 family**: spatial geometry (`uniform`, `cluster`, `annulus`) crossed with mingling (`uniform`, `cluster`, `annulus`).
All three mingling patterns of each geometry realization reuse the same point set.

## Install

```bash
pip install chromabench
```

## At a glance

`load_dataset()` builds the benchmark, and `evaluate(method, dataset)` scores a method using balanced accuracy under stratified 10-fold cross-validation, grouped by shared geometry. Nine-class chance accuracy is **1/9**.

A method implements `fit(samples, y)` and `predict(samples)`. Each sample is a dict with `"coords"` (an `(n, 2)` array) and `"colours"` (an `(n,)` array of `0`/`1` labels); `y` contains class indices.

```python
from chromabench import evaluate, load_dataset

ds = load_dataset()
print(evaluate(MyChromaticMethod(), ds)["score"])
```

See [`examples/quickstart.py`](examples/quickstart.py).

## The data

Each sample contains 200 points in the unit square, balanced between 100 points of colour A (`0`, blue) and 100 of colour B (`1`, orange).
Rows describe spatial geometry; columns describe mingling. Both axes use the same three pattern names: `uniform`, `cluster`, and `annulus`.

![The nine classes, with shared coordinates across each row](https://raw.githubusercontent.com/subthaumic/chromabench/main/assets/grid.png)

Class names have the form `<geometry>_<mingling>`:

| Geometry / mingling | `uniform` | `cluster` | `annulus` |
| --- | --- | --- | --- |
| `uniform` | `uniform_uniform` | `uniform_cluster` | `uniform_annulus` |
| `cluster` | `cluster_uniform` | `cluster_cluster` | `cluster_annulus` |
| `annulus` | `annulus_uniform` | `annulus_cluster` | `annulus_annulus` |

<details>
<summary><b>Generation details</b></summary>

**Spatial geometry**

- `uniform`: 200 independent uniform points in `[0,1]^2`.
- `cluster`: `K = 6` well-spaced centres, Gaussian clusters with `sigma = 0.065`, and a uniform background fraction of `0.15`. Coordinates are clipped to the unit square. Centres are split evenly between A and B for `cluster` mingling; point counts in each cluster account for the background to preserve the target colour counts.
- `annulus`: 200 points sampled uniformly by area around `(0.5, 0.5)`, between radii `0.20` and `0.45`, with no background. Sample angle uniformly and squared radius uniformly between the squared bounds. The parameters are `annulus_inner_radius` and `annulus_outer_radius`, satisfying `0 < inner < outer <= 0.5`.

**Mingling**

`uniform` selects exactly 100 points uniformly at random for A and assigns the rest to B.

`cluster` produces local colour clusters:

- On `uniform` and `annulus` geometry, complete-linkage hierarchical clustering orders small clusters by increasing merge height and points within them by distance to their centroid; the first 100 points become A. On the annulus this gives local patches along the ring.
- On `cluster` geometry, points inherit their generating centre's colour. Background points inherit their nearest centre's colour. Counts are adjusted if necessary to maintain the target balance.

`annulus` produces nested colour organisation:

- **`uniform` geometry:** sort by distance from `(0.5, 0.5)`. The middle 100 points become A, with 50 B points inside and 50 outside the band.
- **`cluster` geometry:** within each cluster, inner points become B and outer points A, giving a core and surrounding shell. Background points join their nearest centre before radial ranking. A counts are allocated proportionally across clusters, with largest-remainder rounding to keep exactly 100 A points overall.
- **`annulus` geometry:** sort by distance from `(0.5, 0.5)`. The inner 100 points become B and outer 100 become A, giving nested inner and outer rings.

Custom `n_A` and `n_B` are supported. In `uniform_annulus`, `floor(n_B / 2)` B points lie inside the A band, with the remainder outside. In `annulus_annulus`, all B points lie inside all A points. In `cluster_annulus`, shell quotas use the requested A fraction.

The `annulus` mingling pattern describes the colour organisation; it does not guarantee a persistent loop in every finite sample. In particular, sparse or overlapping cluster shells can weaken the surrounding pattern. In `annulus_uniform`, both colour subsets can form loops; `annulus_annulus` tests radial partitioning of the annulus.

</details>

## Data generation

With defaults, `load_dataset()` returns **900 point clouds**: nine classes with 100 samples each. Class indices follow the table in row-major order (`0 = uniform_uniform`, ..., `8 = annulus_annulus`).

```python
from chromabench import load_dataset

ds = load_dataset()          # 9 classes x 100 samples, seed 42
ds.samples                  # list of {"coords": (n, 2), "colours": (n,)}
ds.y                        # class index per sample
ds.class_names              # nine class names
ds.groups                   # shared-geometry IDs: three samples per group
```

`load_dataset(n_per_class=100, seed=42, params=None)` controls sample count, reproducibility, and generator parameters. Each geometry is generated once from `seed + group_id`; separate random streams assign its mingling patterns. The three samples share identical point sets, though array order may differ because points are stacked by colour. Samples are returned in class-major order.

`generate()` returns a `(samples, y, class_names)` tuple. Use `Dataset(samples, y, class_names)` for custom datasets; evaluation identifies identical point sets when `groups` is omitted.

For a single realization:

```python
from chromabench import simulate
import numpy as np

sample = simulate("cluster_annulus", rng=np.random.default_rng(0), params=None)
sample["coords"], sample["colours"]
sample["geometry"]           # "cluster"
sample["mingling"]           # "annulus"
```

## Evaluation

`evaluate(method, dataset)` uses **stratified group 10-fold cross-validation**. All three mingling patterns of a geometry stay together in either training or testing, preventing shared point sets from crossing folds. Each class needs at least `n_splits` independent geometry groups. Balanced accuracy is computed from pooled out-of-fold predictions; chance is `1/9`.

```python
from chromabench import evaluate, load_dataset

result = evaluate(MyChromaticMethod(), load_dataset())
result["score"]      # {"balanced_accuracy": ...}
```

See [`examples/validate_method.py`](examples/validate_method.py).

## The baseline: Ordinary Persistent Homology

Persistent homology of pooled points discards colour. All three mingling patterns of a geometry are therefore identical to this baseline. A method that perfectly identifies spatial geometry while ignoring mingling reaches **1/3** balanced accuracy; the measured baseline can be lower when it confuses geometries.

`PHBaseline` computes alpha-complex persistence in dimensions 0 and 1, vectorizes it with persistence images, and trains logistic regression. Its internal cross-validation also groups identical geometries. Point coordinates are canonically ordered before PH computation so colour-dependent array ordering cannot affect the result.

```python
from chromabench import load_dataset, run_baseline

print(run_baseline(load_dataset())["score"])
```

[`PHBaseline`](src/chromabench/baseline.py) implements the same `fit`/`predict` interface as a submission. `run_baseline` is a convenience wrapper around `evaluate(PHBaseline())`.

<p align="center"><img src="https://raw.githubusercontent.com/subthaumic/chromabench/main/assets/ph_confusion.png" width="650"></p>

The confusion matrix contains pooled out-of-fold predictions, normalized by true class. Identical geometries receive identical predictions across their mingling patterns; the baseline need not guess each mingling equally often.

Regenerate the class panels, overview, and baseline confusion matrix with `uv run python scripts/render_figures.py`.

## Citation

This is exactly what GitHub's "Cite this repository" button emits from [`CITATION.cff`](CITATION.cff) -- keep the two in sync.

```bibtex
@software{Bleher_chromabench_2026,
author = {Bleher, Michael},
license = {MIT},
month = jun,
title = {{chromabench}},
url = {https://github.com/subthaumic/chromabench},
version = {0.1.0},
year = {2026}
}
```

## License

MIT.
