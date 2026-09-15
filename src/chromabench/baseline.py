"""The colour-blind persistent-homology baseline.

`PHBaseline` implements the :class:`~chromabench.evaluation.Method`. It is subsequently
scored by :func:`~chromabench.evaluation.evaluate`.

This baseline deliberately throws colour away and works only from ``coords``:
   alpha complex on the pooled points 
-> persistent homology in dimensions 0 and 1 
-> per-dimension persistence images 
-> standardized, L2-regularized multinomial logistic regression (the regularization
strength is chosen by inner cross-validation).

Because the point sets are identical across the three mingling patterns, this
baseline can recover spatial layout but cannot distinguish mingling.
"""

from __future__ import annotations

from typing import Any

import gudhi as gd
import numpy as np
from gudhi.representations import PersistenceImage
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import Dataset, _canonical_coords, _geometry_groups
from .evaluation import evaluate

PH_DIMS = (0, 1)
PI_RESOLUTION = [10, 10]
PI_BANDWIDTH = 0.02
C_VALUES = (0.01, 0.1, 1.0, 10.0, 100.0)
MAX_ITER = 5000
INNER_CV_SPLITS = 5


def _finite(diagram: np.ndarray) -> np.ndarray:
    if diagram is None or len(diagram) == 0:
        return np.empty((0, 2))
    d = np.asarray(diagram, dtype=float).reshape(-1, 2)
    return d[np.isfinite(d).all(axis=1)]


def _alpha_ph(coords: np.ndarray, dims: tuple[int, ...]) -> dict[int, np.ndarray]:
    """Persistent homology of the alpha complex on the points, as finite
    (birth, death) diagrams per dimension."""
    st = gd.AlphaComplex(points=_canonical_coords(coords)).create_simplex_tree()
    st.compute_persistence()
    out: dict[int, np.ndarray] = {}
    for d in dims:
        intervals = np.asarray(st.persistence_intervals_in_dimension(d), dtype=float).reshape(-1, 2)
        out[d] = _finite(intervals) if intervals.size else np.empty((0, 2))
    return out


def _fit_vectorizer(diagrams: list[np.ndarray], vectorizer: PersistenceImage) -> None:
    """Fit a persistence-image vectorizer on a set of diagrams (training fold
    only)."""
    nonempty = [d for d in (_finite(x) for x in diagrams) if len(d) > 0]
    vectorizer.fit(nonempty if nonempty else [np.array([[0.0, 1e-6]])])


def _transform_diagrams(diagrams: list[np.ndarray], vectorizer: PersistenceImage) -> np.ndarray:
    """Transform diagrams with an already-fitted vectorizer; empty diagrams map
    to (near-)zero vectors."""
    rows, zero = [], None
    for x in diagrams:
        d = _finite(x)
        if len(d) == 0:
            if zero is None:
                zero = np.zeros_like(vectorizer.transform([np.array([[0.0, 0.0]])])[0])
            rows.append(zero)
        else:
            rows.append(vectorizer.transform([d])[0])
    return np.asarray(rows, dtype=float)


def _pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(solver="lbfgs", max_iter=MAX_ITER)),
        ]
    )


class PHBaseline:
    """Reference :class:`~chromabench.evaluation.Method`: colour-blind PH with
    persistence-image features and logistic regression. See the module docstring
    for what it computes; it is a complete, copyable example of the ``fit`` /
    ``predict`` interface a submission implements."""

    def __init__(self, dims: tuple[int, ...] = PH_DIMS, random_state: int = 0):
        self.dims = dims
        self.random_state = random_state

    def fit(self, samples: list[dict[str, np.ndarray]], y: np.ndarray) -> "PHBaseline":
        diagrams = [_alpha_ph(s["coords"], self.dims) for s in samples]
        self._vectorizers: dict[int, PersistenceImage] = {}
        blocks = []
        for d in self.dims:
            per_dim = [dg[d] for dg in diagrams]
            vec = PersistenceImage(bandwidth=PI_BANDWIDTH, resolution=PI_RESOLUTION, weight=lambda x: x[1])
            _fit_vectorizer(per_dim, vec)
            self._vectorizers[d] = vec
            blocks.append(_transform_diagrams(per_dim, vec))
        groups = _geometry_groups(samples)
        n_splits = min(INNER_CV_SPLITS, min(len(np.unique(groups[y == label])) for label in np.unique(y)))
        if n_splits < 2:
            raise ValueError("PHBaseline requires at least two training geometry groups per class")
        inner = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        self._clf = GridSearchCV(
            _pipeline(), {"clf__C": list(C_VALUES)}, cv=inner, scoring="balanced_accuracy", n_jobs=1,
        )
        self._clf.fit(np.hstack(blocks), y, groups=groups)
        return self

    def predict(self, samples: list[dict[str, np.ndarray]]) -> np.ndarray:
        diagrams = [_alpha_ph(s["coords"], self.dims) for s in samples]
        blocks = [_transform_diagrams([dg[d] for dg in diagrams], self._vectorizers[d]) for d in self.dims]
        return self._clf.predict(np.hstack(blocks))


def run_baseline(dataset: Dataset | None = None, *, n_splits: int = 10, seed: int = 42) -> dict[str, Any]:
    """Score the colour-blind PH baseline. Convenience wrapper around 
    ``evaluate(PHBaseline(), ...)``."""
    return evaluate(PHBaseline(), dataset, n_splits=n_splits, seed=seed)
