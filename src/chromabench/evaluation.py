"""Evaluation.

``score`` is the benchmark metric (balanced accuracy).
``evaluate`` runs any object implementing the :class:`Method` against the benchmark
under stratified group cross-validation and scores its out-of-fold predictions.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

from .dataset import Dataset, _geometry_groups, load_dataset


class Method(Protocol):
    """The method under test: anything that can fit on point clouds and predict
    class indices. ``samples`` is a list of ``{"coords", "colours"}`` dicts."""

    def fit(self, samples: list[dict[str, np.ndarray]], y: np.ndarray) -> "Method": ...
    def predict(self, samples: list[dict[str, np.ndarray]]) -> np.ndarray: ...


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """The canonical chromabench score: balanced accuracy (the macro-average of
    per-class recall; chance = 1/9 on the nine balanced classes)."""
    return {"balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred))}


def evaluate(method: Method, dataset: Dataset | None = None, *, n_splits: int = 10, seed: int = 42) -> dict[str, Any]:
    """Score a method against the benchmark.

    For each of ``n_splits`` stratified group folds, ``method`` is fit on the training
    point clouds and asked to label the held-out ones; the out-of-fold
    predictions are pooled and scored with the canonical metric. The method
    receives the raw ``{"coords", "colours"}`` samples, so it is free to use
    colour -- which is the whole point. The folds match ``run_baseline``'s (same
    ``n_splits`` and ``seed``), so a method's score and the baseline's are
    directly comparable. All mingling patterns of a geometry stay in the same fold."""
    if dataset is None:
        dataset = load_dataset(seed=seed)
    y = dataset.y
    groups = dataset.groups if dataset.groups is not None else _geometry_groups(dataset.samples)
    if any(len(np.unique(groups[y == label])) < n_splits for label in np.unique(y)):
        raise ValueError("each class must have at least n_splits independent geometry groups")
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_pred = np.empty_like(y)
    for train_idx, test_idx in outer.split(np.zeros(len(y)), y, groups):
        train_samples = [dataset.samples[i] for i in train_idx]
        test_samples = [dataset.samples[i] for i in test_idx]
        method.fit(train_samples, y[train_idx])
        y_pred[test_idx] = np.asarray(method.predict(test_samples))
    return {"score": score(y, y_pred), "y_true": y, "y_pred": y_pred, "n_splits": n_splits}
