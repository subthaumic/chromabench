"""Dataset construction for chromabench.

The frozen configuration is the four factorial classes with ``n_per_class =
100`` and the default ``FactorialParams`` (``n_A = n_B = 100``, ``K = 6``,
``sigma = 0.065``, ``background_fraction = 0.15``, ...). Each realization is
seeded by a global running index, so the whole dataset is determined by the
seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .simulator import CLASS_NAMES, simulate

# Frozen benchmark defaults.
N_PER_CLASS = 100
SEED = 42

PointCloud = dict[str, np.ndarray]


@dataclass
class Dataset:
    """A built chromabench dataset.

    ``samples`` is a list of point clouds, each a dict with ``coords`` of shape
    ``(n, 2)`` and ``colours`` of shape ``(n,)`` (0 = colour A, 1 = colour B).
    ``y`` holds the integer class index of each sample; ``class_names`` maps
    those indices to the factorial class strings.
    """

    samples: list[PointCloud]
    y: np.ndarray
    class_names: tuple[str, ...]


def generate(
    n_per_class: int = N_PER_CLASS,
    seed: int = SEED,
    params: dict[str, Any] | None = None,
) -> tuple[list[PointCloud], np.ndarray, tuple[str, ...]]:
    """Generate the raw factorial point clouds.

    Returns ``(samples, y, class_names)``. Each realization ``i`` is drawn from
    ``np.random.default_rng(seed + i)`` with ``i`` a global running index across
    all classes.
    """
    samples: list[PointCloud] = []
    labels: list[int] = []
    global_idx = 0
    for class_idx, class_name in enumerate(CLASS_NAMES):
        for _ in range(n_per_class):
            rng = np.random.default_rng(seed + global_idx)
            result = simulate(class_name, rng=rng, params=params)
            samples.append({"coords": result["coords"], "colours": result["colours"]})
            labels.append(class_idx)
            global_idx += 1
    return samples, np.asarray(labels, dtype=int), CLASS_NAMES


def load_dataset(
    n_per_class: int = N_PER_CLASS,
    seed: int = SEED,
    params: dict[str, Any] | None = None,
) -> Dataset:
    """Build the benchmark dataset: four classes x ``n_per_class`` point clouds."""
    samples, y, class_names = generate(n_per_class, seed, params)
    return Dataset(samples=samples, y=y, class_names=class_names)
