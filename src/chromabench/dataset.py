"""Build nine factorial classes in groups of three shared geometries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .simulator import (
    CLASS_NAMES, GEOMETRIES, MINGLING_PATTERNS,
    _apply_mingling, _generate_geometry, _params_from_dict,
)

N_PER_CLASS = 100
SEED = 42

PointCloud = dict[str, np.ndarray]


@dataclass
class Dataset:
    """Point clouds, class indices, and optional shared-geometry group IDs.

    Each sample contains ``coords`` of shape ``(n, 2)`` and ``colours`` of
    shape ``(n,)`` (0 = A, 1 = B). ``class_names`` maps the entries in ``y``.
    ``groups`` keeps all mingling patterns of a geometry in the same evaluation fold.
    If omitted for a custom dataset, evaluation identifies identical point sets.
    """

    samples: list[PointCloud]
    y: np.ndarray
    class_names: tuple[str, ...]
    groups: np.ndarray | None = None


def _canonical_coords(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    return coords[np.lexsort((coords[:, 1], coords[:, 0]))]


def _geometry_groups(samples: list[PointCloud]) -> np.ndarray:
    """Identify duplicate geometries independently of point order and colour."""
    known: dict[bytes, int] = {}
    groups = []
    for sample in samples:
        key = _canonical_coords(sample["coords"]).tobytes()
        groups.append(known.setdefault(key, len(known)))
    return np.asarray(groups, dtype=int)


def generate(
    n_per_class: int = N_PER_CLASS,
    seed: int = SEED,
    params: dict[str, Any] | None = None,
) -> tuple[list[PointCloud], np.ndarray, tuple[str, ...]]:
    """Return ``(samples, y, class_names)`` with shared points across mingling patterns.

    Use ``load_dataset`` to also obtain the explicit geometry group IDs.
    """
    ds = load_dataset(n_per_class, seed, params)
    return ds.samples, ds.y, ds.class_names


def load_dataset(
    n_per_class: int = N_PER_CLASS,
    seed: int = SEED,
    params: dict[str, Any] | None = None,
) -> Dataset:
    """Build nine classes x ``n_per_class`` clouds, in class-major order.

    Each geometry realization is generated once with seed ``seed + group_id``.
    Its three mingling patterns share that point set, though their array order can
    differ because samples are stacked by colour. Mingling RNGs use separate
    streams independently of the sampled geometry.
    """
    if not isinstance(n_per_class, (int, np.integer)) or n_per_class < 1:
        raise ValueError("n_per_class must be a positive integer")
    p = _params_from_dict(params)
    by_class: list[list[PointCloud]] = [[] for _ in CLASS_NAMES]
    group_ids: list[list[int]] = [[] for _ in CLASS_NAMES]
    for geometry_idx, geometry in enumerate(GEOMETRIES):
        for realization in range(n_per_class):
            group_id = geometry_idx * n_per_class + realization
            coords, meta = _generate_geometry(geometry, np.random.default_rng(seed + group_id), p)
            for mingling_idx, mingling in enumerate(MINGLING_PATTERNS):
                rng = np.random.default_rng(np.random.SeedSequence([seed, group_id, mingling_idx, 1]))
                result = _apply_mingling(coords, geometry, meta, mingling, rng, p)
                class_idx = geometry_idx * len(MINGLING_PATTERNS) + mingling_idx
                by_class[class_idx].append({"coords": result["coords"], "colours": result["colours"]})
                group_ids[class_idx].append(group_id)
    return Dataset(
        samples=[sample for bucket in by_class for sample in bucket],
        y=np.repeat(np.arange(len(CLASS_NAMES)), n_per_class),
        class_names=CLASS_NAMES,
        groups=np.asarray([group for bucket in group_ids for group in bucket], dtype=int),
    )
