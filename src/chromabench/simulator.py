"""Factorial two-colour point-cloud generator.

The simulation separates the unlabelled point geometry from the colour
assignment:

    geometry:   uniform | cluster
    colouring:  mixed   | separated

This gives four classes:

    uniform_mixed
    uniform_separated
    cluster_mixed
    cluster_separated

The returned colour convention is 0 = colour A and 1 = colour B.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import linkage

COLOUR_A = 0
COLOUR_B = 1

DOMAIN_BOUNDARY = np.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]],
    dtype=float,
)

GEOMETRIES = ("uniform", "cluster")
COLOURINGS = ("mixed", "separated")
CLASS_NAMES = tuple(f"{geometry}_{colouring}" for geometry in GEOMETRIES for colouring in COLOURINGS)


@dataclass(frozen=True)
class FactorialParams:
    n_A: int = 100
    n_B: int = 100
    K: int = 6
    sigma: float = 0.065
    background_fraction: float = 0.15
    cluster_margin: float = 0.12
    cluster_min_distance: float = 0.22
    hclust_method: str = "complete"
    hclust_min_cluster_size: int = 4
    hclust_max_cluster_size: int = 12


def _balanced_counts(total: int, bins: int) -> np.ndarray:
    if bins <= 0:
        raise ValueError("number of bins must be positive")
    counts = np.full(bins, total // bins, dtype=np.int64)
    counts[: total % bins] += 1
    return counts


def _well_spaced_centres(
    rng: np.random.Generator,
    n: int,
    *,
    margin: float,
    min_distance: float,
) -> np.ndarray:
    if n <= 0:
        raise ValueError("number of centres must be positive")

    centres: list[np.ndarray] = []
    attempts = 0
    max_attempts = max(500, 200 * n)
    low = float(margin)
    high = float(1.0 - margin)
    if not low < high:
        raise ValueError("centre margin leaves no sampling area")

    while len(centres) < n and attempts < max_attempts:
        attempts += 1
        candidate = rng.uniform(low, high, size=2)
        if all(np.linalg.norm(candidate - centre) >= min_distance for centre in centres):
            centres.append(candidate)

    if len(centres) == n:
        return np.asarray(centres, dtype=float)

    side = int(np.ceil(np.sqrt(n)))
    xs = np.linspace(low, high, side)
    grid = np.array([(x, y) for y in xs for x in xs], dtype=float)
    rng.shuffle(grid, axis=0)
    return grid[:n]


def _sample_at_centres(
    rng: np.random.Generator,
    centres: np.ndarray,
    cluster_ids: np.ndarray,
    sigma: float,
) -> np.ndarray:
    if len(cluster_ids) == 0:
        return np.empty((0, 2), dtype=float)
    pts = centres[cluster_ids] + sigma * rng.standard_normal(size=(len(cluster_ids), 2))
    return np.clip(pts, 0.0, 1.0)


def _generate_geometry(
    geometry: str,
    rng: np.random.Generator,
    params: FactorialParams,
) -> tuple[np.ndarray, dict[str, Any]]:
    if geometry not in GEOMETRIES:
        raise ValueError(f"unknown geometry {geometry!r}; expected one of {GEOMETRIES}")

    n_total = int(params.n_A + params.n_B)
    if geometry == "uniform":
        return rng.uniform(0.0, 1.0, size=(n_total, 2)), {"geometry": geometry}

    if params.K % 2 != 0:
        raise ValueError("cluster geometry uses an even K so centres split evenly by colour")

    half = params.K // 2
    n_bg = int(round(n_total * params.background_fraction))
    n_cluster = n_total - n_bg
    centres = _well_spaced_centres(
        rng,
        params.K,
        margin=params.cluster_margin,
        min_distance=params.cluster_min_distance,
    )
    cluster_colours = np.repeat([COLOUR_A, COLOUR_B], half).astype(np.int64)

    bg_coords = rng.uniform(0.0, 1.0, size=(n_bg, 2))
    bg_cluster_ids = np.empty(0, dtype=np.int64)
    bg_counts = np.zeros(2, dtype=np.int64)
    if n_bg:
        d2 = np.sum((bg_coords[:, None, :] - centres[None, :, :]) ** 2, axis=2)
        bg_cluster_ids = np.argmin(d2, axis=1).astype(np.int64)
        bg_counts = np.bincount(cluster_colours[bg_cluster_ids], minlength=2)

    target_cluster_counts = np.array([params.n_A, params.n_B], dtype=np.int64) - bg_counts
    if np.any(target_cluster_counts < 0):
        raise ValueError("background fraction is too large to preserve exact colour counts")
    if int(target_cluster_counts.sum()) != n_cluster:
        raise ValueError("cluster/background split is inconsistent with colour counts")

    cluster_counts = np.concatenate(
        [
            _balanced_counts(int(target_cluster_counts[COLOUR_A]), half),
            _balanced_counts(int(target_cluster_counts[COLOUR_B]), half),
        ]
    )
    cluster_ids = np.repeat(np.arange(params.K, dtype=np.int64), cluster_counts)
    cluster_coords = _sample_at_centres(rng, centres, cluster_ids, params.sigma)
    coords = np.vstack([cluster_coords, bg_coords])
    cluster_ids = np.concatenate([cluster_ids, bg_cluster_ids])

    order = rng.permutation(n_total)
    coords = coords[order]
    cluster_ids = cluster_ids[order]
    return coords, {
        "geometry": geometry,
        "centres": centres,
        "cluster_ids": cluster_ids,
        "cluster_colours": cluster_colours,
        "cluster_counts": cluster_counts,
        "background_counts_by_colour": bg_counts,
        "n_cluster": n_cluster,
        "n_background": n_bg,
    }


def _hierarchical_cluster_candidates(
    coords: np.ndarray,
    params: FactorialParams,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    n = len(coords)
    z = linkage(coords, method=params.hclust_method, metric="euclidean")
    members: dict[int, np.ndarray] = {i: np.array([i], dtype=np.int64) for i in range(n)}
    heights = np.zeros(2 * n - 1, dtype=float)
    parent_heights = np.full(2 * n - 1, np.inf, dtype=float)

    for row_idx, (left, right, dist, _count) in enumerate(z):
        left = int(left)
        right = int(right)
        node = n + row_idx
        members[node] = np.concatenate([members[left], members[right]])
        heights[node] = float(dist)
        parent_heights[left] = float(dist)
        parent_heights[right] = float(dist)

    eps = np.finfo(float).eps
    candidates = []
    for node in range(n, 2 * n - 1):
        idx = members[node]
        size = len(idx)
        if size < params.hclust_min_cluster_size or size > params.hclust_max_cluster_size:
            continue
        if size > params.n_A:
            continue

        height = heights[node]
        parent_height = parent_heights[node]
        if np.isinf(parent_height):
            parent_height = height
        persistence = max(parent_height - height, 0.0)
        strength = persistence / max(height, eps)
        candidates.append(
            {
                "node": node,
                "members": idx,
                "size": size,
                "height": height,
                "parent_height": parent_height,
                "persistence": persistence,
                "strength": strength,
            }
        )

    candidates.sort(key=lambda c: (c["height"], c["size"], -c["persistence"], c["node"]))
    return candidates, z


def _hierarchical_point_order(
    coords: np.ndarray,
    params: FactorialParams,
) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray]:
    candidates, z = _hierarchical_cluster_candidates(coords, params)
    used = np.zeros(len(coords), dtype=bool)
    ordered: list[int] = []
    ordered_clusters: list[dict[str, Any]] = []

    for candidate in candidates:
        idx = candidate["members"]
        idx = idx[~used[idx]]
        if len(idx) == 0:
            continue
        centroid = coords[idx].mean(axis=0)
        local_order = idx[np.argsort(np.linalg.norm(coords[idx] - centroid, axis=1), kind="stable")]
        ordered.extend(int(i) for i in local_order)
        used[local_order] = True
        ordered_clusters.append({**candidate, "members": local_order})

    remaining = np.flatnonzero(~used)
    if len(remaining):
        if ordered_clusters:
            centroids = np.array([coords[c["members"]].mean(axis=0) for c in ordered_clusters])
            d2 = np.sum((coords[remaining, None, :] - centroids[None, :, :]) ** 2, axis=2)
            remaining = remaining[np.argsort(np.min(d2, axis=1), kind="stable")]
        ordered.extend(int(i) for i in remaining)

    return np.asarray(ordered, dtype=np.int64), ordered_clusters, z


def _assign_uniform_hclust(
    coords: np.ndarray,
    params: FactorialParams,
) -> tuple[np.ndarray, dict[str, Any]]:
    colours = np.full(len(coords), COLOUR_B, dtype=np.int64)
    order, ordered_clusters, z = _hierarchical_point_order(coords, params)
    colours[order[: params.n_A]] = COLOUR_A
    return colours, {
        "colouring": "separated",
        "colour_rule": "hierarchical_small_cluster_order",
        "hclust_linkage": z,
        "hclust_order": order,
        "hclust_blue_indices": order[: params.n_A],
        "hclust_ordered_cluster_sizes": np.array([len(c["members"]) for c in ordered_clusters]),
        "hclust_ordered_cluster_heights": np.array([c["height"] for c in ordered_clusters]),
    }


def _enforce_exact_counts(
    coords: np.ndarray,
    colours: np.ndarray,
    params: FactorialParams,
) -> np.ndarray:
    counts = np.bincount(colours, minlength=2)
    if counts[COLOUR_A] == params.n_A and counts[COLOUR_B] == params.n_B:
        return colours

    colours = colours.copy()
    if counts[COLOUR_A] > params.n_A:
        candidates = np.flatnonzero(colours == COLOUR_A)
        remove_n = int(counts[COLOUR_A] - params.n_A)
        centroid = coords[candidates].mean(axis=0)
        remove = candidates[np.argsort(-np.linalg.norm(coords[candidates] - centroid, axis=1), kind="stable")[:remove_n]]
        colours[remove] = COLOUR_B
    elif counts[COLOUR_A] < params.n_A:
        candidates = np.flatnonzero(colours == COLOUR_B)
        add_n = int(params.n_A - counts[COLOUR_A])
        if np.any(colours == COLOUR_A):
            centroid = coords[colours == COLOUR_A].mean(axis=0)
            add = candidates[np.argsort(np.linalg.norm(coords[candidates] - centroid, axis=1), kind="stable")[:add_n]]
        else:
            add = candidates[:add_n]
        colours[add] = COLOUR_A

    return colours


def _assign_cluster_colours(
    coords: np.ndarray,
    geometry_meta: dict[str, Any],
    params: FactorialParams,
) -> tuple[np.ndarray, dict[str, Any]]:
    cluster_ids = geometry_meta.get("cluster_ids")
    cluster_colours = geometry_meta.get("cluster_colours")
    if cluster_ids is None or cluster_colours is None:
        centres = geometry_meta.get("centres")
        if centres is None:
            raise ValueError("cluster_separated requires cluster IDs or centres in geometry metadata")
        centres = np.asarray(centres, dtype=float)
        half = len(centres) // 2
        if len(centres) % 2 != 0:
            raise ValueError("cluster_separated requires an even number of centres")
        cluster_colours = np.repeat([COLOUR_A, COLOUR_B], half).astype(np.int64)
        d2 = np.sum((coords[:, None, :] - centres[None, :, :]) ** 2, axis=2)
        cluster_ids = np.argmin(d2, axis=1)

    cluster_ids = np.asarray(cluster_ids, dtype=np.int64)
    cluster_colours = np.asarray(cluster_colours, dtype=np.int64)
    colours = cluster_colours[cluster_ids]
    colours = _enforce_exact_counts(coords, colours, params)
    return colours, {
        "colouring": "separated",
        "colour_rule": "cluster_colour_split",
    }


def _assign_colours(
    coords: np.ndarray,
    colouring: str,
    rng: np.random.Generator,
    params: FactorialParams,
    *,
    geometry: str,
    geometry_meta: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if colouring not in COLOURINGS:
        raise ValueError(f"unknown colouring {colouring!r}; expected one of {COLOURINGS}")

    n = len(coords)
    if params.n_A + params.n_B != n:
        raise ValueError("n_A + n_B must equal the number of uncoloured points")

    colours = np.full(n, COLOUR_B, dtype=np.int64)
    if colouring == "mixed":
        idx_A = rng.choice(n, size=params.n_A, replace=False)
        colours[idx_A] = COLOUR_A
        return colours, {"colouring": colouring}

    if geometry == "uniform":
        return _assign_uniform_hclust(coords, params)
    if geometry == "cluster":
        return _assign_cluster_colours(coords, geometry_meta or {}, params)
    raise ValueError(f"unknown geometry {geometry!r}")


def _stack_by_colour(coords: np.ndarray, colours: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(colours, kind="stable")
    return coords[order], colours[order], order


def _reorder_pointwise_metadata(metadata: dict[str, Any], order: np.ndarray) -> dict[str, Any]:
    n = len(order)
    out = {}
    for key, value in metadata.items():
        if isinstance(value, np.ndarray) and len(value) == n:
            out[key] = value[order]
        else:
            out[key] = value
    return out


def _params_from_dict(params: dict[str, Any] | None) -> FactorialParams:
    if params is None:
        return FactorialParams()
    allowed = FactorialParams.__dataclass_fields__
    return FactorialParams(**{key: value for key, value in params.items() if key in allowed})


def simulate(class_name: str, rng: np.random.Generator, params: dict[str, Any] | None) -> dict:
    """Generate one realization of a specified class."""
    if class_name not in CLASS_NAMES:
        raise ValueError(f"unknown class: {class_name!r}; expected one of {CLASS_NAMES}")

    p = _params_from_dict(params)
    geometry, colouring = class_name.split("_", 1)
    coords, geometry_meta = _generate_geometry(geometry, rng, p)
    colours, colour_meta = _assign_colours(
        coords,
        colouring,
        rng,
        p,
        geometry=geometry,
        geometry_meta=geometry_meta,
    )
    metadata = _reorder_pointwise_metadata({**geometry_meta, **colour_meta}, np.arange(len(coords)))
    coords, colours, order = _stack_by_colour(coords, colours)
    metadata = _reorder_pointwise_metadata(metadata, order)

    return {
        "coords": coords,
        "colours": colours,
        "domain_boundary": DOMAIN_BOUNDARY.copy(),
        "geometry": geometry,
        "colouring": colouring,
        "metadata": metadata,
    }
