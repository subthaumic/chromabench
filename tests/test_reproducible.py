import numpy as np

from chromabench import generate, load_dataset
from chromabench.simulator import CLASS_NAMES, GEOMETRIES, MINGLING_PATTERNS


def canonical(coords):
    return coords[np.lexsort((coords[:, 1], coords[:, 0]))]


def test_dataset_is_deterministic():
    a = load_dataset(n_per_class=3)
    b = load_dataset(n_per_class=3)
    assert np.array_equal(a.y, b.y)
    assert np.array_equal(a.groups, b.groups)
    for sa, sb in zip(a.samples, b.samples):
        assert np.array_equal(sa['coords'], sb['coords'])
        assert np.array_equal(sa['colours'], sb['colours'])
    c = load_dataset(n_per_class=3, seed=43)
    assert not np.array_equal(a.samples[0]['coords'], c.samples[0]['coords'])


def test_class_balance_shape_and_shared_geometry():
    assert GEOMETRIES == MINGLING_PATTERNS == ("uniform", "cluster", "annulus")
    ds = load_dataset()
    assert len(ds.samples) == 9 * 100
    assert len(ds.class_names) == 9
    assert tuple(ds.class_names) == CLASS_NAMES
    assert (np.bincount(ds.y) == 100).all()
    for s in ds.samples:
        assert s['coords'].shape == (200, 2)
        assert ((s['coords'] >= 0) & (s['coords'] <= 1)).all()
        assert np.array_equal(np.bincount(s['colours']), [100, 100])
    for group in np.unique(ds.groups):
        idx = np.flatnonzero(ds.groups == group)
        assert len(idx) == 3
        assert {ds.class_names[ds.y[i]].split('_')[1] for i in idx} == set(MINGLING_PATTERNS)
        reference = canonical(ds.samples[idx[0]]['coords'])
        for i in idx[1:]:
            assert np.array_equal(reference, canonical(ds.samples[i]['coords']))
            assert not np.shares_memory(ds.samples[idx[0]]['coords'], ds.samples[i]['coords'])


def test_generate_tuple_interface():
    samples, y, names = generate(n_per_class=2)
    ds = load_dataset(n_per_class=2)
    assert names == ds.class_names
    assert np.array_equal(y, ds.y)
    for a, b in zip(samples, ds.samples):
        assert np.array_equal(a['coords'], b['coords'])
        assert np.array_equal(a['colours'], b['colours'])
