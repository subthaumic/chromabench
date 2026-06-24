import numpy as np

from chromabench import load_dataset
from chromabench.simulator import CLASS_NAMES


def test_dataset_is_deterministic():
    a = load_dataset()
    b = load_dataset()
    assert np.array_equal(a.y, b.y)
    for sa, sb in zip(a.samples, b.samples):
        assert np.allclose(sa["coords"], sb["coords"])
        assert np.array_equal(sa["colours"], sb["colours"])


def test_class_balance_and_shape():
    ds = load_dataset()
    assert len(ds.samples) == 4 * 100
    assert tuple(ds.class_names) == CLASS_NAMES
    counts = np.bincount(ds.y, minlength=4)
    assert (counts == 100).all()
    for s in ds.samples:
        assert s["coords"].ndim == 2 and s["coords"].shape[1] == 2
        assert set(np.unique(s["colours"])).issubset({0, 1})
