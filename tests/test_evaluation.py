import numpy as np
import pytest

from chromabench import Dataset, evaluate, load_dataset, run_baseline


def geometry_key(sample):
    coords = sample['coords']
    return coords[np.lexsort((coords[:, 1], coords[:, 0]))].tobytes()


class CheckHeldOutGeometries:
    def __init__(self):
        self.held_out = []

    def fit(self, samples, y):
        self.training = {geometry_key(s) for s in samples}
        return self

    def predict(self, samples):
        keys = [geometry_key(s) for s in samples]
        assert self.training.isdisjoint(keys)
        self.held_out.extend(keys)
        return np.zeros(len(samples), dtype=int)


@pytest.mark.parametrize('explicit_groups', [True, False])
def test_shared_geometries_never_cross_folds(explicit_groups):
    ds = load_dataset(n_per_class=6)
    if not explicit_groups:
        ds = Dataset(ds.samples, ds.y, ds.class_names)
    method = CheckHeldOutGeometries()
    result = evaluate(method, ds, n_splits=3)
    assert len(method.held_out) == len(ds.samples)
    assert result['score']['balanced_accuracy'] == pytest.approx(1 / 9)


def test_too_few_groups():
    with pytest.raises(ValueError, match='geometry groups'):
        evaluate(CheckHeldOutGeometries(), load_dataset(n_per_class=2), n_splits=3)


def test_ph_baseline_cannot_distinguish_minglings():
    ds = load_dataset(n_per_class=6)
    result = run_baseline(ds, n_splits=3)
    assert result['score']['balanced_accuracy'] <= 1 / 3 + 1e-12
    for group in np.unique(ds.groups):
        assert len(np.unique(result['y_pred'][ds.groups == group])) == 1
