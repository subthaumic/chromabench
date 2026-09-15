from importlib.metadata import version

import numpy as np
import pytest

import chromabench
from chromabench import load_dataset, simulate


def test_version_matches_package_metadata():
    assert chromabench.__version__ == version('chromabench')


def test_unknown_class():
    with pytest.raises(ValueError, match='unknown class'):
        simulate('uniform_mixed', np.random.default_rng(0), None)


def test_odd_cluster_count():
    with pytest.raises(ValueError, match='even K'):
        simulate('cluster_uniform', np.random.default_rng(0), {'K': 5})


@pytest.mark.parametrize('n_per_class', [0, -1, 2.5])
def test_invalid_n_per_class(n_per_class):
    with pytest.raises(ValueError, match='n_per_class'):
        load_dataset(n_per_class=n_per_class)


def test_unknown_param_key():
    with pytest.raises(ValueError, match='unknown params'):
        simulate('annulus_uniform', np.random.default_rng(0), {'annulus_inner_raduis': .1})
    with pytest.raises(ValueError, match='unknown params'):
        load_dataset(n_per_class=1, params={'sigmaa': .1})
