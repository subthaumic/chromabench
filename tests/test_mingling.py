import numpy as np
import pytest
from scipy.spatial import cKDTree

from chromabench import simulate
from chromabench.simulator import GEOMETRIES

SEEDS = range(10)


def same_colour_neighbours(sample, k=5):
    """Mean fraction of each point's k nearest neighbours that share its colour."""
    coords, colours = sample['coords'], sample['colours']
    _, nn = cKDTree(coords).query(coords, k=k + 1)
    return np.mean(colours[nn[:, 1:]] == colours[:, None])


def mean_over_seeds(name):
    return np.mean([same_colour_neighbours(simulate(name, np.random.default_rng(s), None)) for s in SEEDS])


@pytest.mark.parametrize('geometry', GEOMETRIES)
def test_uniform_mingling_is_mixed(geometry):
    assert abs(mean_over_seeds(f'{geometry}_uniform') - .5) < .05


@pytest.mark.parametrize('mingling', ['cluster', 'annulus'])
@pytest.mark.parametrize('geometry', GEOMETRIES)
def test_structured_mingling_is_separated(geometry, mingling):
    assert mean_over_seeds(f'{geometry}_{mingling}') > .65
