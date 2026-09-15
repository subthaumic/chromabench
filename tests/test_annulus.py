import numpy as np
import pytest

from chromabench import simulate
from chromabench.simulator import CLASS_NAMES


@pytest.mark.parametrize('name', CLASS_NAMES)
@pytest.mark.parametrize('seed', [0, 42, 123])
def test_all_classes_preserve_counts(name, seed):
    s = simulate(name, np.random.default_rng(seed), {'n_A': 73, 'n_B': 127})
    geometry, mingling = name.split('_')
    assert s['geometry'] == geometry
    assert s['mingling'] == s['metadata']['mingling'] == mingling
    assert np.array_equal(np.bincount(s['colours']), [73, 127])
    assert s['coords'].shape == (200, 2)


@pytest.mark.parametrize('inner,outer', [(0, .45), (.45, .2), (.2, .6)])
def test_invalid_annulus_radii(inner, outer):
    with pytest.raises(ValueError, match='annulus radii'):
        simulate('annulus_uniform', np.random.default_rng(0),
                 {'annulus_inner_radius': inner, 'annulus_outer_radius': outer})


def test_spatial_annulus_bounds_and_area_sampling():
    s = simulate('annulus_uniform', np.random.default_rng(42),
                 {'n_A': 5000, 'n_B': 5000, 'annulus_inner_radius': .15, 'annulus_outer_radius': .4})
    r2 = np.sum((s['coords'] - .5)**2, axis=1)
    assert (r2 >= .15**2).all() and (r2 <= .4**2).all()
    # Squared radius, rather than radius, is uniform for area-uniform sampling.
    assert abs(np.mean(r2) - (.15**2 + .4**2) / 2) < .002


@pytest.mark.parametrize('name,start', [('uniform_annulus', 50), ('annulus_annulus', 100)])
def test_global_annular_mingling(name, start):
    s = simulate(name, np.random.default_rng(42), None)
    order = np.argsort(np.linalg.norm(s['coords'] - .5, axis=1))
    expected = np.ones(200, dtype=int)
    expected[start:start + 100] = 0
    assert np.array_equal(s['colours'][order], expected)


@pytest.mark.parametrize('n_a,n_b', [(100, 100), (73, 127)])
def test_cluster_shells_surround_cores(n_a, n_b):
    s = simulate('cluster_annulus', np.random.default_rng(42), {'n_A': n_a, 'n_B': n_b})
    ids = s['metadata']['cluster_ids']
    for i, centre in enumerate(s['metadata']['centres']):
        mask = ids == i
        radii = np.linalg.norm(s['coords'][mask] - centre, axis=1)
        colours = s['colours'][mask]
        assert radii[colours == 1].max() <= radii[colours == 0].min()
        assert abs(np.sum(colours == 0) - mask.sum() * n_a / (n_a + n_b)) < 1
    assert np.array_equal(np.bincount(s['colours']), [n_a, n_b])
