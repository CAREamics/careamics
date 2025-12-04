import numpy as np
import pytest

from careamics.dataset_ng.normalization.statistics import (
    _compute_mean_std,
    _compute_min_max,
    _compute_quantiles,
)
from careamics.dataset_ng.patching_strategies import WholeSamplePatchingStrategy
from careamics.dataset_ng.image_stack_loader import load_arrays
from careamics.dataset_ng.patch_extractor.patch_extractor import PatchExtractor


@pytest.mark.parametrize(
    "array_shape,axes,c_axis",
    [
        [(1, 10, 10), "CYX", 0],
        [(3, 10, 10), "CYX", 0],
        [(3, 1, 10, 10), "SCYX", 1],
        [(3, 3, 10, 10), "SCYX", 1],
        [(3, 1, 10, 10, 10), "SCZYX", 1],
        [(3, 3, 10, 10, 10), "SCZYX", 1],
    ],
)
def test_compute_mean_std(array_shape, axes, c_axis):
    test_arr = np.random.randint(0, 100, array_shape)
    loader = load_arrays([test_arr], axes=axes)
    extractor = PatchExtractor(loader)
    strategy = WholeSamplePatchingStrategy(extractor.shape)

    computed_means, computed_stds = _compute_mean_std(extractor, strategy)

    assert len(computed_means) == array_shape[c_axis]
    assert len(computed_stds) == array_shape[c_axis]

    axes_for_mean = list(range(test_arr.ndim))
    axes_for_mean.remove(c_axis)
    real_means = test_arr.mean(axis=tuple(axes_for_mean))
    real_stds = test_arr.std(axis=tuple(axes_for_mean))

    assert np.allclose(computed_means, real_means)
    assert np.allclose(computed_stds, real_stds)


@pytest.mark.parametrize(
    "array_shape,axes,c_axis",
    [
        [(1, 10, 10), "CYX", 0],
        [(3, 10, 10), "CYX", 0],
        [(3, 1, 10, 10), "SCYX", 1],
        [(3, 3, 10, 10), "SCYX", 1],
        [(3, 1, 10, 10, 10), "SCZYX", 1],
        [(3, 3, 10, 10, 10), "SCZYX", 1],
    ],
)
def test_compute_min_max(array_shape, axes, c_axis):
    test_arr = np.random.randint(0, 100, array_shape)
    loader = load_arrays([test_arr], axes=axes)
    extractor = PatchExtractor(loader)
    strategy = WholeSamplePatchingStrategy(extractor.shape)

    computed_mins, computed_maxes = _compute_min_max(extractor, strategy)

    assert len(computed_mins) == array_shape[c_axis]
    assert len(computed_maxes) == array_shape[c_axis]

    axes_for_minmax = list(range(test_arr.ndim))
    axes_for_minmax.remove(c_axis)
    real_mins = test_arr.min(axis=tuple(axes_for_minmax))
    real_maxes = test_arr.max(axis=tuple(axes_for_minmax))

    assert np.allclose(computed_mins, real_mins)
    assert np.allclose(computed_maxes, real_maxes)


@pytest.mark.parametrize(
    "array_shape,axes,c_axis",
    [
        [(1, 10, 10), "CYX", 0],
        [(3, 10, 10), "CYX", 0],
        [(3, 1, 10, 10), "SCYX", 1],
        [(3, 3, 10, 10), "SCYX", 1],
        [(3, 1, 10, 10, 10), "SCZYX", 1],
        [(3, 3, 10, 10, 10), "SCZYX", 1],
    ],
)
def test_compute_quantiles(array_shape, axes, c_axis):
    test_arr = np.random.randint(0, 100, array_shape)
    loader = load_arrays([test_arr], axes=axes)
    extractor = PatchExtractor(loader)
    strategy = WholeSamplePatchingStrategy(extractor.shape)

    n_channels = array_shape[c_axis]
    lower_levels = [0.01] * n_channels
    upper_levels = [0.99] * n_channels

    computed_lower, computed_upper = _compute_quantiles(
        extractor, strategy, lower_levels, upper_levels
    )

    assert len(computed_lower) == n_channels
    assert len(computed_upper) == n_channels

    axes_for_quantiles = list(range(test_arr.ndim))
    axes_for_quantiles.remove(c_axis)
    real_lower_quantiles = np.quantile(test_arr, 0.01, axis=tuple(axes_for_quantiles))
    real_upper_quantiles = np.quantile(test_arr, 0.99, axis=tuple(axes_for_quantiles))

    assert np.allclose(computed_lower, real_lower_quantiles)
    assert np.allclose(computed_upper, real_upper_quantiles)
