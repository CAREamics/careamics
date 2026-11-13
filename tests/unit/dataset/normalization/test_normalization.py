from typing import Any, Literal

import numpy as np
import pytest

from careamics.config.factories.config_discriminators import instantiate_norm_config
from careamics.dataset.normalization import create_normalization

NORMS = ["mean_std", "quantile", "min_max"]

# ------------------------ Test utilities --------------------------


def get_arrays(dims: Literal[2, 3]) -> tuple[np.ndarray, np.ndarray]:
    """Generate test arrays."""
    rng = np.random.default_rng(42)

    shape = (1, 32, 32) if dims == 2 else (1, 16, 32, 32)
    min_in, max_in = 0.0, 255.0
    min_tar, max_tar = 0.0, 65535.0

    input_array = rng.uniform(min_in, max_in, size=shape).astype(np.float32)
    target_array = rng.uniform(min_tar, max_tar, size=shape).astype(np.float32)

    return input_array, target_array


def get_norm_parameters(
    norm: str,
    arr_in: np.ndarray,
    arr_tar: np.ndarray,
    target_mode: str = "independent",
) -> dict[str, Any]:
    """Generate normalization parameters."""
    norm_dict = {}
    match norm:
        case "mean_std":
            norm_dict = {
                "name": norm,
                "input_means": [arr_in.mean()],
                "input_stds": [arr_in.std()],
                "target_means": (
                    [arr_tar.mean()]
                    if target_mode != "same_as_input"
                    else [arr_in.mean()]
                ),
                "target_stds": (
                    [arr_tar.std()]
                    if target_mode != "same_as_input"
                    else [arr_in.std()]
                ),
                "target_mode": target_mode,
            }
        case "quantile":
            norm_dict = {
                "name": norm,
                "input_lower_quantile_values": [np.quantile(arr_in, 0.01)],
                "input_upper_quantile_values": [np.quantile(arr_in, 0.99)],
                "target_lower_quantile_values": (
                    [np.quantile(arr_tar, 0.2)]
                    if target_mode != "same_as_input"
                    else [np.quantile(arr_in, 0.01)]
                ),
                "target_upper_quantile_values": (
                    [np.quantile(arr_tar, 0.8)]
                    if target_mode != "same_as_input"
                    else [np.quantile(arr_in, 0.99)]
                ),
                "target_mode": target_mode,
            }
        case "min_max":
            norm_dict = {
                "name": norm,
                "input_mins": [arr_in.min()],
                "input_maxes": [arr_in.max()],
                "target_mins": (
                    [arr_tar.min()]
                    if target_mode != "same_as_input"
                    else [arr_in.min()]
                ),
                "target_maxes": (
                    [arr_tar.max()]
                    if target_mode != "same_as_input"
                    else [arr_in.max()]
                ),
                "target_mode": target_mode,
            }
        case _:
            raise ValueError(f"Unknown normalization strategy: {norm}")

    return instantiate_norm_config(norm_dict)


@pytest.mark.parametrize("norm", NORMS)
def test_normalization_parameters(norm):
    """Test that the test utilities work."""
    arr_in, arr_tar = get_arrays(2)
    _ = get_norm_parameters(norm, arr_in, arr_tar)


# -------------------------- Unit tests ----------------------------


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("norm", NORMS)
def test_normalization_changes_range(norm, dim):
    """Test that the normalization changes the range of the data."""
    arr_in, arr_tar = get_arrays(dim)
    norm_params = get_norm_parameters(norm, arr_in, arr_tar)
    norm_func = create_normalization(norm_params)

    # normalize arrays
    arr_in_norm, arr_tar_norm = norm_func(arr_in, arr_tar)

    # range normalizations will normalize to range [0, 1]
    # mean-std will normalize to have mean 0 and std 1, range ~[-1.7, 1.7]
    assert (arr_in_norm.max() - arr_in_norm.min()) < 4
    assert (arr_tar_norm.max() - arr_tar_norm.min()) < 4


@pytest.mark.parametrize("norm", NORMS)
def test_no_target_normalization(norm):
    """Test that target data is unchanged when target normalization is disabled."""
    arr_in, arr_tar = get_arrays(2)
    norm_params = get_norm_parameters(norm, arr_in, arr_tar, target_mode="none")
    norm_func = create_normalization(norm_params)
    assert norm_func.skip_target

    # normalize arrays
    arr_in_norm, arr_tar_norm = norm_func(arr_in, arr_tar)

    # range normalizations will normalize to range [0, 1]
    # mean-std will normalize to have mean 0 and std 1, range ~[-1.7, 1.7]
    assert (arr_in_norm.max() - arr_in_norm.min()) < 4
    np.testing.assert_almost_equal(arr_tar_norm, arr_tar)


@pytest.mark.parametrize("norm", NORMS)
def test_same_as_input_uses_input_statistics(norm):
    """Test that target statistics are copied from input statistics."""
    arr_in, arr_tar = get_arrays(2)
    norm_params = get_norm_parameters(
        norm, arr_in, arr_tar, target_mode="same_as_input"
    )
    norm_func = create_normalization(norm_params)

    # normalize arrays, using input as target as well
    arr_in_norm, arr_tar_norm = norm_func(arr_in, arr_in)
    np.testing.assert_almost_equal(arr_in_norm, arr_tar_norm)
