import itertools
from typing import Any

import pytest

from careamics.config.data.normalization_config import (
    MeanStdConfig,
    MinMaxConfig,
    NoNormConfig,
    QuantileConfig,
)
from careamics.config.ng_factories.ng_config_discriminator import validate_norm_config

NORMS_WO_NONE_QUANT = ["mean_std", "min_max"]
NORMS_WO_NONE = NORMS_WO_NONE_QUANT + ["quantile"]
NORMS_W_NONE = NORMS_WO_NONE + ["none"]
NORMS_ORDERED = ["quantile", "min_max"]
NORMS_W_NONE_CLASSES = [MeanStdConfig, QuantileConfig, MinMaxConfig, NoNormConfig]


def create_optional_params(
    norm: str, length: int = 2, per_channel: bool = True
) -> dict[str, Any]:
    """Create normalization config with all optional parameters.

    These parameters are expected to pass validation unless `per_channel` is set to
    False and `length` is greater than 1.
    """
    match norm:
        case "mean_std":
            return {
                "name": norm,
                "per_channel": per_channel,
                "input_means": [0 for _ in range(length)],
                "input_stds": [1 for _ in range(length)],
                "target_means": [0 for _ in range(length)],
                "target_stds": [1 for _ in range(length)],
            }
        case "quantile":
            return {
                "name": norm,
                "per_channel": per_channel,
                "lower_quantiles": [0.01 for _ in range(length)],
                "upper_quantiles": [0.99 for _ in range(length)],
                "input_lower_quantile_values": [0 for _ in range(length)],
                "input_upper_quantile_values": [1 for _ in range(length)],
                "target_lower_quantile_values": [0 for _ in range(length)],
                "target_upper_quantile_values": [1 for _ in range(length)],
            }
        case "min_max":
            return {
                "name": norm,
                "per_channel": per_channel,
                "input_mins": [0 for _ in range(length)],
                "input_maxes": [1 for _ in range(length)],
                "target_mins": [0 for _ in range(length)],
                "target_maxes": [1 for _ in range(length)],
            }
        case "none":
            return {"name": norm, "per_channel": per_channel}
        case _:
            raise ValueError(f"Invalid normalization name: {norm}")


def create_pruned_dicts(norm: dict[str, Any]) -> list[dict[str, Any]]:
    """Create dicts with one optional parameter removed.

    Used to test when a parameter is missing, but its counterpart is present.
    """
    if norm["name"] == "none":
        raise ValueError("NoNorm is not compatible with this function.")

    # no need to prune those
    protected_keys = {"name", "per_channel"}

    if norm["name"] == "quantile":
        # these cannot be `None``
        protected_keys = protected_keys.union(
            {
                "lower_quantiles",
                "upper_quantiles",
            }
        )

    # create dicts with one optional parameter removed
    dict_lst = []
    for k in norm.keys():
        if k not in protected_keys:
            dict_lst.append({key: val for key, val in norm.items() if key != k})

    return dict_lst


def create_extra_element_dicts(norm: dict[str, Any]) -> list[dict[str, Any]]:
    """Create dicts with one extra element added to a list parameter.

    This method assumes that all non-protected parameters are lists, and adds an extra
    element to one of them.
    """
    if norm["name"] == "none":
        raise ValueError("NoNorm is not compatible with this function.")

    # no need to prune those
    protected_keys = {"name", "per_channel"}

    # create dicts with one optional parameter removed
    dict_lst = []
    for k in norm.keys():
        if k not in protected_keys:
            new_dict = {key: val for key, val in norm.items() if key != k}
            new_dict[k] = norm[k] + [norm[k][-1]]  # add extra element (same as last)
            dict_lst.append(new_dict)

    return dict_lst


def _swap_keys(d: dict[str, Any], key1: str, key2: str) -> dict[str, Any]:
    """Swap the values of two keys in a dict."""
    new_dict = d.copy()
    new_dict[key1], new_dict[key2] = d[key2], d[key1]
    return new_dict


def create_wrong_values_dicts(norm: dict[str, Any]) -> list[dict[str, Any]]:
    """Create dicts with wrong values for the parameters by swapping quantities."""
    if norm["name"] not in NORMS_ORDERED:
        raise ValueError(f"{norm['name']} is not compatible with this function.")

    match norm["name"]:
        case "quantile":
            key_pairs = [
                ("lower_quantiles", "upper_quantiles"),
                ("input_lower_quantile_values", "input_upper_quantile_values"),
                ("target_lower_quantile_values", "target_upper_quantile_values"),
            ]
        case "min_max":
            key_pairs = [("input_mins", "input_maxes"), ("target_mins", "target_maxes")]
        case _:
            raise ValueError(f"Invalid normalization name: {norm['name']}")

    dict_lst = []
    for lower_key, upper_key in key_pairs:
        # swap the keys to create wrong values (e.g. lower quantiles > upper quantiles)
        dict_lst.append(_swap_keys(norm, lower_key, upper_key))

    return dict_lst


# ------------------------ Test utilities --------------------------


@pytest.mark.parametrize(
    "norm, n_channels, per_channel",
    list(itertools.product(NORMS_W_NONE, [1, 2, 3], [True]))
    + list(itertools.product(NORMS_W_NONE, [1], [False])),
)
def test_optional_parameters_creation(norm, n_channels, per_channel):
    """Test the creation of a normalization using the defaults of the utility
    function."""
    cfg = validate_norm_config(
        create_optional_params(norm, length=n_channels, per_channel=per_channel)
    )
    cfg.validate_size(n_channels, n_channels)


@pytest.mark.parametrize(
    "norm, exp_class", list(zip(NORMS_W_NONE, NORMS_W_NONE_CLASSES, strict=True))
)
def test_validation_class(norm, exp_class):
    """Test norm validation."""
    cfg = validate_norm_config(create_optional_params(norm))
    assert isinstance(cfg, exp_class)


# -------------------------- Unit tests ----------------------------


# TODO this is not testing the parameters individually, a single failure is enough...
@pytest.mark.parametrize("norm", NORMS_WO_NONE)
def test_per_channel_false_length_one(norm):
    """Test that when per_channel is False, the length of the parameters must be 1."""
    with pytest.raises(ValueError, match="Global statistics"):
        validate_norm_config(create_optional_params(norm, length=2, per_channel=False))


@pytest.mark.parametrize(
    "norm",
    list(
        itertools.chain.from_iterable(
            [create_pruned_dicts(create_optional_params(n)) for n in NORMS_WO_NONE]
        )
    ),
)
def test_missing_parameter(norm):
    """Test error for when a parameter is missing, but its counterpart is present."""
    with pytest.raises(ValueError, match="must be both provided or both None"):
        validate_norm_config(norm)


@pytest.mark.parametrize(
    "norm",
    list(
        itertools.chain.from_iterable(
            [
                create_extra_element_dicts(create_optional_params(n))
                for n in NORMS_WO_NONE
            ]
        )
    ),
)
def test_mismatching_lengths(norm):
    """Test error for when a parameter has a different length than its counterpart."""
    with pytest.raises(ValueError, match="must have same length"):
        validate_norm_config(norm)


# TODO quantile has a blind spot here, we don't check that the error is raised by
# both the quantiles and their values, but it's probably good enough for now.
@pytest.mark.parametrize(
    "norm, n_channels_in, n_channels_out, exp_error",
    # input channels mismatching
    list(
        itertools.product(
            NORMS_WO_NONE_QUANT,
            [1, 3],
            [2],
            [pytest.raises(ValueError, match="does not match number")],
        )
    )
    # target channels mismatching
    + list(
        itertools.product(
            NORMS_WO_NONE_QUANT,
            [2],
            [1, 3],
            [pytest.raises(ValueError, match="does not match number")],
        )
    )
    # quantile with same number of input and output channels
    + [
        ("quantile", 1, 1, pytest.raises(ValueError, match="does not match number")),
        ("quantile", 3, 3, pytest.raises(ValueError, match="does not match number")),
    ]
    # quantile throws a different error when there is a input/output mismatch
    + list(
        itertools.product(
            ["quantile"],
            [2],
            [1, 3],
            [pytest.raises(ValueError, match="Quantile normalization per channel")],
        )
    ),
)
def test_validate_size(norm, n_channels_in, n_channels_out, exp_error):
    """Test that validate_size raises an error when the number of channels does not
    match the length of the parameters.
    """
    with exp_error:
        cfg = validate_norm_config(create_optional_params(norm, length=2))
        cfg.validate_size(n_channels_in, n_channels_out)


@pytest.mark.parametrize(
    "norm",
    list(
        itertools.chain.from_iterable(
            [
                create_wrong_values_dicts(create_optional_params(n))
                for n in NORMS_ORDERED
            ]
        )
    ),
)
def test_values_wrong_order(norm):
    """Test error for when quantiles or min/max values are in the wrong order."""
    with pytest.raises(ValueError, match="must be less"):
        validate_norm_config(norm)
