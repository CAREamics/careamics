import itertools
from typing import Any

import pytest

from careamics.config.data.normalization_config import (
    MeanStdConfig,
    MinMaxConfig,
    NoNormConfig,
    QuantileConfig,
)
from careamics.config.ng_factories.ng_config_discriminator import (
    instantiate_norm_config,
)

NORMS_WO_NONE_QUANT = ["mean_std", "min_max"]
NORMS_WO_NONE = NORMS_WO_NONE_QUANT + ["quantile"]
NORMS_W_NONE = NORMS_WO_NONE + ["none"]
NORMS_ORDERED = ["quantile", "min_max"]
NORMS_W_NONE_CLASSES = [MeanStdConfig, MinMaxConfig, QuantileConfig, NoNormConfig]


def create_norm_dict(
    norm: str, length: int = 2, per_channel: bool = True
) -> dict[str, Any]:
    """Create a normalization config dictionary with all optional parameters.

    These parameters are expected to pass validation unless `per_channel` is set to
    False and `length` is greater than 1.

    Parameters
    ----------
    norm : str
        The name of the normalization method. Must be one of "mean_std", "quantile",
        "min_max", or "none".
    length : int, optional
        The length of the list parameters (e.g. input_means, input_stds, etc.).
    per_channel : bool, optional
        Whether the normalization parameters are specified per channel.

    Returns
    -------
    dict[str, Any]
        A dictionary corresponding to a normalization conffiguration.
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
    """Create a list of normalization config dictionaries with missing parameters.

    Specifically, this function removes a single parameter (except for protected keys)
    from a normalization config dictionary. It is used to test the validation of
    normalization configs when some parameters are missing, but their counterparts are
    present. For example, for mean/std normalization, we want to test that an error is
    raised if input_means is provided but input_stds is missing, or vice versa.

    Parameters
    ----------
    norm : dict[str, Any]
        A dictionary corresponding to a normalization configuration.

    Returns
    -------
    list[dict[str, Any]]
        A list of dictionaries, each with one optional parameter removed.
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
    """
    Create a list of normalization config dict with parameters of different length.

    Specifically, this function adds an extra element to one of the list parameters.
    It is used to test the validation of normalization configs when parameters have
    different lengths. For example, for mean/std normalization, we want to test that an
    error is raised if input_means has a different length than input_stds.

    Parameters
    ----------
    norm : dict[str, Any]
        A dictionary corresponding to a normalization configuration.

    Returns
    -------
    list[dict[str, Any]]
        A list of dictionaries, each with one optional parameter having an extra
        element.
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
    """
    Create a list of normalization config dicts with wrong values.

    Specifically, this function swaps the values of certain key pairs to create
    invalid configurations. It is used to test the validation of normalization
    configs when parameters have incorrect relationships. For example, for
    quantile normalization, we want to test that an error is raised if lower
    quantiles are greater than upper quantiles.

    Parameters
    ----------
    norm : dict[str, Any]
        A dictionary corresponding to a normalization configuration.

    Returns
    -------
    list[dict[str, Any]]
        A list of dictionaries, each with one pair of keys swapped.
    """
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
    # test all normalizations with different number of channels
    list(itertools.product(NORMS_W_NONE, [1, 2, 3], [True]))
    # test all normalizations with per_channel=False (which requires n_channels=1)
    + list(itertools.product(NORMS_W_NONE, [1], [False])),
)
def test_optional_parameters_creation(norm, n_channels, per_channel):
    """Test creating normalization config with optional parameters.

    `create_optional_params` should return a valid normalization config as long as
    the combination of `n_channels` and `per_channel` is valid.
    """
    # create a Normalization configuration
    cfg = instantiate_norm_config(
        create_norm_dict(norm, length=n_channels, per_channel=per_channel)
    )
    # validate the size, this validation method is not run by the normalization config
    # itself, but by the NGConfiguration validation
    cfg.validate_size(n_channels, n_channels)


@pytest.mark.parametrize(
    "norm, exp_class", list(zip(NORMS_W_NONE, NORMS_W_NONE_CLASSES, strict=True))
)
def test_validation_class(norm, exp_class):
    """Test that `instantiate_norm_config` returns the correct class."""
    # create a Normalization configuration
    cfg = instantiate_norm_config(create_norm_dict(norm))
    assert isinstance(cfg, exp_class)


# -------------------------- Unit tests ----------------------------


# TODO this is not testing the parameters individually, since all parameters will have
# same length, it is sufficient that one of them triggers an error for this test to
# pass. However, if we had a blind spot for one of the parameter (e.g. length not
# validated), this test would not catch it.
@pytest.mark.parametrize("norm", NORMS_WO_NONE)
def test_per_channel_false_length_one(norm):
    """Test that when `per_channel` is False, the length of the parameters must be 1."""
    with pytest.raises(ValueError, match="Global statistics"):
        instantiate_norm_config(
            # since `per_channel` is False, validation should fail if parameters have
            # length greater than 1, so we set length to 2 to trigger the error.
            create_norm_dict(norm, length=2, per_channel=False)
        )


@pytest.mark.parametrize(
    "norm",
    list(
        itertools.chain.from_iterable(  # flatten the list of lists
            # generate a list of normalization config dicts with missing parameters
            # for each normalization, one of the parameter has been removed.
            [create_pruned_dicts(create_norm_dict(n)) for n in NORMS_WO_NONE]
        )
    ),
)
def test_missing_parameter(norm):
    """Test that an error is raised when a parameter is missing, but its counterpart is
    present.

    For instance, for mean/std normalization, an error should be raised if input_means
    is provided but input_stds is missing, or vice versa.
    """
    with pytest.raises(ValueError, match="must be both provided or both None"):
        instantiate_norm_config(norm)


@pytest.mark.parametrize(
    "norm",
    list(
        itertools.chain.from_iterable(  # flatten list of lists
            [
                # generate a list of normalization config dicts with parameters of
                # different lengths.
                # for each normalization, one of the parameters has an extra element.
                create_extra_element_dicts(create_norm_dict(n))
                for n in NORMS_WO_NONE
            ]
        )
    ),
)
def test_mismatching_lengths(norm):
    """Test that an error is raised when a parameter has a different length than its
    counterpart.

    For instance, for mean/std normalization, an error should be raised if input_means
    has a different length than input_stds.
    """
    with pytest.raises(ValueError, match="must have same length"):
        instantiate_norm_config(norm)


# TODO quantile has a blind spot here, we don't check that the error is raised by
# both the quantiles and their values, but it's probably good enough for now.
@pytest.mark.parametrize(
    "norm, n_channels_in, n_channels_out, exp_error",
    # all norms but none and quantile, input channels mismatching with input stats
    list(
        itertools.product(
            NORMS_WO_NONE_QUANT,
            [1, 3],  # mismatching
            [2],  # matching
            [pytest.raises(ValueError, match="does not match number")],
        )
    )
    # all norms but none and quantile, output channels mismatching with target stats
    + list(
        itertools.product(
            NORMS_WO_NONE_QUANT,
            [2],  # matching
            [1, 3],  # mismatching
            [pytest.raises(ValueError, match="does not match number")],
        )
    )
    # quantile with same input/output channels, but mismatching with quantiles length
    + [
        ("quantile", 1, 1, pytest.raises(ValueError, match="does not match number")),
        ("quantile", 3, 3, pytest.raises(ValueError, match="does not match number")),
    ]
    # quantile throws a different error when there is an input/output mismatch
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

    For instance, we ask to validate with 3 input channels, but min_max normalization's
    input_maxes has length 2.

    Here the length of all parameters is fixed to 2, the mismatching only coming from
    n_channels_in or n_channels_out.
    """
    with exp_error:
        # instantiate the configuration and validate it
        cfg = instantiate_norm_config(create_norm_dict(norm, length=2))
        cfg.validate_size(n_channels_in, n_channels_out)


@pytest.mark.parametrize(
    "norm",
    list(
        itertools.chain.from_iterable(  # flatter the list of lists
            [
                # for all normalizations that expect a relationship between parameters
                # (e.g. lower quantiles must be less than upper quantiles), create a
                # list of normalization config dicts with wrong values (e.g. lower
                # quantiles greater than upper quantiles) by swapping pairs.
                create_wrong_values_dicts(create_norm_dict(n))
                for n in NORMS_ORDERED
            ]
        )
    ),
)
def test_values_wrong_order(norm):
    """Test that an error is raised when parameters are in the wrong order."""
    with pytest.raises(ValueError, match="must be less"):
        instantiate_norm_config(norm)
