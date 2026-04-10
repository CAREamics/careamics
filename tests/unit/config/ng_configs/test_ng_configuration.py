"""Unit tests for the Configuration Pydantic model."""

import itertools
from contextlib import nullcontext

import pytest

from careamics.config.configuration import Configuration
from careamics.config.n2v_configuration import N2VConfiguration
from careamics.config.ng_factories.ng_config_discriminator import instantiate_config
from tests.unit.config.data.test_normalization_config import (
    NORMS_W_NONE,
    NORMS_WO_NONE,
    create_norm_dict,
)
from tests.unit.lightning.dataset_ng.lightning_modules.constraints import (
    _compatible_shapes,
    _incompatible_shapes,
)
from tests.utils import unet_ng_config_dict_testing

# algorithms and their expected config classes for testing
ALGORITHMS = ["care", "n2n", "n2v"]
ALGORITHMS_CONFIGS = [Configuration, Configuration, N2VConfiguration]


# ------------------------ Test utilities --------------------------


def test_default_unet_config():
    """Test that the default Configuration can be created."""
    unet_config_dict = unet_ng_config_dict_testing()
    instantiate_config(unet_config_dict)


@pytest.mark.parametrize(
    "algorithm, config_class", list(zip(ALGORITHMS, ALGORITHMS_CONFIGS, strict=True))
)
def test_unet_configs(algorithm, config_class):
    """Test that an Configuration can be created for each UNet-based algorithm."""
    unet_config_dict = unet_ng_config_dict_testing(algorithm=algorithm)
    cfg = instantiate_config(unet_config_dict)
    assert isinstance(cfg, config_class)


# -------------------------- Unit tests ----------------------------


@pytest.mark.parametrize(
    "name, exp_error",
    list(itertools.product(["Sn4K3", "C4_M e-L"], [nullcontext(0)]))
    + list(
        itertools.product(
            [
                "",
                "   ",
                "name#",
                "name/",
                "name^",
                "name%",
                "name,",
                "name.",
                "namea=b",
            ],
            [pytest.raises(ValueError, match="Experiment name")],
        )
    ),
)
def test_experiment_name(name, exp_error):
    """Test the validation of the experiment name."""
    with exp_error:
        unet_config_dict = unet_ng_config_dict_testing(experiment_name=name)
        instantiate_config(unet_config_dict)


@pytest.mark.parametrize(
    "axes, conv_dims, exp_error",
    [
        # no error, both 3D or both 2D
        ("YX", 2, nullcontext()),
        ("ZYX", 3, nullcontext()),
        # errors for mismatches
        ("YX", 3, pytest.raises(ValueError, match="Mismatch between data")),
        ("ZYX", 2, pytest.raises(ValueError, match="Mismatch between data")),
    ],
)
def test_validate_3D(axes, conv_dims, exp_error):
    """Test that validate_3D raises an error when data and model dimensionality
    mismatch."""
    unet_config_dict = unet_ng_config_dict_testing(
        axes=axes,
        model_kwargs={"conv_dims": conv_dims},
    )

    with exp_error:
        instantiate_config(unet_config_dict)


@pytest.mark.parametrize(
    "mode, patching, patch_size, exp_error",
    # valid patch sizes for training
    list(
        itertools.product(
            ["training"],
            ["stratified"],
            _compatible_shapes(depth=2),
            [nullcontext(0)],
        )
    )
    # invalid patch sizes for training
    + list(
        itertools.product(
            ["training"],
            ["stratified"],
            _incompatible_shapes(depth=2),
            [pytest.raises(ValueError, match="Input data dimension")],
        )
    )
    # valid patching for predicting (whole sample, no patch size)
    + list(
        itertools.product(
            ["predicting"],
            ["whole"],
            [None],
            [nullcontext(0)],
        )
    )
    + list(
        itertools.product(
            ["predicting"],
            ["tiled"],
            _compatible_shapes(depth=2),
            [nullcontext(0)],
        )
    )
    # invalid prediction patching
    + list(
        itertools.product(
            ["predicting"],
            ["tiled"],
            _incompatible_shapes(depth=2),
            [pytest.raises(ValueError, match="Input data dimension")],
        )
    ),
)
def test_validate_patch_against_model(mode, patching, patch_size, exp_error):
    """Test that `validate_patch_against_model` raises an error when the patch size is
    not compatible with the model constraints, and does not raise an error when it is
    compatible or when patching is whole sample (no patch size).
    """
    unet_config_dict = unet_ng_config_dict_testing(
        algorithm="care",  # avoid pixel manipulation validation error in N2V
        mode=mode,
        patching=patching,
        patch_size=(patch_size, patch_size),
    )

    with exp_error:
        instantiate_config(unet_config_dict)


@pytest.mark.parametrize(
    "channels, n_input, exp_error",
    # no channels, skip validation
    list(itertools.product([None], [1, 3], [nullcontext(0)]))
    # channels match model input channels
    + [
        ([1], 1, nullcontext(0)),
        ([0, 3], 2, nullcontext(0)),
    ]
    # channels do not match model input channels
    + list(
        itertools.product(
            [[0, 1]],
            [1, 3],
            [pytest.raises(ValueError, match="Number of channels in input image")],
        )
    ),
)
def test_validate_channels_against_inputs(channels, n_input, exp_error):
    """Test that `validate_channels_against_inputs` raises an error when the number of
    channels in the data does not match the model input channels, and does not raise an
    error when they match or when there are no channels specified in the data.
    """
    unet_config_dict = unet_ng_config_dict_testing(
        algorithm="care",  # avoid validation error from mismathcing channels in N2V
        axes="CYX",
        data_kwargs={"channels": channels} if channels is not None else {},
        n_channels_in=n_input,
    )

    with exp_error:
        instantiate_config(unet_config_dict)


@pytest.mark.parametrize(
    "norm, length, n_input, exp_error",
    # matching channels and norm stats length
    list(itertools.product(NORMS_W_NONE, [2], [2], [nullcontext(0)]))
    # mismatching channels and norm stats length
    + list(
        itertools.product(
            NORMS_WO_NONE,
            [2],
            [3],
            [pytest.raises(ValueError, match="does not match number")],
        )
    ),
)
def test_validate_norm_against_channels(norm, length, n_input, exp_error):
    """Test that `validate_norm_against_channels` raises an error when the number of
    channels in the data does not match the model input channels, and does not raise an
    error when they match or when there are no channels specified in the data.

    Note that we are not testing mismatching channels here, those are tested in the
    specific unit tests relative to the normalization validation themselves. Same for
    `per_channel=False`.
    """
    unet_config_dict = unet_ng_config_dict_testing(
        axes="CYX",
        n_channels_in=n_input,
        n_channels_out=n_input,
        data_kwargs=(
            {"normalization": create_norm_dict(norm=norm, length=length)}
            if norm is not None
            else {}
        ),
    )

    with exp_error:
        instantiate_config(unet_config_dict)
