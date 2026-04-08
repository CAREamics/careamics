"""Unit tests for the NGConfiguration Pydantic model."""

import itertools
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest

from careamics.config.ng_configs import N2VConfiguration, NGConfiguration
from careamics.config.ng_factories.ng_config_discriminator import instantiate_config
from tests.utils import unet_ng_config_dict_testing

# algorithms and their expected config classes for testing
ALGORITHMS = ["care", "n2n", "n2v"]
ALGORITHMS_CONFIGS = [NGConfiguration, NGConfiguration, N2VConfiguration]

# path to the model constraints, used for mocking
GET_MODEL_CONSTRAINTS_PATH = (
    "careamics.config.ng_configs.ng_configuration.get_model_constraints"
)

# path to the default training factory, used for mocking
DEFAULT_TRAINING_DICT_PATH = (
    "careamics.config.ng_configs.ng_training_configuration.default_training_dict"
)


# ------------------------ Test utilities --------------------------


def test_get_model_constraints_path():
    """Test that the path to get model constraints is correct."""
    with patch(GET_MODEL_CONSTRAINTS_PATH) as mock_get_constraints:
        from careamics.config.ng_configs.ng_configuration import get_model_constraints

        # call the function to ensure the path is correct
        get_model_constraints("dummy_model_config")

        mock_get_constraints.assert_called_once_with("dummy_model_config")


def test_get_default_training_dict_path():
    """Test that the path to the default training dict is correct."""
    with patch(DEFAULT_TRAINING_DICT_PATH) as mock_factory:
        from careamics.config.ng_configs.ng_training_configuration import (
            default_training_dict,
        )

        # call the function to ensure the path is correct
        default_training_dict(algorithm="care")

        mock_factory.assert_called_once_with(algorithm="care")


def test_default_unet_config():
    """Test that the default NGConfiguration can be created."""
    unet_config_dict = unet_ng_config_dict_testing()
    instantiate_config(unet_config_dict)


@pytest.mark.parametrize(
    "algorithm, config_class", list(zip(ALGORITHMS, ALGORITHMS_CONFIGS, strict=True))
)
def test_unet_configs(algorithm, config_class):
    """Test that an NGConfiguration can be created for each UNet-based algorithm."""
    unet_config_dict = unet_ng_config_dict_testing(algorithm=algorithm)
    cfg = instantiate_config(unet_config_dict)
    assert isinstance(cfg, config_class)


# -------------------------- Unit tests ----------------------------


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_training_config_default(algorithm):
    """Test that the default training dict is called with the correct algorithm.

    Actual `default_training_dict` is unit tested separately in its own module
    test file. This test checks that during NGConfiguration instantiation,
    `default_training_dict` is called once with the correct algorithm name.
    """
    from careamics.config.ng_configs.ng_training_configuration import (
        default_training_dict,
    )

    unet_config_dict = unet_ng_config_dict_testing(algorithm=algorithm)

    with patch(DEFAULT_TRAINING_DICT_PATH, wraps=default_training_dict) as mock:
        instantiate_config(unet_config_dict)

        mock.assert_called_once()
        assert mock.call_args.kwargs["algorithm"] == algorithm


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
    "data_is_3D, model_is_3D, exp_error",
    [
        # no error, both 3D or both 2D
        (False, False, nullcontext()),
        (True, True, nullcontext()),
        # errors for mismatches
        (False, True, pytest.raises(ValueError, match="Mismatch between data")),
        (True, False, pytest.raises(ValueError, match="Mismatch between data")),
    ],
)
def test_validate_3D(data_is_3D, model_is_3D, exp_error):
    """Test that validate_3D raises an error when data and model dimensionality
    mismatch."""
    unet_config_dict = unet_ng_config_dict_testing()
    cfg = instantiate_config(unet_config_dict)

    # patch members methods to avoid Pydantic validation
    with (
        patch.object(type(cfg.data_config), "is_3D", return_value=data_is_3D),
        patch.object(
            type(cfg.algorithm_config.model), "is_3D", return_value=model_is_3D
        ),
        exp_error,
    ):
        # directly call the validator
        NGConfiguration.validate_3D(cfg)


@pytest.mark.parametrize(
    "mode, patching, n_calls",
    [
        ("training", "stratified", 1),
        ("predicting", "whole", 0),  # does not have `patch_size`
    ],
)
def test_validate_patch_against_model(mode, patching, n_calls):
    """Test that `validate_patch_against_model` calls model constraints
    `validate_spatial_shape`.

    Actual `validate_spatial_shape` is unit tested separately in its own module
    test file. This test simply check that if there is a `patch_size`, the validation
    from model constraints is skipped, otherwise it is called with the correct
    parameters.
    """
    unet_config_dict = unet_ng_config_dict_testing(mode=mode, patching=patching)

    # mock model constraints
    with patch(GET_MODEL_CONSTRAINTS_PATH) as mock_get_constraints:
        mock_constraints = MagicMock()
        mock_get_constraints.return_value = mock_constraints

        cfg = instantiate_config(unet_config_dict)

        # reset mock to clear calls from the validation at instantiation of the config
        mock_get_constraints.reset_mock()
        mock_constraints.reset_mock()

        # directly call the validator
        NGConfiguration.validate_patch_against_model(cfg)

        if n_calls > 0:
            # check it instantiated the model constraints with the correct model config
            assert mock_get_constraints.call_args.args[0] == cfg.algorithm_config.model

            # check that it called with validation with the patch size
            mock_constraints.validate_spatial_shape.assert_called_once_with(
                cfg.data_config.patching.patch_size
            )
        else:
            # no call to model constraints
            mock_get_constraints.assert_not_called()


@pytest.mark.parametrize(
    "channels, n_calls",
    [
        ([0, 1, 2], 1),
        (None, 0),  # no channels, skip validation
    ],
)
def test_validate_channels_against_inputs(channels, n_calls):
    """Test that `validate_channels_against_inputs` calls model constraints
    `validate_input_channels`.

    Actual `validate_input_channels` is unit tested separately in its own module
    test file. This test simply checks that if there are channels, the validation
    from model constraints is called with the correct parameters, otherwise it is
    skipped.
    """
    # ensure validation of the configuration
    unet_config_dict = unet_ng_config_dict_testing(
        axes="CYX" if channels is not None else "YX",
        data_kwargs={"channels": channels} if channels is not None else {},
    )

    # mock model constraints
    with patch(GET_MODEL_CONSTRAINTS_PATH) as mock_get_constraints:
        mock_constraints = MagicMock()
        mock_get_constraints.return_value = mock_constraints

        cfg = instantiate_config(unet_config_dict)

        # reset mock to clear calls from the validation at instantiation of the config
        mock_get_constraints.reset_mock()
        mock_constraints.reset_mock()

        # directly call the validator
        NGConfiguration.validate_channels_against_inputs(cfg)

        if n_calls > 0:
            # check it instantiated the model constraints with the correct model config
            assert mock_get_constraints.call_args.args[0] == cfg.algorithm_config.model

            # check that it called validation with the number of channels
            mock_constraints.validate_input_channels.assert_called_once_with(
                len(channels)
            )
        else:
            # no call to model constraints
            mock_get_constraints.assert_not_called()


def test_validate_norm_against_channels():
    """Test that `validate_norm_against_channels` calls
    `normalization.validate_size` with the model's input and output channels.

    Actual `validate_size` is unit tested separately in normalization config
    test files. This test simply checks that the validator delegates to
    `normalization.validate_size` with the correct parameters from the model.
    """
    # create a valid configuration with different input and output channels
    n_in = 2
    n_out = 3
    unet_config_dict = unet_ng_config_dict_testing(
        algorithm="care", axes="CYX", n_channels_in=n_in, n_channels_out=n_out
    )

    # mock model constraints during instantiation
    with patch(GET_MODEL_CONSTRAINTS_PATH):
        cfg = instantiate_config(unet_config_dict)

    # mock validate size of the normalization config
    with patch.object(
        type(cfg.data_config.normalization), "validate_size"
    ) as mock_validate_size:
        NGConfiguration.validate_norm_against_channels(cfg)

        mock_validate_size.assert_called_once_with(n_in, n_out)
