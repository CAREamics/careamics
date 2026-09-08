import pytest

from careamics.config.factories.config_discriminators import instantiate_norm_config
from careamics.config.factories.seg_factory import (
    _get_expected_target_axes,
    _get_input_size,
    _get_norm_dict_with_target_skipped,
    create_advanced_seg_config,
)
from careamics.config.seg_configuration import SegConfiguration

# --- Test utilities


def create_configuration(**kwargs) -> SegConfiguration:
    """Wrapper around `create_advanced_seg_config`."""
    min_params = {
        "experiment_name": "test_seg",
        "data_type": "array",
        "axes": "YX",
        "patch_size": (16, 16),
        "batch_size": 2,
        "n_classes": 1,
    }

    min_params.update(**kwargs)
    return create_advanced_seg_config(**min_params)


# --- Unit tests


@pytest.mark.parametrize(
    "axes, exp_axes",
    [("YX", "YX"), ("YXC", "YX"), ("STZYXC", "STZYX"), ("SCYX", "SYX")],
)
def test_expected_target_axes(axes, exp_axes):
    """Test expected target axes."""
    target_axes = _get_expected_target_axes(axes)
    assert target_axes == exp_axes


@pytest.mark.parametrize(
    "axes, channels, n_channels_in, exp_n_channels",
    [
        ("YX", None, None, 1),
        ("YX", None, 1, 1),
        ("CYX", [0, 1], 2, 2),
        ("CYX", [0, 1], None, 2),
        ("CYX", None, 2, 2),
    ],
)
def test_get_input_size(axes, channels, n_channels_in, exp_n_channels):
    """Test _get_input_size."""
    result = _get_input_size(axes, channels, n_channels_in)
    assert result == exp_n_channels


@pytest.mark.parametrize("normalization", ["quantile", "mean_std", "min_max", "none"])
@pytest.mark.parametrize(
    "norm_params", [None, {}, {"skip_target": False}, {"skip_target": True}]
)
def test_get_norm_dict_with_target_skipped(normalization, norm_params):
    """Test that the returned dictionary always has target skipped."""
    norm_dict = _get_norm_dict_with_target_skipped(normalization, norm_params)

    # check that one can instantiate a configuration
    cfg = instantiate_norm_config(norm_dict)

    if normalization != "none":
        assert cfg.skip_target


def test_loss_parameters_are_configured():
    """Test that segmentation loss parameters are passed to the algorithm config."""
    cfg = create_configuration(
        n_classes=2,
        loss="dice_ce",
        loss_parameters={
            "class_weights": [1.0, 2.0, 3.0],
            "dice_weight": 0.25,
            "ce_weights": 0.75,
        },
    )

    assert cfg.algorithm_config.loss.name == "dice_ce"
    assert cfg.algorithm_config.loss.class_weights == [1.0, 2.0, 3.0]
    assert cfg.algorithm_config.loss.dice_weight == 0.25
    assert cfg.algorithm_config.loss.ce_weights == 0.75


def test_loss_class_weights_match_model_classes():
    """Test that class weights have one value per model output class."""
    with pytest.raises(ValueError, match="Class weights must have length 3"):
        create_configuration(
            n_classes=2,
            loss_parameters={"class_weights": [1.0, 2.0]},
        )


class TestSegFactory:

    @pytest.mark.parametrize("n_classes", [1, 2])
    def test_n_classes_to_model_inputs(self, n_classes):
        """Test that the model inputs is background + foreground classes."""
        cfg: SegConfiguration = create_configuration(n_classes=n_classes)
        assert cfg.algorithm_config.model.num_classes == n_classes + 1
