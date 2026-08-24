import pytest

from careamics.config.factories.seg_factory import (
    _get_expected_target_axes,
    _get_input_size,
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


class TestSegFactory:

    @pytest.mark.parametrize("n_classes", [1, 2])
    def test_n_classes_to_model_inputs(self, n_classes):
        """Test that the model inputs is background + foreground classes."""
        cfg: SegConfiguration = create_configuration(n_classes=n_classes)
        assert cfg.algorithm_config.model.num_classes == n_classes + 1

    @pytest.mark.parametrize(
        "norm_params", [None, {}, {"skip_target": False}, {"skip_target": True}]
    )
    def test_skip_target_enforced(self, norm_params):
        """Test that `skip_target` is always enforced."""
        cfg: SegConfiguration = create_configuration(normalization_params=norm_params)
        assert cfg.data_config.normalization.skip_target
