import pytest

from careamics.config.architectures import LVAEConfig
from careamics.models.constraints import LVAEConstraints, get_model_constraints


def _lvae_config(**kwargs) -> LVAEConfig:
    """Build an LVAE config, overriding defaults with kwargs."""
    params = {
        "architecture": "LVAE",
        "input_shape": (64, 64),
        "encoder_conv_strides": [2, 2],
        "decoder_conv_strides": [2, 2],
        "z_dims": [128, 128],  # 2 hierarchy levels -> factor 2**2 = 4
        "output_channels": 1,
    }
    params.update(kwargs)
    return LVAEConfig(**params)


def test_factory_returns_lvae_constraints():
    """`get_model_constraints` dispatches LVAE configs to `LVAEConstraints`."""
    constraints = get_model_constraints(_lvae_config())
    assert isinstance(constraints, LVAEConstraints)


@pytest.mark.parametrize("dim", [4, 8, 64, 128])
def test_validate_spatial_shape_compatible(dim):
    """Dimensions divisible by stride ** n_levels are accepted."""
    # z_dims length 2 -> factor 2**2 = 4
    constraints = LVAEConstraints(_lvae_config())
    constraints.validate_spatial_shape((dim, dim))


@pytest.mark.parametrize("dim", [65, 66, 70, 130])
def test_validate_spatial_shape_incompatible(dim):
    """Dimensions not divisible by stride ** n_levels are rejected."""
    constraints = LVAEConstraints(_lvae_config())
    with pytest.raises(ValueError, match="Input data dimension"):
        constraints.validate_spatial_shape((dim, dim))


def test_validate_spatial_shape_more_levels():
    """Factor scales with the number of hierarchy levels (len(z_dims))."""
    # 4 levels, stride 2 -> factor 2**4 = 16
    constraints = LVAEConstraints(_lvae_config(z_dims=[128, 128, 128, 128]))
    constraints.validate_spatial_shape((64, 64))  # 64 % 16 == 0
    with pytest.raises(ValueError, match="Input data dimension"):
        constraints.validate_spatial_shape((72, 72))  # 72 % 16 != 0


def test_validate_spatial_shape_stride_one_unconstrained():
    """A dimension with stride 1 (e.g. Z in 2.5D) is unconstrained."""
    constraints = LVAEConstraints(
        _lvae_config(
            input_shape=(64, 64, 64),
            encoder_conv_strides=[1, 2, 2],
            decoder_conv_strides=[1, 2, 2],
        )
    )
    # Z has stride 1 -> factor 1 -> any odd depth accepted
    constraints.validate_spatial_shape((5, 64, 64))


@pytest.mark.parametrize("length", [1, 4])
def test_validate_spatial_shape_wrong_length(length):
    """Only spatial shapes of length 2 or 3 are accepted."""
    constraints = LVAEConstraints(_lvae_config())
    with pytest.raises(ValueError, match="Spatial input shape"):
        constraints.validate_spatial_shape((64,) * length)


def test_validate_spatial_shape_dim_mismatch():
    """A spatial shape whose length differs from the encoder strides is rejected."""
    # 2D model (strides length 2) fed a 3D shape
    constraints = LVAEConstraints(_lvae_config())
    with pytest.raises(ValueError, match="does not match the model"):
        constraints.validate_spatial_shape((64, 64, 64))


def test_validate_input_channels_is_noop():
    """LVAE accepts any input channel count (single mixed input)."""
    constraints = LVAEConstraints(_lvae_config())
    constraints.validate_input_channels(1)
    constraints.validate_input_channels(7)


def test_validate_target_channels_match():
    """Target channel count must equal the model output channels."""
    constraints = LVAEConstraints(_lvae_config(output_channels=3))
    constraints.validate_target_channels(3)


def test_validate_target_channels_mismatch():
    """A target channel count differing from output channels is rejected."""
    constraints = LVAEConstraints(_lvae_config(output_channels=3))
    with pytest.raises(ValueError, match="target image"):
        constraints.validate_target_channels(2)
