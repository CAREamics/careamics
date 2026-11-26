import pytest

from careamics.config.architectures import COSDDConfig
from careamics.models.constraints import COSDDConstraints


def _constraints(input_shape: tuple[int, ...], in_channels: int) -> COSDDConstraints:
    return COSDDConstraints(
        COSDDConfig(
            architecture="COSDD", input_shape=input_shape, in_channels=in_channels
        )
    )


@pytest.mark.parametrize(
    "input_shape, shape",
    [((256, 256), (37, 101)), ((8, 64, 64), (5, 17, 130))],
)
def test_any_spatial_shape_accepted(input_shape, shape):
    """Any spatial shape with the model dimensionality is accepted."""
    _constraints(input_shape, 1).validate_spatial_shape(shape)


@pytest.mark.parametrize(
    "input_shape, shape",
    [((256, 256), (5, 64, 64)), ((8, 64, 64), (64, 64))],
)
def test_spatial_dimensionality_mismatch_raises(input_shape, shape):
    """A spatial shape with another dimensionality is rejected."""
    with pytest.raises(ValueError, match="dimensionality"):
        _constraints(input_shape, 1).validate_spatial_shape(shape)


def test_channels():
    """Input and target channels must match the model channels."""
    constraints = _constraints((64, 64), 2)
    constraints.validate_input_channels(2)
    constraints.validate_target_channels(2)

    with pytest.raises(ValueError, match="input image"):
        constraints.validate_input_channels(1)
    with pytest.raises(ValueError, match="target image"):
        constraints.validate_target_channels(3)
