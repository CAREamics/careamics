"""Shape bookkeeping in the LVAE that `torch.compile` is sensitive to."""

import pytest

from careamics.config.architectures import LVAEConfig
from careamics.models.lvae.layers import _downscaled_latent_shape
from careamics.models.model_factory import model_factory


def _create_lvae(
    input_shape: tuple[int, ...] = (64, 64),
    encoder_conv_strides: tuple[int, ...] = (2, 2),
    decoder_conv_strides: tuple[int, ...] = (2, 2),
    multiscale_count: int = 3,
):
    return model_factory(
        LVAEConfig(
            architecture="LVAE",
            input_shape=input_shape,
            z_dims=(128, 128),
            encoder_conv_strides=encoder_conv_strides,
            decoder_conv_strides=decoder_conv_strides,
            multiscale_count=multiscale_count,
            output_channels=2,
            predict_logvar=True,
        )
    )


@pytest.mark.lvae
@pytest.mark.parametrize(
    "latent_shape, expected",
    [
        # a 2D encoder stores only the X extent; it is broadcast over (Y, X)
        ((64,), (32, 32)),
        ((128,), (64, 64)),
        # the full 2D tile shape, as set by `reset_for_inference`
        ((64, 64), (32, 32)),
        ((64, 128), (32, 64)),
        # 3D: Z is retained, Y and X are halved
        ((16, 64, 64), (16, 32, 32)),
    ],
)
def test_downscaled_latent_shape(latent_shape, expected):
    """The crop target halves Y and X, broadcasting a 1-tuple over both."""
    assert _downscaled_latent_shape(latent_shape) == expected


@pytest.mark.lvae
def test_downscaled_latent_shape_returns_plain_ints():
    """The result holds Python ints, so the caller's comparison stays static.

    Routing this through NumPy makes Dynamo treat `x.shape[-1] > shape[-1]` as
    data-dependent control flow and refuse to capture a single graph.
    """
    shape = _downscaled_latent_shape((64,))
    assert all(type(dim) is int for dim in shape)


@pytest.mark.lvae
def test_output_expected_shape_survives_repeated_reads():
    """`output_expected_shape` is a tuple, not a single-use generator.

    `BottomUpLayer.forward` reads it on every pass, so a generator would be
    exhausted after the first one.
    """
    model = _create_lvae()

    for layer in model.bottom_up_layers:
        assert isinstance(layer.output_expected_shape, tuple)
        assert tuple(layer.output_expected_shape) == tuple(
            layer.output_expected_shape
        )


@pytest.mark.lvae
def test_reset_for_inference_sets_reusable_shapes():
    """`reset_for_inference` also stores tuples rather than generators."""
    model = _create_lvae()

    model.reset_for_inference((64, 64))

    for i, layer in enumerate(model.bottom_up_layers):
        assert layer.output_expected_shape == tuple(
            64 // 2 ** (i + 1) for _ in range(2)
        )
    for layer in model.top_down_layers:
        assert layer.latent_shape == (64, 64)
