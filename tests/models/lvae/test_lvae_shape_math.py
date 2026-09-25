"""Shape bookkeeping in the LVAE that `torch.compile` is sensitive to."""

import pytest

from careamics.config.architectures import LVAEConfig
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
def test_output_expected_shape_survives_repeated_reads():
    """`output_expected_shape` is a tuple, not a single-use generator.

    `BottomUpLayer.forward` reads it on every pass, so a generator would be
    exhausted after the first one.
    """
    model = _create_lvae()

    for layer in model.bottom_up_layers:
        assert isinstance(layer.output_expected_shape, tuple)
        assert tuple(layer.output_expected_shape) == tuple(layer.output_expected_shape)


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
