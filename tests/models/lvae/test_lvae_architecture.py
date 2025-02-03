from typing import Literal, Union

import numpy as np
import pytest
import torch
from torch import nn

from careamics.config import VAEBasedAlgorithm
from careamics.config.architectures import LVAEModel
from careamics.config.likelihood_model import GaussianLikelihoodConfig
from careamics.config.loss_model import LVAELossConfig
from careamics.models.model_factory import model_factory


# TODO move to conftest as a fixture
def create_LVAE_model(
    input_shape: tuple = (1, 64, 64),
    z_dims: list[int] = (128, 128, 128, 128),
    encoder_conv_strides=(2, 2),
    decoder_conv_strides=(2, 2),
    multiscale_count: int = 0,
    output_channels: int = 1,
    analytical_kl: bool = False,
    predict_logvar: Union[Literal["pixelwise"], None] = None,
) -> nn.Module:
    lvae_model_config = LVAEModel(
        architecture="LVAE",
        input_shape=input_shape,
        z_dims=z_dims,
        encoder_conv_strides=encoder_conv_strides,
        decoder_conv_strides=decoder_conv_strides,
        multiscale_count=multiscale_count,
        output_channels=output_channels,
        predict_logvar=predict_logvar,
        analytical_kl=analytical_kl,
    )

    config = VAEBasedAlgorithm(
        algorithm_type="vae",
        algorithm="musplit",
        loss=LVAELossConfig(loss_type="musplit"),
        model=lvae_model_config,
        gaussian_likelihood=GaussianLikelihoodConfig(
            predict_logvar=predict_logvar, logvar_lowerbound=0.0
        ),
    )
    return model_factory(config.model)


@pytest.mark.skip(reason="Needs to be updated")
@pytest.mark.parametrize(
    "img_size, encoder_conv_strides, decoder_conv_strides",
    [
        ([1, 64, 64], [2, 2], [2, 2]),
        ([1, 128, 128], [2, 2], [2, 2]),
        ([1, 16, 64, 64], [1, 2, 2], [1, 2, 2]),
        ([1, 16, 128, 128], [1, 2, 2], [1, 2, 2]),
        ([1, 16, 64, 64], [1, 2, 2], [2, 2]),
    ],
)
def test_first_bottom_up(
    img_size: int,
    encoder_conv_strides,
    decoder_conv_strides,
    tmp_path,
    create_dummy_noise_model,
) -> None:
    model = create_LVAE_model(
        tmp_path=tmp_path,
        encoder_conv_strides=encoder_conv_strides,
        decoder_conv_strides=decoder_conv_strides,
        create_dummy_noise_model=create_dummy_noise_model,
        input_shape=img_size,
    )
    first_bottom_up = model.first_bottom_up
    inputs = torch.ones((1, *img_size))
    output = first_bottom_up(inputs)
    assert output.shape == (1, model.encoder_n_filters, *img_size[1:])


@pytest.mark.skip(reason="Needs to be updated")
@pytest.mark.parametrize(
    "img_size, z_dims, multiscale_count, encoder_conv_stride, decoder_conv_stride",
    [
        ([1, 64, 64], [128, 128, 128], 0, (2, 2), (2, 2)),
        ([1, 64, 64], [128, 128, 128], 2, (2, 2), (2, 2)),
        ([1, 128, 128], [128, 128, 128], 0, (2, 2), (2, 2)),
        ([1, 128, 128], [128, 128, 128], 2, (2, 2), (2, 2)),
        ([1, 64, 64], [128, 128, 128, 128], 0, (2, 2), (2, 2)),
        ([1, 64, 64], [128, 128, 128, 128], 3, (2, 2), (2, 2)),
        ([1, 128, 128], [128, 128, 128, 128], 0, (2, 2), (2, 2)),
        ([1, 128, 128], [128, 128, 128, 128], 3, (2, 2), (2, 2)),
        ([1, 16, 64, 64], [128, 128, 128], 0, (1, 2, 2), (1, 2, 2)),
        ([1, 16, 64, 64], [128, 128, 128], 2, (1, 2, 2), (1, 2, 2)),
        ([1, 16, 128, 128], [128, 128, 128], 0, (1, 2, 2), (1, 2, 2)),
        ([1, 16, 128, 128], [128, 128, 128], 2, (1, 2, 2), (1, 2, 2)),
        ([1, 16, 64, 64], [128, 128, 128, 128], 0, (1, 2, 2), (1, 2, 2)),
        ([1, 16, 64, 64], [128, 128, 128, 128], 3, (1, 2, 2), (1, 2, 2)),
        ([1, 16, 128, 128], [128, 128, 128, 128], 0, (1, 2, 2), (1, 2, 2)),
        ([1, 16, 128, 128], [128, 128, 128, 128], 3, (1, 2, 2), (1, 2, 2)),
    ],
)
def test_bottom_up_layers(
    img_size: list[int],
    z_dims: list[int],
    multiscale_count,
    encoder_conv_stride,
    decoder_conv_stride,
    tmp_path,
    create_dummy_noise_model,
) -> None:
    model = create_LVAE_model(
        tmp_path=tmp_path,
        input_shape=img_size,
        create_dummy_noise_model=create_dummy_noise_model,
        z_dims=z_dims,
        multiscale_count=multiscale_count,
        encoder_conv_strides=encoder_conv_stride,
        decoder_conv_strides=decoder_conv_stride,
    )
    bottom_up_layers = model.bottom_up_layers
    assert len(bottom_up_layers) == len(z_dims)
    n_filters = model.encoder_n_filters
    downscale_factor = [2 for _ in model.image_size[-2:]]
    if len(model.image_size) == 4:
        downscale_factor.insert(0, 1)
    downscale_factor = (1, 1, *downscale_factor)
    expected_img_size = np.array((1, n_filters, *model.image_size[1:])) // np.array(
        downscale_factor
    )
    input_size = (1, n_filters, *model.image_size[1:])
    inputs = torch.ones(input_size)
    for layer in bottom_up_layers:
        input1, input2 = layer(inputs)
        assert input1.shape == tuple(expected_img_size)
        assert input2.shape == tuple(expected_img_size)
        # this is cuz /= gives ufunc error
        # expected_img_size = expected_img_size / np.array(downscale_factor)
        # TODO why do we need to do this?


@pytest.mark.skip(reason="Needs to be updated")
@pytest.mark.parametrize(
    "z_dims, multiscale_count",
    [
        ([128, 128], 4),
        ([128, 128, 128], 5),
    ],
)
def test_LC_init(
    z_dims: list[int],
    multiscale_count: int,
    tmp_path,
    create_dummy_noise_model,
) -> None:

    with pytest.raises(AssertionError):
        create_LVAE_model(
            tmp_path=tmp_path,
            create_dummy_noise_model=create_dummy_noise_model,
            z_dims=z_dims,
            multiscale_count=multiscale_count,
        )


@pytest.mark.skip(reason="Needs to be updated")
@pytest.mark.parametrize(
    "img_size, z_dims, multiscale_count, encoder_conv_stride, decoder_conv_stride",
    [
        ([1, 64, 64], [128, 128, 128], 0, (2, 2), (2, 2)),
        ([1, 64, 64], [128, 128, 128], 2, (2, 2), (2, 2)),
        ([1, 128, 128], [128, 128, 128], 0, (2, 2), (2, 2)),
        ([1, 128, 128], [128, 128, 128], 2, (2, 2), (2, 2)),
        ([1, 64, 64], [128, 128, 128, 128], 0, (2, 2), (2, 2)),
        ([1, 64, 64], [128, 128, 128, 128], 3, (2, 2), (2, 2)),
        ([1, 128, 128], [128, 128, 128, 128], 0, (2, 2), (2, 2)),
        ([1, 128, 128], [128, 128, 128, 128], 3, (2, 2), (2, 2)),
        ([1, 16, 64, 64], [128, 128, 128], 0, (1, 2, 2), (1, 2, 2)),
        ([1, 16, 64, 64], [128, 128, 128], 2, (1, 2, 2), (1, 2, 2)),
        ([1, 16, 128, 128], [128, 128, 128], 0, (1, 2, 2), (1, 2, 2)),
        ([1, 16, 128, 128], [128, 128, 128], 2, (1, 2, 2), (1, 2, 2)),
        ([1, 16, 64, 64], [128, 128, 128, 128], 0, (1, 2, 2), (1, 2, 2)),
        ([1, 16, 64, 64], [128, 128, 128, 128], 3, (1, 2, 2), (1, 2, 2)),
        ([1, 16, 128, 128], [128, 128, 128, 128], 0, (1, 2, 2), (1, 2, 2)),
        ([1, 16, 128, 128], [128, 128, 128, 128], 3, (1, 2, 2), (1, 2, 2)),
    ],
)
def test_bottom_up_pass(
    img_size: list[int],
    z_dims: list[int],
    multiscale_count: int,
    encoder_conv_stride,
    decoder_conv_stride,
    tmp_path,
    create_dummy_noise_model,
) -> None:

    model = create_LVAE_model(
        input_shape=img_size,
        tmp_path=tmp_path,
        create_dummy_noise_model=create_dummy_noise_model,
        z_dims=z_dims,
        encoder_conv_strides=encoder_conv_stride,
        decoder_conv_strides=decoder_conv_stride,
        multiscale_count=multiscale_count,
    )
    first_bottom_up_layer = model.first_bottom_up
    lowres_first_bottom_up_layers = model.lowres_first_bottom_ups
    bottom_up_layers = model.bottom_up_layers

    assert len(bottom_up_layers) == len(
        z_dims
    ), "Different number of bottom_up_layers and z_dims"
    # Check we have the right number of lowres_net's
    for i in range(multiscale_count - 1):
        assert bottom_up_layers[i].lowres_net is not None, "Missing lowres_net"

    img_size = model.image_size
    n_filters = model.encoder_n_filters
    inputs = torch.ones((1, *img_size))
    outputs = model._bottomup_pass(
        inp=inputs,
        first_bottom_up=first_bottom_up_layer,
        lowres_first_bottom_ups=lowres_first_bottom_up_layers,
        bottom_up_layers=bottom_up_layers,
    )
    exp_img_size = img_size
    for i in range(len(bottom_up_layers)):
        if i + 1 > multiscale_count - 1:
            exp_img_size[1:] = [s // 2 for s in exp_img_size[1:]]
        assert outputs[i].shape == (1, n_filters, *exp_img_size[1:])


@pytest.mark.skip(reason="Needs to be updated")
@pytest.mark.parametrize("img_size", [64, 128])
@pytest.mark.parametrize("multiscale_count", [1, 3, 5])
def test_topmost_top_down_layer(
    img_size: int, multiscale_count: int, tmp_path, create_dummy_noise_model
) -> None:
    model = create_LVAE_model(
        input_shape=img_size,
        tmp_path=tmp_path,
        create_dummy_noise_model=create_dummy_noise_model,
        multiscale_count=multiscale_count,
    )
    topmost_top_down = model.top_down_layers[-1]
    n_filters = model.encoder_n_filters

    downscaling = 2 ** (model.n_layers + 1 - multiscale_count)
    downscaled_size = img_size // downscaling
    bu_value = torch.ones((1, n_filters, downscaled_size, downscaled_size))
    output, data = topmost_top_down(bu_value=bu_value, inference_mode=True)

    retain_sp_dims = (
        topmost_top_down.retain_spatial_dims and downscaled_size == img_size
    )
    exp_out_size = downscaled_size if retain_sp_dims else 2 * downscaled_size
    expected_out_shape = (1, n_filters, exp_out_size, exp_out_size)
    expected_z_shape = (1, model.z_dims[0], downscaled_size, downscaled_size)
    assert output.shape == expected_out_shape
    assert data["z"].shape == expected_z_shape


@pytest.mark.skip(reason="Needs to be updated")
@pytest.mark.parametrize(
    "img_size, z_dims, multiscale_count, encoder_conv_stride, decoder_conv_stride",
    [
        ((1, 64, 64), (64, 64), 1, (2, 2), (2, 2)),
        ((1, 128, 128), (64, 64), 1, (2, 2), (2, 2)),
        ((1, 64, 64), (64, 64, 64, 64), 3, (2, 2), (2, 2)),
        ((1, 128, 128), (64, 64, 64, 64), 3, (2, 2), (2, 2)),
        ((1, 64, 64), (64, 64, 64, 64, 64), 5, (2, 2), (2, 2)),
        ((1, 1, 64, 64), (64, 64), 1, (1, 2, 2), (1, 2, 2)),
        ((1, 1, 64, 64), (64, 64), 1, (1, 2, 2), (2, 2)),
    ],
)
def test_all_top_down_layers(
    img_size: int,
    z_dims: tuple[int],
    multiscale_count: int,
    encoder_conv_stride,
    decoder_conv_stride,
    tmp_path,
    create_dummy_noise_model,
) -> None:
    model = create_LVAE_model(
        tmp_path=tmp_path,
        create_dummy_noise_model=create_dummy_noise_model,
        input_shape=img_size,
        z_dims=z_dims,
        multiscale_count=multiscale_count,
        encoder_conv_strides=encoder_conv_stride,
        decoder_conv_strides=decoder_conv_stride,
    )
    top_down_layers = model.top_down_layers
    n_filters = model.encoder_n_filters
    downscaling = 2 ** (model.n_layers + 1 - multiscale_count)
    downscaled_size = [img_sz // downscaling for img_sz in img_size[-2:]]
    if len(img_size) == 4:
        downscaled_size = downscaled_size.insert(1, img_size[1])
    inputs = skip_input = None
    bu_value = torch.ones((1, img_size[1], *downscaled_size))
    for i in reversed(range(model.n_layers)):
        td_layer = top_down_layers[i]
        output, data = td_layer(
            input_=inputs,
            bu_value=bu_value,
            inference_mode=True,
            skip_connection_input=skip_input,
        )
        inputs = bu_value = skip_input = output

        retain_sp_dims = (
            td_layer.retain_spatial_dims and downscaled_size == img_size[-2:]
        )
        exp_out_size = (
            downscaled_size if retain_sp_dims else [2 * s for s in downscaled_size]
        )
        expected_out_shape = (1, n_filters, *exp_out_size)
        expected_z_shape = (1, model.z_dims[0], *downscaled_size)
        assert (
            output.shape == expected_out_shape
        ), f"Found problem in layer {i+1}, retain={td_layer.retain_spatial_dims},"
        f"dwsc={downscaled_size}"
        assert data["z"].shape == expected_z_shape
        downscaled_size = exp_out_size


@pytest.mark.skip(reason="Needs to be updated")
@pytest.mark.parametrize("img_size", [64, 128])
@pytest.mark.parametrize("multiscale_count", [1, 3, 5])
def test_final_top_down(
    img_size: int, multiscale_count: int, tmp_path, create_dummy_noise_model
) -> None:
    model = create_LVAE_model(
        tmp_path=tmp_path,
        create_dummy_noise_model=create_dummy_noise_model,
        input_shape=img_size,
        multiscale_count=multiscale_count,
    )
    final_top_down = model.final_top_down
    n_filters = model.encoder_n_filters
    final_upsampling = not model.no_initial_downscaling
    inp_size = img_size // 2 if final_upsampling else img_size
    inputs = torch.ones((1, n_filters, inp_size, inp_size))
    output = final_top_down(inputs)
    expected_out_shape = (1, n_filters, img_size, img_size)
    assert output.shape == expected_out_shape


@pytest.mark.skip(reason="Needs to be updated")
@pytest.mark.parametrize("img_size", [64, 128])
@pytest.mark.parametrize("multiscale_count", [1, 3, 5])
def test_top_down_pass(
    img_size: int, multiscale_count: int, tmp_path, create_dummy_noise_model
) -> None:
    model = create_LVAE_model(
        tmp_path=tmp_path,
        create_dummy_noise_model=create_dummy_noise_model,
        input_shape=img_size,
        multiscale_count=multiscale_count,
    )
    top_down_layers = model.top_down_layers
    final_top_down = model.final_top_down
    n_filters = model.encoder_n_filters
    n_layers = model.n_layers

    # Compute the bu_values for all the layers
    bu_values = []
    td_sizes = []
    curr_size = img_size
    for i in range(n_layers):
        if i + 1 > multiscale_count - 1:
            curr_size //= 2
        td_sizes.append(curr_size)
        bu_values.append(torch.ones((1, n_filters, curr_size, curr_size)))

    output, data = model.topdown_pass(
        top_down_layers=top_down_layers,
        final_top_down_layer=final_top_down,
        bu_values=bu_values,
    )

    expected_out_shape = (1, n_filters, img_size, img_size)
    assert output.shape == expected_out_shape
    for i in range(n_layers):
        expected_z_shape = (1, model.z_dims[i], td_sizes[i], td_sizes[i])
        assert data["z"][i].shape == expected_z_shape


@pytest.mark.skip(reason="Needs to be updated")
@pytest.mark.parametrize("img_size", [64, 128])
@pytest.mark.parametrize("multiscale_count", [1, 3, 5])
@pytest.mark.parametrize("analytical_kl", [False, True])
@pytest.mark.parametrize("batch_size", [1, 8])
def test_KL_shape(
    img_size: int,
    multiscale_count: int,
    analytical_kl: bool,
    batch_size: int,
    tmp_path,
    create_dummy_noise_model,
) -> None:
    model = create_LVAE_model(
        tmp_path=tmp_path,
        create_dummy_noise_model=create_dummy_noise_model,
        input_shape=img_size,
        multiscale_count=multiscale_count,
        analytical_kl=analytical_kl,
    )
    top_down_layers = model.top_down_layers
    final_top_down = model.final_top_down
    n_filters = model.encoder_n_filters
    n_layers = model.n_layers

    # Compute the bu_values for all the layers
    bu_values = []
    td_sizes = []
    curr_size = img_size
    for i in range(n_layers):
        if i + 1 > multiscale_count - 1:
            curr_size //= 2
        td_sizes.append(curr_size)
        bu_values.append(torch.ones((batch_size, n_filters, curr_size, curr_size)))

    _, data = model.topdown_pass(
        top_down_layers=top_down_layers,
        final_top_down_layer=final_top_down,
        bu_values=bu_values,
    )

    exp_keys = [
        "kl",  # samplewise
        "kl_restricted",
        "kl_channelwise",
        "kl_spatial",
    ]
    assert all(k in data.keys() for k in exp_keys)
    for i in range(n_layers):
        expected_z_shape = (batch_size, model.z_dims[i], td_sizes[i], td_sizes[i])
        assert data["z"][i].shape == expected_z_shape
        assert data["kl"][i].shape == (batch_size,)
        if model._restricted_kl:
            assert data["kl_restricted"][i].shape == (batch_size,)
        assert data["kl_channelwise"][i].shape == (batch_size, model.z_dims[i])
        assert data["kl_spatial"][i].shape == (batch_size, td_sizes[i], td_sizes[i])


@pytest.mark.skip(reason="Needs to be updated")
@pytest.mark.parametrize("img_size", [64, 128])
@pytest.mark.parametrize("multiscale_count", [1, 3, 5])
@pytest.mark.parametrize("predict_logvar", [None, "pixelwise"])
@pytest.mark.parametrize("output_channels", [1, 2])
def test_output_layer(
    img_size: int,
    multiscale_count: int,
    predict_logvar: Union[Literal["pixelwise"], None],
    output_channels: int,
    tmp_path,
    create_dummy_noise_model,
) -> None:
    model = create_LVAE_model(
        tmp_path=tmp_path,
        create_dummy_noise_model=create_dummy_noise_model,
        input_shape=img_size,
        multiscale_count=multiscale_count,
        predict_logvar=predict_logvar,
        output_channels=output_channels,
    )
    out_layer = model.output_layer
    n_filters = model.encoder_n_filters
    input_ = torch.ones((1, n_filters, img_size, img_size))
    output = out_layer(input_)

    num_out_ch = output_channels * (2 if predict_logvar == "pixelwise" else 1)
    exp_out_shape = (1, num_out_ch, img_size, img_size)
    assert output.shape == exp_out_shape


@pytest.mark.parametrize(
    "img_size, z_dims, multiscale_count, encoder_conv_stride, decoder_conv_stride",
    [
        ([64, 64], [32, 32, 32], 1, (2, 2), (2, 2)),
        ([64, 64], [32, 32, 32], 2, (2, 2), (2, 2)),
        ([64, 64], [32, 32, 32], 3, (2, 2), (2, 2)),
        ([32, 32], [32, 32, 32], 1, (2, 2), (2, 2)),
        ([32, 32], [32, 32, 32], 2, (2, 2), (2, 2)),
        ([32, 32], [32, 32, 32], 3, (2, 2), (2, 2)),
        ([64, 64], [32, 32, 32, 32], 1, (2, 2), (2, 2)),
        ([64, 64], [32, 32, 32, 32], 3, (2, 2), (2, 2)),
        ([64, 64], [32, 32, 32, 32], 4, (2, 2), (2, 2)),
        ([32, 32], [32, 32, 32, 32], 1, (2, 2), (2, 2)),
        ([32, 32], [32, 32, 32, 32], 3, (2, 2), (2, 2)),
        ([16, 64, 64], [32, 32, 32], 1, (1, 2, 2), (1, 2, 2)),
        ([16, 64, 64], [32, 32, 32], 2, (1, 2, 2), (1, 2, 2)),
        ([16, 64, 64], [32, 32, 32], 3, (1, 2, 2), (1, 2, 2)),
        ([16, 32, 32], [32, 32, 32], 1, (1, 2, 2), (1, 2, 2)),
        ([16, 32, 32], [32, 32, 32], 2, (1, 2, 2), (1, 2, 2)),
        ([16, 32, 32], [32, 32, 32, 32], 1, (1, 2, 2), (1, 2, 2)),
        ([16, 64, 64], [32, 32, 32, 32], 3, (1, 2, 2), (1, 2, 2)),
        ([16, 64, 64], [32, 32, 32, 32], 1, (1, 2, 2), (1, 2, 2)),
        ([16, 64, 64], [32, 32, 32, 32], 3, (1, 2, 2), (1, 2, 2)),
        ((15, 64, 64), [32, 32, 32], 1, (1, 2, 2), (2, 2)),
        ((15, 64, 64), [32, 32, 32], 2, (1, 2, 2), (2, 2)),
        ((15, 64, 64), [32, 32, 32], 3, (1, 2, 2), (2, 2)),
    ],
)  # TODO LC input in channels
def test_lvae(
    img_size: list[int],
    z_dims: list[int],
    multiscale_count: int,
    encoder_conv_stride,
    decoder_conv_stride,
    tmp_path,
    create_dummy_noise_model,
) -> None:
    model = create_LVAE_model(
        input_shape=img_size,
        z_dims=z_dims,
        multiscale_count=multiscale_count,
        encoder_conv_strides=encoder_conv_stride,
        decoder_conv_strides=decoder_conv_stride,
    )
    inputs = torch.ones((1, multiscale_count, *img_size))
    output, td_data = model(inputs)
    assert (
        output.shape == (1, 1, *img_size)
        if len(encoder_conv_stride) == len(decoder_conv_stride)
        else (1, 1, *img_size[1:])
    )
    assert td_data is not None
    assert len(td_data["z"]) == len(z_dims)
    assert len(td_data["kl"]) == len(z_dims)
    assert all(kl is not None for kl in td_data["kl"])
