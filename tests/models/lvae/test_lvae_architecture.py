import pytest
import torch
from torch import nn

from careamics.config.architectures import LVAEModel
from careamics.config import VAEAlgorithmConfig
from careamics.models.model_factory import model_factory
from careamics.config.nm_model import GaussianMixtureNmModel

def create_LVAE_model(
    input_shape: int = 64,
    z_dims: list[int] = [128, 128, 128, 128],
    multiscale_count: int = 0,
) -> nn.Module:
    lvae_model_config = LVAEModel(
        architecture="LVAE",
        conv_dims=2,
        input_shape=input_shape,
        z_dims=z_dims,
        multiscale_count=multiscale_count,
        predict_logvar=None,
        enable_noise_model=False,
    )
    # algorithm, loss and noise_model are not important here since
    # we are only interested in testing the model architecture
    config = VAEAlgorithmConfig(
        algorithm_type="vae",
        algorithm="musplit",
        loss="musplit_loss",
        model=lvae_model_config,
        noise_model=GaussianMixtureNmModel(model_type="GaussianMixtureNoiseModel"),
    )
    return model_factory(config.model)

  
@pytest.mark.parametrize("img_size", [32, 64, 128, 256])
def test_first_bottom_up(img_size: int) -> None:
    model = create_LVAE_model(input_shape=img_size)
    first_bottom_up = model.first_bottom_up
    input = torch.ones((1, 1, img_size, img_size))
    output = first_bottom_up(input)
    assert output.shape == (1, model.encoder_n_filters, img_size, img_size)

@pytest.mark.parametrize(
    "z_dims", 
    [
        [128, 128],
        [128, 128, 128],
        [128, 128, 128, 128],
        [128, 128, 128, 128, 128],
    ]
)
def test_bottom_up_layers_no_LC(z_dims: list[int]) -> None:
    model = create_LVAE_model(z_dims=z_dims)
    bottom_up_layers = model.bottom_up_layers
    assert len(bottom_up_layers) == len(z_dims)
    img_size = model.image_size
    n_filters = model.encoder_n_filters
    input = torch.ones((1, n_filters, img_size, img_size))
    exp_img_size = img_size // 2
    for layer in bottom_up_layers:
        input, input2 = layer(input)
        assert input.shape == (1, n_filters, exp_img_size, exp_img_size)
        assert input2.shape == (1, n_filters, exp_img_size, exp_img_size)
        exp_img_size //= 2
        
@pytest.mark.parametrize(
    "z_dims", 
    [
        [128, 128],
        [128, 128, 128],
        [128, 128, 128, 128],
        [128, 128, 128, 128, 128],
    ]
)
def test_bottom_up_layers_no_LC(z_dims: list[int]) -> None:
    model = create_LVAE_model(z_dims=z_dims)
    bottom_up_layers = model.bottom_up_layers
    assert len(bottom_up_layers) == len(z_dims)
    img_size = model.image_size
    n_filters = model.encoder_n_filters
    input = torch.ones((1, n_filters, img_size, img_size))
    exp_img_size = img_size // 2
    for layer in bottom_up_layers:
        input, input2 = layer(input)
        assert input.shape == (1, n_filters, exp_img_size, exp_img_size)
        assert input2.shape == (1, n_filters, exp_img_size, exp_img_size)
        exp_img_size //= 2


@pytest.mark.parametrize(
    "z_dims, multiscale_count", 
    [
        ([128, 128], 4),
        ([128, 128, 128], 5),
    ]
)
def test_LC_init(
    z_dims: list[int], 
    multiscale_count: int
) -> None:
    # pattern = re.compile(
    #     r"Multiscale count \((\d+)\) should not exceed the number of bottom up layers \((\d+)\) by more than 1\.\n"
    # ) 
    with pytest.raises(AssertionError):
        create_LVAE_model(z_dims=z_dims, multiscale_count=multiscale_count)
        
@pytest.mark.parametrize(
    "z_dims, multiscale_count", 
    [
        ([128, 128, 128], 2),
        ([128, 128, 128], 4),
        ([128, 128, 128, 128], 4),
        ([128, 128, 128, 128], 5),
    ]
)
def test_bottom_up_layers_with_LC(
    z_dims: list[int], 
    multiscale_count: int
) -> None:

    model = create_LVAE_model(z_dims=z_dims, multiscale_count=multiscale_count)
    bottom_up_layers = model.bottom_up_layers
    
    assert len(bottom_up_layers) == len(z_dims), "Different number of bottom_up_layers and z_dims"
    # Check we have the right number of lowres_net's
    for i in range(multiscale_count - 1):
        assert bottom_up_layers[i].lowres_net is not None, "Missing lowres_net"
    
    img_size = model.image_size
    n_filters = model.encoder_n_filters
    input = torch.ones((1, n_filters, img_size, img_size))
    lowres_input = torch.ones((1, n_filters, img_size, img_size))
    exp_img_size = img_size
    for i, layer in enumerate(bottom_up_layers):
        if i + 1 > multiscale_count - 1:
            # assert layer.enable_multiscale is False, f"mc={multiscale_count}, i={i}"
            exp_img_size //= 2
            lowres_input = None
        input, merged = layer(input, lowres_input)
        assert input.shape == (1, n_filters, exp_img_size, exp_img_size)
        assert merged.shape == (1, n_filters, exp_img_size, exp_img_size)
        
        
@pytest.mark.parametrize(
    "z_dims, multiscale_count", 
    [
        ([128, 128, 128], 0),
        ([128, 128, 128], 1),
        ([128, 128, 128], 2),
        ([128, 128, 128], 4),
        ([128, 128, 128, 128], 0),
        ([128, 128, 128, 128], 1),
        ([128, 128, 128, 128], 4),
        ([128, 128, 128, 128], 5),
    ]
)
def test_bottom_up_pass(
    z_dims: list[int], 
    multiscale_count: int
) -> None:

    model = create_LVAE_model(z_dims=z_dims, multiscale_count=multiscale_count)
    first_bottom_up_layer = model.first_bottom_up
    lowres_first_bottom_up_layers = model.lowres_first_bottom_ups
    bottom_up_layers = model.bottom_up_layers
    
    assert len(bottom_up_layers) == len(z_dims), "Different number of bottom_up_layers and z_dims"
    # Check we have the right number of lowres_net's
    for i in range(multiscale_count - 1):
        assert bottom_up_layers[i].lowres_net is not None, "Missing lowres_net"
    
    img_size = model.image_size
    n_filters = model.encoder_n_filters
    n_channels = multiscale_count if multiscale_count else 1
    input = torch.ones((1, n_channels, img_size, img_size))
    outputs = model._bottomup_pass(
        inp=input,
        first_bottom_up=first_bottom_up_layer,
        lowres_first_bottom_ups=lowres_first_bottom_up_layers,
        bottom_up_layers=bottom_up_layers
    )
    exp_img_size = img_size
    for i in range(len(bottom_up_layers)):
        if i + 1 > multiscale_count - 1:
            exp_img_size //= 2
        assert outputs[i].shape == (1, n_filters, exp_img_size, exp_img_size)
    
        
@pytest.mark.parametrize("img_size", [64, 128])
@pytest.mark.parametrize("multiscale_count", [1, 3, 5])
def test_topmost_top_down_layer(img_size: int, multiscale_count: int) -> None:
    model = create_LVAE_model(input_shape=img_size, multiscale_count=multiscale_count)
    topmost_top_down = model.top_down_layers[-1]
    n_filters = model.encoder_n_filters

    downscaling = 2**(model.n_layers + 1 - multiscale_count)
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
    
    
@pytest.mark.parametrize("img_size", [64, 128])
@pytest.mark.parametrize("multiscale_count", [1, 3, 5])
def test_all_top_down_layers(img_size: int, multiscale_count: int) -> None:
    model = create_LVAE_model(input_shape=img_size, multiscale_count=multiscale_count)
    top_down_layers = model.top_down_layers
    n_filters = model.encoder_n_filters
    downscaling = 2**(model.n_layers + 1 - multiscale_count)
    downscaled_size = img_size // downscaling
    input = skip_input = None
    bu_value = torch.ones((1, n_filters, downscaled_size, downscaled_size))
    for i in reversed(range(model.n_layers)):
        td_layer = top_down_layers[i]
        output, data = td_layer(
            input_=input, 
            bu_value=bu_value, 
            inference_mode=True, 
            skip_connection_input=skip_input
        )
        input = bu_value = skip_input = output
        
        retain_sp_dims = (
            td_layer.retain_spatial_dims and downscaled_size == img_size 
        )
        exp_out_size = downscaled_size if retain_sp_dims else 2 * downscaled_size
        expected_out_shape = (1, n_filters, exp_out_size, exp_out_size) 
        expected_z_shape = (1, model.z_dims[0], downscaled_size, downscaled_size)
        assert output.shape == expected_out_shape, f"Found problem in layer {i+1}, retain={td_layer.retain_spatial_dims}, dwsc={downscaled_size}"
        assert data["z"].shape == expected_z_shape
        downscaled_size = exp_out_size


@pytest.mark.parametrize("img_size", [64, 128])
@pytest.mark.parametrize("multiscale_count", [1, 3, 5])
def test_final_top_down(img_size: int, multiscale_count: int) -> None:
    model = create_LVAE_model(input_shape=img_size, multiscale_count=multiscale_count)
    final_top_down = model.final_top_down
    n_filters = model.encoder_n_filters
    final_upsampling = not model.no_initial_downscaling
    inp_size = img_size // 2 if final_upsampling else img_size
    input = torch.ones((1, n_filters, inp_size, inp_size))
    output = final_top_down(input)
    expected_out_shape = (1, n_filters, img_size, img_size) 
    assert output.shape == expected_out_shape
    
@pytest.mark.parametrize("img_size", [64, 128])
@pytest.mark.parametrize("multiscale_count", [1, 3, 5])
def test_top_down_pass(img_size: int, multiscale_count: int) -> None:
    model = create_LVAE_model(input_shape=img_size, multiscale_count=multiscale_count)
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
        bu_values.append(
            torch.ones((1, n_filters, curr_size, curr_size))
        )
    
    output, data = model.topdown_pass(
        top_down_layers=top_down_layers,
        final_top_down_layer=final_top_down,
        bu_values=bu_values
    )
    
    expected_out_shape = (1, n_filters, img_size, img_size)  
    assert output.shape == expected_out_shape
    for i in range(n_layers):
        expected_z_shape = (1, model.z_dims[i], td_sizes[i], td_sizes[i])
        assert data["z"][i].shape == expected_z_shape