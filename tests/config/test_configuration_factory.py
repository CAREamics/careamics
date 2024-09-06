import pytest

from careamics.config import (
    create_care_configuration,
    create_n2n_configuration,
    create_n2v_configuration,
)
from careamics.config.configuration_factory import (
    _create_configuration,
    _create_supervised_configuration,
    _create_unet_configuration,
    _list_augmentations,
)
from careamics.config.support import (
    SupportedPixelManipulation,
    SupportedStructAxis,
    SupportedTransform,
)
from careamics.config.transformations import (
    N2VManipulateModel,
    XYFlipModel,
    XYRandomRotate90Model,
)


def test_list_aug_default():
    """Test that the default augmentations are present."""
    list_aug = _list_augmentations(augmentations=None)

    assert len(list_aug) == 2
    assert list_aug[0].name == SupportedTransform.XY_FLIP.value
    assert list_aug[1].name == SupportedTransform.XY_RANDOM_ROTATE90.value


def test_list_aug_no_aug():
    """Test that disabling augmentation results in empty transform list."""
    list_aug = _list_augmentations(augmentations=[])
    assert len(list_aug) == 0


def test_list_aug_error_duplicate_transforms():
    """Test that an error is raised when there are duplicate transforms."""
    with pytest.raises(ValueError):
        _list_augmentations(
            augmentations=[XYFlipModel(), XYRandomRotate90Model(), XYFlipModel()],
        )


def test_list_aug_error_wrong_transform():
    """Test that an error is raised when the wrong transform is passed."""
    with pytest.raises(ValueError):
        _list_augmentations(
            augmentations=[XYFlipModel(), N2VManipulateModel()],
        )


def test_supervised_configuration_passing_transforms():
    """Test that transforms can be passed to the configuration."""
    config = _create_supervised_configuration(
        algorithm="n2n",
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        augmentations=[XYFlipModel()],
    )
    assert len(config.data_config.transforms) == 1
    assert config.data_config.transforms[0].name == SupportedTransform.XY_FLIP.value


def test_model_creation():
    """Test that the correct parameters are passed to the model."""
    model_kwargs = {
        "depth": 4,
        "conv_dims": 2,
        "n2v2": False,
        "in_channels": 2,
        "num_classes": 5,
        "independent_channels": False,
    }

    # choose different parameters
    axes = "XYZ"
    conv_dims = 3
    in_channels = 3
    num_classes = 4
    independent_channels = True
    use_n2v2 = True

    model = _create_unet_configuration(
        axes=axes,
        n_channels_in=in_channels,
        n_channels_out=num_classes,
        independent_channels=independent_channels,
        use_n2v2=use_n2v2,
        model_params=model_kwargs,
    )

    assert model.depth == model_kwargs["depth"]
    assert model.conv_dims == conv_dims
    assert model.n2v2 == use_n2v2
    assert model.in_channels == in_channels
    assert model.num_classes == num_classes
    assert model.independent_channels == independent_channels


def test_create_configuration():
    """Test that the methods correctly passes all parameters."""
    algorithm = "care"
    experiment_name = "test"
    data_type = "tiff"
    axes = "CYX"
    patch_size = [128, 128]
    batch_size = 8
    num_epochs = 100
    transform_list = [XYFlipModel(), XYRandomRotate90Model()]
    independent_channels = False
    loss = "mse"
    n_channels_in = 2
    n_channels_out = 3
    logger = "tensorboard"
    model_params = {
        "depth": 5,
    }
    dataloader_params = {
        "num_workers": 4,
    }

    # instantiate config
    config = _create_configuration(
        algorithm=algorithm,
        experiment_name=experiment_name,
        data_type=data_type,
        axes=axes,
        patch_size=patch_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        augmentations=transform_list,
        independent_channels=independent_channels,
        loss=loss,
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
        logger=logger,
        model_params=model_params,
        dataloader_params=dataloader_params,
    )

    assert config.algorithm_config.algorithm == algorithm
    assert config.experiment_name == experiment_name
    assert config.data_config.data_type == data_type
    assert config.data_config.axes == axes
    assert config.data_config.patch_size == patch_size
    assert config.data_config.batch_size == batch_size
    assert config.training_config.num_epochs == num_epochs
    assert config.data_config.transforms == transform_list
    assert config.algorithm_config.model.independent_channels == independent_channels
    assert config.algorithm_config.loss == loss
    assert config.algorithm_config.model.in_channels == n_channels_in
    assert config.algorithm_config.model.num_classes == n_channels_out
    assert config.training_config.logger == logger
    assert config.algorithm_config.model.depth == model_params["depth"]
    assert config.data_config.dataloader_params == dataloader_params


def test_supervised_configuration_error_with_channel_axes():
    """Test that an error is raised if channels are in axes, but the input channel
    number is not specified."""
    with pytest.raises(ValueError):
        _create_supervised_configuration(
            algorithm="n2n",
            experiment_name="test",
            data_type="tiff",
            axes="CYX",
            patch_size=[64, 64],
            batch_size=8,
            num_epochs=100,
        )


def test_supervised_configuration_error_without_channel_axes():
    """Test that an error is raised if channels are not in axes, but the input channel
    number is specified and greater than 1."""
    with pytest.raises(ValueError):
        _create_supervised_configuration(
            algorithm="n2n",
            experiment_name="test",
            data_type="tiff",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
            num_epochs=100,
            n_channels_in=2,
        )


def test_supervised_configuration_channels():
    """Test that no error is raised if channels are in axes and the input channel
    are specified."""
    _create_supervised_configuration(
        algorithm="n2n",
        experiment_name="test",
        data_type="tiff",
        axes="CYX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        n_channels_in=4,
    )


def test_n2n_configuration():
    """Test that N2N configuration can be created."""
    config = create_n2n_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
    )
    assert config.algorithm_config.algorithm == "n2n"


def test_n2n_configuration_n_channels():
    """Test the behaviour of the number of channels in and out."""
    n_channels_in = 4
    n_channels_out = 5

    # n_channels_out not specified
    config = create_n2n_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="CYX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        n_channels_in=n_channels_in,
    )
    assert config.algorithm_config.model.in_channels == n_channels_in
    assert config.algorithm_config.model.num_classes == n_channels_in

    # specify n_channels_out
    config = create_n2n_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="CYX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
    )
    assert config.algorithm_config.model.in_channels == n_channels_in
    assert config.algorithm_config.model.num_classes == n_channels_out


def test_care_configuration():
    """Test that CARE configuration can be created."""
    config = create_care_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
    )
    assert config.algorithm_config.algorithm == "care"


def test_care_configuration_n_channels():
    """Test the behaviour of the number of channels in and out."""
    n_channels_in = 4
    n_channels_out = 5

    # n_channels_out not specified
    config = create_care_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="CYX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        n_channels_in=n_channels_in,
    )
    assert config.algorithm_config.model.in_channels == n_channels_in
    assert config.algorithm_config.model.num_classes == n_channels_in

    # specify n_channels_out
    config = create_care_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="CYX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
    )
    assert config.algorithm_config.model.in_channels == n_channels_in
    assert config.algorithm_config.model.num_classes == n_channels_out


def test_n2v_configuration():
    """Test that N2V configuration can be created."""
    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
    )
    assert config.algorithm_config.algorithm == "n2v"
    assert config.algorithm_config.loss == "n2v"
    assert (
        config.data_config.transforms[-1].name
        == SupportedTransform.N2V_MANIPULATE.value
    )


def test_n2v_configuration_default_transforms():
    """Test the default n2v transforms."""
    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
    )
    assert len(config.data_config.transforms) == 3
    assert config.data_config.transforms[0].name == SupportedTransform.XY_FLIP.value
    assert (
        config.data_config.transforms[1].name
        == SupportedTransform.XY_RANDOM_ROTATE90.value
    )
    assert (
        config.data_config.transforms[2].name == SupportedTransform.N2V_MANIPULATE.value
    )


def test_n2v_configuration_no_aug():
    """Test the default n2v transforms."""
    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        augmentations=[],
    )
    assert len(config.data_config.transforms) == 1
    assert (
        config.data_config.transforms[-1].name
        == SupportedTransform.N2V_MANIPULATE.value
    )


def test_n2v_configuration_n2v2_structn2v():
    """Test that N2V manipulate is ignored when explicitely passed."""
    use_n2v2 = True
    roi_size = 15
    masked_pixel_percentage = 0.5
    struct_mask_axis = SupportedStructAxis.HORIZONTAL.value
    struct_n2v_span = 15

    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        use_n2v2=use_n2v2,  # median strategy
        roi_size=roi_size,
        masked_pixel_percentage=masked_pixel_percentage,
        struct_n2v_axis=struct_mask_axis,
        struct_n2v_span=struct_n2v_span,
    )
    assert len(config.data_config.transforms) == 3
    assert (
        config.data_config.transforms[-1].name
        == SupportedTransform.N2V_MANIPULATE.value
    )
    assert (
        config.data_config.transforms[-1].name
        == SupportedTransform.N2V_MANIPULATE.value
    )
    assert (
        config.data_config.transforms[-1].strategy
        == SupportedPixelManipulation.MEDIAN.value
    )
    assert config.data_config.transforms[-1].roi_size == roi_size
    assert (
        config.data_config.transforms[-1].masked_pixel_percentage
        == masked_pixel_percentage
    )
    assert config.data_config.transforms[-1].struct_mask_axis == struct_mask_axis
    assert config.data_config.transforms[-1].struct_mask_span == struct_n2v_span
