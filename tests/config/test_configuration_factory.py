import pytest

from careamics.config import create_n2v_configuration
from careamics.config.support import (
    SupportedPixelManipulation,
    SupportedStructAxis,
    SupportedTransform,
)


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
    assert (
        config.data_config.transforms[-1].name
        == SupportedTransform.N2V_MANIPULATE.value
    )
    assert (
        config.data_config.transforms[-1].strategy
        == SupportedPixelManipulation.UNIFORM.value
    )
    assert not config.data_config.transforms[-2].is_3D  # XY_RANDOM_ROTATE90
    assert not config.data_config.transforms[-3].is_3D  # NDFLIP
    assert not config.algorithm_config.model.is_3D()


def test_n2v_3d_configuration():
    """Test that N2V configuration can be created in 3D."""
    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="ZYX",
        patch_size=[64, 64, 64],
        batch_size=8,
        num_epochs=100,
    )
    assert (
        config.data_config.transforms[-1].name
        == SupportedTransform.N2V_MANIPULATE.value
    )
    assert (
        config.data_config.transforms[-1].strategy
        == SupportedPixelManipulation.UNIFORM.value
    )
    assert config.data_config.transforms[-2].is_3D  # XY_RANDOM_ROTATE90
    assert config.data_config.transforms[-3].is_3D  # NDFLIP
    assert config.algorithm_config.model.is_3D()


def test_n2v_3d_error():
    """Test that errors are raised if algorithm `is_3D` and data axes are
    incompatible."""
    with pytest.raises(ValueError):
        create_n2v_configuration(
            experiment_name="test",
            data_type="tiff",
            axes="ZYX",
            patch_size=[64, 64],
            batch_size=8,
            num_epochs=100,
        )

    with pytest.raises(ValueError):
        create_n2v_configuration(
            experiment_name="test",
            data_type="tiff",
            axes="YX",
            patch_size=[64, 64, 64],
            batch_size=8,
            num_epochs=100,
        )


def test_n2v_model_parameters():
    """Test passing N2V UNet parameters, and that explicit parameters override the
    model_kwargs ones."""
    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        use_n2v2=False,
        model_kwargs={
            "depth": 4,
            "n2v2": True,
            "in_channels": 2,
            "num_classes": 5,
        },
    )
    assert config.algorithm_config.model.depth == 4
    assert not config.algorithm_config.model.n2v2

    # set to 1 because no C specified
    assert config.algorithm_config.model.in_channels == 1
    assert config.algorithm_config.model.num_classes == 1


def test_n2v_model_parameters_channels():
    """Test that the number of channels in the function call has priority over the
    model kwargs."""
    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YXC",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        n_channels=4,
        model_kwargs={
            "depth": 4,
            "n2v2": True,
            "in_channels": 2,
            "num_classes": 5,
        },
    )
    assert config.algorithm_config.model.in_channels == 4
    assert config.algorithm_config.model.num_classes == 4


def test_n2v_model_parameters_channels_error():
    """Test that an error is raised if the number of channels is not specified and
    C in axes, or C in axes and number of channels not specified."""
    with pytest.raises(ValueError):
        create_n2v_configuration(
            experiment_name="test",
            data_type="tiff",
            axes="YXC",
            patch_size=[64, 64],
            batch_size=8,
            num_epochs=100,
        )

    with pytest.raises(ValueError):
        create_n2v_configuration(
            experiment_name="test",
            data_type="tiff",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
            num_epochs=100,
            n_channels=5,
        )


def test_n2v_no_aug():
    """Test that N2V configuration can be created without augmentation."""
    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        use_augmentations=False,
    )
    assert len(config.data_config.transforms) == 2
    assert (
        config.data_config.transforms[-1].name
        == SupportedTransform.N2V_MANIPULATE.value
    )
    assert config.data_config.transforms[-2].name == SupportedTransform.NORMALIZE.value


def test_n2v_augmentation_parameters():
    """Test that N2V configuration can be created with augmentation parameters."""
    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        roi_size=17,
        masked_pixel_percentage=0.5,
    )
    assert config.data_config.transforms[-1].roi_size == 17
    assert config.data_config.transforms[-1].masked_pixel_percentage == 0.5


def test_n2v2():
    """Test that N2V2 configuration can be created."""
    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        use_n2v2=True,
    )
    assert (
        config.data_config.transforms[-1].strategy
        == SupportedPixelManipulation.MEDIAN.value
    )


def test_structn2v():
    """Test that StructN2V configuration can be created."""
    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        struct_n2v_axis=SupportedStructAxis.HORIZONTAL.value,
        struct_n2v_span=7,
    )
    assert (
        config.data_config.transforms[-1].struct_mask_axis
        == SupportedStructAxis.HORIZONTAL.value
    )
    assert config.data_config.transforms[-1].struct_mask_span == 7
