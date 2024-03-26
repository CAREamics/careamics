import pytest

from careamics.config import create_n2v_configuration
from careamics.config.support import (
    SupportedTransform, SupportedPixelManipulation, SupportedStructAxis
)


def test_n2v_configuration(tmp_path):
    """Test that N2V configuration can be created."""
    config = create_n2v_configuration(
        experiment_name="test",
        working_directory=tmp_path,
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
    )
    assert config.data.transforms[-1].name == SupportedTransform.N2V_MANIPULATE.value
    assert config.data.transforms[-1].parameters.strategy == \
        SupportedPixelManipulation.UNIFORM.value
    assert not config.data.transforms[-2].parameters.is_3D # XY_RANDOM_ROTATE90
    assert not config.data.transforms[-3].parameters.is_3D # NDFLIP
    assert not config.algorithm.model.is_3D()


def test_n2v_3d_configuration(tmp_path):
    """Test that N2V configuration can be created in 3D."""
    config = create_n2v_configuration(
        experiment_name="test",
        working_directory=tmp_path,
        data_type="tiff",
        axes="ZYX",
        patch_size=[64, 64, 64],
        batch_size=8,
        num_epochs=100,
    )
    assert config.data.transforms[-1].name == SupportedTransform.N2V_MANIPULATE.value
    assert config.data.transforms[-1].parameters.strategy == \
        SupportedPixelManipulation.UNIFORM.value
    assert config.data.transforms[-2].parameters.is_3D # XY_RANDOM_ROTATE90
    assert config.data.transforms[-3].parameters.is_3D # NDFLIP
    assert config.algorithm.model.is_3D()


def test_n2v_3d_error(tmp_path):
    """Test that errors are raised if algorithm `is_3D` and data axes are
    incompatible."""
    with pytest.raises(ValueError):
        create_n2v_configuration(
            experiment_name="test",
            working_directory=tmp_path,
            data_type="tiff",
            axes="ZYX",
            patch_size=[64, 64],
            batch_size=8,
            num_epochs=100,
        )

    with pytest.raises(ValueError):
        create_n2v_configuration(
            experiment_name="test",
            working_directory=tmp_path,
            data_type="tiff",
            axes="YX",
            patch_size=[64, 64, 64],
            batch_size=8,
            num_epochs=100,
        )


def test_n2v_model_parameters(tmp_path):
    """Test passing N2V UNet parameters, and that explicit parameters override the 
    model_kwargs ones."""
    config = create_n2v_configuration(
        experiment_name="test",
        working_directory=tmp_path,
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        use_n2v2=False,
        n_channels=3,
        model_kwargs={
            "depth": 4,
            "n2v2": True,
            "in_channels": 2,
            "num_classes": 5,
        },
    )
    assert config.algorithm.model.depth == 4
    assert not config.algorithm.model.n2v2
    assert config.algorithm.model.in_channels == 3
    assert config.algorithm.model.num_classes == 3


def test_n2v_no_aug(tmp_path):
    """Test that N2V configuration can be created without augmentation."""
    config = create_n2v_configuration(
        experiment_name="test",
        working_directory=tmp_path,
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        use_augmentations=False,
    )
    assert len(config.data.transforms) == 2
    assert config.data.transforms[-1].name == SupportedTransform.N2V_MANIPULATE.value
    assert config.data.transforms[-2].name == SupportedTransform.NORMALIZE.value


def test_n2v_augmentation_parameters(tmp_path):
    """Test that N2V configuration can be created with augmentation parameters."""
    config = create_n2v_configuration(
        experiment_name="test",
        working_directory=tmp_path,
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        roi_size = 17,
        masked_pixel_percentage = 0.5,
    )
    assert config.data.transforms[-1].parameters.roi_size == 17
    assert config.data.transforms[-1].parameters.masked_pixel_percentage == 0.5


def test_n2v2(tmp_path):
    """Test that N2V2 configuration can be created."""
    config = create_n2v_configuration(
        experiment_name="test",
        working_directory=tmp_path,
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        use_n2v2=True,
    )
    assert config.data.transforms[-1].parameters.strategy == \
        SupportedPixelManipulation.MEDIAN.value


def test_structn2v(tmp_path):
    """Test that StructN2V configuration can be created."""
    config = create_n2v_configuration(
        experiment_name="test",
        working_directory=tmp_path,
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        struct_n2v_axis=SupportedStructAxis.HORIZONTAL.value,
        struct_n2v_span=7,
    )
    assert config.data.transforms[-1].parameters.struct_mask_axis == \
        SupportedStructAxis.HORIZONTAL.value
    assert config.data.transforms[-1].parameters.struct_mask_span == 7