import pytest

from careamics.config import (
    create_care_configuration,
    create_inference_configuration,
    create_n2n_configuration,
    create_n2v_configuration,
)
from careamics.config.support import (
    SupportedPixelManipulation,
    SupportedStructAxis,
    SupportedTransform,
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

    assert config.data_config.transforms[0].name == SupportedTransform.XY_FLIP.value
    assert (
        config.data_config.transforms[1].name
        == SupportedTransform.XY_RANDOM_ROTATE90.value
    )
    assert not config.algorithm_config.model.is_3D()


def test_n2n_3d_configuration():
    """Test that a 3D N2N configuration can be created."""
    config = create_care_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="ZYX",
        patch_size=[64, 64, 64],
        batch_size=8,
        num_epochs=100,
    )
    assert config.data_config.transforms[0].name == SupportedTransform.XY_FLIP.value
    assert (
        config.data_config.transforms[1].name
        == SupportedTransform.XY_RANDOM_ROTATE90.value
    )
    assert config.algorithm_config.model.is_3D()


def test_n2n_3d_errors():
    """Test that errors are raised if the axes are incompatible with the patches."""
    with pytest.raises(ValueError):
        create_n2n_configuration(
            experiment_name="test",
            data_type="tiff",
            axes="ZYX",
            patch_size=[64, 64],
            batch_size=8,
            num_epochs=100,
        )

    with pytest.raises(ValueError):
        create_n2n_configuration(
            experiment_name="test",
            data_type="tiff",
            axes="YX",
            patch_size=[64, 64, 64],
            batch_size=8,
            num_epochs=100,
        )


def test_n2n_channels_errors():
    """Test that error are raised if the number of input channel and the axes are not
    compatible."""
    with pytest.raises(ValueError):
        create_n2n_configuration(
            experiment_name="test",
            data_type="tiff",
            axes="YXC",
            patch_size=[64, 64],
            batch_size=8,
            num_epochs=100,
        )

    with pytest.raises(ValueError):
        create_n2n_configuration(
            experiment_name="test",
            data_type="tiff",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
            num_epochs=100,
            n_channels_in=5,
        )


def test_n2n_aug_off():
    """Test that the augmentations are correctly disabled."""
    config = create_n2n_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        use_augmentations=False,
    )
    assert len(config.data_config.transforms) == 0


@pytest.mark.parametrize("ind_channels", [True, False])
def test_n2n_independent_channels(ind_channels):
    """Test that independent channels are correctly passed."""
    config = create_n2n_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YXC",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        n_channels_in=4,
        independent_channels=ind_channels,
    )
    assert config.algorithm_config.model.independent_channels == ind_channels


def test_n2n_channels_equal():
    """Test that channels in and out are equal if only channels_in is set."""
    config = create_n2n_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YXC",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        n_channels_in=4,
    )
    assert config.algorithm_config.model.in_channels == 4
    assert config.algorithm_config.model.num_classes == 4


def test_n2n_channels_different():
    """Test that channels in and out can be different."""
    config = create_n2n_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YXC",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        n_channels_in=4,
        n_channels_out=5,
    )
    assert config.algorithm_config.model.in_channels == 4
    assert config.algorithm_config.model.num_classes == 5


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

    assert config.data_config.transforms[0].name == SupportedTransform.XY_FLIP.value
    assert (
        config.data_config.transforms[1].name
        == SupportedTransform.XY_RANDOM_ROTATE90.value
    )
    assert not config.algorithm_config.model.is_3D()


def test_care_3d_configuration():
    """Test that a 3D care configuration can be created."""
    config = create_care_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="ZYX",
        patch_size=[64, 64, 64],
        batch_size=8,
        num_epochs=100,
    )
    assert config.data_config.transforms[0].name == SupportedTransform.XY_FLIP.value
    assert config.algorithm_config.model.is_3D()


def test_care_3d_errors():
    """Test that errors are raised if the axes are incompatible with the patches."""
    with pytest.raises(ValueError):
        create_care_configuration(
            experiment_name="test",
            data_type="tiff",
            axes="ZYX",
            patch_size=[64, 64],
            batch_size=8,
            num_epochs=100,
        )

    with pytest.raises(ValueError):
        create_care_configuration(
            experiment_name="test",
            data_type="tiff",
            axes="YX",
            patch_size=[64, 64, 64],
            batch_size=8,
            num_epochs=100,
        )


def test_care_channels_errors():
    """Test that error are raised if the number of input channel and the axes are not
    compatible."""
    with pytest.raises(ValueError):
        create_care_configuration(
            experiment_name="test",
            data_type="tiff",
            axes="YXC",
            patch_size=[64, 64],
            batch_size=8,
            num_epochs=100,
        )

    with pytest.raises(ValueError):
        create_care_configuration(
            experiment_name="test",
            data_type="tiff",
            axes="YX",
            patch_size=[64, 64],
            batch_size=8,
            num_epochs=100,
            n_channels_in=5,
        )


def test_care_aug_off():
    """Test that the augmentations are correctly disabled."""
    config = create_care_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        use_augmentations=False,
    )
    assert len(config.data_config.transforms) == 0


@pytest.mark.parametrize("ind_channels", [True, False])
def test_care_independent_channels(ind_channels):
    """Test that independent channels are correctly passed."""
    config = create_care_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YXC",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        n_channels_in=4,
        independent_channels=ind_channels,
    )
    assert config.algorithm_config.model.independent_channels == ind_channels


def test_care_chanels_out():
    """Test that channels out are set to channels in if not speicifed."""
    config = create_care_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YXC",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        n_channels_in=4,
    )
    assert config.algorithm_config.model.num_classes == 4

    # otherwise set independently
    config = create_care_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YXC",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        n_channels_in=4,
        n_channels_out=5,
    )
    assert config.algorithm_config.model.num_classes == 5


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
        independent_channels=False,
        use_n2v2=False,
        model_kwargs={
            "depth": 4,
            "n2v2": True,
            "in_channels": 2,
            "num_classes": 5,
            "independent_channels": True,
        },
    )
    assert config.algorithm_config.model.depth == 4
    assert not config.algorithm_config.model.n2v2
    assert not config.algorithm_config.model.independent_channels

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


@pytest.mark.parametrize("ind_channels", [True, False])
def test_n2v_independent_channels(ind_channels):
    """Test that the idnependent channels parameter is passed correctly."""
    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YXC",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
        n_channels=4,
        independent_channels=ind_channels,
    )
    assert config.algorithm_config.model.independent_channels == ind_channels


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
    assert len(config.data_config.transforms) == 1
    assert (
        config.data_config.transforms[0].name == SupportedTransform.N2V_MANIPULATE.value
    )


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


def test_inference_config_no_stats():
    """Test that an inference configuration fails if no statistics are present."""
    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
    )

    with pytest.raises(ValueError):
        create_inference_configuration(
            configuration=config,
        )


def test_inference_config():
    """Test that an inference configuration can be created."""
    config = create_n2v_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
    )
    config.data_config.set_mean_and_std([0.5], [0.2])

    create_inference_configuration(
        configuration=config,
    )


def test_inference_tile_size():
    """Test that an inference configuration can be created for a UNet model."""
    config = create_care_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
    )
    config.data_config.set_mean_and_std([0.5], [0.2])

    # check UNet depth, tile increment must then be a factor of 4
    assert config.algorithm_config.model.depth == 2

    # error if not a factor of 4
    with pytest.raises(ValueError):
        create_inference_configuration(
            configuration=config,
            tile_size=[6, 6],
            tile_overlap=[2, 2],
        )

    # no error if a factor of 4
    create_inference_configuration(
        configuration=config,
        tile_size=[8, 8],
        tile_overlap=[2, 2],
    )


def test_inference_tile_no_overlap():
    """Test that an error is raised if the tile overlap is not specified, but the tile
    size is."""
    config = create_care_configuration(
        experiment_name="test",
        data_type="tiff",
        axes="YX",
        patch_size=[64, 64],
        batch_size=8,
        num_epochs=100,
    )
    config.data_config.set_mean_and_std([0.5], [0.2])

    with pytest.raises(ValueError):
        create_inference_configuration(
            configuration=config,
            tile_size=[8, 8],
        )
