import pytest

from careamics.config.data.ng_data_config import NGDataConfig
from careamics.config.support.supported_patching_strategies import (
    SupportedPatchingStrategy,
)


def default_patching(mode: str) -> dict:
    """Return default patching strategy based on mode."""
    if mode == "training":
        return {"name": SupportedPatchingStrategy.RANDOM, "patch_size": [16, 16]}
    elif mode == "validating":
        return {"name": SupportedPatchingStrategy.FIXED_RANDOM, "patch_size": [16, 16]}
    elif mode == "predicting":
        return {"name": SupportedPatchingStrategy.WHOLE}
    else:
        raise ValueError(f"Unknown mode: {mode}")


@pytest.mark.parametrize(
    "patching_strategy, mode",
    [
        (
            {"name": SupportedPatchingStrategy.RANDOM, "patch_size": [16, 16]},
            "training",
        ),
        (
            {"name": SupportedPatchingStrategy.FIXED_RANDOM, "patch_size": [16, 16]},
            "validating",
        ),
        (
            {
                "name": SupportedPatchingStrategy.TILED,
                "patch_size": [16, 16],
                "overlaps": [4, 4],
            },
            "predicting",
        ),
        (
            {
                "name": SupportedPatchingStrategy.WHOLE,
            },
            "predicting",
        ),
    ],
)
def test_config_strategy(patching_strategy, mode):

    # Test the DataModel class
    data_config = NGDataConfig(
        mode=mode,
        data_type="array",
        axes="YX",
        patching=patching_strategy,
    )

    assert data_config.patching.name == patching_strategy["name"]


@pytest.mark.parametrize(
    "patching_strategy, mode",
    [
        (
            {"name": SupportedPatchingStrategy.RANDOM, "patch_size": [16, 16]},
            "validating",
        ),
        (
            {"name": SupportedPatchingStrategy.RANDOM, "patch_size": [16, 16]},
            "predicting",
        ),
        (
            {"name": SupportedPatchingStrategy.FIXED_RANDOM, "patch_size": [16, 16]},
            "training",
        ),
        (
            {"name": SupportedPatchingStrategy.FIXED_RANDOM, "patch_size": [16, 16]},
            "predicting",
        ),
        (
            {
                "name": SupportedPatchingStrategy.TILED,
                "patch_size": [16, 16],
                "overlaps": [4, 4],
            },
            "training",
        ),
        (
            {
                "name": SupportedPatchingStrategy.TILED,
                "patch_size": [16, 16],
                "overlaps": [4, 4],
            },
            "validating",
        ),
        (
            {
                "name": SupportedPatchingStrategy.WHOLE,
            },
            "training",
        ),
        (
            {
                "name": SupportedPatchingStrategy.WHOLE,
            },
            "validating",
        ),
    ],
)
def test_wrong_config_strategy(patching_strategy, mode):
    # Test the DataModel class
    with pytest.raises(ValueError):
        _ = NGDataConfig(
            mode=mode,
            data_type="array",
            axes="YX",
            patching=patching_strategy,
        )


@pytest.mark.parametrize(
    "in_memory, data_type, error",
    [
        # accepted combinations
        (True, "array", False),
        (True, "tiff", False),
        (True, "custom", False),
        (False, "tiff", False),
        (False, "custom", False),
        (False, "zarr", False),
        (False, "czi", False),
        # defaults are valid
        (None, "array", False),
        (None, "tiff", False),
        (None, "custom", False),
        (None, "zarr", False),
        (None, "czi", False),
        # validation errors
        (False, "array", True),
        (True, "zarr", True),
        (True, "czi", True),
    ],
)
def test_config_in_memory(in_memory, data_type, error):
    if error:
        with pytest.raises(ValueError):
            NGDataConfig(
                mode="predicting",
                data_type=data_type,
                axes="YX" if data_type != "czi" else "SCYX",
                in_memory=in_memory,
                patching={"name": SupportedPatchingStrategy.WHOLE},
            )
    else:
        # if in_memory is None, check the default value
        if in_memory is None:
            config = NGDataConfig(
                mode="predicting",
                data_type=data_type,
                axes="YX" if data_type != "czi" else "SCYX",
                patching={"name": SupportedPatchingStrategy.WHOLE},
            )
            if data_type in ("array", "tiff", "custom"):
                assert config.in_memory is True
            else:
                assert config.in_memory is False
        else:
            _ = NGDataConfig(
                mode="predicting",
                data_type=data_type,
                axes="YX" if data_type != "czi" else "SCYX",
                in_memory=in_memory,
                patching={"name": SupportedPatchingStrategy.WHOLE},
            )


@pytest.mark.parametrize(
    "channels, error",
    [
        # valid
        (None, False),
        ([1], False),
        ([0, 4, 6], False),
        ((0, 1), False),
        # validator changes to valid value
        (3, False),
        ([], False),
        ((), False),
        # not a sequence
        ("invalid", True),
        ([0, -1], True),
        ([0, 3.3], True),
        ([0, 3, 0], True),
    ],
)
def test_channels(channels, error):
    if error:
        with pytest.raises(ValueError):
            NGDataConfig(
                mode="predicting",
                data_type="array",
                axes="CYX",
                patching=default_patching("predicting"),
                channels=channels,
            )
    else:
        _ = NGDataConfig(
            mode="predicting",
            data_type="array",
            axes="CYX",
            patching=default_patching("predicting"),
            channels=channels,
        )


def test_propagate_seed():
    global_seed = 42
    config = NGDataConfig(
        mode="training",
        data_type="array",
        axes="CYX",
        patching=default_patching("training"),
        patch_filter={
            "name": "shannon",
            "threshold": 0.5,
        },
        transforms=[{"name": "XYFlip"}],
        seed=global_seed,
    )

    assert config.seed == global_seed
    assert config.patching.seed == global_seed
    assert config.patch_filter.seed == global_seed
    for transform in config.transforms:
        assert transform.seed == global_seed


@pytest.mark.parametrize(
    "filter_config, mode, error",
    [
        ({"name": "mask"}, "training", False),
        ({"name": "mask"}, "validating", True),
        ({"name": "mask"}, "predicting", True),
    ],
)
def test_validate_coord_filters(filter_config, mode, error):

    if error:
        with pytest.raises(ValueError):
            NGDataConfig(
                mode=mode,
                data_type="array",
                axes="CYX",
                patching=default_patching(mode),
                coord_filter=filter_config,
            )
    else:
        _ = NGDataConfig(
            mode=mode,
            data_type="array",
            axes="CYX",
            patching=default_patching(mode),
            coord_filter=filter_config,
        )


class TestDimensions:

    @pytest.mark.parametrize(
        "mode, axes, patching, patch_size",
        [
            ("training", "ZYX", "random", [16, 16]),
            ("training", "CYX", "random", [16, 16, 16]),
            ("training", "SYX", "random", [16, 16, 16]),
            ("predicting", "ZYX", "tiled", [16, 16]),
            ("predicting", "CYX", "tiled", [16, 16, 16]),
            ("predicting", "SYX", "tiled", [16, 16, 16]),
        ],
    )
    def test_dimensions(self, mode, axes, patching, patch_size):
        if patching == "random":
            patching = {
                "name": SupportedPatchingStrategy.RANDOM,
                "patch_size": patch_size,
            }
        elif patching == "tiled":
            patching = {
                "name": SupportedPatchingStrategy.TILED,
                "patch_size": patch_size,
                "overlaps": [4 for _ in patch_size],
            }
        else:
            patching = {"name": SupportedPatchingStrategy.WHOLE}

        with pytest.raises(ValueError):
            NGDataConfig(
                mode=mode,
                data_type="array",
                axes=axes,
                patching=patching,
            )

    @pytest.mark.parametrize(
        "data_type, mode, axes, patching, patch_size, is_3D",
        [
            ("array", "training", "CYX", "random", [16, 16], False),
            ("array", "training", "CZYX", "random", [8, 16, 16], True),
            ("array", "predicting", "CYX", "tiled", [16, 16], False),
            ("array", "predicting", "CZYX", "tiled", [8, 16, 16], True),
            ("array", "predicting", "CYX", "whole", None, False),
            ("array", "predicting", "CZYX", "whole", None, True),
            # czi has some specificities due to T being a 3D axis
            ("czi", "training", "SCYX", "random", [16, 16], False),
            ("czi", "training", "SCZYX", "random", [8, 16, 16], True),
            ("czi", "training", "SCTYX", "random", [8, 16, 16], True),
            ("czi", "predicting", "SCYX", "tiled", [16, 16], False),
            ("czi", "predicting", "SCZYX", "tiled", [8, 16, 16], True),
            ("czi", "predicting", "SCTYX", "tiled", [8, 16, 16], True),
            ("czi", "predicting", "SCYX", "whole", None, False),
            ("czi", "predicting", "SCZYX", "whole", None, True),
            ("czi", "predicting", "SCTYX", "whole", None, True),
        ],
    )
    def test_is_3D(self, data_type, mode, axes, patching, patch_size, is_3D):
        if patching == "random":
            patching = {
                "name": SupportedPatchingStrategy.RANDOM,
                "patch_size": patch_size,
            }
        elif patching == "tiled":
            patching = {
                "name": SupportedPatchingStrategy.TILED,
                "patch_size": patch_size,
                "overlaps": [4 for _ in patch_size],
            }
        else:
            patching = {"name": SupportedPatchingStrategy.WHOLE}

        config_2D = NGDataConfig(
            mode=mode,
            data_type=data_type,
            axes=axes,
            patching=patching,
        )
        assert config_2D.is_3D() == is_3D

    @pytest.mark.parametrize(
        "mode, axes, patching, patch_size, error",
        [
            # accepted combinations
            ("training", "SCYX", "random", [16, 16], False),
            ("training", "SCZYX", "random", [8, 16, 16], False),
            ("training", "SCTYX", "random", [8, 16, 16], False),
            ("predicting", "SCYX", "tiled", [16, 16], False),
            ("predicting", "SCZYX", "tiled", [8, 16, 16], False),
            ("predicting", "SCTYX", "tiled", [8, 16, 16], False),
            ("predicting", "SCYX", "whole", None, False),
            ("predicting", "SCZYX", "whole", None, False),
            ("predicting", "SCTYX", "whole", None, False),
            # validation errors
            ("training", "SCZYX", "random", [16, 16], True),
            ("training", "SCTYX", "random", [16, 16], True),
            ("predicting", "SCZYX", "tiled", [16, 16], True),
            ("predicting", "SCTYX", "tiled", [16, 16], True),
        ],
    )
    def test_czi_zt_3D(self, mode, axes, patching, patch_size, error):
        if patching == "random":
            patching = {
                "name": SupportedPatchingStrategy.RANDOM,
                "patch_size": patch_size,
            }
        elif patching == "tiled":
            patching = {
                "name": SupportedPatchingStrategy.TILED,
                "patch_size": patch_size,
                "overlaps": [4 for _ in patch_size],
            }
        else:
            patching = {"name": SupportedPatchingStrategy.WHOLE}

        if error:
            with pytest.raises(ValueError):
                _ = NGDataConfig(
                    mode=mode,
                    data_type="czi",
                    axes=axes,
                    patching=patching,
                )
        else:
            NGDataConfig(
                mode=mode,
                data_type="czi",
                axes=axes,
                patching=patching,
            )


class TestConvertMode:

    def test_default(self):
        """Test converting mode with default parameters."""
        original_config = NGDataConfig(
            mode="training",
            data_type="array",
            axes="CYX",
            patching=default_patching("training"),
        )

        val_config = original_config.convert_mode("validating")
        assert val_config.mode == "validating"
        assert val_config.patching.name == SupportedPatchingStrategy.FIXED_RANDOM
        assert val_config.patching.patch_size == [16, 16]
        assert val_config.axes == original_config.axes
        assert val_config.data_type == original_config.data_type
        assert "shuffle" not in val_config.val_dataloader_params

        pred_config = original_config.convert_mode("predicting")
        assert pred_config.mode == "predicting"
        assert pred_config.patching.name == SupportedPatchingStrategy.WHOLE
        assert pred_config.axes == original_config.axes
        assert pred_config.data_type == original_config.data_type
        assert "shuffle" not in pred_config.pred_dataloader_params

    def test_tiled_patching(self):
        """Test converting mode for tiling."""
        original_config = NGDataConfig(
            mode="training",
            data_type="array",
            axes="CYX",
            patching=default_patching("training"),
        )

        assert (
            original_config.convert_mode("predicting").patching.name
            == SupportedPatchingStrategy.WHOLE
        )

        assert (
            original_config.convert_mode(
                "predicting",
                new_patch_size=[32, 32],
                overlap_size=[8, 8],
            ).patching.name
            == SupportedPatchingStrategy.TILED
        )

        with pytest.raises(ValueError):
            original_config.convert_mode(
                "predicting",
                new_patch_size=[32, 32],
            )

    def test_conservation_means_stds(self):
        """Test converting mode conserves means and stds."""
        original_config = NGDataConfig(
            mode="training",
            data_type="array",
            axes="CYX",
            patching=default_patching("training"),
            image_means=[0.5],
            image_stds=[0.2],
            target_means=[0.3],
            target_stds=[0.1],
        )

        val_config = original_config.convert_mode("validating")
        assert val_config.image_means == original_config.image_means
        assert val_config.image_stds == original_config.image_stds
        assert val_config.target_means == original_config.target_means
        assert val_config.target_stds == original_config.target_stds

        pred_config = original_config.convert_mode("predicting")
        assert pred_config.image_means == original_config.image_means
        assert pred_config.image_stds == original_config.image_stds
        assert pred_config.target_means == original_config.target_means
        assert pred_config.target_stds == original_config.target_stds

    def test_with_dataloader_params(self):
        """Test converting mode with new dataloader parameters."""
        original_config = NGDataConfig(
            mode="training",
            data_type="array",
            axes="CYX",
            patching=default_patching("training"),
            val_dataloader_params={"pin_memory": True},
            pred_dataloader_params={"num_workers": 2},
        )

        val_config = original_config.convert_mode("validating")
        assert val_config.val_dataloader_params["pin_memory"]

        pred_config = original_config.convert_mode("predicting")
        assert pred_config.pred_dataloader_params["num_workers"] == 2

    def test_passing_inoffensive_parameters(self):
        """Test converting mode with parameters."""
        original_config = NGDataConfig(
            mode="training",
            data_type="array",
            axes="YX",
            patching=default_patching("training"),
        )

        val_config = original_config.convert_mode(
            "validating",
            new_batch_size=5,
            new_data_type="tiff",
        )
        assert val_config.mode == "validating"
        assert val_config.batch_size == 5
        assert val_config.data_type == "tiff"

        pred_config = original_config.convert_mode(
            "predicting",
            new_batch_size=10,
            new_data_type="tiff",
        )
        assert pred_config.mode == "predicting"
        assert pred_config.batch_size == 10
        assert pred_config.data_type == "tiff"

    def test_in_memory_change(self):
        """Test converting mode with in_memory change."""
        original_config = NGDataConfig(
            mode="training",
            data_type="tiff",
            axes="YX",
            patching=default_patching("training"),
            in_memory=True,
        )

        val_config = original_config.convert_mode(
            "validating",
            new_in_memory=False,
        )
        assert val_config.mode == "validating"
        assert val_config.in_memory is False

        pred_config = original_config.convert_mode(
            "predicting",
            new_in_memory=False,
        )
        assert pred_config.mode == "predicting"
        assert pred_config.in_memory is False

    @pytest.mark.parametrize("mode", ["training", "validating", "predicting"])
    def test_convert_mode_to_training_error(self, mode):
        """Test converting mode to training raises error."""
        original_config = NGDataConfig(
            mode=mode,
            data_type="array",
            axes="CYX",
            patching=default_patching(mode),
        )

        with pytest.raises(ValueError):
            _ = original_config.convert_mode("training")

    @pytest.mark.parametrize("mode", ["validating", "predicting"])
    def test_changing_spatial_axes_error(self, mode):
        """Test converting mode while changing spatial axes."""
        original_config = NGDataConfig(
            mode="training",
            data_type="array",
            axes="CYX",
            patching=default_patching("training"),
        )

        with pytest.raises(ValueError):
            _ = original_config.convert_mode(
                mode,
                new_axes="CZYX",
            )

    def test_adding_channels(self):
        """Test converting mode while adding channels."""
        original_config = NGDataConfig(
            mode="training",
            data_type="array",
            axes="YX",
            patching=default_patching("training"),
        )

        # adding "C" with multiple channels specified
        with pytest.raises(ValueError):
            _ = original_config.convert_mode(
                "validating",
                new_axes="CYX",
                new_channels=[0, 1],
            )

        # specifying all channels is ambiguous due to singleton case, warning is raised
        with pytest.warns(UserWarning):
            _ = original_config.convert_mode(
                "validating",
                new_axes="CYX",
                new_channels="all",
            )

    def test_removing_channels(self):
        """Test converting mode while removing channels."""
        original_config = NGDataConfig(
            mode="training",
            data_type="array",
            axes="CYX",
            channels=[0, 1],
            patching=default_patching("training"),
        )

        #
        with pytest.raises(ValueError):
            _ = original_config.convert_mode(
                "validating",
                new_axes="YX",
            )
