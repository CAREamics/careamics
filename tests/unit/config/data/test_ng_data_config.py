"""Unit tests for the NGDataConfig Pydantic model."""

import sys
import types
from collections.abc import Callable

import pytest

from careamics.config.data.ng_data_config import (
    NGDataConfig,
    _are_spatial_dims_maintained,
    _validate_channel_conversion,
    default_in_memory,
)

# Notes:
# - Not testing `np_float_to_scientific_str`, as it only calls a numpy function
# - Not testing default values of Pydantic models, unless they depend on each other
# - Not testing constraints of parameter that are directly declared to Pydantic (e.g.
# `Field(..., ge=1)` or `Mode` enum values)
# - Not testing the error raised when calling `are_spatial_dims_maintained` in
# `convert_mode`, since it has its own test
# - Not testing change of channels in `convert_mode`, since it relies on
# `_validate_channel_conversion` which has its own tests


class TestAxesValidation:
    """Tests for the `axes_valid` field validator."""

    # TODO make global fixture? See array_shape for other use of many axes strings

    ORDERED_AXES = (
        "YX",
        "CYX",
        "TYX",
        "SYX",
        "SCYX",
        "STYX",
        "TCYX",
        "SCTYX",
        "ZYX",
        "CZYX",
        "TZYX",
        "SZYX",
        "SCZYX",
        "STZYX",
        "TCZYX",
        "SCTZYX",
    )

    DISORDERED_AXES = (
        "XY",
        "YXC",
        "YXZ",
        "TSYX",
        "YXCZ",
        "TCYXZS",
    )

    CZI_AXES = (
        "SCYX",
        "SCZYX",
        "SCTYX",
    )

    DISALLOWED_AXES = (
        "X",  # too few
        "ZT",  # no YX
        "YYX",  # duplicates
        "YXm",  # invalid char
        "YZX",  # non-consecutive XY
    )

    DISALLOWED_CZI_AXES = (
        "CZYX",
        "TCYX",
        "CYX",
        "SZYX",
        "STCZYX",
        "SZYXC",
        "TCZYX",
        "TCYXZ",
    )

    @pytest.mark.parametrize("axes", ORDERED_AXES + DISORDERED_AXES)
    def test_valid_axes(self, minimum_train_data_cfg, axes):
        """Test that valid axes are accepted."""
        cfg_dict = minimum_train_data_cfg(
            axes=axes,
            patching={
                "name": "stratified",
                "patch_size": [16, 16] if "Z" not in axes else [8, 16, 16],
            },
        )
        cfg = NGDataConfig(**cfg_dict)

        assert cfg.axes == axes

    @pytest.mark.parametrize("axes", DISALLOWED_AXES)
    def test_invalid_axes(self, minimum_train_data_cfg, axes):
        cfg_dict = minimum_train_data_cfg(
            axes=axes,
            patching={
                "name": "stratified",
                "patch_size": [16, 16] if "Z" not in axes else [8, 16, 16],
            },
        )

        with pytest.raises(ValueError, match="Invalid axes"):
            NGDataConfig(**cfg_dict)

    @pytest.mark.parametrize("axes", CZI_AXES)
    def test_czi_valid_axes(self, minimum_train_data_cfg, axes):
        cfg_dict = minimum_train_data_cfg(
            data_type="czi",
            axes=axes,
            patching={
                "name": "stratified",
                "patch_size": (
                    [16, 16] if "Z" not in axes and "T" not in axes else [8, 16, 16]
                ),
            },
        )
        cfg = NGDataConfig(**cfg_dict)

        assert cfg.axes == axes

    @pytest.mark.parametrize("axes", DISALLOWED_CZI_AXES)
    def test_czi_invalid_axes(self, minimum_train_data_cfg, axes):
        cfg_dict = minimum_train_data_cfg(
            data_type="czi",
            axes=axes,
            patching={
                "name": "stratified",
                "patch_size": (
                    [16, 16] if "Z" not in axes and "T" not in axes else [8, 16, 16]
                ),
            },
        )

        with pytest.raises(ValueError, match="Invalid axes"):
            NGDataConfig(**cfg_dict)


class TestDefaultInMemory:
    """Tests for the `default_in_memory` factory function.

    This method is used to set default `in_memory` field depending on `data_type`.
    """

    @pytest.mark.parametrize("data_type", ["array", "tiff", "custom", None])
    def test_default_true(self, data_type):
        assert default_in_memory({"data_type": data_type})

    @pytest.mark.parametrize("data_type", ["zarr", "czi"])
    def test_default_false(self, data_type):
        assert not default_in_memory({"data_type": data_type})


class TestInMemoryValidation:
    """Tests for validation of `in_memory` parameter."""

    def test_array_must_be_in_memory(self, minimum_train_data_cfg):
        cfg = minimum_train_data_cfg(data_type="array", in_memory=False)
        with pytest.raises(ValueError, match="`in_memory` must be True"):
            NGDataConfig(**cfg)

    @pytest.mark.parametrize("data_type", ["zarr", "czi"])
    def test_zarr_czi_cannot_be_in_memory(self, minimum_train_data_cfg, data_type):
        cfg = minimum_train_data_cfg(data_type=data_type, in_memory=True)
        with pytest.raises(ValueError, match="`in_memory` not supported"):
            NGDataConfig(**cfg)

    @pytest.mark.parametrize("data_type", ["tiff", "custom"])
    @pytest.mark.parametrize("in_memory", [True, False])
    def test_tiff_custom_both_modes(self, minimum_train_data_cfg, data_type, in_memory):
        cfg_dict = minimum_train_data_cfg(
            data_type=data_type,
            in_memory=in_memory,
        )
        cfg = NGDataConfig(**cfg_dict)
        assert cfg.in_memory == in_memory


class TestChannelsValidation:
    """Tests for validate_channels."""

    @pytest.mark.parametrize(
        "channels, axes",
        [
            ([1], "YX"),
            ("1, 2", "CYX"),
            ([1, 3.14], "CYX"),
            ([1, -1], "CYX"),
            ([0, 1, 0], "CYX"),
        ],
    )
    def test_validation_errors(self, minimum_train_data_cfg, channels, axes):
        cfg_dict = minimum_train_data_cfg(axes=axes, channels=channels)
        with pytest.raises(ValueError, match="Channels must"):
            NGDataConfig(**cfg_dict)

    @pytest.mark.parametrize(
        "channels, coerced_channels",
        [
            (1, [1]),
            ([], None),
            ((), None),
        ],
    )
    def test_coercision(self, minimum_train_data_cfg, channels, coerced_channels):
        cfg_dict = minimum_train_data_cfg(axes="CYX", channels=channels)
        cfg = NGDataConfig(**cfg_dict)
        assert cfg.channels == coerced_channels

    @pytest.mark.parametrize(
        "channels, axes",
        [
            (None, "YX"),
            (None, "CYX"),
            ([1], "CYX"),
            ([0, 2], "CYX"),
        ],
    )
    def test_valid_channels(self, minimum_train_data_cfg, channels, axes):
        cfg_dict = minimum_train_data_cfg(axes=axes, channels=channels)
        cfg = NGDataConfig(**cfg_dict)
        assert cfg.channels == channels


@pytest.mark.parametrize(
    "strategy, mode, valid",
    [
        # valid
        ("stratified", "training", True),
        ("random", "training", True),
        ("fixed_random", "validating", True),
        ("whole", "predicting", True),
        ("tiled", "predicting", True),
        # invalid
        ("stratified", "validating", False),
        ("stratified", "predicting", False),
        ("random", "validating", False),
        ("random", "predicting", False),
        ("fixed_random", "training", False),
        ("fixed_random", "predicting", False),
        ("whole", "training", False),
        ("whole", "validating", False),
        ("tiled", "training", False),
        ("tiled", "validating", False),
    ],
)
def test_patching_strategy_vs_mode(
    minimum_train_data_cfg, patching_config, strategy, mode, valid
):
    """Test the validation of patching strategies against mode."""
    cfg_dict = minimum_train_data_cfg(
        patching=patching_config(strategy, is_3D=False), mode=mode
    )
    if valid:
        cfg = NGDataConfig(**cfg_dict)
        assert cfg.patching.name == strategy
    else:
        with pytest.raises(ValueError, match="is not compatible with mode"):
            NGDataConfig(**cfg_dict)


@pytest.mark.parametrize("name", ["shannon", "mean_std", "max"])
def test_patch_filter_vs_mode(minimum_mode_data_cfg, patch_filter_config, name):
    """Test that patch filters are only allowed in training mode."""
    cfg_dict = minimum_mode_data_cfg(
        mode="training", patch_filter=patch_filter_config(name)
    )

    # test that it is valid during training
    cfg = NGDataConfig(**cfg_dict)
    assert cfg.patch_filter is not None
    assert cfg.patch_filter.name == name

    # test that it raises an error during validation and prediction
    for mode in ["validating", "predicting"]:
        cfg_dict = minimum_mode_data_cfg(
            mode=mode, patch_filter=patch_filter_config(name)
        )
        with pytest.raises(ValueError, match="only allowed in 'training' mode"):
            NGDataConfig(**cfg_dict)


def test_coords_filter_vs_mode(minimum_mode_data_cfg, coord_filter_config):
    """Test that coordinate filters are only allowed in training mode."""
    cfg_dict = minimum_mode_data_cfg(
        mode="training", coord_filter=coord_filter_config()
    )

    # test that it is valid during training
    cfg = NGDataConfig(**cfg_dict)
    assert cfg.coord_filter is not None
    assert cfg.coord_filter.name == "mask"

    # test that it raises an error during validation and prediction
    for mode in ["validating", "predicting"]:
        cfg_dict = minimum_mode_data_cfg(mode=mode, coord_filter=coord_filter_config())
        with pytest.raises(ValueError, match="only allowed in 'training' mode"):
            NGDataConfig(**cfg_dict)


class TestDataloaderParams:
    """Tests for dataloader parameters validator."""

    @pytest.fixture
    def mock_cuda(self, mocker) -> Callable:
        """Fixture used to mock `torch.cuda.is_available`."""

        def _mock(cuda_available: bool):
            # mock torch.cuda.is_available
            fake_is_available = mocker.Mock(return_value=cuda_available)

            fake_cuda = types.SimpleNamespace(is_available=fake_is_available)
            fake_torch = types.ModuleType("torch")
            fake_torch.cuda = fake_cuda

            mocker.patch.dict(sys.modules, {"torch": fake_torch})

        return _mock

    @pytest.mark.parametrize("shuffle", [True, False])
    def test_valid_shuffle(self, minimum_train_data_cfg, shuffle):
        """Test valid `shuffle` values in `train_dataloader_params`."""
        NGDataConfig(
            **minimum_train_data_cfg(train_dataloader_params={"shuffle": shuffle})
        )

    def test_shuffle_validation_error(self, minimum_train_data_cfg):
        """Test that the absence of `shuffle` in `train_dataloader_params` raises a
        validation error."""
        with pytest.raises(ValueError, match="`train_dataloader_params` must include"):
            NGDataConfig(**minimum_train_data_cfg(train_dataloader_params={}))

    def test_shuffle_warning(self, minimum_train_data_cfg):
        """Test that a warning is raised with `shuffle=False`."""
        with pytest.warns(match="`train_dataloader_params` includes `shuffle=False`"):
            NGDataConfig(
                **minimum_train_data_cfg(train_dataloader_params={"shuffle": False})
            )

    @pytest.mark.parametrize(
        "dataloader", ["train_dataloader_params", "val_dataloader_params"]
    )
    @pytest.mark.parametrize("cuda_available", [True, False])
    def test_pin_memory_added(
        self, minimum_train_data_cfg, mock_cuda, dataloader, cuda_available
    ):
        """Test that `pin_memory` is set to `torch.cuda.is_available()` in training
        and validation dataloader parameters."""
        mock_cuda(cuda_available)

        # create config
        cfg_dict = minimum_train_data_cfg(
            **{dataloader: {"shuffle": True}}  # passes for both validation and train
        )
        cfg = NGDataConfig(**cfg_dict)
        assert "pin_memory" in getattr(cfg, dataloader)

    def test_pin_memory_not_overridden(self, minimum_train_data_cfg, mock_cuda):
        """Test that `pin_memory` is not overridden if already set in dataloader
        params."""
        mock_cuda(False)

        cfg_dict = minimum_train_data_cfg(
            train_dataloader_params={"shuffle": True, "pin_memory": True},
            val_dataloader_params={"num_workers": 2, "pin_memory": True},
        )
        cfg = NGDataConfig(**cfg_dict)
        assert cfg.train_dataloader_params["pin_memory"] is True
        assert cfg.val_dataloader_params["pin_memory"] is True

    def test_val_dataloader_num_workers_set_to_train(self, minimum_train_data_cfg):
        """Test that `num_workers` in `val_dataloader_params` is set to match
        `train_dataloader_params` if not explicitly set."""
        # set to train
        cfg_dict = minimum_train_data_cfg(
            train_dataloader_params={"shuffle": True, "num_workers": 3},
        )
        cfg = NGDataConfig(**cfg_dict)
        assert cfg.val_dataloader_params["num_workers"] == 3

    def test_val_dataloader_num_workers_not_overwritten(self, minimum_train_data_cfg):
        """Test that `num_workers` in `val_dataloader_params` is not overwritten
        if explicitly set."""
        # set to train
        cfg_dict = minimum_train_data_cfg(
            train_dataloader_params={"shuffle": True, "num_workers": 3},
            val_dataloader_params={"num_workers": 5},
        )
        cfg = NGDataConfig(**cfg_dict)
        assert cfg.val_dataloader_params["num_workers"] == 5


@pytest.mark.parametrize(
    "data_type, axes, is_3D, error",
    [
        # valid cases
        ("array", "YX", False, False),
        ("array", "ZYX", True, False),
        ("czi", "SCYX", False, False),
        ("czi", "SCZYX", True, False),
        ("czi", "SCTYX", True, False),
        # invalid cases
        ("array", "YX", True, True),
        ("array", "ZYX", False, True),
        ("czi", "SCYX", True, True),
        ("czi", "SCZYX", False, True),
        ("czi", "SCTYX", False, True),
    ],
)
def test_validate_dimensions(
    minimum_train_data_cfg, patching_config, data_type, axes, is_3D, error
):
    """Test the `validate_dimensions` model validator."""
    cfg = minimum_train_data_cfg(
        data_type=data_type,
        axes=axes,
        patching=patching_config("stratified", is_3D=is_3D),
    )
    if error:
        with pytest.raises(ValueError, match="`patch_size` in `patching` must have"):
            NGDataConfig(**cfg)
    else:
        NGDataConfig(**cfg)


class TestSeed:
    """Tests for seed and seed propagation model validators."""

    def test_seed(self, minimum_train_data_cfg):
        """Test that the seed default factory generates different seeds."""
        seeds = [NGDataConfig(**minimum_train_data_cfg()).seed for _ in range(5)]
        assert len(set(seeds)) == len(seeds)

    def test_seed_propagation(self, minimum_train_data_cfg):
        """Test seed propagation to filters, augmentations and patching."""
        cfg = minimum_train_data_cfg(
            seed=42,
            patch_filter={"name": "shannon", "threshold": 0.5},
            coord_filter={"name": "mask"},
        )
        cfg = NGDataConfig(**cfg)
        assert cfg.seed == 42
        assert cfg.patch_filter.seed == 42
        assert cfg.coord_filter.seed == 42
        assert cfg.patching.seed == 42
        for aug in cfg.transforms:
            assert aug.seed == 42


@pytest.mark.parametrize(
    "data_type, axes, is_3d",
    [
        ("array", "YX", False),
        ("array", "ZYX", True),
        ("czi", "SCYX", False),
        ("czi", "SCZYX", True),
        ("czi", "SCTYX", True),
    ],
)
def test_is_3d(minimum_train_data_cfg, patching_config, data_type, axes, is_3d):
    """Test the `is_3D` method."""
    cfg_dict = minimum_train_data_cfg(
        data_type=data_type,
        axes=axes,
        patching=patching_config("stratified", is_3D=is_3d),  # for a valid config
    )
    cfg = NGDataConfig(**cfg_dict)
    assert cfg.is_3D() == is_3d


@pytest.mark.parametrize(
    "old_data_type, old_axes, new_data_type, new_axes, expected",
    [
        # maintaining spatial dimensions
        ("array", "YX", "array", "YX", True),
        ("array", "ZYX", "array", "ZYX", True),
        ("czi", "SCYX", "czi", "SCYX", True),
        ("czi", "SCZYX", "czi", "SCZYX", True),
        ("czi", "SCTYX", "czi", "SCTYX", True),
        ("czi", "SCYX", "array", "YX", True),
        ("czi", "SCZYX", "array", "ZYX", True),
        ("czi", "SCTYX", "array", "ZYX", True),
        ("array", "YX", "czi", "SCYX", True),
        ("array", "ZYX", "czi", "SCZYX", True),
        ("array", "ZYX", "czi", "SCTYX", True),
        # not maintaining spatial dimensions
        ("array", "YX", "array", "ZYX", False),
        ("array", "ZYX", "array", "YX", False),
        ("czi", "SCYX", "czi", "SCZYX", False),
        ("czi", "SCYX", "czi", "SCTYX", False),
        ("czi", "SCZYX", "czi", "SCYX", False),
        ("czi", "SCTYX", "czi", "SCYX", False),
        ("czi", "SCZYX", "czi", "SCTYX", False),
        ("czi", "SCTYX", "czi", "SCZYX", False),
        ("czi", "SCYX", "array", "ZYX", False),
        ("czi", "SCZYX", "array", "YX", False),
        ("czi", "SCTYX", "array", "YX", False),
        ("array", "YX", "czi", "ZYX", False),
        ("array", "YX", "czi", "TYX", False),
        ("array", "ZYX", "czi", "SCYX", False),
    ],
)
def test_are_spatial_dims_maintained(
    old_data_type, old_axes, new_data_type, new_axes, expected
):
    """Test the output of `are_spatial_dims_maintained`."""
    assert (
        _are_spatial_dims_maintained(old_data_type, old_axes, new_data_type, new_axes)
        == expected
    )


@pytest.mark.parametrize(
    "old_axes, old_channels, new_axes, new_channels, raise_error",
    [
        # No error: no change or we just cannot tell
        ("CYX", None, "CYX", None, False),
        ("CYX", None, "CYX", [1], False),
        ("CYX", None, "CYX", [0, 2], False),
        ("CYX", [1], "CYX", None, False),
        ("CYX", [1], "CYX", [1], False),
        ("CYX", [0, 2], "CYX", None, False),
        ("CYX", [0, 2], "CYX", [0, 2], False),
        # Error: different number of channels
        ("CYX", [1], "CYX", [0, 2], True),
        ("CYX", [0, 2], "CYX", [1], True),
        # removing C axis
        ("CYX", None, "YX", None, False),
        ("CYX", [1], "YX", None, False),
        ("CYX", [0, 2], "YX", None, True),
        # adding C axis
        ("YX", None, "CYX", None, False),
        ("YX", None, "CYX", [1], False),
        ("YX", None, "CYX", [0, 2], True),
    ],
)
def test_validate_channel_conversion(
    old_axes, old_channels, new_axes, new_channels, raise_error
):
    """Test channel conversion validation."""
    if raise_error:
        with pytest.raises(ValueError, match="Cannot switch"):
            _validate_channel_conversion(old_axes, old_channels, new_axes, new_channels)
    else:
        # should not raise an error
        _validate_channel_conversion(old_axes, old_channels, new_axes, new_channels)


class TestConvertMode:
    """Tests for the `convert_mode` method."""

    @pytest.mark.parametrize("mode", ["validating", "predicting"])
    def test_convert_mode(self, minimum_train_data_cfg, mode):
        """Test that `convert_mode` correctly converts the mode."""
        cfg_dict = minimum_train_data_cfg(
            mode="training",
            batch_size=5,
            data_type="tiff",
            axes="SCYX",
            channels=[0, 2],
            in_memory=True,
            val_dataloader_params={"shuffle": False, "num_workers": 3},
            pred_dataloader_params={"num_workers": 2},
        )
        cfg = NGDataConfig(**cfg_dict)

        converted_cfg = cfg.convert_mode(mode)
        assert converted_cfg.mode == mode
        assert converted_cfg.batch_size == cfg.batch_size
        assert converted_cfg.data_type == cfg.data_type
        assert converted_cfg.axes == cfg.axes
        assert converted_cfg.channels == cfg.channels
        assert converted_cfg.in_memory == cfg.in_memory
        assert converted_cfg.val_dataloader_params == cfg.val_dataloader_params
        assert converted_cfg.pred_dataloader_params == cfg.pred_dataloader_params

    @pytest.mark.parametrize("mode", ["validating", "predicting"])
    def test_convert_mode_parameter_update(self, minimum_train_data_cfg, mode):
        """Test that `convert_mode` correctly updates parameters when specified."""
        cfg_dict = minimum_train_data_cfg(
            mode="training",
            batch_size=5,
            data_type="tiff",
            axes="SCYX",
            channels=[0, 2],
            in_memory=False,
            val_dataloader_params={"shuffle": False, "num_workers": 3},
            pred_dataloader_params={"num_workers": 2},
        )
        new_batch_size = 3
        new_axes = "CYX"
        new_data_type = "array"
        new_channels = [1, 5]
        new_in_memory = True
        new_dataloader_params = {"num_workers": 1}

        cfg = NGDataConfig(**cfg_dict)
        converted_cfg = cfg.convert_mode(
            mode,
            new_batch_size=new_batch_size,
            new_axes=new_axes,
            new_data_type=new_data_type,
            new_channels=new_channels,
            new_in_memory=new_in_memory,
            new_dataloader_params=new_dataloader_params,
        )
        assert converted_cfg.mode == mode
        assert converted_cfg.batch_size == new_batch_size
        assert converted_cfg.data_type == new_data_type
        assert converted_cfg.axes == new_axes
        assert converted_cfg.channels == new_channels
        assert converted_cfg.in_memory == new_in_memory
        if mode == "validating":
            assert converted_cfg.val_dataloader_params == new_dataloader_params
            assert converted_cfg.pred_dataloader_params == cfg.pred_dataloader_params
        elif mode == "predicting":
            assert converted_cfg.pred_dataloader_params == new_dataloader_params
            assert converted_cfg.val_dataloader_params == cfg.val_dataloader_params

    def test_normalization_conservation(self, minimum_train_data_cfg):
        """Test that normalization is conserved when converting mode."""
        cfg_dict = minimum_train_data_cfg(
            mode="training",
            normalization={
                "name": "mean_std",
                "input_means": [0.5],
                "input_stds": [0.2],
                "target_means": [0.3],
                "target_stds": [0.1],
            },
        )
        cfg = NGDataConfig(**cfg_dict)
        converted_cfg = cfg.convert_mode("validating")
        assert converted_cfg.normalization == cfg.normalization

    def test_cannot_convert_to_training(self, minimum_train_data_cfg):
        """Test that converting to 'training' mode raises an error."""
        cfg_dict = minimum_train_data_cfg(mode="training")
        cfg = NGDataConfig(**cfg_dict)

        with pytest.raises(
            ValueError, match="Conversion to 'training' mode is not supported"
        ):
            cfg.convert_mode("training")

    @pytest.mark.parametrize("mode", ["validating", "predicting"])
    def test_cannot_convert_from_val_and_pred(self, minimum_mode_data_cfg, mode):
        """Test that converting from 'validating' or 'predicting' mode raises an
        error."""
        cfg_dict = minimum_mode_data_cfg(mode=mode)
        cfg = NGDataConfig(**cfg_dict)

        with pytest.raises(
            ValueError, match="Only conversion from 'training' mode is supported"
        ):
            cfg.convert_mode("training")

    @pytest.mark.parametrize(
        "old_channels, new_channels, expected",
        [
            (None, None, None),
            (None, [1], [1]),
            (None, "all", None),
            ([0], None, [0]),
            ([0], [1], [1]),
            ([0], "all", None),
        ],
    )
    def test_channels_conversion(
        self, minimum_train_data_cfg, old_channels, new_channels, expected
    ):
        """Test that `convert_mode` correctly converts the `channels` field.

        Note that this only test that the `channels` ambiguity is correctly handled
        by using `channels=None` has using the old `channels`, and `channels="all"` as
        signifying that we want to keep all channels (final value `None`).
        """
        cfg_dict = minimum_train_data_cfg(
            mode="training", axes="CYX", channels=old_channels
        )
        cfg = NGDataConfig(**cfg_dict)

        converted_cfg = cfg.convert_mode("validating", new_channels=new_channels)
        assert converted_cfg.channels == expected

    def test_patching_tiling_error(self, minimum_train_data_cfg):
        """Test error converting to prediction while passing `None` overlap."""
        cfg_dict = minimum_train_data_cfg(
            mode="training",
        )
        cfg = NGDataConfig(**cfg_dict)
        with pytest.raises(
            ValueError, match="`overlap_size` parameter must be specified"
        ):
            cfg.convert_mode("predicting", new_patch_size=[128, 128], overlap_size=None)

    @pytest.mark.parametrize(
        "mode, params, expected_strategy",
        [
            (
                "predicting",
                {"new_patch_size": [128, 128], "overlap_size": [48, 48]},
                "tiled",
            ),
            ("predicting", {}, "whole"),
            ("validating", {}, "fixed_random"),
        ],
    )
    def test_patching_prediction(
        self, minimum_train_data_cfg, mode, params, expected_strategy
    ):
        """Test converting to prediction while passing tiling parameters."""
        cfg_dict = minimum_train_data_cfg(
            mode="training",
        )
        cfg = NGDataConfig(**cfg_dict)
        cfg = cfg.convert_mode(mode, **params)
        assert cfg.patching.name == expected_strategy

    @pytest.mark.parametrize("mode", ["validating", "predicting"])
    def test_patch_coord_filters_removal(
        self, minimum_train_data_cfg, patch_filter_config, coord_filter_config, mode
    ):
        """Test that patch and coordinate filters are removed upon conversion."""
        cfg_dict = minimum_train_data_cfg(
            mode="training",
            patch_filter=patch_filter_config("shannon"),
            coord_filter=coord_filter_config(),
        )
        cfg = NGDataConfig(**cfg_dict)
        converted_cfg = cfg.convert_mode(mode)
        assert converted_cfg.patch_filter is None
        assert converted_cfg.coord_filter is None
