"""Unit tests for the NGDataConfig Pydantic model."""

import itertools
import sys
from collections.abc import Callable
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest

from careamics.config.data.ng_data_config import (
    NGDataConfig,
    _are_spatial_dims_maintained,
    _validate_channel_conversion,
    default_in_memory,
)
from tests.utils import (
    DEFAULT_MODE,
    DEFAULT_PATCHING,
    coord_filter_dict_testing,
    ng_data_config_dict_testing,
    patch_filter_dict_testing,
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

# -- Stages
TRAINING = "training"
VALIDATING = "validating"
PREDICTING = "predicting"
NON_TRAINING_STAGES = [VALIDATING, PREDICTING]

# -- Patch filters
PATCH_FILTERS = ["shannon", "max", "mean_std"]
COORD_FILTERS = ["mask"]

# -- Patching strategies
TRAIN_PATCHING = ["stratified", "random"]
VAL_PATCHING = ["fixed_random"]
PRED_PATCHING = ["tiled", "whole"]

# -- Axes
AXES_WO_CHANNELS_2D: tuple[str, ...] = (
    "YX",
    "TYX",
    "SYX",
    "STYX",
    # disordered
    "XY",
    "TSYX",
)

AXES_W_CHANNELS_2D: tuple[str, ...] = (
    "CYX",
    "SCYX",
    "TCYX",
    # disordered
    "YXC",
    "SCTYX",
)

AXES_WO_CHANNELS_3D: tuple[str, ...] = (
    "ZYX",
    "TZYX",
    "SZYX",
    # disordered
    "YXZ",
)

AXES_W_CHANNELS_3D: tuple[str, ...] = (
    "CZYX",
    "TCZYX",
    "SCZYX",
    "STZYX",
    "TCZYX",
    # disordered
    "YXCZ",
    "TCYXZS",
    "SCTZYX",
)

AXES_DISALLOWED: tuple[str, ...] = (
    "X",  # too few
    "ZT",  # no YX
    "YYX",  # duplicates
    "YXm",  # invalid char
    "YZX",  # non-consecutive XY
)

AXES_CZI_2D = ("SCYX",)

AXES_CZI_3D_Z = ("SCZYX",)
AXES_CZI_3D_T = ("SCTYX",)
AXES_CZI_3D = AXES_CZI_3D_Z + AXES_CZI_3D_T

AXES_CZI_DISALLOWED = (
    "CZYX",
    "TCYX",
    "CYX",
    "SZYX",
    "STCZYX",
    "SZYXC",
    "TCZYX",
    "TCYXZ",
)


# ------------------------ Test utilities --------------------------


def test_default_data_config():
    """Test that the default NGDataConfig can be created."""
    ng_data_config_dict = ng_data_config_dict_testing()
    NGDataConfig(**ng_data_config_dict)


# -------------------------- Unit tests ----------------------------


@pytest.mark.parametrize(
    "data_type, axes, expected_error",
    list(
        itertools.product(
            ["array"],
            AXES_WO_CHANNELS_2D + AXES_W_CHANNELS_2D,
            [nullcontext(0)],
        )
    )
    + list(
        itertools.product(
            ["array"],
            AXES_WO_CHANNELS_3D + AXES_W_CHANNELS_3D,
            [nullcontext(0)],
        )
    )
    + list(
        itertools.product(
            ["array"],
            AXES_DISALLOWED,
            [pytest.raises(ValueError, match="Invalid axes")],
        )
    )
    + list(
        itertools.product(
            ["czi"],
            AXES_CZI_2D,
            [nullcontext(0)],
        )
    )
    + list(
        itertools.product(
            ["czi"],
            AXES_CZI_3D,
            [nullcontext(0)],
        )
    )
    + list(
        itertools.product(
            ["czi"],
            AXES_CZI_DISALLOWED,
            [pytest.raises(ValueError, match="Invalid axes")],
        )
    ),
)
def test_axes(data_type, axes, expected_error):
    """Test that valid axes are accepted."""
    cfg_dict = ng_data_config_dict_testing(
        data_type=data_type,
        axes=axes,
    )

    with expected_error:
        cfg = NGDataConfig(**cfg_dict)
        assert cfg.axes == axes


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

    def test_array_must_be_in_memory(self):
        cfg = ng_data_config_dict_testing(data_type="array", in_memory=False)
        with pytest.raises(ValueError, match="`in_memory` must be True"):
            NGDataConfig(**cfg)

    @pytest.mark.parametrize("data_type", ["zarr", "czi"])
    def test_zarr_czi_cannot_be_in_memory(self, data_type):
        cfg = ng_data_config_dict_testing(data_type=data_type, in_memory=True)
        with pytest.raises(ValueError, match="`in_memory` not supported"):
            NGDataConfig(**cfg)

    @pytest.mark.parametrize("data_type", ["tiff", "custom"])
    @pytest.mark.parametrize("in_memory", [True, False])
    def test_tiff_custom_both_modes(self, data_type, in_memory):
        cfg_dict = ng_data_config_dict_testing(
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
    def test_validation_errors(self, channels, axes):
        cfg_dict = ng_data_config_dict_testing(axes=axes, channels=channels)
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
    def test_coercision(self, channels, coerced_channels):
        cfg_dict = ng_data_config_dict_testing(axes="CYX", channels=channels)
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
    def test_valid_channels(self, channels, axes):
        cfg_dict = ng_data_config_dict_testing(axes=axes, channels=channels)
        cfg = NGDataConfig(**cfg_dict)
        assert cfg.channels == channels


@pytest.mark.parametrize(
    "mode, patching, expectation",
    # valid
    list(itertools.product(["training"], TRAIN_PATCHING, [nullcontext(0)]))
    + list(itertools.product(["validating"], VAL_PATCHING, [nullcontext(0)]))
    + list(itertools.product(["predicting"], PRED_PATCHING, [nullcontext(0)]))
    # invalid
    + list(
        itertools.product(
            ["training"],
            VAL_PATCHING + PRED_PATCHING,
            [pytest.raises(ValueError, match="Patching strategy ")],
        )
    )
    + list(
        itertools.product(
            ["validating"],
            TRAIN_PATCHING + PRED_PATCHING,
            [pytest.raises(ValueError, match="Patching strategy ")],
        )
    )
    + list(
        itertools.product(
            ["predicting"],
            TRAIN_PATCHING + VAL_PATCHING,
            [pytest.raises(ValueError, match="Patching strategy ")],
        )
    ),
)
def test_validate_patching_mode(mode: str, patching: str, expectation):
    ng_data_config_dict = ng_data_config_dict_testing(mode=mode, patching=patching)
    with expectation:
        cfg = NGDataConfig(**ng_data_config_dict)
        assert cfg.mode == ng_data_config_dict["mode"]
        assert cfg.patching.name == ng_data_config_dict["patching"]["name"]


@pytest.mark.parametrize(
    "stage, filter_name, expected_error",
    # valid
    list(itertools.product([TRAINING], PATCH_FILTERS, [nullcontext(0)]))
    # invalid
    + list(
        itertools.product(
            NON_TRAINING_STAGES,
            PATCH_FILTERS,
            [pytest.raises(ValueError, match="only allowed in 'training' mode")],
        )
    ),
)
def test_patch_filter_vs_mode(stage, filter_name, expected_error):
    """Test that patch filters are only allowed in training mode."""
    cfg_dict = ng_data_config_dict_testing(
        mode=stage, patch_filter=patch_filter_dict_testing(filter_name)
    )

    with expected_error:
        cfg = NGDataConfig(**cfg_dict)
        assert cfg.patch_filter is not None
        assert cfg.patch_filter.name == filter_name


@pytest.mark.parametrize(
    "stage, filter_name, expected_error",
    # valid
    list(itertools.product([TRAINING], COORD_FILTERS, [nullcontext(0)]))
    # invalid
    + list(
        itertools.product(
            NON_TRAINING_STAGES,
            COORD_FILTERS,
            [pytest.raises(ValueError, match="only allowed in 'training' mode")],
        )
    ),
)
def test_coords_filter_vs_mode(stage, filter_name, expected_error):
    """Test that coordinate filters are only allowed in training mode."""
    cfg_dict = ng_data_config_dict_testing(
        mode=stage, coord_filter=coord_filter_dict_testing(filter_name)
    )

    with expected_error:
        cfg = NGDataConfig(**cfg_dict)
        assert cfg.coord_filter is not None
        assert cfg.coord_filter.name == filter_name


class TestDataloaderParams:
    """Tests for dataloader parameters validator."""

    @pytest.fixture
    def mock_cuda(self, mocker) -> Callable:
        """Fixture used to mock `torch.cuda.is_available`."""

        def _mock(cuda_available: bool):
            # mock torch.cuda.is_available
            fake_is_available = mocker.Mock(return_value=cuda_available)

            fake_cuda = SimpleNamespace(is_available=fake_is_available)
            fake_torch = ModuleType("torch")
            fake_torch.cuda = fake_cuda

            mocker.patch.dict(sys.modules, {"torch": fake_torch})

        return _mock

    @pytest.mark.parametrize("shuffle", [True, False])
    def test_valid_shuffle(self, shuffle):
        """Test valid `shuffle` values in `train_dataloader_params`."""
        NGDataConfig(
            **ng_data_config_dict_testing(train_dataloader_params={"shuffle": shuffle})
        )

    def test_shuffle_validation_error(self):
        """Test that the absence of `shuffle` in `train_dataloader_params` raises a
        validation error."""
        with pytest.raises(ValueError, match="`train_dataloader_params` must include"):
            NGDataConfig(**ng_data_config_dict_testing(train_dataloader_params={}))

    def test_shuffle_warning(self):
        """Test that a warning is raised with `shuffle=False`."""
        with pytest.warns(match="`train_dataloader_params` includes `shuffle=False`"):
            NGDataConfig(
                **ng_data_config_dict_testing(
                    train_dataloader_params={"shuffle": False}
                )
            )

    @pytest.mark.parametrize(
        "dataloader", ["train_dataloader_params", "val_dataloader_params"]
    )
    @pytest.mark.parametrize("cuda_available", [True, False])
    def test_pin_memory_added(self, mock_cuda, dataloader, cuda_available):
        """Test that `pin_memory` is set to `torch.cuda.is_available()` in training
        and validation dataloader parameters."""
        mock_cuda(cuda_available)

        # create config
        cfg_dict = ng_data_config_dict_testing(
            **{dataloader: {"shuffle": True}}  # passes for both validation and train
        )
        cfg = NGDataConfig(**cfg_dict)
        assert "pin_memory" in getattr(cfg, dataloader)

    def test_pin_memory_not_overridden(self, mock_cuda):
        """Test that `pin_memory` is not overridden if already set in dataloader
        params."""
        mock_cuda(False)

        cfg_dict = ng_data_config_dict_testing(
            train_dataloader_params={"shuffle": True, "pin_memory": True},
            val_dataloader_params={"num_workers": 2, "pin_memory": True},
        )
        cfg = NGDataConfig(**cfg_dict)
        assert cfg.train_dataloader_params["pin_memory"] is True
        assert cfg.val_dataloader_params["pin_memory"] is True

    def test_val_dataloader_num_workers_set_to_train(self):
        """Test that `num_workers` in `val_dataloader_params` is set to match
        `train_dataloader_params` if not explicitly set."""
        # set to train
        cfg_dict = ng_data_config_dict_testing(
            train_dataloader_params={"shuffle": True, "num_workers": 3},
        )
        cfg = NGDataConfig(**cfg_dict)
        assert cfg.val_dataloader_params["num_workers"] == 3

    def test_val_dataloader_num_workers_not_overwritten(self):
        """Test that `num_workers` in `val_dataloader_params` is not overwritten
        if explicitly set."""
        # set to train
        cfg_dict = ng_data_config_dict_testing(
            train_dataloader_params={"shuffle": True, "num_workers": 3},
            val_dataloader_params={"num_workers": 5},
        )
        cfg = NGDataConfig(**cfg_dict)
        assert cfg.val_dataloader_params["num_workers"] == 5


@pytest.mark.parametrize(
    "data_type, axes, patches, expected_error",
    # valid
    list(
        itertools.product(
            ["array"],
            AXES_W_CHANNELS_2D + AXES_WO_CHANNELS_2D,
            [(16, 16)],
            [nullcontext(0)],
        )
    )
    + list(
        itertools.product(
            ["array"],
            AXES_W_CHANNELS_3D + AXES_WO_CHANNELS_3D,
            [(16, 16, 16)],
            [nullcontext(0)],
        )
    )
    + list(itertools.product(["czi"], AXES_CZI_2D, [(16, 16)], [nullcontext(0)]))
    + list(itertools.product(["czi"], AXES_CZI_3D, [(16, 16, 16)], [nullcontext(0)]))
    # invalid
    + list(
        itertools.product(
            ["array"],
            AXES_W_CHANNELS_2D + AXES_WO_CHANNELS_2D,
            [(16, 16, 16)],
            [pytest.raises(ValueError, match="`patch_size` in `patching` must have")],
        )
    )
    + list(
        itertools.product(
            ["array"],
            AXES_W_CHANNELS_3D + AXES_WO_CHANNELS_3D,
            [(16, 16)],
            [pytest.raises(ValueError, match="`patch_size` in `patching` must have")],
        )
    )
    + list(
        itertools.product(
            ["czi"],
            AXES_CZI_2D,
            [(16, 16, 16)],
            [pytest.raises(ValueError, match="`patch_size` in `patching` must have")],
        )
    )
    + list(
        itertools.product(
            ["czi"],
            AXES_CZI_3D,
            [(16, 16)],
            [pytest.raises(ValueError, match="`patch_size` in `patching` must have")],
        )
    ),
)
def test_validate_dimensions(data_type, axes, patches, expected_error):
    """Test the `validate_dimensions` model validator comparing axes and patch sizes."""
    cfg_dict = ng_data_config_dict_testing(
        data_type=data_type,
        axes=axes,
        patching=DEFAULT_PATCHING,
        patch_size=patches,
    )
    with expected_error:
        NGDataConfig(**cfg_dict)


class TestSeed:
    """Tests for seed and seed propagation model validators."""

    def test_seed(self):
        """Test that the seed default factory generates different seeds."""
        seeds = [NGDataConfig(**ng_data_config_dict_testing()).seed for _ in range(5)]
        assert len(set(seeds)) == len(seeds)

    def test_seed_propagation(self):
        """Test seed propagation to filters, augmentations and patching."""
        cfg = ng_data_config_dict_testing(
            seed=42,
            patch_filter={"name": "shannon", "threshold": 0.5},
            coord_filter={"name": "mask"},
        )
        cfg = NGDataConfig(**cfg)
        assert cfg.seed == 42
        assert cfg.patch_filter.seed == 42
        assert cfg.coord_filter.seed == 42
        assert cfg.patching.seed == 42
        for aug in cfg.augmentations:
            assert aug.seed == 42


@pytest.mark.parametrize(
    "data_type, axes, is_3d",
    # no need to test with channels as the `is_3D` is not influenced by them
    list(itertools.product(["array"], AXES_WO_CHANNELS_2D, [False]))
    + list(itertools.product(["array"], AXES_WO_CHANNELS_3D, [True]))
    + list(itertools.product(["czi"], AXES_CZI_2D, [False]))
    + list(itertools.product(["czi"], AXES_CZI_3D, [True])),
)
def test_is_3d(data_type, axes, is_3d):
    """Test the `is_3D` method."""
    cfg_dict = ng_data_config_dict_testing(
        data_type=data_type,
        axes=axes,
        patching=DEFAULT_PATCHING,
        patch_size=(16, 16, 16) if is_3d else (16, 16),
    )
    cfg = NGDataConfig(**cfg_dict)
    assert cfg.is_3D() == is_3d


@pytest.mark.parametrize(
    "old_data_type, old_axes, new_data_type, new_axes, dims_maintained",
    # array -> array with same spatial dimensions
    list(
        itertools.product(
            ["array"], AXES_WO_CHANNELS_2D, ["array"], AXES_WO_CHANNELS_2D, [True]
        )
    )
    + list(
        itertools.product(
            ["array"], AXES_WO_CHANNELS_3D, ["array"], AXES_WO_CHANNELS_3D, [True]
        )
    )
    # czi -> czi with same spatial dimensions, T and Z can't be swapped
    + list(itertools.product(["czi"], AXES_CZI_2D, ["czi"], AXES_CZI_2D, [True]))
    + list(itertools.product(["czi"], AXES_CZI_3D_Z, ["czi"], AXES_CZI_3D_Z, [True]))
    + list(itertools.product(["czi"], AXES_CZI_3D_T, ["czi"], AXES_CZI_3D_T, [True]))
    # array <-> czi with same spatial dimensions
    + list(
        itertools.product(["array"], AXES_WO_CHANNELS_2D, ["czi"], AXES_CZI_2D, [True])
    )
    + list(
        itertools.product(["array"], AXES_WO_CHANNELS_3D, ["czi"], AXES_CZI_3D, [True])
    )
    + list(
        itertools.product(["czi"], AXES_CZI_2D, ["array"], AXES_WO_CHANNELS_2D, [True])
    )
    + list(
        itertools.product(["czi"], AXES_CZI_3D, ["array"], AXES_WO_CHANNELS_3D, [True])
    )
    # array -> array with different spatial dimensions
    + list(
        itertools.product(
            ["array"], AXES_WO_CHANNELS_2D, ["array"], AXES_WO_CHANNELS_3D, [False]
        )
    )
    + list(
        itertools.product(
            ["array"], AXES_WO_CHANNELS_3D, ["array"], AXES_WO_CHANNELS_2D, [False]
        )
    )
    # czi -> czi with different spatial dimensions
    + list(itertools.product(["czi"], AXES_CZI_2D, ["czi"], AXES_CZI_3D, [False]))
    + list(itertools.product(["czi"], AXES_CZI_3D, ["czi"], AXES_CZI_2D, [False]))
    # czi -> czi with different depth dimension
    + list(itertools.product(["czi"], AXES_CZI_3D_Z, ["czi"], AXES_CZI_3D_T, [False]))
    + list(itertools.product(["czi"], AXES_CZI_3D_T, ["czi"], AXES_CZI_3D_Z, [False]))
    # array <-> czi with different spatial dimensions
    + list(
        itertools.product(["array"], AXES_WO_CHANNELS_2D, ["czi"], AXES_CZI_3D, [False])
    )
    + list(
        itertools.product(["array"], AXES_WO_CHANNELS_3D, ["czi"], AXES_CZI_2D, [False])
    )
    + list(
        itertools.product(["czi"], AXES_CZI_2D, ["array"], AXES_WO_CHANNELS_3D, [False])
    )
    + list(
        itertools.product(["czi"], AXES_CZI_3D, ["array"], AXES_WO_CHANNELS_2D, [False])
    ),
)
def test_are_spatial_dims_maintained(
    old_data_type, old_axes, new_data_type, new_axes, dims_maintained
):
    """Test the output of `are_spatial_dims_maintained`."""
    assert (
        _are_spatial_dims_maintained(old_data_type, old_axes, new_data_type, new_axes)
        == dims_maintained
    )


@pytest.mark.parametrize(
    "old_axes, old_channels, new_axes, new_channels, expected_error",
    # No error: no change or we just cannot tell
    list(
        itertools.product(
            ["CYX"], [None], ["CYX"], [None, [1], [0, 2]], [nullcontext(0)]
        )
    )
    + list(itertools.product(["CYX"], [[1]], ["CYX"], [None, [1]], [nullcontext(0)]))
    + list(
        itertools.product(["CYX"], [[0, 2]], ["CYX"], [None, [0, 2]], [nullcontext(0)])
    )
    # Error: different number of channels
    + [
        ("CYX", [1], "CYX", [0, 2], pytest.raises(ValueError, match="Cannot switch")),
        ("CYX", [0, 2], "CYX", [1], pytest.raises(ValueError, match="Cannot switch")),
    ]
    # removing C axis, no error
    + list(itertools.product(["CYX"], [None, [1]], ["YX"], [None], [nullcontext(0)]))
    # removing C axis, with error
    + [
        ("CYX", [0, 2], "YX", None, pytest.raises(ValueError, match="Cannot switch")),
    ]
    # adding C axis, no error
    + list(itertools.product(["YX"], [None], ["CYX"], [None, [1]], [nullcontext(0)]))
    # adding C axis, with error
    + [
        ("YX", None, "CYX", [0, 2], pytest.raises(ValueError, match="Cannot switch")),
    ],
)
def test_validate_channel_conversion(
    old_axes, old_channels, new_axes, new_channels, expected_error
):
    """Test channel conversion validation."""
    with expected_error:
        _validate_channel_conversion(old_axes, old_channels, new_axes, new_channels)


class TestConvertMode:
    """Tests for the `convert_mode` method."""

    @pytest.mark.parametrize("new_mode", NON_TRAINING_STAGES)
    def test_convert_mode(self, new_mode):
        """Test that `convert_mode` correctly converts the mode."""
        cfg_dict = ng_data_config_dict_testing(
            mode=TRAINING,
            patching=DEFAULT_PATCHING,
            data_type="tiff",
            axes="SCYX",
            batch_size=5,
            channels=[0, 2],
            in_memory=True,
            val_dataloader_params={"shuffle": False, "num_workers": 3},
            pred_dataloader_params={"num_workers": 2},
        )
        cfg = NGDataConfig(**cfg_dict)

        converted_cfg = cfg.convert_mode(new_mode=new_mode)
        assert converted_cfg.mode == new_mode
        assert converted_cfg.batch_size == cfg.batch_size
        assert converted_cfg.data_type == cfg.data_type
        assert converted_cfg.axes == cfg.axes
        assert converted_cfg.channels == cfg.channels
        assert converted_cfg.in_memory == cfg.in_memory
        assert converted_cfg.val_dataloader_params == cfg.val_dataloader_params
        assert converted_cfg.pred_dataloader_params == cfg.pred_dataloader_params
        assert converted_cfg.patching.name != DEFAULT_PATCHING

    @pytest.mark.parametrize("mode", NON_TRAINING_STAGES)
    def test_convert_mode_parameter_update(self, mode):
        """Test that `convert_mode` correctly updates parameters when specified."""
        cfg_dict = ng_data_config_dict_testing(
            mode=TRAINING,
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
            new_mode=mode,
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

    def test_normalization_conservation(self):
        """Test that normalization is conserved when converting mode."""
        cfg_dict = ng_data_config_dict_testing(
            mode=TRAINING,
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

    def test_cannot_convert_to_training(self):
        """Test that converting to 'training' mode raises an error."""
        cfg_dict = ng_data_config_dict_testing(mode=DEFAULT_MODE)
        cfg = NGDataConfig(**cfg_dict)

        with pytest.raises(
            ValueError, match="Conversion to 'training' mode is not supported"
        ):
            cfg.convert_mode(TRAINING)

    @pytest.mark.parametrize("mode", NON_TRAINING_STAGES)
    def test_cannot_convert_from_val_and_pred(self, mode):
        """Test that converting from 'validating' or 'predicting' mode raises an
        error."""
        cfg_dict = ng_data_config_dict_testing(mode=mode)
        cfg = NGDataConfig(**cfg_dict)

        with pytest.raises(
            ValueError, match="Only conversion from 'training' mode is supported"
        ):
            cfg.convert_mode(TRAINING)

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
    def test_channels_conversion(self, old_channels, new_channels, expected):
        """Test that `convert_mode` correctly converts the `channels` field.

        Note that this only test that the `channels` ambiguity is correctly handled
        by using `channels=None` has using the old `channels`, and `channels="all"` as
        signifying that we want to keep all channels (final value `None`).
        """
        cfg_dict = ng_data_config_dict_testing(
            mode=TRAINING, axes="CYX", channels=old_channels
        )
        cfg = NGDataConfig(**cfg_dict)

        converted_cfg = cfg.convert_mode("validating", new_channels=new_channels)
        assert converted_cfg.channels == expected

    def test_patching_tiling_error(self):
        """Test error converting to prediction while passing `None` overlap."""
        cfg_dict = ng_data_config_dict_testing(
            mode="training",
        )
        cfg = NGDataConfig(**cfg_dict)
        with pytest.raises(
            ValueError, match="`overlap_size` parameter must be specified"
        ):
            cfg.convert_mode("predicting", new_patch_size=[128, 128], overlap_size=None)

    @pytest.mark.parametrize(
        "mode, patch_size, overlap, expected_error",
        # validation throws no error
        list(
            itertools.product([VALIDATING], [None, (64, 64)], [None], [nullcontext(0)])
        )
        # predicting with tiling and valid patch size and overlap throws no error,
        # same for predicting with no tiling (patch_size=None)
        + [
            (PREDICTING, (64, 64), (32, 32), nullcontext(0)),
            (PREDICTING, None, None, nullcontext(0)),
        ]
        # prediction with tiling but no overlap throws error
        + [
            (
                PREDICTING,
                (64, 64),
                None,
                pytest.raises(
                    ValueError, match="`overlap_size` parameter must be specified"
                ),
            ),
        ],
    )
    def test_new_patch(self, mode, patch_size, overlap, expected_error):
        """Test converting to prediction while passing tiling parameters."""
        cfg_dict = ng_data_config_dict_testing(
            mode=TRAINING,
        )
        cfg = NGDataConfig(**cfg_dict)
        with expected_error:
            cfg = cfg.convert_mode(
                mode, new_patch_size=patch_size, overlap_size=overlap
            )

            match mode:
                case "validating":
                    assert cfg.patching.name == "fixed_random"
                case "predicting":
                    if overlap is not None:
                        assert cfg.patching.name == "tiled"
                    else:
                        assert cfg.patching.name == "whole"

    @pytest.mark.parametrize("mode", NON_TRAINING_STAGES)
    def test_patch_coord_filters_removal(self, mode):
        """Test that patch and coordinate filters are removed upon conversion."""
        cfg_dict = ng_data_config_dict_testing(
            mode=TRAINING,
            patch_filter=patch_filter_dict_testing("shannon"),
            coord_filter=coord_filter_dict_testing(),
        )
        cfg = NGDataConfig(**cfg_dict)
        converted_cfg = cfg.convert_mode(mode)
        assert converted_cfg.patch_filter is None
        assert converted_cfg.coord_filter is None
