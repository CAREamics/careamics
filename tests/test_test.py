import itertools
from contextlib import nullcontext
from typing import Any

import pytest

from careamics.config.data.ng_data_config import NGDataConfig

# --- can go in conftest.py


TRAIN_PATCHING = ["stratified", "random"]
VAL_PATCHING = ["fixed_random"]
PRED_PATCHING = ["tiled", "whole"]

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

AXES_CZI_3D = (
    "SCZYX",
    "SCTYX",
)

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


DEFAULT_PATCHING = "stratified"
DEFAULT_AXES = "YX"
DEFAULT_DATA_TYPE = "array"
DEFAULT_MODE = "training"


def patch_size_testing(data_type: str = DEFAULT_DATA_TYPE, axes: str = DEFAULT_AXES):
    if data_type == "czi" and axes == "SCTYX":
        return (8, 16, 16)
    elif "Z" in axes:
        return (8, 16, 16)
    else:
        return (16, 16)


def patching_config_dict_testing(
    patching: str = DEFAULT_PATCHING,
    data_type: str = DEFAULT_DATA_TYPE,
    axes: str = DEFAULT_AXES,
    patch_size: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    if patch_size is None:
        patch_size = patch_size_testing(data_type, axes)
    match patching:
        case "random" | "fixed_random" | "stratified":
            # with seed
            return {
                "name": patching,
                "patch_size": patch_size,
                "seed": 42,
            }
        case "tiled":
            # with overlaps
            return {
                "name": patching,
                "patch_size": patch_size,
                "overlaps": tuple(ps // 2 for ps in patch_size),
            }
        case _:
            return {
                "name": patching,
                "patch_size": patch_size,
            }


def ng_data_config_dict_testing(
    # parameters `patching` and `patch_size` can be passed through
    mode: str = DEFAULT_MODE,
    data_type: str = DEFAULT_DATA_TYPE,
    axes: str = DEFAULT_AXES,
    patching: str = DEFAULT_PATCHING,
    patch_size: tuple[int, ...] | None = None,
    # TODO: add normalization
) -> dict[str, Any]:
    patching_config_dict = patching_config_dict_testing(
        patching, data_type, axes, patch_size
    )
    return {
        "mode": mode,
        "data_type": data_type,
        "axes": axes,
        "patching": patching_config_dict,
        "normalization": {"name": "mean_std"},
    }


# ---


def test_default_data_config():
    ng_data_config_dict = ng_data_config_dict_testing()
    _ = NGDataConfig(**ng_data_config_dict)


@pytest.mark.parametrize(
    "data_type, axes, expectation",
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
            [pytest.raises(ValueError, match=r"axes .* are not valid")],
        )
    ),
)
def test_axes_valid(data_type: str, axes: str, expectation):
    ng_data_config_dict = ng_data_config_dict_testing(data_type=data_type, axes=axes)
    with expectation:
        cfg = NGDataConfig(**ng_data_config_dict)
        assert cfg.axes == ng_data_config_dict["axes"]


@pytest.mark.parametrize(
    "mode, patching, expectation",
    list(itertools.product(["training"], TRAIN_PATCHING, [nullcontext(0)]))
    + list(itertools.product(["validating"], VAL_PATCHING, [nullcontext(0)]))
    + list(itertools.product(["predicting"], PRED_PATCHING, [nullcontext(0)]))
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
