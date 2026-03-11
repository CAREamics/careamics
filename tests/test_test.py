import itertools
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any, TypedDict

import pytest
from _pytest.mark.structures import ParameterSet

from careamics.config.data.ng_data_config import NGDataConfig

# --- can go in conftest.py


class Params(TypedDict):
    params: Sequence[Any]
    labels: Sequence[str]


def parameter_cartesian_prod(*param_sets: Sequence[Any] | Params):
    params: list[list[Any]] = []
    for param_set in param_sets:
        if isinstance(param_set, dict):
            params_list = [
                pytest.param(p, id=id_)
                for p, id_ in zip(param_set["params"], param_set["labels"], strict=True)
            ]
            params.append(params_list)
        else:
            params.append([pytest.param(p, id=str(p)) for p in param_set])

    result: list[ParameterSet] = []
    for combo in itertools.product(*params):
        values = tuple(v for p in combo for v in p.values)  # flatten values
        ids = [p.id for p in combo]
        id_ = "-".join(ids) if ids else None
        result.append(pytest.param(*values, id=id_))
    return result


# ---

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


# fixtures to act as defaults
@pytest.fixture
def patching():
    return "stratified"


@pytest.fixture
def mode():
    return "training"


@pytest.fixture
def data_type():
    return "array"


@pytest.fixture
def axes():
    return "YX"


@pytest.fixture
def patch_size():
    return (16, 16)


@pytest.fixture
def patching_config_dict(patching: str, patch_size: tuple[int, ...]) -> dict[str, Any]:
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


@pytest.fixture
def ng_data_config_dict(
    # parameters `patching` and `patch_size` can be passed through
    patching_config_dict: dict[str, Any],
    mode: str,
    data_type: str,
    axes: str,
    # TODO: add normalization
) -> dict[str, Any]:
    return {
        "mode": mode,
        "data_type": data_type,
        "axes": axes,
        "patching": patching_config_dict,
        "normalization": {"name": "mean_std"},
    }


def test_default_data_config(ng_data_config_dict):
    _ = NGDataConfig(**ng_data_config_dict)


@pytest.mark.parametrize(
    "data_type, axes, patch_size, expectation",
    parameter_cartesian_prod(
        ["array"],
        AXES_WO_CHANNELS_2D + AXES_W_CHANNELS_2D,
        {"params": [(16, 16)], "labels": ["2D_patch"]},
        {"params": [nullcontext(0)], "labels": ["pass"]},
    )
    + parameter_cartesian_prod(
        ["array"],
        AXES_WO_CHANNELS_3D + AXES_W_CHANNELS_3D,
        {"params": [(8, 16, 16)], "labels": ["3D_patch"]},
        {"params": [nullcontext(0)], "labels": ["pass"]},
    )
    + parameter_cartesian_prod(
        ["array"],
        AXES_DISALLOWED,
        # patch dims do not matter because axes is validated first
        {"params": [(16, 16)], "labels": ["patch"]},
        {
            "params": [pytest.raises(ValueError, match="Invalid axes")],
            "labels": ["error"],
        },
    )
    + parameter_cartesian_prod(
        ["czi"],
        AXES_CZI_2D,
        {"params": [(16, 16)], "labels": ["2D_patch"]},
        {"params": [nullcontext(0)], "labels": ["pass"]},
    )
    + parameter_cartesian_prod(
        ["czi"],
        AXES_CZI_3D,
        {"params": [(8, 16, 16)], "labels": ["3D_patch"]},
        {"params": [nullcontext(0)], "labels": ["pass"]},
    )
    + parameter_cartesian_prod(
        ["czi"],
        AXES_CZI_DISALLOWED,
        # patch dims do not matter because axes is validated first
        {"params": [(16, 16)], "labels": ["patch"]},
        {
            "params": [pytest.raises(ValueError, match=r"axes .* are not valid")],
            "labels": ["error"],
        },
    ),
)
def test_axes_valid(ng_data_config_dict: dict[str, Any], expectation):
    with expectation:
        cfg = NGDataConfig(**ng_data_config_dict)
        assert cfg.axes == ng_data_config_dict["axes"]


@pytest.mark.parametrize(
    "mode, patching, expectation",
    parameter_cartesian_prod(
        ["training"],
        TRAIN_PATCHING,
        {"params": [nullcontext(0)], "labels": ["pass"]},
    )
    + parameter_cartesian_prod(
        ["validating"],
        VAL_PATCHING,
        {"params": [nullcontext(0)], "labels": ["pass"]},
    )
    + parameter_cartesian_prod(
        ["predicting"],
        PRED_PATCHING,
        {"params": [nullcontext(0)], "labels": ["pass"]},
    )
    + parameter_cartesian_prod(
        ["training"],
        VAL_PATCHING + PRED_PATCHING,
        {
            "params": [pytest.raises(ValueError, match="Patching strategy ")],
            "labels": ["error"],
        },
    )
    + parameter_cartesian_prod(
        ["validating"],
        TRAIN_PATCHING + PRED_PATCHING,
        {
            "params": [pytest.raises(ValueError, match="Patching strategy ")],
            "labels": ["error"],
        },
    )
    + parameter_cartesian_prod(
        ["predicting"],
        TRAIN_PATCHING + VAL_PATCHING,
        {
            "params": [pytest.raises(ValueError, match="Patching strategy ")],
            "labels": ["error"],
        },
    ),
)
def test_validate_patching_mode(ng_data_config_dict: dict[str, Any], expectation):
    with expectation:
        cfg = NGDataConfig(**ng_data_config_dict)
        assert cfg.mode == ng_data_config_dict["mode"]
        assert cfg.patching.name == ng_data_config_dict["patching"]["name"]
