"""Functional tests for MicroSplit dataset factories."""

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest
import tifffile
from numpy.typing import NDArray

from careamics.config.data import MicroSplitDataConfig
from careamics.dataset.factory import (
    MicroSplitMultiplexedTargetData,
    MicroSplitPairedData,
    MicroSplitSeparateTargetData,
    create_microsplit_dataset,
    create_microsplit_pred_dataset,
)
from careamics.dataset.patching import is_uncorrelated_specs

MicroSplitSource = list[NDArray[Any]] | list[Path]


def _format_sources(
    arrays: list[NDArray[Any]],
    data_type: Literal["array", "tiff"],
    tmp_path: Path,
    name: str,
) -> MicroSplitSource:
    """Return array sources or write them as TIFF sources."""
    if data_type == "array":
        return arrays

    paths = []
    for index, array in enumerate(arrays):
        path = tmp_path / f"{name}_{index}.tiff"
        tifffile.imwrite(path, array, metadata={"axes": "SCYX"})
        paths.append(path)
    return paths


def _dummy_multiplexed_data(tmp_path: Path, data_type: Literal["array", "tiff"]):
    rng = np.random.default_rng(42)
    n_channels = 2
    target_data = _format_sources(
        [rng.random(size=(2, n_channels, 256, 256)).astype(np.float32)],
        data_type,
        tmp_path,
        "multiplexed_target",
    )
    return MicroSplitMultiplexedTargetData(target_data), n_channels


def _dummy_separate_channel_data(tmp_path: Path, data_type: Literal["array", "tiff"]):
    rng = np.random.default_rng(42)
    target_channel_data = [
        _format_sources(
            [rng.random(size=(2, 1, 256, 256)).astype(np.float32)],
            data_type,
            tmp_path,
            "separate_target_0",
        ),
        _format_sources(
            [rng.random(size=(2, 1, 200, 300)).astype(np.float32)],
            data_type,
            tmp_path,
            "separate_target_1",
        ),
    ]
    n_channels = len(target_channel_data)
    return MicroSplitSeparateTargetData(target_channel_data), n_channels


def _dummy_paired_data(tmp_path: Path, data_type: Literal["array", "tiff"]):
    rng = np.random.default_rng(42)
    n_channels = 2
    input_data = _format_sources(
        [rng.random(size=(2, 1, 256, 256)).astype(np.float32)],
        data_type,
        tmp_path,
        "paired_input",
    )
    target_data = _format_sources(
        [rng.random(size=(2, n_channels, 256, 256)).astype(np.float32)],
        data_type,
        tmp_path,
        "paired_target",
    )
    return (
        MicroSplitPairedData(
            input_data=input_data,
            target_data=target_data,
        ),
        n_channels,
    )


def _dummy_pred_data(tmp_path: Path, data_type: Literal["array", "tiff"]):
    rng = np.random.default_rng(42)
    input_data = _format_sources(
        [
            rng.random(size=(2, 1, 256, 256)).astype(np.float32),
            rng.random(size=(1, 1, 200, 300)).astype(np.float32),
        ],
        data_type,
        tmp_path,
        "pred",
    )
    return input_data, 1


def _microsplit_data_from_mode(
    mode: Literal["multiplexed", "separate", "paired", "prediction"],
    data_type: Literal["array", "tiff"],
    tmp_path: Path,
) -> tuple[
    MicroSplitMultiplexedTargetData[MicroSplitSource]
    | MicroSplitSeparateTargetData[MicroSplitSource]
    | MicroSplitPairedData[MicroSplitSource],
    int,
]:
    """Return factory input data and expected constructor for a MicroSplit mode."""

    if mode == "multiplexed":
        return _dummy_multiplexed_data(tmp_path, data_type)

    elif mode == "separate":
        return _dummy_separate_channel_data(tmp_path, data_type)

    elif mode == "paired":
        return _dummy_paired_data(tmp_path, data_type)

    else:
        raise ValueError(f"Mode {mode} unrecognized.")


@pytest.mark.parametrize("data_type", ["array", "tiff"])
@pytest.mark.parametrize(
    "mode,uncorrelated_channel_prob",
    [("multiplexed", 0), ("multiplexed", 1), ("separate", 1), ("paired", 0)],
)
@pytest.mark.parametrize("multiscale_count", [1, 2, 3])
def test_microsplit_factory_dataset_all_indices(
    tmp_path: Path,
    data_type: Literal["array", "tiff"],
    uncorrelated_channel_prob: float,
    multiscale_count: int,
    mode: Literal["multiplexed", "separate", "paired"],
) -> None:
    """Test MicroSplit factory datasets can produce output for every index."""
    data, n_channels = _microsplit_data_from_mode(mode, data_type, tmp_path)
    patch_size = (16, 16)
    config = MicroSplitDataConfig(
        mode="training",
        data_type=data_type,
        axes="SCYX",
        patching={"name": "stratified", "patch_size": patch_size, "seed": 13},
        normalization={"name": "none"},
        seed=42,
        multiscale_count=multiscale_count,
        uncorrelated_channel_prob=uncorrelated_channel_prob,
    )

    dataset = create_microsplit_dataset(
        config=config,
        data=data,
        loading=None,
        rng=np.random.default_rng(23),
    )
    for index in range(len(dataset)):
        input_region, target_region = dataset[index]
        if mode == "multiplexed":
            expected_uncorrelated = bool(uncorrelated_channel_prob)
        elif mode == "separate":
            expected_uncorrelated = True
        elif mode == "paired":
            expected_uncorrelated = False

        assert input_region.data.shape == (multiscale_count, *patch_size)
        assert target_region.data.shape == (n_channels, *patch_size)
        assert is_uncorrelated_specs(input_region.region_spec) is expected_uncorrelated
        assert is_uncorrelated_specs(target_region.region_spec) is expected_uncorrelated


@pytest.mark.parametrize("data_type", ["array", "tiff"])
@pytest.mark.parametrize("multiscale_count", [1, 2, 3])
def test_microsplit_pred_factory_dataset_all_indices(
    tmp_path: Path,
    data_type: Literal["array", "tiff"],
    multiscale_count: int,
) -> None:
    """Test MicroSplit factory datasets can produce output for every index."""
    data, _ = _dummy_pred_data(tmp_path, data_type)
    patch_size = (16, 16)
    config = MicroSplitDataConfig(
        mode="predicting",
        data_type=data_type,
        axes="SCYX",
        patching={"name": "tiled", "patch_size": patch_size, "overlaps": (8, 8)},
        normalization={"name": "none"},
        seed=42,
        multiscale_count=multiscale_count,
    )
    dataset = create_microsplit_pred_dataset(
        config=config,
        input_data=data,
    )
    for index in range(len(dataset)):
        (input_region,) = dataset[index]

        assert input_region.data.shape == (multiscale_count, *patch_size)
