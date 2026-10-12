"""Functional tests for MicroSplit dataset factories."""

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest
import tifffile
from numpy.typing import NDArray

from careamics.config.data import MicroSplitDataConfig
from careamics.dataset.factory import (
    IndependentTargets,
    MultiChannelTarget,
    PairedInputTarget,
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
    return MultiChannelTarget(target_data), n_channels


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
    return IndependentTargets(target_channel_data), n_channels


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
        PairedInputTarget(
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
    MultiChannelTarget[MicroSplitSource]
    | IndependentTargets[MicroSplitSource]
    | PairedInputTarget[MicroSplitSource],
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
def test_train_dataset_all_indices(
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
        patching={"name": "stratified", "patch_size": patch_size, "seed": 42},
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
def test_pred_dataset_all_indices(
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


def test_train_factory_rejects_predict_config() -> None:
    """Test training factory rejects prediction configs."""
    config = MicroSplitDataConfig(
        mode="predicting",
        data_type="array",
        axes="SCYX",
        patching={"name": "tiled", "patch_size": (16, 16), "overlaps": (8, 8)},
        normalization={"name": "none"},
    )
    data = [np.zeros((1, 1, 32, 32), dtype=np.float32)]

    with pytest.raises(
        ValueError,
        match="Use `create_microsplit_pred_dataset` to create prediction datasets",
    ):
        create_microsplit_dataset(
            config=config,
            data=MultiChannelTarget(data),
        )


def test_pred_factory_rejects_train_config() -> None:
    """Test prediction factory rejects training configs."""
    config = MicroSplitDataConfig(
        mode="training",
        data_type="array",
        axes="SCYX",
        patching={"name": "stratified", "patch_size": (16, 16), "seed": 42},
        normalization={"name": "none"},
    )
    data = [np.zeros((1, 1, 32, 32), dtype=np.float32)]

    with pytest.raises(
        ValueError,
        match=(
            "`create_microsplit_pred_dataset` requires a config with mode='predicting'"
        ),
    ):
        create_microsplit_pred_dataset(config=config, input_data=data)


@pytest.mark.parametrize(
    "data",
    [
        IndependentTargets([]),
        IndependentTargets([np.ones((128, 128))]),
    ],
)
def test_less_than_two_separate_channels_error(
    data: IndependentTargets,
) -> None:
    """Test training factory rejects unsupported MicroSplit data inputs."""
    config = MicroSplitDataConfig(
        mode="training",
        data_type="array",
        axes="YX",
        patching={"name": "stratified", "patch_size": (16, 16), "seed": 42},
        normalization={"name": "none"},
    )

    with pytest.raises((TypeError, ValueError), match="two target channel sources"):
        create_microsplit_dataset(config=config, data=data)


def test_pred_factory_rejects_non_sequence() -> None:
    """Test prediction factory requires sequences for standard loading."""
    config = MicroSplitDataConfig(
        mode="predicting",
        data_type="array",
        axes="SCYX",
        patching={"name": "tiled", "patch_size": (16, 16), "overlaps": (8, 8)},
        normalization={"name": "none"},
    )

    with pytest.raises(TypeError, match="Prediction input must be a sequence"):
        create_microsplit_pred_dataset(
            config=config,
            input_data=np.zeros((1, 1, 32, 32), dtype=np.float32),
        )


@pytest.mark.parametrize(
    ("mode", "config_kwargs", "warning_field"),
    [
        ("separate", {"channels": [0]}, "channels"),
        ("separate", {"uncorrelated_channel_prob": 1.0}, "uncorrelated_channel_prob"),
        (
            "paired",
            {"alpha_ranges": [(0.5, 0.5), (0.5, 0.5)]},
            "alpha_ranges",
        ),
        ("paired", {"channels": [0]}, "channels"),
        ("paired", {"uncorrelated_channel_prob": 1.0}, "uncorrelated_channel_prob"),
    ],
)
def test_factory_warns_unused_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: Literal["separate", "paired"],
    config_kwargs: dict[str, Any],
    warning_field: str,
) -> None:
    """Test factory warns when explicitly set config fields are unused."""
    warnings: list[str] = []
    monkeypatch.setattr(
        "careamics.dataset.factory.microsplit_factory.logger.warning",
        lambda message, *args: warnings.append(message % args),
    )
    data, _ = _microsplit_data_from_mode(mode, "array", tmp_path)
    config = MicroSplitDataConfig(
        mode="training",
        data_type="array",
        axes="SCYX",
        patching={"name": "stratified", "patch_size": (16, 16), "seed": 42},
        normalization={"name": "none"},
        **config_kwargs,
    )

    create_microsplit_dataset(
        config=config,
        data=data,
        loading=None,
        rng=np.random.default_rng(23),
    )

    assert len(warnings) == 1
    assert warning_field in warnings[0]
