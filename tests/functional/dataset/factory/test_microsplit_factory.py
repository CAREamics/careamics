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
    PredData,
    TrainValData,
    create_microsplit_dataset,
    create_microsplit_pred_dataset,
    create_microsplit_pred_dataset_from_data,
    create_microsplit_train_val_datasets,
)
from careamics.dataset.patch_constructor.microsplit_patch_constructors import (
    PairedInputTargetMsPatchConstr,
    PredMsPatchConstr,
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


def _paired_train_val_data(
    tmp_path: Path, data_type: Literal["array", "tiff"] = "array"
) -> tuple[TrainValData[Any], int]:
    """Return a `TrainValData` container holding paired input/target sources."""
    train, n_channels = _dummy_paired_data(tmp_path / "train", data_type)
    val, _ = _dummy_paired_data(tmp_path / "val", data_type)
    return (
        TrainValData(
            train_data=train.input_data,
            train_data_target=train.target_data,
            val_data=val.input_data,
            val_data_target=val.target_data,
        ),
        n_channels,
    )


def _training_config(**kwargs: Any) -> MicroSplitDataConfig:
    """Return a MicroSplit training configuration with test defaults."""
    params: dict[str, Any] = {
        "mode": "training",
        "data_type": "array",
        "axes": "SCYX",
        "patching": {"name": "stratified", "patch_size": (16, 16), "seed": 42},
        "normalization": {"name": "none"},
        "seed": 42,
        "multiscale_count": 1,
    }
    params.update(kwargs)
    return MicroSplitDataConfig(**params)


@pytest.mark.parametrize("data_type", ["array", "tiff"])
@pytest.mark.parametrize("multiscale_count", [1, 3])
def test_train_val_datasets_build_paired_constructors(
    tmp_path: Path,
    data_type: Literal["array", "tiff"],
    multiscale_count: int,
) -> None:
    """Test both datasets use the paired constructor and carry lateral context."""
    # Arrange
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    data, n_channels = _paired_train_val_data(tmp_path, data_type)
    patch_size = (16, 16)
    config = _training_config(data_type=data_type, multiscale_count=multiscale_count)

    # Act
    train_dataset, val_dataset = create_microsplit_train_val_datasets(config, data)

    # Assert
    assert isinstance(train_dataset.patch_constructor, PairedInputTargetMsPatchConstr)
    assert isinstance(val_dataset.patch_constructor, PairedInputTargetMsPatchConstr)

    input_region, target_region = train_dataset[0]
    assert input_region.data.shape == (multiscale_count, *patch_size)
    assert target_region.data.shape == (n_channels, *patch_size)


def test_train_val_datasets_convert_validation_config(tmp_path: Path) -> None:
    """Test the validation dataset keeps MicroSplit fields in validating mode."""
    # Arrange
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    data, _ = _paired_train_val_data(tmp_path)
    config = _training_config(multiscale_count=3, padding_mode="wrap")

    # Act
    _, val_dataset = create_microsplit_train_val_datasets(config, data)

    # Assert
    assert isinstance(val_dataset.config, MicroSplitDataConfig)
    assert val_dataset.config.mode == "validating"
    assert val_dataset.config.multiscale_count == 3
    assert val_dataset.config.padding_mode == "wrap"


def test_train_val_datasets_statistics_computed_on_training_data(
    tmp_path: Path,
) -> None:
    """Test the validation dataset reuses the training normalization statistics.

    The training dataset must be built before the validation configuration is derived,
    otherwise the validation data would be normalized by its own statistics.
    """
    # Arrange
    train_target = np.ones((2, 2, 64, 64), dtype=np.float32)
    val_target = np.full((2, 2, 64, 64), 100.0, dtype=np.float32)
    data = TrainValData(
        train_data=[train_target.sum(axis=1, keepdims=True)],
        train_data_target=[train_target],
        val_data=[val_target.sum(axis=1, keepdims=True)],
        val_data_target=[val_target],
    )
    config = _training_config(normalization={"name": "mean_std"})

    # Act
    train_dataset, val_dataset = create_microsplit_train_val_datasets(config, data)

    # Assert: statistics come from the training data (input sums to 2.0), not the
    # validation data (which would sum to 200.0).
    assert train_dataset.config.normalization.input_means == [2.0]
    assert (
        val_dataset.config.normalization.input_means
        == train_dataset.config.normalization.input_means
    )


def test_train_val_datasets_reject_non_training_config(tmp_path: Path) -> None:
    """Test a non-training configuration cannot be used to build training datasets."""
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    data, _ = _paired_train_val_data(tmp_path)
    config = _training_config().convert_mode("validating")

    with pytest.raises(ValueError, match="cannot be used for training"):
        create_microsplit_train_val_datasets(config, data)


@pytest.mark.parametrize(
    ("missing_field", "match"),
    [
        ("train_data_target", "MicroSplit is supervised"),
        ("val_data_target", "`val_data_target` must be provided"),
    ],
)
def test_train_val_datasets_require_targets(
    tmp_path: Path, missing_field: str, match: str
) -> None:
    """Test MicroSplit requires both training and validation targets."""
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    data, _ = _paired_train_val_data(tmp_path)
    setattr(data, missing_field, None)

    with pytest.raises(ValueError, match=match):
        create_microsplit_train_val_datasets(_training_config(), data)


def test_train_val_datasets_reject_mask(tmp_path: Path) -> None:
    """Test mask filtering is rejected, MicroSplit constructors do not support it."""
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    data, _ = _paired_train_val_data(tmp_path)
    data.train_data_mask = [np.ones((2, 1, 256, 256), dtype=np.float32)]

    with pytest.raises(NotImplementedError, match="Mask-based patch filtering"):
        create_microsplit_train_val_datasets(_training_config(), data)


def test_paired_mode_rejects_multichannel_input(tmp_path: Path) -> None:
    """Test the paired mode rejects an input with several channels.

    The MicroSplit input is the superimposed image, so it must be single-channel.
    """
    data, _ = _dummy_paired_data(tmp_path, "array")
    multichannel_input = PairedInputTarget(
        input_data=data.target_data, target_data=data.target_data
    )

    with pytest.raises(ValueError, match="requires a single-channel input"):
        create_microsplit_dataset(config=_training_config(), data=multichannel_input)


def test_pred_dataset_from_data_builds_prediction_constructor(tmp_path: Path) -> None:
    """Test a prediction container yields the MicroSplit prediction constructor."""
    input_data, _ = _dummy_pred_data(tmp_path, "array")
    config = _training_config(
        mode="predicting",
        patching={"name": "tiled", "patch_size": (16, 16), "overlaps": (8, 8)},
    )

    dataset = create_microsplit_pred_dataset_from_data(config, PredData(input_data))

    assert isinstance(dataset.patch_constructor, PredMsPatchConstr)


def test_pred_dataset_from_data_converts_training_config(tmp_path: Path) -> None:
    """Test a training configuration is converted to whole-image prediction."""
    input_data, _ = _dummy_pred_data(tmp_path, "array")

    dataset = create_microsplit_pred_dataset_from_data(
        _training_config(), PredData(input_data)
    )

    assert dataset.config.mode == "predicting"
    assert dataset.config.patching.name == "whole"


def test_pred_dataset_from_data_rejects_validating_config(tmp_path: Path) -> None:
    """Test a validation configuration cannot be used for prediction."""
    input_data, _ = _dummy_pred_data(tmp_path, "array")
    config = _training_config().convert_mode("validating")

    with pytest.raises(ValueError, match="cannot be used for prediction"):
        create_microsplit_pred_dataset_from_data(config, PredData(input_data))


def test_pred_dataset_from_data_warns_on_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test a prediction target is ignored with a warning."""
    warnings: list[str] = []
    monkeypatch.setattr(
        "careamics.dataset.factory.microsplit_factory.logger.warning",
        lambda message, *args: warnings.append(message % args),
    )
    input_data, _ = _dummy_pred_data(tmp_path, "array")

    create_microsplit_pred_dataset_from_data(
        _training_config(), PredData(input_data, pred_data_target=input_data)
    )

    assert len(warnings) == 1
    assert "pred_data_target" in warnings[0]


def test_factory_does_not_warn_for_default_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test explicitly setting a field to its default does not warn.

    Configuration factories set every MicroSplit field explicitly, so warning on
    `model_fields_set` alone would warn about fields the user never chose.
    """
    warnings: list[str] = []
    monkeypatch.setattr(
        "careamics.dataset.factory.microsplit_factory.logger.warning",
        lambda message, *args: warnings.append(message % args),
    )
    data, _ = _dummy_paired_data(tmp_path, "array")
    # `uncorrelated_channel_prob` is unused by the paired mode, but it is set to its
    # own default value here, so the user did not really choose it.
    config = _training_config(uncorrelated_channel_prob=0.0)

    create_microsplit_dataset(config=config, data=data)

    assert warnings == []
