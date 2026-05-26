"""Functional tests for MicroSplit dataset factories."""

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest
import tifffile
from numpy.typing import NDArray

from careamics.config.data import (
    MicroSplitDataConfig,
    SlidingWindowTiledPatchingConfig,
)
from careamics.dataset.factory import (
    MicroSplitMultiplexedTargetData,
    MicroSplitPairedData,
    MicroSplitSeparateTargetData,
    create_microsplit_dataset,
    create_microsplit_pred_dataset,
)
from careamics.dataset.image_stack import InMemoryImageStack
from careamics.dataset.patch_constructor import MsPredPatchConstructor
from careamics.dataset.patch_extractor import PatchExtractor
from careamics.dataset.patching import (
    SlidingWindowTiledPatching,
    WholeSamplePatching,
    is_tile_specs,
    is_uncorrelated_specs,
)

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


def _microsplit_data_from_mode(
    mode: Literal["multiplexed", "separate", "paired"],
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


@pytest.mark.parametrize("data_type", ["array", "tiff"])
@pytest.mark.parametrize(
    "mode,uncorrelated_channel_prob",
    [("multiplexed", 0), ("multiplexed", 1), ("separate", 1), ("paired", 0)],
)
@pytest.mark.parametrize("multiscale_count", [1, 2, 3])
def test_microsplit_factory_dataset_outputs_all_indices(
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
        patching={"name": "random", "patch_size": patch_size, "seed": 13},
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


# ---------------------------------------------------------------------------
# Sliding-window tiled prediction x MicroSplit
# ---------------------------------------------------------------------------


def _pred_sliding_window_config(
    *,
    patch_size: tuple[int, int],
    overlaps: tuple[int, int],
    stride: tuple[int, int],
    multiscale_count: int,
) -> MicroSplitDataConfig:
    return MicroSplitDataConfig(
        mode="predicting",
        data_type="array",
        axes="SCYX",
        patching=SlidingWindowTiledPatchingConfig(
            patch_size=list(patch_size),
            overlaps=list(overlaps),
            stride=list(stride),
        ),
        normalization={"name": "none"},
        multiscale_count=multiscale_count,
        padding_mode="reflect",
        seed=42,
    )


@pytest.mark.parametrize("multiscale_count", [1, 3])
def test_microsplit_pred_factory_with_sliding_window_tiled(multiscale_count: int):
    """SW-tiled prediction dataset yields LC patches with consistent TileSpecs."""
    patch_size = (16, 16)
    overlaps = (8, 8)
    stride = (4, 4)

    rng = np.random.default_rng(0)
    image = rng.random(size=(1, 1, 32, 32)).astype(np.float32)

    config = _pred_sliding_window_config(
        patch_size=patch_size,
        overlaps=overlaps,
        stride=stride,
        multiscale_count=multiscale_count,
    )
    dataset = create_microsplit_pred_dataset(config=config, input_data=[image])

    # n_patches matches what SlidingWindowTiledPatching reports.
    sw_patching = SlidingWindowTiledPatching(
        data_shapes=[image.shape],
        patch_size=list(patch_size),
        overlaps=list(overlaps),
        stride=list(stride),
    )
    assert len(dataset) == sw_patching.n_patches
    assert len(dataset) > 0

    # Each item is a TileSpecs-bearing LC patch with shape (L, Y, X).
    seen_total_tiles: set[int] = set()
    for idx in range(len(dataset)):
        (input_region,) = dataset[idx]
        spec = input_region.region_spec
        assert is_tile_specs(spec)
        assert input_region.data.shape == (multiscale_count, *patch_size)
        seen_total_tiles.add(int(spec["total_tiles"]))

    # All tiles for the single image carry the same total_tiles value and it
    # matches n_patches.
    assert seen_total_tiles == {len(dataset)}


def test_microsplit_convert_mode_stride_builds_sliding_window_config():
    """`convert_mode(..., stride=...)` switches to SlidingWindowTiledPatchingConfig."""
    train_config = MicroSplitDataConfig(
        mode="training",
        data_type="array",
        axes="SCYX",
        patching={"name": "random", "patch_size": [16, 16], "seed": 13},
        normalization={"name": "none"},
        multiscale_count=2,
        padding_mode="reflect",
        seed=42,
    )

    pred_config = train_config.convert_mode(
        new_mode="predicting",
        new_patch_size=[16, 16],
        overlap_size=[8, 8],
        stride=[4, 4],
    )

    assert isinstance(pred_config, MicroSplitDataConfig)
    assert pred_config.mode == "predicting"
    assert isinstance(pred_config.patching, SlidingWindowTiledPatchingConfig)
    assert list(pred_config.patching.stride) == [4, 4]
    assert list(pred_config.patching.patch_size) == [16, 16]
    assert list(pred_config.patching.overlaps) == [8, 8]
    # MicroSplit-specific fields preserved.
    assert pred_config.multiscale_count == 2
    assert pred_config.padding_mode == "reflect"


def test_microsplit_convert_mode_stride_requires_predicting_mode_and_sizes():
    """`stride` is only valid for predicting with patch_size and overlap_size."""
    train_config = MicroSplitDataConfig(
        mode="training",
        data_type="array",
        axes="SCYX",
        patching={"name": "random", "patch_size": [16, 16], "seed": 13},
        normalization={"name": "none"},
    )
    with pytest.raises(ValueError, match="stride"):
        train_config.convert_mode(
            new_mode="predicting",
            new_patch_size=[16, 16],
            overlap_size=None,
            stride=[4, 4],
        )
    with pytest.raises(ValueError, match="stride"):
        train_config.convert_mode(
            new_mode="validating",
            new_patch_size=[16, 16],
            overlap_size=[8, 8],
            stride=[4, 4],
        )


def test_ms_pred_patch_constructor_rejects_non_tile_specs_patching():
    """The guard raises TypeError when the patching strategy isn't TileSpecs-valued."""
    rng = np.random.default_rng(0)
    image = rng.random(size=(1, 1, 32, 32)).astype(np.float32)
    extractor = PatchExtractor(
        image_stacks=[InMemoryImageStack.from_array(image, axes="SCYX")]
    )
    whole_patching = WholeSamplePatching(data_shapes=[image.shape])

    with pytest.raises(TypeError, match="TileSpecs"):
        MsPredPatchConstructor(
            patching_strategy=whole_patching,
            input_extractor=extractor,
            multiscale_count=1,
            padding_mode="reflect",
        )
