"""Tests for MicroSplit patch constructor behavior."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from careamics.dataset.image_stack import InMemoryImageStack
from careamics.dataset.patch_constructor.microsplit_patch_constructors import (
    MsPredPatchConstructor,
    MsT1PatchConstructor,
    MsT2PatchConstructor,
    MsT3PatchConstructor,
    _get_uncorrelated_metadata,
)
from careamics.dataset.patch_extractor import PatchExtractor
from careamics.dataset.patching import (
    StratifiedPatching,
    UncorrelatedPatchSpecs,
    is_uncorrelated_specs,
)

MULTIPLEXED_TARGET_SHAPES = [(2, 3, 512, 512), (1, 3, 620, 256)]
INPUT_SHAPES = [
    # make channel dimension equal to 1
    tuple(dim if i != 1 else 1 for i, dim in enumerate(shape))
    for shape in MULTIPLEXED_TARGET_SHAPES
]
SEPARATE_C1_TARGET_SHAPES = [(2, 1, 512, 512), (1, 1, 620, 256)]
SEPARATE_C2_TARGET_SHAPES = [(1, 1, 480, 512), (1, 1, 512, 512), (1, 1, 256, 620)]
SEPARATE_C3_TARGET_SHAPES = [(3, 1, 512, 512)]


# channels will have all the same value
C1_VALUE = 1
C2_VALUE = 5
C3_VALUE = 10
REAL_INPUT_VALUE = 16.2

PATCH_SIZE = (64, 64)

MULTIPLEXED_TARGET_SOURCE = "multiplexed_target_{idx}.tiff"
REAL_INPUT_SOURCE = "input_{idx}.tiff"
SEPARATE_CHANNEL_SOURCE = "separate_target_{channel_idx}_{data_idx}.tiff"


def _image_stack_from_array(
    source: str,
    array: np.ndarray,
    axes: str,
) -> InMemoryImageStack:
    """Return an image stack with a distinct source."""
    return InMemoryImageStack(
        source=Path(source),
        data=array,
        original_axes=axes,
        original_data_shape=array.shape,
    )


@pytest.fixture
def multiplexed_target_extractor():
    axes = "SCYX"
    arrays = [
        np.ones(shape) * np.array([C1_VALUE, C2_VALUE, C3_VALUE]).reshape(1, 3, 1, 1)
        for shape in MULTIPLEXED_TARGET_SHAPES
    ]
    image_stacks = [
        _image_stack_from_array(MULTIPLEXED_TARGET_SOURCE.format(idx=idx), array, axes)
        for idx, array in enumerate(arrays)
    ]
    return PatchExtractor(image_stacks)


@pytest.fixture
def input_target_extractor():
    axes = "SCYX"
    arrays = [np.full(shape, fill_value=REAL_INPUT_VALUE) for shape in INPUT_SHAPES]
    image_stacks = [
        _image_stack_from_array(REAL_INPUT_SOURCE.format(idx=idx), array, axes)
        for idx, array in enumerate(arrays)
    ]
    return PatchExtractor(image_stacks)


@pytest.fixture
def separate_target_extractors():
    axes = "SCYX"
    patch_extractors: list[PatchExtractor] = []
    target_shapes = [
        SEPARATE_C1_TARGET_SHAPES,
        SEPARATE_C2_TARGET_SHAPES,
        SEPARATE_C3_TARGET_SHAPES,
    ]
    target_values = [C1_VALUE, C2_VALUE, C3_VALUE]
    for channel_idx, (shapes, value) in enumerate(
        zip(target_shapes, target_values, strict=True)
    ):
        arrays = [np.full(shape, fill_value=value) for shape in shapes]
        image_stacks = [
            _image_stack_from_array(
                SEPARATE_CHANNEL_SOURCE.format(
                    channel_idx=channel_idx, data_idx=data_idx
                ),
                array,
                axes,
            )
            for data_idx, array in enumerate(arrays)
        ]
        patch_extractors.append(PatchExtractor(image_stacks))
    return patch_extractors


@pytest.fixture
def patching_strategy():
    return StratifiedPatching(MULTIPLEXED_TARGET_SHAPES, PATCH_SIZE, seed=42)


@pytest.fixture
def separate_patching_strategies():
    return [
        StratifiedPatching(SEPARATE_C1_TARGET_SHAPES, PATCH_SIZE, seed=42),
        StratifiedPatching(SEPARATE_C2_TARGET_SHAPES, PATCH_SIZE, seed=42),
        StratifiedPatching(SEPARATE_C3_TARGET_SHAPES, PATCH_SIZE, seed=42),
    ]


@pytest.mark.parametrize("multiscale_count", [1, 2, 3])
@pytest.mark.parametrize(
    "alpha_ranges",
    [
        None,
        [(1.2, 1.2), (1.2, 1.2), (1.2, 1.2)],
        [(0.2, 0.2), (0.5, 0.5), (0.8, 0.8)],
        [(0.5, 1), (0.5, 1), (0.5, 1)],
    ],
)
@pytest.mark.parametrize("uncorrelated_channel_prob", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("channels", [None, [0, 1], [2, 0]])
def test_t1_construct_patch(
    multiscale_count: int,
    alpha_ranges: Sequence[tuple[float, float]] | None,
    uncorrelated_channel_prob: float,
    channels: Sequence[int] | None,
    multiplexed_target_extractor: PatchExtractor,
    patching_strategy: StratifiedPatching,
):
    """Test that the MicroSplit T1 patch constructor outputs patches as expected."""
    rng = np.random.default_rng(42)
    patch_constructor = MsT1PatchConstructor(
        patching_strategy,
        multiplexed_target_extractor,
        multiscale_count,
        "reflect",
        alpha_ranges,
        uncorrelated_channel_prob,
        channels,
        rng=rng,
    )

    # values for testing
    n_channels = len(channels) if channels is not None else 3
    channel_values = [C1_VALUE, C2_VALUE, C3_VALUE]
    if channels is not None:
        channel_values = [channel_values[c_idx] for c_idx in channels]
    # find input max value
    if alpha_ranges is not None:
        channel_indices = channels if channels is not None else range(3)
        max_input_value = sum(
            alpha_ranges[c_idx][1] * value
            for c_idx, value in zip(channel_indices, channel_values, strict=True)
        )
    else:
        max_input_value = sum(value / n_channels for value in channel_values)

    # test each index
    for index in range(patch_constructor.n_patches):
        input_patch, target_patch, patch_specs = patch_constructor.construct_patch(
            index
        )

        # lateral context patches
        assert input_patch.shape[0] == multiscale_count

        # channel subset
        assert target_patch.shape[0] == n_channels

        # uncorrelated patch
        if uncorrelated_channel_prob == 0:
            assert not is_uncorrelated_specs(patch_specs)
        if uncorrelated_channel_prob == 1:
            assert is_uncorrelated_specs(patch_specs)

        # alpha weighting
        assert (input_patch <= max_input_value).all()

        # correct target order
        assert (
            target_patch == np.array(channel_values).reshape(1, n_channels, 1, 1)
        ).all()


@pytest.mark.parametrize("multiscale_count", [1, 2, 3])
@pytest.mark.parametrize(
    "alpha_ranges",
    [
        None,
        [(1.2, 1.2), (1.2, 1.2), (1.2, 1.2)],
        [(0.2, 0.2), (0.5, 0.5), (0.8, 0.8)],
        [(0.5, 1), (0.5, 1), (0.5, 1)],
    ],
)
def test_t2_construct_patch(
    multiscale_count: int,
    alpha_ranges: Sequence[tuple[float, float]] | None,
    separate_target_extractors: Sequence[PatchExtractor[Any]],
    separate_patching_strategies: Sequence[StratifiedPatching],
):
    """Test that the MicroSplit T2 patch constructor outputs patches as expected."""
    rng = np.random.default_rng(42)
    patch_constructor = MsT2PatchConstructor(
        separate_patching_strategies,
        separate_target_extractors,
        multiscale_count,
        "reflect",
        alpha_ranges,
        rng=rng,
    )

    n_channels = len(separate_target_extractors)
    channel_values = [C1_VALUE, C2_VALUE, C3_VALUE]
    # find input max value
    if alpha_ranges is not None:
        max_input_value = sum(
            alpha_range[1] * value
            for alpha_range, value in zip(alpha_ranges, channel_values, strict=True)
        )
    else:
        max_input_value = sum(value / n_channels for value in channel_values)

    # test each index
    for index in range(patch_constructor.n_patches):
        input_patch, target_patch, patch_specs = patch_constructor.construct_patch(
            index
        )

        # lateral context patches
        assert input_patch.shape[0] == multiscale_count

        # channel subset
        assert target_patch.shape[0] == n_channels

        # uncorrelated patch
        assert is_uncorrelated_specs(patch_specs)

        # alpha weighting
        assert (input_patch <= max_input_value).all()

        # correct target order
        assert (
            target_patch == np.array(channel_values).reshape(1, n_channels, 1, 1)
        ).all()


@pytest.mark.parametrize("multiscale_count", [1, 2, 3])
def test_t3_construct_patch(
    multiscale_count: int,
    input_target_extractor: PatchExtractor,
    multiplexed_target_extractor: PatchExtractor,
    patching_strategy: StratifiedPatching,
):
    """Test that the MicroSplit T3 patch constructor outputs patches as expected."""
    patch_constructor = MsT3PatchConstructor(
        patching_strategy,
        input_target_extractor,
        multiplexed_target_extractor,
        multiscale_count,
        "reflect",
    )

    # values for testing
    n_channels = 3
    channel_values = [C1_VALUE, C2_VALUE, C3_VALUE]

    # test each index
    for index in range(patch_constructor.n_patches):
        input_patch, target_patch, patch_specs = patch_constructor.construct_patch(
            index
        )

        # lateral context patches
        assert input_patch.shape[0] == multiscale_count

        # channel subset
        assert target_patch.shape[0] == n_channels

        # uncorrelated patch
        assert not is_uncorrelated_specs(patch_specs)

        # input_value
        assert (input_patch == REAL_INPUT_VALUE).all()

        # correct target order
        assert (
            target_patch == np.array(channel_values).reshape(1, n_channels, 1, 1)
        ).all()


@pytest.mark.parametrize("multiscale_count", [1, 2, 3])
def test_pred_construct_patch(
    multiscale_count: int,
    input_target_extractor: PatchExtractor,
    patching_strategy: StratifiedPatching,
):
    """Test that the MicroSplit T3 patch constructor outputs patches as expected."""
    patch_constructor = MsPredPatchConstructor(
        patching_strategy,
        input_target_extractor,
        multiscale_count,
        "reflect",
    )

    # test each index
    for index in range(patch_constructor.n_patches):
        input_patch, target_patch, patch_specs = patch_constructor.construct_patch(
            index
        )

        # lateral context patches
        assert input_patch.shape[0] == multiscale_count

        # no target
        assert target_patch is None

        # uncorrelated patch
        assert not is_uncorrelated_specs(patch_specs)

        # input_value
        assert (input_patch == REAL_INPUT_VALUE).all()


@pytest.mark.parametrize("principal_channel", [0, 1])
@pytest.mark.parametrize(
    "data_indices",
    [
        [0, 0, 1],
        [1, 0, 1],
        # testing with only 2 channels to emulate when `channels` option is used
        [0, 0],
        [1, 0],
    ],
)
def test_get_uncorrelated_metadata_t1(
    multiplexed_target_extractor: PatchExtractor,
    data_indices: Sequence[int],
    principal_channel: int,
) -> None:
    """
    Test uncorrelated metadata records all channel sources and shapes.

    Code path for training mode 1.
    """
    patch_spec = UncorrelatedPatchSpecs(
        data_idx=1,
        sample_idx=0,
        coords=(0, 0),
        patch_size=PATCH_SIZE,
        principal_channel=principal_channel,
        all_data_idx=data_indices,
        all_sample_idx=[0, 0],
        all_coords=[(0, 0), (0, 0)],
    )

    metadata = _get_uncorrelated_metadata(multiplexed_target_extractor, patch_spec)

    expected_sources = [
        MULTIPLEXED_TARGET_SOURCE.format(idx=data_idx) for data_idx in data_indices
    ]
    expected_shapes = [MULTIPLEXED_TARGET_SHAPES[idx] for idx in data_indices]
    assert metadata["source"] == expected_sources[principal_channel]
    assert metadata["additional_metadata"]["all_sources"] == expected_sources
    assert metadata["additional_metadata"]["all_data_shapes"] == expected_shapes


@pytest.mark.parametrize("principal_channel", [0, 1, 2])
@pytest.mark.parametrize("data_indices", [[0, 0, 0], [1, 2, 0], [0, 1, 0]])
def test_get_uncorrelated_metadata_t2(
    separate_target_extractors: Sequence[PatchExtractor[Any]],
    data_indices: Sequence[int],
    principal_channel: int,
) -> None:
    """Test uncorrelated metadata records sources from multiple extractors.

    Code path for training mode 2.
    """
    patch_spec = UncorrelatedPatchSpecs(
        data_idx=0,
        sample_idx=0,
        coords=(0, 0),
        patch_size=PATCH_SIZE,
        principal_channel=principal_channel,
        all_data_idx=data_indices,
        all_sample_idx=[0, 0, 0],
        all_coords=[(0, 0), (0, 0), (0, 0)],
    )

    metadata = _get_uncorrelated_metadata(separate_target_extractors, patch_spec)

    expected_sources = [
        SEPARATE_CHANNEL_SOURCE.format(channel_idx=c_idx, data_idx=data_idx)
        for c_idx, data_idx in enumerate(data_indices)
    ]
    seperate_target_shapes = [
        SEPARATE_C1_TARGET_SHAPES,
        SEPARATE_C2_TARGET_SHAPES,
        SEPARATE_C3_TARGET_SHAPES,
    ]
    expected_shapes = [
        seperate_target_shapes[c_idx][data_idx]
        for c_idx, data_idx in enumerate(data_indices)
    ]

    assert metadata["source"] == expected_sources[principal_channel]
    assert metadata["additional_metadata"]["all_sources"] == expected_sources
    assert metadata["additional_metadata"]["all_data_shapes"] == expected_shapes
