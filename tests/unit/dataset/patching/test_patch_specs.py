"""Tests for patch specification helpers."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from torch.utils.data import default_collate

from careamics.dataset.patching import (
    Patching,
    PatchSpecs,
    PerChannelPatchSpecs,
    RandomPatching,
    SequentialPatching,
    StratifiedPatching,
    TiledPatching,
    WholeSamplePatching,
    is_per_channel_specs,
    is_tile_specs,
    pack_patch_spec,
    unpack_patch_spec,
)


@pytest.mark.parametrize(
    ("specs", "expected_tile", "expected_uncorrelated"),
    [
        (
            {
                "data_idx": 0,
                "sample_idx": 0,
                "coords": (0, 0),
                "patch_size": (16, 16),
            },
            False,
            False,
        ),
        (
            {
                "data_idx": 0,
                "sample_idx": 0,
                "coords": (0, 0),
                "patch_size": (16, 16),
                "crop_coords": (0, 0),
                "crop_size": (8, 8),
                "stitch_coords": (0, 0),
                "total_tiles": 4,
            },
            True,
            False,
        ),
        (
            {
                "data_idx": 0,
                "sample_idx": 0,
                "coords": (0, 0),
                "patch_size": (16, 16),
                "principal_channel": 0,
                "all_data_idx": (0, 1),
                "all_sample_idx": (0, 0),
                "all_coords": ((0, 0), (8, 8)),
            },
            False,
            True,
        ),
    ],
)
def test_patch_spec_type_guards(
    specs: dict[str, object],
    expected_tile: bool,
    expected_uncorrelated: bool,
) -> None:
    """Test patch spec helpers identify tile and uncorrelated patch specs."""
    assert is_tile_specs(specs) is expected_tile
    assert is_per_channel_specs(specs) is expected_uncorrelated


def _as_tuples(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _as_tuples(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(_as_tuples(item) for item in value)
    return int(value)


def _collate_and_unpack(specs: Sequence[PatchSpecs]) -> list[PatchSpecs]:
    batch = default_collate([pack_patch_spec(spec) for spec in specs])
    return [unpack_patch_spec(row) for row in batch]


@pytest.mark.parametrize(
    "data_shapes, patch_size",
    [
        ([(2, 1, 32, 40), (1, 1, 48, 32)], (16, 16)),
        ([(2, 1, 16, 32, 40), (1, 1, 24, 48, 32)], (8, 16, 16)),
    ],
)
@pytest.mark.parametrize(
    "create_strategy",
    [
        lambda shapes, size: RandomPatching(shapes, size, seed=42),
        lambda shapes, size: StratifiedPatching(shapes, size, seed=42),
        lambda shapes, size: SequentialPatching(shapes, size, [2] * len(size)),
        lambda shapes, size: TiledPatching(shapes, size, [4] * len(size)),
        lambda shapes, size: WholeSamplePatching(shapes),
    ],
    ids=["random", "stratified", "sequential", "tiled", "whole_sample"],
)
def test_packed_patch_spec_round_trip(
    data_shapes: Sequence[Sequence[int]],
    patch_size: Sequence[int],
    create_strategy,
) -> None:
    """Test that collated packed specs unpack to the original specs."""
    strategy: Patching = create_strategy(data_shapes, patch_size)
    specs = [strategy.get_patch_spec(i) for i in range(strategy.n_patches)]

    unpacked = _collate_and_unpack(specs)

    assert unpacked == [_as_tuples(spec) for spec in specs]


@pytest.mark.parametrize("ndim", [2, 3])
def test_packed_per_channel_spec_round_trip(ndim: int) -> None:
    """Test that per-channel specs with same and different locations collate."""
    coords = tuple(range(ndim))
    same_locations = PerChannelPatchSpecs(
        data_idx=1,
        sample_idx=2,
        coords=coords,
        patch_size=(16,) * ndim,
        principal_channel=0,
        all_data_idx=(1, 1, 1),
        all_sample_idx=(2, 2, 2),
        all_coords=(coords, coords, coords),
    )
    different_locations = PerChannelPatchSpecs(
        data_idx=0,
        sample_idx=3,
        coords=coords,
        patch_size=(16,) * ndim,
        principal_channel=1,
        all_data_idx=(2, 0, 1),
        all_sample_idx=(1, 3, 0),
        all_coords=((5,) * ndim, coords, (9,) * ndim),
    )
    specs = [same_locations, different_locations]

    assert _collate_and_unpack(specs) == specs


def test_unpack_unknown_kind() -> None:
    """Test that an unknown packed spec kind raises an error."""
    packed = pack_patch_spec(
        PatchSpecs(data_idx=0, sample_idx=0, coords=(0, 0), patch_size=(8, 8))
    )
    packed[0] = 7

    with pytest.raises(ValueError, match="Unknown"):
        unpack_patch_spec(packed)
