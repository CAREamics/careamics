"""Tests for patch specification type guards."""

import pytest

from careamics.dataset.patching import is_tile_specs, is_uncorrelated_specs


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
    assert is_uncorrelated_specs(specs) is expected_uncorrelated
