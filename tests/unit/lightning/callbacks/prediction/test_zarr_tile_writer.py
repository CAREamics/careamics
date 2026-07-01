from collections.abc import Sequence

import numpy as np
import pytest

from careamics.dataset.image_region_data import ImageRegionData
from careamics.dataset.patching import TileSpecs
from careamics.lightning.callbacks.prediction.zarr_tile_write_strategy import (
    _auto_chunks,
)

# TODO add tests for TileHandler methods


@pytest.fixture
def tile(
    original_shape: Sequence[int],
    original_axes: str,
    tile_shape: Sequence[int],
    chunks: Sequence[int] | None,
    shards: Sequence[int] | None,
) -> ImageRegionData:
    """Fixture for a sample ImageRegionData object representing a tile."""
    is_3d = "Z" in original_axes
    has_samples = "S" in original_axes or "T" in original_axes

    return ImageRegionData(
        data=np.random.rand(*tile_shape).astype(np.float32),
        source="test_source",
        data_shape=tile_shape,
        dtype="float32",
        axes=original_axes,
        original_data_shape=original_shape,
        region_spec=TileSpecs(
            data_idx=2,
            sample_idx=0 if has_samples else None,
            coords=(0, 0, 0) if is_3d else (0, 0),
            patch_size=tile_shape,
            crop_coords=(2, 2, 2) if is_3d else (2, 2),
            crop_size=(10, 10, 10) if is_3d else (10, 10),
            stitch_coords=(4, 4, 4) if is_3d else (4, 4),
            total_tiles=1,
        ),
        additional_metadata={"chunks": chunks, "shards": shards},
    )


@pytest.mark.parametrize(
    "axes, data_shape, expected_chunks",
    [
        # axes are original data, can be STCZYX in any order
        # data_shape is in format SC(Z)YX with potential singleton dimensions
        # expected_chunks is in format SC(Z)YX as data is currently not reshaped
        # simple usual shapes
        ("YX", (32, 64), (32, 64)),
        ("YX", (128, 32), (128, 32)),
        ("ZYX", (32, 64, 64), (1, 64, 64)),
        ("ZYX", (64, 128, 64), (1, 128, 64)),
        ("CYX", (5, 64, 64), (1, 64, 64)),
        ("SYX", (5, 64, 256), (1, 64, 128)),
        ("SCYX", (8, 5, 64, 64), (1, 1, 64, 64)),
        ("SCZYX", (5, 5, 32, 256, 64), (1, 1, 1, 128, 64)),
        # different orders (but YX together)
        ("YXZ", (64, 64, 32), (64, 64, 1)),
        ("YXC", (64, 64, 3), (64, 64, 1)),
        ("SYXZ", (4, 64, 64, 32), (1, 64, 64, 1)),
        ("CSYX", (3, 5, 64, 64), (1, 1, 64, 64)),
        ("SZCYX", (8, 16, 3, 256, 64), (1, 1, 1, 128, 64)),
        # T dimension
        ("TYX", (5, 64, 64), (1, 64, 64)),
        ("TCYX", (5, 3, 64, 64), (1, 1, 64, 64)),
        ("STYX", (5, 4, 64, 64), (1, 1, 64, 64)),
        ("STCYX", (5, 4, 3, 256, 64), (1, 1, 1, 128, 64)),
    ],
)
def test_auto_chunks(axes, data_shape, expected_chunks):
    chunks = _auto_chunks(axes, data_shape)
    assert chunks == expected_chunks
