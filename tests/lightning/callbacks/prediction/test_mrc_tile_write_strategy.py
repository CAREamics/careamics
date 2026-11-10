from pathlib import Path

import mrcfile
import numpy as np

from careamics.dataset.image_region_data import ImageRegionData
from careamics.dataset.image_stack import MrcImageStack
from careamics.lightning.callbacks.prediction.mrc_image_write_strategy import (
    MrcImageWriteStrategy,
)
from careamics.lightning.callbacks.prediction.mrc_tile_write_strategy import (
    MrcTileWriteStrategy,
)


def _region(
    source: Path, data: np.ndarray, x: int, y: int, metadata: dict
) -> ImageRegionData:
    return ImageRegionData(
        data=data,
        source=str(source),
        dtype=str(data.dtype),
        data_shape=(4, 4),
        axes="YX",
        target_axes="YX",
        original_data_shape=(4, 4),
        region_spec={
            "data_idx": 0,
            "sample_idx": 0,
            "coords": (y, x),
            "patch_size": (2, 2),
            "crop_coords": (0, 0),
            "crop_size": (2, 2),
            "stitch_coords": (y, x),
            "total_tiles": 4,
        },
        additional_metadata=metadata,
    )


def test_writes_tiles_with_mmap_and_restores_mrc_metadata(tmp_path: Path):
    source = tmp_path / "source.mrc"
    with mrcfile.new(source, overwrite=True) as mrc:
        mrc.set_data(np.zeros((4, 4), dtype=np.uint16))
        mrc.header.nxstart = 11

    metadata = MrcImageStack(source, axes="YX").additional_metadata

    writer = MrcTileWriteStrategy()
    tiles = [
        _region(source, np.ones((1, 2, 2), dtype=np.float32), 0, 0, metadata),
        _region(source, np.full((1, 2, 2), 2, dtype=np.float32), 2, 0, metadata),
    ]
    writer.write_batch(tmp_path / "predictions", tiles)
    writer.close()

    with mrcfile.open(tmp_path / "predictions" / "source.mrc", permissive=True) as mrc:
        np.testing.assert_array_equal(
            mrc.data,
            [[1, 1, 2, 2], [1, 1, 2, 2], [0, 0, 0, 0], [0, 0, 0, 0]],
        )
        assert mrc.header.nxstart == 11
        assert mrc.header.mode == 2  # float32 prediction data, not uint16 source data


def test_writes_complete_image_and_restores_mrc_metadata(tmp_path: Path):
    source = tmp_path / "source.mrc"
    with mrcfile.new(source, overwrite=True) as mrc:
        mrc.set_data(np.zeros((4, 4), dtype=np.uint16))
        mrc.header.nxstart = 13

    metadata = MrcImageStack(source, axes="YX").additional_metadata
    prediction = _region(
        source,
        np.full((1, 4, 4), 3, dtype=np.float32),
        0,
        0,
        metadata,
    )._replace(
        region_spec={
            "data_idx": 0,
            "sample_idx": 0,
            "coords": (0, 0),
            "patch_size": (4, 4),
        }
    )

    writer = MrcImageWriteStrategy()
    writer.write_batch(tmp_path / "predictions", [prediction])

    with mrcfile.open(tmp_path / "predictions" / "source.mrc") as mrc:
        np.testing.assert_array_equal(mrc.data, np.full((4, 4), 3, dtype=np.float32))
        assert mrc.header.nxstart == 13
        assert mrc.header.mode == 2
