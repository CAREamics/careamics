from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from careamics.config.support import SupportedData
from careamics.dataset_ng.legacy_interoperability import tilespecs_to_tileinfos
from careamics.dataset_ng.patch_extractor.patch_extractor_factory import (
    create_patch_extractor,
)
from careamics.dataset_ng.patching_strategies import TileSpecs, TilingStrategy
from careamics.prediction_utils.stitch_prediction import stitch_prediction


def _test_smoke_tiling(
    data_shapes: list[tuple[int, ...]],
    patch_size: tuple[int, ...],
    overlaps: tuple[int, ...],
    axes: str,
):
    data = [
        np.arange(np.prod(data_shape)).reshape(data_shape) for data_shape in data_shapes
    ]
    patch_extractor = create_patch_extractor(
        source=data, axes=axes, data_type=SupportedData.ARRAY
    )
    tiling_strategy = TilingStrategy(
        data_shapes=data_shapes, tile_size=patch_size, overlaps=overlaps
    )
    tile_specs: list[TileSpecs] = []
    tiles: list[NDArray[Any]] = []
    n_tiles = tiling_strategy.n_patches
    for i in range(n_tiles):
        tile_spec = tiling_strategy.get_patch_spec(i)
        if tile_spec["data_idx"] == 1:
            pass
        tile = patch_extractor.extract_patch(
            data_idx=tile_spec["data_idx"],
            sample_idx=tile_spec["sample_idx"],
            coords=tile_spec["coords"],
            patch_size=tile_spec["patch_size"],
        )
        tile_specs.append(tile_spec)
        tiles.append(tile)

    tile_infos = tilespecs_to_tileinfos(tile_specs)

    stitched_samples = stitch_prediction(tiles, tile_infos)
    samples = [sample for d in data for sample in np.split(d, d.shape[0])]

    for sample, stitched_sample in zip(samples, stitched_samples):
        np.testing.assert_array_equal(sample, stitched_sample)


@pytest.mark.parametrize("overlaps", [(2, 2), (3, 4), (6, 3)])
@pytest.mark.parametrize(
    "data_shapes,patch_size",
    [
        [[(2, 1, 32, 32), (1, 1, 19, 37), (3, 1, 14, 9)], (8, 8)],
        [[(2, 1, 32, 32), (1, 1, 19, 37), (3, 1, 14, 9)], (8, 5)],
    ],
)
def test_smoke_tiling_2D(
    data_shapes: list[tuple[int, int, int, int]],
    patch_size: tuple[int, int],
    overlaps: tuple[int, int],
):
    _test_smoke_tiling(data_shapes, patch_size, overlaps, axes="SCYX")


@pytest.mark.parametrize("overlaps", [(2, 2, 2), (3, 4, 2), (6, 3, 5)])
@pytest.mark.parametrize(
    "data_shapes,patch_size",
    [
        [[(2, 1, 32, 32, 32), (1, 1, 19, 37, 23), (3, 1, 14, 9, 12)], (8, 8, 8)],
        [[(2, 1, 32, 32, 32), (1, 1, 19, 37, 23), (3, 1, 14, 9, 12)], (8, 5, 7)],
    ],
)
def test_smoke_tiling_3D(
    data_shapes: list[tuple[int, int, int, int, int]],
    patch_size: tuple[int, int, int],
    overlaps: tuple[int, int, int],
):
    _test_smoke_tiling(data_shapes, patch_size, overlaps, axes="SCZYX")
