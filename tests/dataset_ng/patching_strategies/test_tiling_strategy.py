import numpy as np
import pytest

from careamics.dataset_ng.dataset import ImageRegionData
from careamics.dataset_ng.legacy_interoperability import imageregions_to_tileinfos
from careamics.dataset_ng.patch_extractor.patch_extractor_factory import (
    create_array_extractor,
)
from careamics.dataset_ng.patching_strategies import TilingStrategy
from careamics.prediction_utils.stitch_prediction import stitch_prediction


def _test_tiling_output(
    data_shapes: list[tuple[int, ...]],
    patch_size: tuple[int, ...],
    overlaps: tuple[int, ...],
    axes: str,
):
    """
    Test that TilingStrategy creates tiles that can be stitched to the original data

    Note this is called by `test_tiling_output_2D` and test_tiling_output_3D
    """
    data = [
        np.arange(np.prod(data_shape)).reshape(data_shape) for data_shape in data_shapes
    ]
    patch_extractor = create_array_extractor(source=data, axes=axes)
    tiling_strategy = TilingStrategy(
        data_shapes=data_shapes, tile_size=patch_size, overlaps=overlaps
    )
    image_regions: list[ImageRegionData] = []
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
        # simulate image region creation in the dataset
        image_region = ImageRegionData(
            data=tile,
            source="array",
            dtype=str(patch_extractor.image_stacks[tile_spec["data_idx"]].data_dtype),
            data_shape=patch_extractor.image_stacks[tile_spec["data_idx"]].data_shape,
            axes=axes,
            region_spec=tile_spec,
        )
        image_regions.append(image_region)

    tile_infos = imageregions_to_tileinfos(image_regions)
    tile_infos = [tile_info[0] for data, tile_info in tile_infos]
    tiles = [image_region.data for image_region in image_regions]

    stitched_samples = stitch_prediction(tiles, tile_infos)
    samples = [sample for d in data for sample in np.split(d, d.shape[0])]

    for sample, stitched_sample in zip(samples, stitched_samples, strict=False):
        np.testing.assert_array_equal(stitched_sample, sample)


@pytest.mark.parametrize("overlaps", [(2, 2), (3, 4), (6, 3)])
@pytest.mark.parametrize(
    "data_shapes,patch_size",
    [
        [[(2, 1, 32, 32), (1, 1, 19, 37), (3, 1, 14, 9), (2, 1, 6, 5)], (8, 8)],
        [[(2, 1, 32, 32), (1, 1, 19, 37), (3, 1, 14, 9), (2, 1, 6, 5)], (8, 5)],
    ],
)
def test_tiling_output_2D(
    data_shapes: list[tuple[int, int, int, int]],
    patch_size: tuple[int, int],
    overlaps: tuple[int, int],
):
    _test_tiling_output(data_shapes, patch_size, overlaps, axes="SCYX")


@pytest.mark.parametrize("overlaps", [(2, 2, 2), (3, 4, 2), (6, 3, 5)])
@pytest.mark.parametrize(
    "data_shapes,patch_size",
    [
        [
            [
                (2, 1, 32, 32, 32),
                (1, 1, 19, 37, 23),
                (3, 1, 14, 9, 12),
                (2, 1, 6, 5, 4),
            ],
            (8, 8, 8),
        ],
        [
            [
                (2, 1, 32, 32, 32),
                (1, 1, 19, 37, 23),
                (3, 1, 14, 9, 12),
                (2, 1, 6, 5, 4),
            ],
            (8, 5, 7),
        ],
    ],
)
def test_tiling_output_3D(
    data_shapes: list[tuple[int, int, int, int, int]],
    patch_size: tuple[int, int, int],
    overlaps: tuple[int, int, int],
):
    _test_tiling_output(data_shapes, patch_size, overlaps, axes="SCZYX")


@pytest.mark.parametrize("overlaps", [(2, 2), (3, 4), (6, 3)])
@pytest.mark.parametrize(
    "data_shapes,patch_size",
    [
        [[(2, 1, 32, 32), (1, 1, 19, 37), (3, 1, 14, 9)], (8, 8)],
        [[(2, 1, 32, 32), (1, 1, 19, 37), (3, 1, 14, 9)], (8, 5)],
    ],
)
def test_whole_image_covered_2d(
    data_shapes: list[tuple[int, int, int, int]],
    patch_size: tuple[int, int],
    overlaps: tuple[int, int],
):
    patching_strategy = TilingStrategy(data_shapes, patch_size, overlaps)
    patch_specs = patching_strategy.tile_specs

    # track where patches have been sampled from
    tracking_arrays = [np.zeros(data_shape, dtype=bool) for data_shape in data_shapes]
    for patch_spec in patch_specs:
        tracking_array = tracking_arrays[patch_spec["data_idx"]]
        spatial_slice = tuple(
            slice(c, c + ps)
            for c, ps in zip(
                patch_spec["coords"], patch_spec["patch_size"], strict=False
            )
        )
        # set to true where the patches would be sampled from
        tracking_array[(patch_spec["sample_idx"], slice(None), *spatial_slice)] = True

    for tracking_array in tracking_arrays:
        # if the patch specs covered all the image all the values should be true
        assert tracking_array.all()


@pytest.mark.parametrize("overlaps", [(2, 2, 2), (3, 4, 2), (6, 3, 5)])
@pytest.mark.parametrize(
    "data_shapes,patch_size",
    [
        [[(2, 1, 32, 32, 32), (1, 1, 19, 37, 23), (3, 1, 14, 9, 12)], (8, 8, 8)],
        [[(2, 1, 32, 32, 32), (1, 1, 19, 37, 23), (3, 1, 14, 9, 12)], (8, 5, 7)],
    ],
)
def test_whole_image_covered_3d(
    data_shapes: list[tuple[int, int, int, int]],
    patch_size: tuple[int, int],
    overlaps: tuple[int, int],
):
    patching_strategy = TilingStrategy(data_shapes, patch_size, overlaps)
    patch_specs = patching_strategy.tile_specs

    # track where patches have been sampled from
    tracking_arrays = [np.zeros(data_shape, dtype=bool) for data_shape in data_shapes]
    for patch_spec in patch_specs:
        tracking_array = tracking_arrays[patch_spec["data_idx"]]
        spatial_slice = tuple(
            slice(c, c + ps)
            for c, ps in zip(
                patch_spec["coords"], patch_spec["patch_size"], strict=False
            )
        )
        # set to true where the patches would be sampled from
        tracking_array[(patch_spec["sample_idx"], slice(None), *spatial_slice)] = True

    for tracking_array in tracking_arrays:
        # if the patch specs covered all the image all the values should be true
        assert tracking_array.all()
