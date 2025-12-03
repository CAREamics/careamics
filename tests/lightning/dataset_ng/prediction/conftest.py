import numpy as np
import pytest
from numpy.typing import NDArray

from careamics.dataset_ng.dataset import ImageRegionData
from careamics.dataset_ng.image_stack_loader import load_arrays
from careamics.dataset_ng.patch_extractor import PatchExtractor
from careamics.dataset_ng.patching_strategies import TileSpecs, TilingStrategy


@pytest.fixture
def tiles(n_data, shape, axes, channels) -> tuple[NDArray, list[ImageRegionData]]:
    """A fixture that mimicks the output of CAREamics NG Dataset with tiling.

    Parameters
    ----------
    n_data : int
        Number of images to create, corresponding to data_idx.
    shape : tuple of int
        Shape of each image.
    axes : str
        Axes of each image.
    channels : list of int | None
        Channels to extract, or None to extract all channels.

    Returns
    -------
    numpy.ndarray
        The original array of shape (n_data, *shape).
    list of ImageRegionData
        List of extracted tiles as ImageRegionData.
    """
    # create data
    array = np.arange(n_data * np.prod(shape)).reshape((n_data, *shape))

    # create tiling strategy
    if "Z" in axes:
        tile_size = (8, 16, 16)
        overlaps = (2, 4, 4)
    else:
        tile_size = (16, 16)
        overlaps = (4, 4)

    if "S" in axes:
        if "C" in axes:
            shape_with_sc = shape
        else:
            shape_with_sc = (shape[0], 1, *shape[1:])
    else:
        if "C" in axes:
            shape_with_sc = (1, *shape)
        else:
            shape_with_sc = (1, 1, *shape)

    tiling_strategy = TilingStrategy(
        data_shapes=[shape_with_sc] * n_data, patch_size=tile_size, overlaps=overlaps
    )
    n_tiles = tiling_strategy.n_patches

    # create patch extractor
    image_stacks = load_arrays(source=[array[i] for i in range(n_data)], axes=axes)
    patch_extractor = PatchExtractor(image_stacks)

    # extract tiles
    tiles: list[ImageRegionData] = []
    for i in range(n_tiles):
        tile_spec: TileSpecs = tiling_strategy.get_patch_spec(i)
        tile = patch_extractor.extract_channel_patch(
            data_idx=tile_spec["data_idx"],
            sample_idx=tile_spec["sample_idx"],
            channels=channels,
            coords=tile_spec["coords"],
            patch_size=tile_spec["patch_size"],
        )

        # adjust channels as done in dataset
        data_shape = list(shape)
        if channels is not None:  # this means "C" is in axes
            c_idx = axes.index("C")
            data_shape[c_idx] = len(channels)

        tiles.append(
            ImageRegionData(
                data=tile,
                source=str(tile_spec["data_idx"]),  # for testing purposes
                dtype=str(tile.dtype),
                data_shape=data_shape,
                axes=axes,
                region_spec=tile_spec,
            )
        )
    return array, tiles


@pytest.fixture
def zarr_tiles(
    tmp_path, tiles, axes, channels
) -> tuple[NDArray, list[ImageRegionData]]:
    """A fixture that mimicks the output of CAREamics NG Dataset with tiling."""
    # create data
    array, all_tiles = tiles

    if "S" in axes:
        if "C" in axes:
            if channels is not None and len(channels) == 1:
                chunks = (1, 8, 8)
            else:
                chunks = (1, 8, 8)
        else:
            chunks = (1, 8, 8)
    else:
        if "C" in axes:
            if channels is not None and len(channels) == 1:
                chunks = (8, 8)
            else:
                chunks = (1, 8, 8)
        else:
            chunks = (8, 8)

    zarr_uri = "file://" + str(tmp_path / "test.zarr")

    z_tiles: list[ImageRegionData] = []
    for tile in all_tiles:
        z_tiles.append(
            ImageRegionData(
                data=tile.data,
                source=str(zarr_uri) + "/" + str(tile.region_spec["data_idx"]),
                dtype=tile.dtype,
                data_shape=tile.data_shape,
                axes=tile.axes,
                region_spec=tile.region_spec,
                chunks=chunks,
            )
        )
    return array, z_tiles
