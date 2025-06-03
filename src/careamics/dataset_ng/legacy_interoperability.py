"""
A module for utility functions that adapts the new dataset outputs to work with previous
code until it is updated.
"""

from collections.abc import Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray

from careamics.config.tile_information import TileInformation

from .dataset import ImageRegionData
from .patching_strategies import TileSpecs


def imageregions_to_tileinfos(
    image_regions: Sequence[ImageRegionData],
) -> list[tuple[NDArray, list[TileInformation]]]:
    """
    Converts a series of `TileSpecs` dictionaries to `TileInformation` pydantic class.

    Parameters
    ----------
    image_regions : sequence of ImageRegionData
        A list of ImageRegionData, it must have an instance of `TileSpecs` as it's
        `region_data` field.

    Returns
    -------
    list of TileInformation
        The converted tile information.
    """

    tile_infos: list[TileInformation] = []

    data = [image_region.data for image_region in image_regions]
    tile_specs = [image_region.region_spec for image_region in image_regions]

    data_indices: NDArray[np.int_] = np.array(
        [tile_spec["data_idx"] for tile_spec in tile_specs], dtype=int
    )
    unique_data_indices = np.unique(data_indices)
    # data_idx denotes which image stack a patch belongs to
    # separate TileSpecs by image_stack
    for data_idx in unique_data_indices:
        # collect all ImageRegions
        data_image_regions: list[ImageRegionData] = [
            image_region
            for image_region in image_regions
            if image_region.region_spec["data_idx"] == data_idx
        ]

        # --- find last indices
        # make sure tiles belonging to the same sample are together
        data_image_regions.sort(
            key=lambda image_region: image_region.region_spec["sample_idx"]
        )
        sample_indices = np.array(
            [
                image_region.region_spec["sample_idx"]
                for image_region in data_image_regions
            ]
        )
        # reverse array so indices returned are at far edge
        _, unique_indices = np.unique(sample_indices[::-1], return_index=True)
        # un reverse indices
        last_indices = len(sample_indices) - 1 - unique_indices

        # convert each ImageRegionData to tile_info
        for i, image_region in enumerate(data_image_regions):
            last_tile = i in last_indices
            tile_info = _imageregion_to_tileinfo(image_region, last_tile)
            tile_infos.append(tile_info)

    return [
        (data, [tile_info]) for data, tile_info in zip(data, tile_infos, strict=False)
    ]


def _imageregion_to_tileinfo(
    image_region: ImageRegionData, last_tile: bool
) -> TileInformation:
    """
    Convert a single `ImageRegionData` instance to a `TileInformation` instance. Whether
    it is the last tile in a sequence needs to be supplied.

    Parameters
    ----------
    image_region : ImageRegionData
        An instance of `ImageRegionData`, it must have an instance of `TileSpecs` as
        it's `region_data` field.
    last_tile : bool
        Whether a tile is the last tile in a sequence, for stitching.

    Returns
    -------
    TileInformation
        A tile information object.

    Raises
    ------
    KeyError
        If `image_region.region_spec` does not contain the keys: {'crop_coords',
        'crop_size', 'stitch_coords'}.
    """
    patch_spec = image_region.region_spec
    data_shape = image_region.data_shape

    # TODO: In python 3.11 and greater, NamedTuples can inherit from Generic
    #   so we could do image_region: ImageRegionData[TileSpecs]
    #   and not have to do this check here + cast
    # make sure image_region.region_spec is TileSpec
    if (
        ("crop_coords" not in patch_spec)
        or ("crop_size" not in patch_spec)
        or ("stitch_coords" not in patch_spec)
    ):
        raise KeyError(
            "Could not find all keys: {'crop_coords', 'crop_size', 'stitch_coords'} in "
            "`image_region.region_spec`."
        )
    tile_spec = cast(TileSpecs, patch_spec)  # ugly cast for mypy
    return _tilespec_to_tileinfo(tile_spec, data_shape, last_tile)


def _tilespec_to_tileinfo(
    tile_spec: TileSpecs, data_shape: Sequence[int], last_tile: bool
) -> TileInformation:
    """
    Convert a single `TileSpec` to a `TileInformation`. Whether it is the last tile
    needs to be supplied.

    Parameters
    ----------
    tile_spec : TileSpecs
        A tile spec dictionary.
    data_shape : sequence of int
        The original shape of the data the tile came from, labeling the dimensions of
        axes SC(Z)YX.
    last_tile : bool
        Whether a tile is the last tile in a sequence, for stitching.

    Returns
    -------
    TileInformation
        A tile information object.
    """
    overlap_crop_coords = tuple(
        (
            tile_spec["crop_coords"][i],
            tile_spec["crop_coords"][i] + tile_spec["crop_size"][i],
        )
        for i in range(len(tile_spec["crop_coords"]))
    )
    stitch_coords = tuple(
        (
            tile_spec["stitch_coords"][i],
            tile_spec["stitch_coords"][i] + tile_spec["crop_size"][i],
        )
        for i in range(len(tile_spec["crop_coords"]))
    )
    return TileInformation(
        array_shape=tuple(data_shape[1:]),  # remove sample dimension
        last_tile=last_tile,
        overlap_crop_coords=overlap_crop_coords,
        stitch_coords=stitch_coords,
        sample_id=tile_spec["sample_idx"],
    )
