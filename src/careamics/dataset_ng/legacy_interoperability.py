import numpy as np
from numpy.typing import NDArray

from careamics.config.tile_information import TileInformation

from .patching_strategies import TileSpecs


# when #420 is merged this will actually take a list of ImageRegion
def tilespecs_to_tileinfos(tile_specs: list[TileSpecs]) -> list[TileInformation]:

    tile_infos: list[TileInformation] = []

    data_indices: NDArray[np.int_] = np.array(
        [tile_spec["data_idx"] for tile_spec in tile_specs], dtype=int
    )
    unique_data_indices = np.unique(data_indices)
    for data_idx in unique_data_indices:
        data_tile_specs: list[TileSpecs] = list(
            filter(lambda tile_spec: tile_spec["data_idx"] == data_idx, tile_specs)
        )

        # --- find last indices
        # make sure tiles belonging to the same sample are together
        data_tile_specs.sort(key=lambda tile_spec: tile_spec["sample_idx"])
        sample_indices = np.array(
            [tile_spec["sample_idx"] for tile_spec in data_tile_specs]
        )
        # reverse array so indices returned are at far edge
        _, unique_indices = np.unique(sample_indices[::-1], return_index=True)
        # un reverse indices
        last_indices = len(sample_indices) - 1 - unique_indices

        # convert each tilespec to tile_info

        for i, tile_spec in enumerate(data_tile_specs):
            last_tile = i in last_indices
            tile_info = _tilespec_to_tileinfo(tile_spec, last_tile)
            tile_infos.append(tile_info)

    return tile_infos


def _tilespec_to_tileinfo(tile_spec: TileSpecs, last_tile: bool) -> TileInformation:
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
        array_shape=tuple(tile_spec["data_shape"][1:]),  # remove sample dimension
        last_tile=last_tile,
        overlap_crop_coords=overlap_crop_coords,
        stitch_coords=stitch_coords,
        sample_id=tile_spec["sample_idx"],
    )
