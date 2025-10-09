"""Prediction utility functions."""

import builtins
from typing import Union

import numpy as np
from numpy.typing import NDArray

from careamics.config.tile_information import TileInformation


class TilingMode:
    """Enum for the tiling mode."""

    TrimBoundary = 0
    PadBoundary = 1
    ShiftBoundary = 2


def stitch_prediction_vae(predictions, dset) -> NDArray:
    """
    Stitch predictions back together using dataset's index manager.

    Parameters
    ----------
    predictions : np.ndarray
        Array of predictions with shape (n_tiles, channels, height, width).
    dset : Dataset
        Dataset object with idx_manager containing tiling information.

    Returns
    -------
    np.ndarray
        Stitched array with shape matching the original data shape.
    """
    mng = dset.idx_manager

    # if there are more channels, use all of them.
    shape = list(dset.get_data_shape())
    shape[-1] = max(shape[-1], predictions.shape[1])

    output = np.zeros(shape, dtype=predictions.dtype)
    # frame_shape = dset.get_data_shape()[:-1]
    for dset_idx in range(predictions.shape[0]):
        # loc = get_location_from_idx(dset, dset_idx, predictions.shape[-2],
        # predictions.shape[-1])
        # grid start, grid end
        gs = np.array(mng.get_location_from_dataset_idx(dset_idx), dtype=int)
        ge = gs + mng.grid_shape

        # patch start, patch end
        ps = gs - mng.patch_offset()
        pe = ps + mng.patch_shape

        # valid grid start, valid grid end
        vgs = np.array([max(0, x) for x in gs], dtype=int)
        vge = np.array(
            [min(x, y) for x, y in zip(ge, mng.data_shape, strict=False)], dtype=int
        )

        if mng.tiling_mode == TilingMode.ShiftBoundary:
            for dim in range(len(vgs)):
                if ps[dim] == 0:
                    vgs[dim] = 0
                if pe[dim] == mng.data_shape[dim]:
                    vge[dim] = mng.data_shape[dim]

        # relative start, relative end. This will be used on pred_tiled
        rs = vgs - ps
        re = rs + (vge - vgs)

        for ch_idx in range(predictions.shape[1]):
            if len(output.shape) == 4:
                # channel dimension is the last one.
                output[vgs[0] : vge[0], vgs[1] : vge[1], vgs[2] : vge[2], ch_idx] = (
                    predictions[dset_idx][ch_idx, rs[1] : re[1], rs[2] : re[2]]
                )
            elif len(output.shape) == 5:
                # channel dimension is the last one.
                assert vge[0] - vgs[0] == 1, "Only one frame is supported"
                output[
                    vgs[0], vgs[1] : vge[1], vgs[2] : vge[2], vgs[3] : vge[3], ch_idx
                ] = predictions[dset_idx][
                    ch_idx, rs[1] : re[1], rs[2] : re[2], rs[3] : re[3]
                ]
            else:
                raise ValueError(f"Unsupported shape {output.shape}")

    return output


# TODO: why not allow input and output of torch.tensor ?
def stitch_prediction(
    tiles: list[np.ndarray],
    tile_infos: list[TileInformation],
) -> list[np.ndarray]:
    """
    Stitch tiles back together to form a full image(s).

    Tiles are of dimensions SC(Z)YX, where C is the number of channels and can be a
    singleton dimension.

    Parameters
    ----------
    tiles : list of numpy.ndarray
        Cropped tiles and their respective stitching coordinates. Can contain tiles
        from multiple images.
    tile_infos : list of TileInformation
        List of information and coordinates obtained from
        `dataset.tiled_patching.extract_tiles`.

    Returns
    -------
    list of numpy.ndarray
        Full image(s).
    """
    # Find where to split the lists so that only info from one image is contained.
    # Do this by locating the last tiles of each image.
    last_tiles = [tile_info.last_tile for tile_info in tile_infos]
    last_tile_position = np.where(last_tiles)[0]
    image_slices = [
        slice(
            None if i == 0 else last_tile_position[i - 1] + 1, last_tile_position[i] + 1
        )
        for i in range(len(last_tile_position))
    ]
    image_predictions = []
    # slice the lists and apply stitch_prediction_single to each in turn.
    for image_slice in image_slices:
        image_predictions.append(
            stitch_prediction_single(tiles[image_slice], tile_infos[image_slice])
        )
    return image_predictions


def stitch_prediction_single(
    tiles: list[NDArray],
    tile_infos: list[TileInformation],
) -> NDArray:
    """
    Stitch tiles back together to form a full image.

    Tiles are of dimensions SC(Z)YX, where C is the number of channels and can be a
    singleton dimension.

    Parameters
    ----------
    tiles : list of numpy.ndarray
        Cropped tiles and their respective stitching coordinates.
    tile_infos : list of TileInformation
        List of information and coordinates obtained from
        `dataset.tiled_patching.extract_tiles`.

    Returns
    -------
    numpy.ndarray
        Full image, with dimensions SC(Z)YX.
    """
    # TODO: this is hacky... need a better way to deal with when input channels and
    #   target channels do not match
    if len(tile_infos[0].array_shape) == 4:
        # 4 dimensions => 3 spatial dimensions so -4 is channel dimension
        tile_channels = tiles[0].shape[-4]
    elif len(tile_infos[0].array_shape) == 3:
        # 3 dimensions => 2 spatial dimensions so -3 is channel dimension
        tile_channels = tiles[0].shape[-3]
    else:
        # Note pretty sure this is unreachable because array shape is already
        #   validated by TileInformation
        raise ValueError(
            f"Unsupported number of output dimension {len(tile_infos[0].array_shape)}"
        )
    # retrieve whole array size, add S dim and use number of channels in tile
    input_shape = (1, tile_channels, *tile_infos[0].array_shape[1:])
    predicted_image = np.zeros(input_shape, dtype=np.float32)

    for tile, tile_info in zip(tiles, tile_infos, strict=False):

        # Compute coordinates for cropping predicted tile
        crop_slices: tuple[Union[builtins.ellipsis, slice], ...] = (
            ...,
            *[slice(c[0], c[1]) for c in tile_info.overlap_crop_coords],
        )

        # Crop predited tile according to overlap coordinates
        cropped_tile = tile[crop_slices]

        # Insert cropped tile into predicted image using stitch coordinates
        image_slices = (..., *[slice(c[0], c[1]) for c in tile_info.stitch_coords])

        # TODO fix mypy error here, potentially due to numpy 2
        predicted_image[image_slices] = cropped_tile.astype(np.float32)  # type: ignore

    return predicted_image
