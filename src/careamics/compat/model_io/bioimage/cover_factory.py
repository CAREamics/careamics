"""Convenience function to create covers for the BMZ."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

color_palette = np.array(
    [
        np.array([255, 195, 0]),  # grey
        np.array([189, 226, 240]),
        np.array([96, 60, 76]),
        np.array([193, 225, 193]),
    ]
)


def _get_norm_slice(array: NDArray) -> NDArray:
    """Get the normalized middle slice of a 4D or 5D array (SC(Z)YX).

    Parameters
    ----------
    array : NDArray
        Array from which to get the middle slice.

    Returns
    -------
    NDArray
        Normalized middle slice of the input array.
    """
    if array.ndim not in (4, 5):
        raise ValueError("Array must be 4D or 5D.")

    channels = array.shape[1] > 1
    z_stack = array.ndim == 5

    # get slice
    if z_stack:
        array_slice = array[0, :, array.shape[2] // 2, ...]
    else:
        array_slice = array[0, ...]

    # channels
    if channels:
        array_slice = np.moveaxis(array_slice, 0, -1)
    else:
        array_slice = array_slice[0, ...]

    # normalize
    array_slice = (
        255
        * (array_slice - array_slice.min())
        / (array_slice.max() - array_slice.min())
    )

    return array_slice.astype(np.uint8)


def _four_channel_image(array: NDArray) -> Image:
    """Convert 4-channel array to Image.

    Parameters
    ----------
    array : NDArray
        Normalized array to convert.

    Returns
    -------
    Image
        Converted array.
    """
    colors = color_palette[np.newaxis, np.newaxis, :, :]
    four_c_array = np.sum(array[..., :4, np.newaxis] * colors, axis=-2).astype(np.uint8)

    return Image.fromarray(four_c_array).convert("RGB")


def _convert_to_image(original_shape: tuple[int, ...], array: NDArray) -> Image:
    """Convert to Image.

    Parameters
    ----------
    original_shape : tuple
        Original shape of the array.
    array : NDArray
        Normalized array to convert.

    Returns
    -------
    Image
        Converted array.
    """
    n_channels = original_shape[1]

    if n_channels > 1:
        if n_channels == 3:
            return Image.fromarray(array).convert("RGB")
        elif n_channels == 2:
            # add an empty channel to the numpy array
            array = np.concatenate([np.zeros_like(array[..., 0:1]), array], axis=-1)

            return Image.fromarray(array).convert("RGB")
        else:  # more than 4
            return _four_channel_image(array[..., :4])
    else:
        return Image.fromarray(array).convert("L").convert("RGB")


def create_cover(directory: Path, array_in: NDArray, array_out: NDArray) -> Path:
    """Create a cover image from input and output arrays.

    Input and output arrays are expected to be SC(Z)YX. For images with a Z
    dimension, the middle slice is taken.

    Parameters
    ----------
    directory : Path
        Directory in which to save the cover.
    array_in : numpy.ndarray
        Array from which to create the cover image.
    array_out : numpy.ndarray
        Array from which to create the cover image.

    Returns
    -------
    Path
        Path to the saved cover image.
    """
    # extract slice and normalize arrays
    slice_in = _get_norm_slice(array_in)
    slice_out = _get_norm_slice(array_out)

    horizontal_split = slice_in.shape[-1] == slice_out.shape[-1]
    if not horizontal_split:
        if slice_in.shape[-2] != slice_out.shape[-2]:
            raise ValueError("Input and output arrays have different shapes.")

    # convert to Image
    image_in = _convert_to_image(array_in.shape, slice_in)
    image_out = _convert_to_image(array_out.shape, slice_out)

    # split horizontally or vertically
    if horizontal_split:
        width = image_in.width // 2

        cover = Image.new("RGB", (image_in.width, image_in.height))
        cover.paste(image_in.crop((0, 0, width, image_in.height)), (0, 0))
        cover.paste(
            image_out.crop(
                (image_in.width - width, 0, image_in.width, image_in.height)
            ),
            (width, 0),
        )
    else:
        height = image_in.height // 2

        cover = Image.new("RGB", (image_in.width, image_in.height))
        cover.paste(image_in.crop((0, 0, image_in.width, height)), (0, 0))
        cover.paste(
            image_out.crop(
                (0, image_in.height - height, image_in.width, image_in.height)
            ),
            (0, height),
        )

    # save
    cover_path = directory / "cover.png"
    cover.save(cover_path)

    return cover_path
