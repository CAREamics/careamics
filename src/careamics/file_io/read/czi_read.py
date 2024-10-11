"""Functions to read region of interest from CZI data."""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
from pylibCZIrw import czi as pyczi

from careamics.utils.logging import get_logger

logger = get_logger(__name__)


def squeeze_possible(array: np.ndarray) -> np.ndarray:
    """Dimension correction by squeezing.

    Parameters
    ----------
    array : np.array
        Input array of data.

    Returns
    -------
    np.array
        Output data with axis=1 squeezed.
    """
    if array.shape[1] == 1:
        return array.squeeze(axis=1)
    return array


def read_czi_roi(
    file_path: Path,  # Keep this to match protocol but don't use it
    czi_reader: pyczi.CziReader,
    patch_size: Union[list[int], tuple[int, ...]],
    coords: Union[list[int], tuple[int, ...]],
    plane: dict[str, int],
    scene: Optional[int] = None,
) -> np.ndarray:
    """Read an ROI from a CZI.

    Parameters
    ----------
    file_path : Path
        CZI filepath (not used at the moment).
    czi_reader : pyczi.CziReader
        CZI reader object.
    patch_size : Union[list[int], tuple[int, ...]]
       Patch_size to extract from the data (it has to be 4 dimensional with CZXY).
    coords : Union[list[int], tuple[int, ...]]
        Coordinates of the roi in Z,X and Y dimensions.
    plane : dict[str, int]
        The plane where the data is extracted from.
    scene : int
        The scene point to extract the data from czi, default is None.

    Returns
    -------
    np.array
       The ROI region of data read from the CZI file of size (C(Z)XY).

    Raises
    ------
    ValueError
        When patchsize does not match 4
    ValueError
        when coords length does not match 3
    ValueError
        plane or coords is None
    e
        exception when the czi cannot be read
    """
    # Check if patch_size has 4 dimensions
    if len(patch_size) != 4:
        raise ValueError("patch_size must have 4 dimensions")

    # Check if coords has length 3
    if coords is not None and len(coords) != 3:
        raise ValueError("coords must have length 3")

    # Check if plane and roi are not None
    if plane is None or coords is None:
        raise ValueError("plane and coords cannot be None")

    try:
        roi = (coords[1], coords[2], *patch_size[2:])
        # flip the XY to YX as the czi reader always reads to YX
        patch_size = list(patch_size)
        patch_size = patch_size[:-2] + [patch_size[-1], patch_size[-2]]
        image = np.zeros(patch_size)
        extract_plane = plane.copy()
        for depth in range(patch_size[1]):
            for channel in range(patch_size[0]):
                extract_plane.update(
                    {
                        "Z": (
                            coords[0] + depth if "Z" not in plane.keys() else plane["Z"]
                        ),
                        "C": channel,
                        "T": (
                            coords[0] + depth if "T" not in plane.keys() else plane["T"]
                        ),
                    }
                )
                image_roi = czi_reader.read(roi=roi, plane=extract_plane, scene=scene)
                image[channel, depth] = image_roi[..., 0]
        return squeeze_possible(image)

    except (ValueError, OSError) as e:
        logging.exception(f"Exception in file reader {czi_reader}: {e}, skipping it.")
        raise e
