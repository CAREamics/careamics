from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import zarr
from numpy.typing import NDArray
from zarr.storage import StorePath

from careamics.config.validators import check_czi_axes_validity
from careamics.file_io import ReadFunc

from ..image_stack import (
    FileImageStack,
    InMemoryImageStack,
    ZarrImageStack,
)
from ..image_stack.czi_image_stack import CziImageStack
from .zarr_utils import collect_arrays, decipher_zarr_uri, is_ome_zarr, is_valid_uri

if TYPE_CHECKING:
    from careamics.file_io.read import ReadFunc


def load_arrays(source: Sequence[NDArray[Any]], axes: str) -> list[InMemoryImageStack]:
    """
    Load image stacks from a sequence of numpy arrays.

    Parameters
    ----------
    source: sequence of numpy.ndarray
        The source arrays of the data.
    axes: str
        The original axes of the data, must be a subset of "STCZYX".

    Returns
    -------
    list[InMemoryImageStack]
    """
    return [InMemoryImageStack.from_array(data=array, axes=axes) for array in source]


# TIFF case
def load_tiffs(source: Sequence[Path], axes: str) -> list[InMemoryImageStack]:
    """
    Load image stacks from a sequence of TIFF files.

    Parameters
    ----------
    source: sequence of Path
        The source files for the data.
    axes: str
        The original axes of the data, must be a subset of "STCZYX".

    Returns
    -------
    list[InMemoryImageStack]
    """
    return [InMemoryImageStack.from_tiff(path=path, axes=axes) for path in source]


# TODO: better name
# iter Tiff
def load_iter_tiff(source: Sequence[Path], axes: str) -> list[FileImageStack]:
    # TODO: better docs
    """
    Load image stacks from a sequence of TIFF files.

    Parameters
    ----------
    source: sequence of Path
        The source files for the data.
    axes: str
        The original axes of the data, must be a subset of "STCZYX".

    Returns
    -------
    list[FileImageStack]
    """
    return [FileImageStack.from_tiff(path=path, axes=axes) for path in source]


# Custom file type case (loaded into memory)
def load_custom_file(
    source: Sequence[Path],
    axes: str,
    *,
    read_func: ReadFunc,
    read_kwargs: dict[str, Any],
) -> list[InMemoryImageStack]:
    """
    Load image stacks from a sequence of files of a custom type.

    Parameters
    ----------
    source: sequence of Path
        The source files for the data.
    axes: str
        The original axes of the data, must be a subset of "STCZYX".
    read_func : ReadFunc
        A function to read the custom file type, see the `ReadFunc` protocol.
    read_kwargs : dict of {str: Any}
        Kwargs that will be passed to the custom `read_func`.

    Returns
    -------
    list[InMemoryImageStack]
    """
    # TODO: lazy loading custom files
    return [
        InMemoryImageStack.from_custom_file_type(
            path=path,
            axes=axes,
            read_func=read_func,
            **read_kwargs,
        )
        for path in source
    ]


def load_zarrs(
    source: Sequence[str | Path | StorePath],
    axes: str,
) -> list[ZarrImageStack]:
    """Create a list of ZarrImageStack from a sequence of zarr file paths or URIs.

    File paths must point to a zarr store (ending with .zarr) and URIs must be in the
    format "file://path/to/zarr_store.zarr/group/path/array_name".

    If the zarr file is an OME-Zarr, the specified multiscale level will be used. Note
    that OME-Zarrs are only supported when providing a path to the zarr store, not when
    using a file URI. One can, however, provide a file URI to the specific resolution
    array within the OME-Zarr.

    Parameters
    ----------
    source : sequence of str or Path
        The source zarr file paths or URIs.
    axes : str
        The original axes of the data, must be a subset of "STCZYX".

    Returns
    -------
    list of ZarrImageStack
        A list of ZarrImageStack created from the sources.
    """

    image_stacks: list[ZarrImageStack] = []

    for data_source in source:
        data_str = str(data_source)

        # either a path to a zarr file or a uri "file://path/to/zarr/array_path"
        if data_str.endswith(".zarr"):
            zarr_group = zarr.open_group(data_str, mode="r")

            # test if ome-zarr (minimum assumption of multiscales)
            if is_ome_zarr(zarr_group):
                # TODO placeholder for handling OME-Zarr
                # - Need to potentially select multiscale level
                # - Extract axes and compare with provided ones
                raise NotImplementedError(
                    "OME-Zarr support is not yet implemented when providing a "
                    "path to the zarr store. Please provide a file URI to the "
                    "specific array within the OME-Zarr."
                )
            else:
                # collect all arrays
                array_paths = collect_arrays(zarr_group)

                # sort names
                array_paths.sort()

            # instantiate image stacks
            for array_path in array_paths:
                image_stacks.append(
                    ZarrImageStack(group=zarr_group, data_path=array_path, axes=axes)
                )

        elif is_valid_uri(data_str):
            # decipher the uri and open the group
            store_path, parent_path, name = decipher_zarr_uri(data_str)

            zarr_group = zarr.open_group(store_path, path=parent_path, mode="r")
            content = zarr_group[name]

            # assert if group or array
            if isinstance(content, zarr.Group):
                array_paths = collect_arrays(content)

                # sort the names
                array_paths.sort()

                for array_path in array_paths:
                    image_stacks.append(
                        ZarrImageStack(group=content, data_path=array_path, axes=axes)
                    )
            else:
                if not isinstance(content, zarr.Array):
                    raise TypeError(
                        f"Content at '{data_str}' is neither a zarr.Group nor "
                        f"a zarr.Array."
                    )

                # create image stack from a single array
                image_stacks.append(
                    ZarrImageStack(
                        group=zarr_group,
                        data_path=name,
                        axes=axes,
                    )
                )

        else:
            raise ValueError(
                f"Source '{data_source}' is neither a zarr file nor a file URI."
            )

    return image_stacks


def load_czis(
    source: Sequence[Path],
    axes: str,
) -> list[CziImageStack]:
    """
    Load CZI image stacks from a sequence of CZI files paths.

    If the CZI files contain multiple scenes, one image stack will be created for
    each scene.

    Axes should be in the format "SC(Z/T)YX", where Z or T are optional, and S and C
    can be singleton dimensions, but must be provided.

    Parameters
    ----------
    source: sequence of Path
        The source files for the data.
    axes: str
        Specifies which axes of the data to use and how.
        If this string ends with `"ZYX"` or `"TYX"`, the data will consist of 3-D
        patches, using `Z` or `T` as third dimension, respectively.
        If the string does not end with "ZYX", the data will consist of 2-D patches.

    Returns
    -------
    list[CziImageStack]

    Raises
    ------
    ValueError
        If the provided axes are not valid.
    """
    if check_czi_axes_validity(axes) is False:
        raise ValueError(
            f"Provided axes '{axes}' are not valid. Axes must be in the `SC(Z/T)YX` "
            f"format, where Z or T are optional, and S and C can be singleton "
            f"dimensions, but must be provided."
        )

    depth_axis: Literal["none", "Z", "T"] = "none"
    if axes.endswith("TYX"):
        depth_axis = "T"
    elif axes.endswith("ZYX"):
        depth_axis = "Z"

    image_stacks: list[CziImageStack] = []
    for path in source:
        scene_rectangles = CziImageStack.get_bounding_rectangles(path)
        image_stacks.extend(
            CziImageStack(path, scene=scene, depth_axis=depth_axis)
            for scene in scene_rectangles.keys()
        )
    return image_stacks
