"""Utility functions to construct ImageStacks from different sources."""

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import zarr
from numpy.typing import NDArray
from zarr.storage import StorePath

from careamics.config.validators import check_czi_axes_validity
from careamics.image_io import ReadFunc

from ..image_stack import (
    FileImageStack,
    InMemoryImageStack,
    ZarrImageStack,
)
from ..image_stack.czi_image_stack import CziImageStack
from ..image_stack.zarr_access import ZarrNode, ZarrPythonAccess
from .zarr_utils import is_ome_zarr, is_valid_uri, to_zarr_node

if TYPE_CHECKING:
    from careamics.image_io.read import ReadFunc


def load_arrays(source: Sequence[NDArray[Any]], axes: str) -> list[InMemoryImageStack]:
    """
    Load ImageStacks from a sequence of numpy arrays.

    Parameters
    ----------
    source : sequence of numpy.ndarray
        Source arrays of the data.
    axes : str
        Original axes of the data, must be a subset of "STCZYX".

    Returns
    -------
    list[InMemoryImageStack]
        ImageStacks created from the input arrays.
    """
    return [InMemoryImageStack.from_array(data=array, axes=axes) for array in source]


# TIFF case
def load_tiffs(source: Sequence[Path], axes: str) -> list[InMemoryImageStack]:
    """
    Load ImageStacks from a sequence of TIFF files.

    Parameters
    ----------
    source : sequence of Path
        Source files for the data.
    axes : str
        Original axes of the data, must be a subset of "STCZYX".

    Returns
    -------
    list[InMemoryImageStack]
        ImageStacks created from the source files.
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
    source : sequence of Path
        Source files for the data.
    axes : str
        Original axes of the data, must be a subset of "STCZYX".

    Returns
    -------
    list[FileImageStack]
        Lazily loaded ImageStacks backed by the source files.
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
    Load ImageStacks from a sequence of files of a custom type.

    Parameters
    ----------
    source : sequence of Path
        Source files for the data.
    axes : str
        Original axes of the data, must be a subset of "STCZYX".
    read_func : ReadFunc
        A function to read the custom file type, see the `ReadFunc` protocol.
    read_kwargs : dict of {str: Any}
        Additional arguments passed to the custom `read_func`.

    Returns
    -------
    list[InMemoryImageStack]
        ImageStacks created from the custom files.
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
        Source zarr file paths or URIs.
    axes : str
        Original axes of the data, must be a subset of "STCZYX".

    Returns
    -------
    list[ZarrImageStack]
        Image stacks created from the sources.
    """
    image_stacks: list[ZarrImageStack] = []
    access = ZarrPythonAccess()

    for data_source in source:
        data_str = str(data_source)

        if not (data_str.endswith(".zarr") or is_valid_uri(data_str)):
            raise ValueError(
                f"Source '{data_source}' is neither a zarr file nor a file URI."
            )

        root_node = to_zarr_node(data_source)
        root_type = access.resolve_node_type(root_node)

        if root_type == "array":
            image_stacks.append(
                ZarrImageStack(
                    node=ZarrNode(
                        store_uri=root_node.store_uri,
                        path=root_node.path,
                        node_type="array",
                    ),
                    axes=axes,
                    access=access,
                )
            )
            continue

        opened = access.open_node(root_node, mode="r")
        if not isinstance(opened, zarr.Group):
            raise TypeError(
                f"Content at '{data_str}' is neither a zarr.Group nor a zarr.Array."
            )

        if root_node.path == "" and is_ome_zarr(opened):
            # TODO implement OME-Zarr resolution against NGFF 0.5.
            raise NotImplementedError(
                "OME-Zarr support is not yet implemented when providing a path to the "
                "zarr store."
            )

        array_paths = access.list_array_paths(root_node)
        array_paths.sort()

        for array_path in array_paths:
            full_path = (
                array_path if root_node.path == "" else f"{root_node.path}/{array_path}"
            )
            image_stacks.append(
                ZarrImageStack(
                    node=ZarrNode(
                        store_uri=root_node.store_uri,
                        path=full_path,
                        node_type="array",
                    ),
                    axes=axes,
                    access=access,
                )
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
    source : sequence of Path
        Source files for the data.
    axes : str
        Axes of the data, must be either "SCYX", "SCZYX" or "SCTYX". Depth axis is
        inferred from the axes string. If this string ends with `"ZYX"` or `"TYX"`, the
        data will consist of 3-D.

    Returns
    -------
    list[CziImageStack]
        Image stacks created from the CZI files.

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
