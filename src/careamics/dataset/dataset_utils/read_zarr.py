"""Function to read zarr images."""

from typing import Union

from zarr import Group, core, hierarchy, storage


def read_zarr(
    zarr_source: Group, axes: str
) -> Union[core.Array, storage.DirectoryStore, hierarchy.Group]:
    """Read a file and returns a pointer.

    Parameters
    ----------
    zarr_source : Group
        Zarr storage.
    axes : str
        Axes of the data.

    Returns
    -------
    np.ndarray
        Pointer to zarr storage.

    Raises
    ------
    ValueError, OSError
        if a file is not a valid tiff or damaged.
    ValueError
        if data dimensions are not 2, 3 or 4.
    ValueError
        if axes parameter from config is not consistent with data dimensions.
    """
    if isinstance(zarr_source, hierarchy.Group):
        array = zarr_source[0]

    elif isinstance(zarr_source, storage.DirectoryStore):
        raise NotImplementedError("DirectoryStore not supported yet")

    elif isinstance(zarr_source, core.Array):
        # array should be of shape (S, (C), (Z), Y, X), iterating over S ?
        if zarr_source.dtype == "O":
            raise NotImplementedError("Object type not supported yet")
        else:
            array = zarr_source
    else:
        raise ValueError(f"Unsupported zarr object type {type(zarr_source)}")

    # sanity check on dimensions
    if len(array.shape) < 2 or len(array.shape) > 4:
        raise ValueError(
            f"Incorrect data dimensions. Must be 2, 3 or 4 (got {array.shape})."
        )

    # sanity check on axes length
    if len(axes) != len(array.shape):
        raise ValueError(f"Incorrect axes length (got {axes}).")

    # arr = fix_axes(arr, axes)
    return array
