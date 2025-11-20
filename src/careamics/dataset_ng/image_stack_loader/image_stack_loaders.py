from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from numpy.typing import NDArray

from careamics.file_io import ReadFunc

from ..image_stack import (
    FileImageStack,
    InMemoryImageStack,
)

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
