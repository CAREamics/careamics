"""Memory-mapped MRC image stack implementation."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mrcfile
import numpy as np
from mrcfile import utils as mrc_utils
from numpy.typing import DTypeLike, NDArray

from careamics.utils.reshape_array import (
    AxesTransform,
    get_patch_slices,
    reshape_patch,
)

from .image_utils.image_stack_utils import pad_patch

_IGNORED_HEADER_FIELDS = {
    "nsymbt",
    "exttyp",
    "extra1",
    "extra2",
}
"""Fields intentionally omitted from the MRC metadata.

- nsymbt: number of bytes occupied by the extended header
- exttyp: four-character code identifying the extended-header format
- extra1, extra2: reserved fields with application-specific usage
"""


def _to_python_metadata_value(value: Any) -> Any:
    """Convert a NumPy value to a regular Python metadata value.

    Parameters
    ----------
    value : Any
        NumPy value to convert.

    Returns
    -------
    Any
        Python scalar, sequence, mapping, or string representation.
    """
    if isinstance(value, np.ndarray):
        if value.dtype.names is not None and value.ndim == 0:
            return {
                name: _to_python_metadata_value(value[name])
                for name in value.dtype.names
            }
        if value.ndim == 0:
            return _to_python_metadata_value(value.item())
        return [_to_python_metadata_value(item) for item in value]

    # structured fields such as `cella`, `cellb` and `origin`
    if isinstance(value, np.void):
        if value.dtype.names is not None:
            return {
                name: _to_python_metadata_value(value[name])
                for name in value.dtype.names
            }
        return value.item()

    # np.int32, np.float32, np.bytes_, etc.
    if isinstance(value, np.generic):
        return _to_python_metadata_value(value.item())

    # Converted bytes (formerly np.bytes_); used by `map` and `label`.
    if isinstance(value, bytes):
        return value.rstrip(b"\x00").decode("ascii", errors="replace")
    return value


def _extract_mrc_header(mrc: Any) -> dict[str, Any]:
    """Extract selected mandatory MRC header fields as Python values.

    Parameters
    ----------
    mrc : Any
        Open MRC file object.

    Returns
    -------
    dict[str, Any]
        Header fields converted to regular Python metadata values.
    """
    return {
        name: _to_python_metadata_value(mrc.header[name])
        for name in mrc.header.dtype.names
        if name not in _IGNORED_HEADER_FIELDS
    }


class MrcImageStack:
    """ImageStack backed by a memory-mapped MRC file.

    Parameters
    ----------
    source : Path
        Path to the MRC file.
    axes : str
        Axis order of the MRC data.
    """

    def __init__(self, source: Path, axes: str):
        """Initialize an MRC-backed image stack.

        Parameters
        ----------
        source : Path
            Path to the MRC file.
        axes : str
            Axis order of the MRC data.
        """
        self._source = Path(source)
        self._original_axes = axes
        self._mrc: mrcfile.mrcfile.MrcFile | None = None

        with mrcfile.open(
            self._source,
            mode="r",
            permissive=True,
            header_only=True,
        ) as mrc:
            self._original_data_shape = tuple(
                mrc_utils.data_shape_from_header(mrc.header)
            )
            self._data_dtype = mrc_utils.dtype_from_mode(mrc.header.mode)
            self.additional_metadata = {"mrc": _extract_mrc_header(mrc)}

        self.data_shape = AxesTransform(
            axes, self._original_data_shape
        ).transformed_shape

    @property
    def source(self) -> Path:
        """Return the path to the MRC source file.

        Returns
        -------
        pathlib.Path
            MRC source path.
        """
        return self._source

    @property
    def data_dtype(self) -> DTypeLike:
        """Return the data type of the source MRC data.

        Returns
        -------
        numpy.typing.DTypeLike
            Source data type.
        """
        return self._data_dtype

    @property
    def original_data_shape(self) -> tuple[int, ...]:
        """Return the source shape in its original axis order.

        Returns
        -------
        tuple of int
            Source data shape.
        """
        return self._original_data_shape

    @property
    def original_axes(self) -> str:
        """Return the axis order of the source data.

        Returns
        -------
        str
            Source axis order.
        """
        return self._original_axes

    @property
    def is_loaded(self) -> bool:
        """Return whether the MRC file has an active memory map.

        Returns
        -------
        bool
            Whether the file is currently memory-mapped.
        """
        return self._mrc is not None

    def load(self) -> None:
        """Open the MRC file as a read-only memory map.

        Returns
        -------
        None
            This method opens the map in place.
        """
        if self._mrc is None:
            self._mrc = mrcfile.mmap(
                self._source,
                mode="r",
                permissive=True,
            )

    def close(self) -> None:
        """Close the active MRC memory map.

        Returns
        -------
        None
            This method closes the map in place.
        """
        if self._mrc is not None:
            self._mrc.close()
            self._mrc = None

    def extract_patch(
        self,
        sample_idx: int,
        channels: Sequence[int] | None,
        coords: Sequence[int],
        patch_size: Sequence[int],
    ) -> NDArray[np.float32]:
        """Extract a patch from the memory-mapped MRC data.

        Parameters
        ----------
        sample_idx : int
            Sample index in the transformed `S` dimension.
        channels : sequence of int or None
            Channel indices to extract. `None` selects all channels.
        coords : sequence of int
            Spatial coordinates of the patch origin in `(Z)YX` order.
        patch_size : sequence of int
            Patch size in `(Z)YX` order.

        Returns
        -------
        numpy.ndarray
            The patch in `C(Z)YX` order and `float32` dtype.
        """
        if (coord_dims := len(coords)) != (patch_dims := len(patch_size)):
            raise ValueError(
                "Patch coordinates and patch size must have the same dimensions "
                f"but found {coord_dims} and {patch_dims}."
            )

        self.load()
        assert self._mrc is not None

        patch_slice = get_patch_slices(
            self._original_axes,
            self._original_data_shape,
            sample_idx,
            channels,
            coords,
            patch_size,
        )
        patch_data = np.asarray(self._mrc.data[patch_slice])
        patch_data = reshape_patch(patch_data, self._original_axes)
        return pad_patch(coords, patch_size, self.data_shape, patch_data)
