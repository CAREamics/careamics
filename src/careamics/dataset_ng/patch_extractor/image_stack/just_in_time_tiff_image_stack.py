from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import tifffile
from numpy.typing import NDArray


class JitTiffImageStack:
    """
    A class for extracting patches from an image stack that has been loaded into memory.
    """

    def __init__(self, source: Path, axes: Optional[str] = None):
        self.source: Path = source
        self.file = tifffile.TiffFile(source)
        self._data: Optional[NDArray] = None
        # TODO: is this correct for multiple series
        self._axes = self.file.series[0].axes if axes is None else axes
        self._orginal_shape = self.file.series[0].shape
        # TODO: calculate from original axes and original shape
        self.data_shape: Sequence[int]

    def extract_patch(
        self, sample_idx: int, coords: Sequence[int], patch_size: Sequence[int]
    ) -> NDArray:
        if len(coords) != len(patch_size):
            raise ValueError("Length of coords and extent must match.")
        # TODO: test for 2D or 3D?

        # TODO: if data not loaded load

        return self._data[
            (
                sample_idx,  # type: ignore
                ...,  # type: ignore
                *[slice(c, c + e) for c, e in zip(coords, patch_size)],  # type: ignore
            )
        ]

    def deallocate(self):
        ...
        # TODO: unload data
