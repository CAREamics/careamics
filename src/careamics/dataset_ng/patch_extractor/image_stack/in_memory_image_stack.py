from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Union

import numpy as np
from numpy.typing import DTypeLike, NDArray
from typing_extensions import Self

from careamics.dataset.dataset_utils import reshape_array
from careamics.file_io.read import ReadFunc, read_tiff


class InMemoryImageStack:
    """
    A class for extracting patches from an image stack that has been loaded into memory.
    """

    def __init__(self, source: Union[Path, Literal["array"]], data: NDArray):
        self.source: Union[Path, Literal["array"]] = source
        # data expected to be in SC(Z)YX shape, reason to use from_array constructor
        self._data: NDArray = data
        self.data_shape: Sequence[int] = self._data.shape
        self.data_dtype: DTypeLike = self._data.dtype

    def _composite_patch(self, patches: list[NDArray]) -> NDArray:
        patch = np.stack(patches, axis=0)
        return patch

    def extract_patch(
        self, sample_idx: int, coords: Sequence[int], patch_size: Sequence[int]
    ) -> NDArray:
        # TODO change?
        if not isinstance(coords, list):
            coords = [coords]
        # coords is a list of coordinates for each channel if we need to extract patches
        # from different spatial locations for each channel
        if any(len(c) != len(patch_size) for c in coords):
            raise ValueError("Length of coords and extent must match.")

        # TODO: test for 2D or 3D?
        patches_per_channel = []
        for channel_idx, coord_pair in enumerate(coords):
            patches_per_channel.append(
                self._data[
                    (
                        sample_idx,  # type: ignore
                        channel_idx,  # type: ignore
                        ...,  # type: ignore
                        *[slice(c, c + e) for c, e in zip(coord_pair, patch_size)],  # type: ignore
                    )
                ]
            )
        return self._composite_patch(patches_per_channel)

    @classmethod
    def from_array(cls, data: NDArray, axes: str) -> Self:
        data = reshape_array(data, axes)
        return cls(source="array", data=data)

    @classmethod
    def from_tiff(cls, path: Path, axes: str) -> Self:
        data = read_tiff(path)
        data = reshape_array(data, axes)
        return cls(source=path, data=data)

    @classmethod
    def from_custom_file_type(
        cls, path: Path, axes: str, read_func: ReadFunc, **read_kwargs: Any
    ) -> Self:
        data = read_func(path, **read_kwargs)
        data = reshape_array(data, axes)
        return cls(source=path, data=data)
