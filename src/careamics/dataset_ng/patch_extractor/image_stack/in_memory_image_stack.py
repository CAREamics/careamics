from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Self, Union

import numpy as np
from numpy.typing import DTypeLike, NDArray

from careamics.dataset.dataset_utils import reshape_array
from careamics.file_io.read import ReadFunc, read_tiff

from .utils import pad_patch


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

    def extract_patch(
        self, sample_idx: int, coords: Sequence[int], patch_size: Sequence[int]
    ) -> NDArray:
        if (coord_dims := len(coords)) != (patch_dims := len(patch_size)):
            raise ValueError(
                "Patch coordinates and patch size must have the same dimensions but "
                f"found {coord_dims} and {patch_dims}."
            )
        # TODO: test for 2D or 3D?

        patch_data = self._data[
            (
                sample_idx,  # type: ignore
                ...,  # type: ignore
                *[
                    slice(
                        np.clip(c, 0, self.data_shape[2 + i]),
                        np.clip(c + ps, 0, self.data_shape[2 + i]),
                    )
                    for i, (c, ps) in enumerate(zip(coords, patch_size, strict=False))
                ],  # type: ignore
            )  # type: ignore
        ]
        patch = pad_patch(coords, patch_size, self.data_shape, patch_data)

        return patch

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
