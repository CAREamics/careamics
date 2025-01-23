from pathlib import Path
from typing import Literal, Union

from numpy.typing import NDArray
from typing_extensions import Self

from careamics.file_io.read import ReadFunc, read_tiff

from ..dataset_utils import reshape_array


class InMemoryReader:

    def __init__(self, source: Union[Path, Literal["array"]], data: NDArray):
        self.source: Union[Path, Literal["array"]] = source
        self._data = data
        self.data_shape = self._data.shape

    def extract_patch(
        self, sample_idx: int, coords: tuple[int, ...], extent: tuple[int, ...]
    ) -> NDArray:
        if len(coords) != len(extent):
            raise ValueError("Length of coords and extent must match.")
        # TODO: test for 2D or 3D?
        return self._data[
            (
                sample_idx,  # type: ignore
                ...,  # type: ignore
                *[slice(c, c + e) for c, e in zip(coords, extent)],  # type: ignore
            )
        ]

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
        cls, path: Path, axes: str, read_func: ReadFunc, *read_args, **read_kwargs
    ) -> Self:
        data = read_func(path, *read_args, **read_kwargs)
        data = reshape_array(data, axes)
        return cls(source=path, data=data)
