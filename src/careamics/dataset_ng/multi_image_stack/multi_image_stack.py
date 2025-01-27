from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from numpy.typing import NDArray

from .image_stack import ImageStack, InMemoryImageStack


class PatchSpecs(TypedDict):
    data_idx: int
    sample_idx: int
    coords: Sequence[int]
    patch_size: Sequence[int]


class MultiImageStack:

    def __init__(self, data_readers: Sequence[ImageStack]):
        self.image_stacks: list[ImageStack] = list(data_readers)

    @classmethod
    def from_arrays(cls, arrays: Sequence[NDArray], axes: str):
        data_readers = [
            InMemoryImageStack.from_array(data=array, axes=axes) for array in arrays
        ]
        return cls(data_readers=data_readers)

    @classmethod
    def from_tiff_files(cls, file_paths: Sequence[Path], axes: str):
        data_readers = [
            InMemoryImageStack.from_tiff(path=path, axes=axes) for path in file_paths
        ]
        return cls(data_readers=data_readers)

    def extract_patch(
        self,
        data_idx: int,
        sample_idx: int,
        coords: Sequence[int],
        patch_size: Sequence[int],
    ) -> NDArray:
        return self.image_stacks[data_idx].extract_patch(
            sample_idx=sample_idx, coords=coords, patch_size=patch_size
        )

    def extract_patches(self, patch_specs: Sequence[PatchSpecs]) -> list[NDArray]:
        return [self.extract_patch(**patch_spec) for patch_spec in patch_specs]
