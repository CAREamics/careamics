from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, TypedDict, Union

from numpy.typing import NDArray
from typing_extensions import Self

from careamics.file_io.read import ReadFunc

from .image_stack import ImageStack, InMemoryImageStack


class PatchSpecs(TypedDict):
    data_idx: int
    sample_idx: int
    coords: Sequence[int]
    patch_size: Sequence[int]


class PatchExtractorConstructor(Protocol):

    # TODO: expand Union for new constructors, or just type hint as Any
    def __call__(
        self, source: Union[Sequence[NDArray], Sequence[Path]], *args, **kwargs
    ) -> "PatchExtractor": ...


class PatchExtractor:
    """
    A class for extracting patches from multiple image stacks.
    """

    def __init__(self, data_readers: Sequence[ImageStack]):
        self.image_stacks: list[ImageStack] = list(data_readers)

    @classmethod
    def from_arrays(cls, source: Sequence[NDArray], axes: str) -> Self:
        data_readers = [
            InMemoryImageStack.from_array(data=array, axes=axes) for array in source
        ]
        return cls(data_readers=data_readers)

    # TODO: rename to load_from_tiff_files?
    #   - to distiguish from possible pointer to files
    @classmethod
    def from_tiff_files(cls, source: Sequence[Path], axes: str) -> Self:
        data_readers = [
            InMemoryImageStack.from_tiff(path=path, axes=axes) for path in source
        ]
        return cls(data_readers=data_readers)

    # TODO: similar to tiff - rename to load_from_custom_file_type?
    @classmethod
    def from_custom_file_type(
        cls,
        source: Sequence[Path],
        axes: str,
        read_func: ReadFunc,
        *read_args,
        **read_kwargs,
    ) -> Self:
        data_readers = [
            InMemoryImageStack.from_custom_file_type(
                path=path,
                axes=axes,
                read_func=read_func,
                read_args=read_args,
                read_kwargs=read_kwargs,
            )
            for path in source
        ]
        return cls(data_readers=data_readers)

    @classmethod
    def from_zarr_files(cls, source, *args, **kwargs) -> Self:
        # TODO: will this create a ZarrImageStack for each array in the zarr file?
        raise NotImplementedError("Reading from zarr has not been implemented.")

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
