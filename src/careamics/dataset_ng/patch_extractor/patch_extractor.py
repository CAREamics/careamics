from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from numpy.typing import NDArray
from typing_extensions import Self

from careamics.file_io.read import ReadFunc

from .image_stack import ImageStack, InMemoryImageStack, ZarrImageStack


class PatchSpecs(TypedDict):
    data_idx: int
    sample_idx: int
    coords: Sequence[int]
    patch_size: Sequence[int]


class PatchExtractor:
    """
    A class for extracting patches from multiple image stacks.
    """

    def __init__(self, image_stacks: Sequence[ImageStack]):
        self.image_stacks: list[ImageStack] = list(image_stacks)

    # TODO: do away with all these constructors
    #   create ImageStackConstructor protocol
    #   just have:
    # @classmethod
    # def from_image_stack_constructor(
    #     self,
    #     constructor: ImageStackConstructor,
    #     sources: Sequence[SourceTypes],
    #     **constructor_kwargs,
    # ) -> Self: ...
    #
    # Even though this is a bit abstract users don't interact with this
    # It will be easier for people who want to write their own ImageStack
    #   we can pass their ImageStackConstructor to the PatchExtractor

    @classmethod
    def from_arrays(cls, source: Sequence[NDArray], *, axes: str) -> Self:
        image_stacks = [
            InMemoryImageStack.from_array(data=array, axes=axes) for array in source
        ]
        return cls(image_stacks=image_stacks)

    # TODO: rename to load_from_tiff_files?
    #   - to distiguish from possible pointer to files
    @classmethod
    def from_tiff_files(cls, source: Sequence[Path], *, axes: str) -> Self:
        image_stacks = [
            InMemoryImageStack.from_tiff(path=path, axes=axes) for path in source
        ]
        return cls(image_stacks=image_stacks)

    # TODO: similar to tiff - rename to load_from_custom_file_type?
    @classmethod
    def from_custom_file_type(
        cls,
        source: Sequence[Path],
        axes: str,
        read_func: ReadFunc,
        **read_kwargs,
    ) -> Self:
        image_stacks = [
            InMemoryImageStack.from_custom_file_type(
                path=path,
                axes=axes,
                read_func=read_func,
                **read_kwargs,
            )
            for path in source
        ]
        return cls(image_stacks=image_stacks)

    @classmethod
    def from_ome_zarr_files(cls, source: Sequence[Path]) -> Self:
        image_stacks = [ZarrImageStack.from_ome_zarr(path) for path in source]
        return cls(image_stacks=image_stacks)

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
