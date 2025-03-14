from collections.abc import Sequence
from typing import TypedDict

from numpy.typing import NDArray

from .image_stack import ImageStack


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

    @property
    def shape(self):
        return [stack.data_shape for stack in self.image_stacks]
