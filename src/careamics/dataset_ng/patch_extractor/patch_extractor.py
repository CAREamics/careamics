from collections.abc import Sequence
from typing import Generic

from numpy.typing import NDArray

from ..image_stack import GenericImageStack
from .patch_construction import PatchConstructor, default_patch_constr


class PatchExtractor(Generic[GenericImageStack]):
    """
    A class for extracting patches from multiple image stacks.
    """

    def __init__(
        self,
        image_stacks: Sequence[GenericImageStack],
        patch_constructor: PatchConstructor = default_patch_constr,
    ):
        self.patch_constructor = patch_constructor
        self.image_stacks: list[GenericImageStack] = list(image_stacks)

        # check all image stacks have the same number of dimensions
        # check all image stacks have the same number of channels
        self.n_spatial_dims = len(self.image_stacks[0].data_shape) - 2  # SC(Z)YX
        self.n_channels = self.image_stacks[0].data_shape[1]
        for i, image_stack in enumerate(image_stacks):
            if (ndims := len(image_stack.data_shape) - 2) != self.n_spatial_dims:
                raise ValueError(
                    "All `ImageStack` objects in a `PatchExtractor` must have the same "
                    "number of spatial dimensions. The first image stack is "
                    f"{self.n_spatial_dims}D but found a {ndims}D image stack at index "
                    f"{i}."
                )
            if (n_channels := image_stack.data_shape[1]) != self.n_channels:
                raise ValueError(
                    "All `ImageStack` objects in a `PatchExtractor` must have the same "
                    f"number of channels. The first image stack has {self.n_channels} "
                    f"but found an image stack with {n_channels} at index {i}."
                )

    def extract_patch(
        self,
        data_idx: int,
        sample_idx: int,
        coords: Sequence[int],
        patch_size: Sequence[int],
    ) -> NDArray:
        return self.extract_channel_patch(
            data_idx=data_idx,
            sample_idx=sample_idx,
            channel_idx=None,
            coords=coords,
            patch_size=patch_size,
        )

    def extract_channel_patch(
        self,
        data_idx: int,
        sample_idx: int,
        channel_idx: int | None,
        coords: Sequence[int],
        patch_size: Sequence[int],
    ) -> NDArray:
        return self.patch_constructor(
            self.image_stacks[data_idx],
            sample_idx=sample_idx,
            channel_idx=channel_idx,
            coords=coords,
            patch_size=patch_size,
        )

    @property
    def shape(self):
        return [stack.data_shape for stack in self.image_stacks]
