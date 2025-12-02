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
        """Extract a patch from the specified image stack across all channels.

        Eqauivalent to calling `extract_channel_patch` with `channels=None`.

        Parameters
        ----------
        data_idx : int
            Index of the image stack to extract the patch from.
        sample_idx : int
            Sample index. The first dimension of the image data will be indexed at this
            value.
        coords : Sequence of int
            The coordinates that define the start of a patch.
        patch_size : Sequence of int
            The size of the patch in each spatial dimension.

        Returns
        -------
        numpy.ndarray
            The extracted patch.
        """
        return self.extract_channel_patch(
            data_idx=data_idx,
            sample_idx=sample_idx,
            channels=None,
            coords=coords,
            patch_size=patch_size,
        )

    def extract_channel_patch(
        self,
        data_idx: int,
        sample_idx: int,
        channels: Sequence[int] | None,
        coords: Sequence[int],
        patch_size: Sequence[int],
    ) -> NDArray:
        """Extract a patch from the specified image stack.

        Parameters
        ----------
        data_idx : int
            Index of the image stack to extract the patch from.
        sample_idx : int
            Sample index. The first dimension of the image data will be indexed at this
            value.
        channels : Sequence of int | None
            Channels to extract. If `None`, all channels are extracted.
        coords : Sequence of int
            The coordinates that define the start of a patch.
        patch_size : Sequence of int
            The size of the patch in each spatial dimension.

        Returns
        -------
        numpy.ndarray
            The extracted patch.
        """
        return self.patch_constructor(
            self.image_stacks[data_idx],
            sample_idx=sample_idx,
            channels=channels,
            coords=coords,
            patch_size=patch_size,
        )

    @property
    def shapes(self) -> list[Sequence[int]]:
        return [stack.data_shape for stack in self.image_stacks]
