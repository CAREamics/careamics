from collections.abc import Sequence
from typing import Protocol, TypedDict


class PatchSpecs(TypedDict):
    """A dictionary that specifies a single patch in a series of `ImageStacks`.

    Attributes
    ----------
    data_idx: int
        Determines which `ImageStack` a patch belongs to, within a series of
        `ImageStack`.
    sample_idx: int
        Determines which sample a patch belongs to, within an `ImageStacks`.
    coords: sequence of int
        The top-left (and first z-slice for 3D) of a patch. The sequence will have
        length 2 or 3 for 2D and 3D data respectively.
    patch_size: sequence of int
        The size of the patch. The sequence will have length 2 or 3 for 2D and 3D data
        respectively.
    """

    data_idx: int
    sample_idx: int
    coords: Sequence[int]
    patch_size: Sequence[int]


class PatchingStrategy(Protocol):
    """An interface for Patching Strategies."""

    @property
    def n_patches(self) -> int:
        """
        The number of patches that the patching strategy should return.

        This also determines the maximum index that can be given to `get_patch_spec`,
        and the length of the `CAREamicsDataset`.

        Returns
        -------
        int
            Number of patches.
        """
        ...

    def get_patch_spec(self, index: int) -> PatchSpecs:
        """
        Get a patch specification for a given patch index.

        This method is intended to be called from within the
        `CAREamicsDataset.__getitem__`. The index will be passed through from this
        method.

        Parameters
        ----------
        index : int
            A patch index.

        Returns
        -------
        PatchSpecs
            A dictionary that specifies a single patch in a series of `ImageStacks`.
        """
        ...
