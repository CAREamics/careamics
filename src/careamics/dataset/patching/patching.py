"""A module to contain type definitions relating to patching strategies."""

from collections.abc import Sequence
from typing import Protocol

from .patch_specs import PatchSpecs


class Patching(Protocol):
    """
    An interface for patching strategies.

    Patching strategies are a component of the `CAREamicsDataset`; they determine
    how patches are extracted from the underlying data.

    Attributes
    ----------
    n_patches: int
        The number of patches that the patching strategy will return.

    Methods
    -------
    get_patch_spec(index: int) -> PatchSpecs
        Get a patch specification for a given patch index.
    """

    # TODO: add data_shapes and patch_size as properties (more convenient in tests)

    @property
    def n_patches(self) -> int:
        """
        The number of patches that the patching strategy will return.

        It also determines the maximum index that can be given to `get_patch_spec`,
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

    # Note: this is used by the FileIterSampler
    def get_patch_indices(self, data_idx: int) -> Sequence[int]:
        """
        Get the patch indices will return patches for a specific `image_stack`.

        The `image_stack` corresponds to the given `data_idx`.

        Parameters
        ----------
        data_idx : int
            An index that corresponds to a given `image_stack`.

        Returns
        -------
        sequence of int
            A sequence of patch indices, that when used to index the `CAREamicsDataset
            will return a patch that comes from the `image_stack` corresponding to the
            given `data_idx`.
        """
        ...
