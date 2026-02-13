"""A module for a fixed coordinate patching strategy, useful for validation."""

from collections.abc import Sequence

from .patching_strategy_protocol import PatchSpecs


class FixedPatchingStrategy:

    def __init__(self, fixed_patch_specs: Sequence[PatchSpecs]):
        self.fixed_patch_specs = fixed_patch_specs

    @property
    def n_patches(self):
        """
        The number of patches that this patching strategy will return.

        It also determines the maximum index that can be given to `get_patch_spec`.
        """
        return len(self.fixed_patch_specs)

    def get_patch_spec(self, index: int) -> PatchSpecs:
        """Return the patch specs for a given index.

        Parameters
        ----------
        index : int
            A patch index.

        Returns
        -------
        PatchSpecs
            A dictionary that specifies a single patch in a series of `ImageStacks`.
        """
        if index >= self.n_patches:
            raise IndexError(
                f"Index {index} out of bounds for FixedRandomPatchingStrategy with "
                f"number of patches, {self.n_patches}"
            )
        # simply index the pre-generated patches to get the correct patch
        return self.fixed_patch_specs[index]

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
        return [
            i
            for i, patch_spec in enumerate(self.fixed_patch_specs)
            if patch_spec["data_idx"] == data_idx
        ]
