"""A module for random patching strategies."""

from collections.abc import Sequence
from typing import Optional

import numpy as np

from .patching_strategy_types import PatchSpecs


class RandomPatchingStrategy:
    """
    A patching strategy for sampling random patches.

    The output of `get_patch_spec` will be random, i.e. if the same index is given
    twice the two outputs can be different.

    However the strategy still ensures that there will be a known number of patches for
    each sample in each image stack. This is achieved through defining a set of bins
    that map to each sample in each image stack. Whichever bin an `index` passed to
    `get_patch_spec` falls into, determines the `"data_idx"` and `"sample_idx"` in
    the returned `PatchSpecs`, but the `"coords"` will be random.

    The number of patches in each sample is based on the number of patches that would
    fit if they were sampled sequentially.
    """

    def __init__(
        self,
        data_shapes: Sequence[Sequence[int]],
        patch_size: Sequence[int],
        seed: Optional[int] = None,
    ):
        """
        A patching strategy for sampling random patches.

        Parameters
        ----------
        data_shapes : sequence of (sequence of int)
            The shapes of the underlying data. Each element is the dimension of the
            axes SC(Z)YX.
        patch_size : sequence of int
            The size of the patch. The sequence will have length 2 or 3, for 2D and 3D
            data respectively.
        seed : int, optional
            An optional seed to ensure the reproducibility of the random pathces.
        """
        self.rng = np.random.default_rng(seed=seed)
        self.patch_size = patch_size
        self.data_shapes = data_shapes

        # these bins will determine which image stack and sample a patch comes from
        # the image_stack_index_bins map to each image stack
        # the sample_index_bins map to each sample within each image stack
        self.image_stack_index_bins, self.sample_index_bins = self._calc_patch_bins(
            self.data_shapes, self.patch_size
        )

        # this is needed to calculate the sample_idx relative to the image_stack
        samples_per_image_stack = [data_shape[0] for data_shape in self.data_shapes]
        self.sample_bins = np.cumsum(samples_per_image_stack)

    @property
    def n_patches(self) -> int:
        """
        The number of patches that this patching strategy will return.

        It also determines the maximum index that can be given to `get_patch_spec`.
        """
        # last bin boundary will be total patches
        return self.image_stack_index_bins[-1]

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
        # TODO: break into smaller testable functions?
        if index >= self.n_patches:
            raise IndexError(
                f"Index {index} out of bounds for RandomPatchingStrategy with number "
                f"of patches, {self.n_patches}"
            )
        # digitize returns the bin that `index` belongs to
        data_index = np.digitize(index, bins=self.image_stack_index_bins)
        # maps to a particular sample within the whole series of image stacks
        #   (not just a single image stack)
        total_samples_index = np.digitize(index, bins=self.sample_index_bins)

        data_shape = self.data_shapes[data_index.item()]
        spatial_shape = data_shape[2:]

        # calculate sample index relative to image stack:
        #   subtract the total number of samples in the previous image stacks
        if data_index == 0:
            n_previous_samples = 0
        else:
            n_previous_samples = self.sample_bins[data_index - 1]
        sample_index = total_samples_index - n_previous_samples
        coords = _random_coords(spatial_shape, self.patch_size, self.rng)
        return {
            "data_idx": data_index.item(),
            "sample_idx": sample_index.item(),
            "coords": coords,
            "patch_size": self.patch_size,
        }

    @staticmethod
    def _calc_patch_bins(
        data_shapes: Sequence[Sequence[int]], patch_size: Sequence[int]
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Calculate bins used to map an index to an image_stack and a sample.

        The number of patches in each sample is based on the number of patches that
        would fit if they were sampled sequentially.

        Parameters
        ----------
        data_shapes : sequence of (sequence of int)
            The shapes of the underlying data. Each element is the dimension of the
            axes SC(Z)YX.
        patch_size : sequence of int
            The size of the patch. The sequence will have length 2 or 3, for 2D and 3D
            data respectively.

        Returns
        -------
        image_stack_index_bins: tuple[int, ...]
            The bins that map an index to an image stack.
        sample_index_bins: tuple[int, ...]
            The bins that map an index to a sample.
        """
        patches_per_image_stack: list[int] = []
        patches_per_sample: list[int] = []
        for data_shape in data_shapes:
            spatial_shape = data_shape[2:]
            n_single_sample_patches = _n_patches(spatial_shape, patch_size)
            # multiply by number of samples in image_stack
            patches_per_image_stack.append(n_single_sample_patches * data_shape[0])
            # list of length `sample` filled with `n_single_sample_patches`
            patches_per_sample.extend([n_single_sample_patches] * data_shape[0])

        # cumulative sum creates the bins
        image_stack_index_bins = np.cumsum(patches_per_image_stack)
        sample_index_bins = np.cumsum(patches_per_sample)
        return tuple(image_stack_index_bins), tuple(sample_index_bins)


class FixedRandomPatchingStrategy:
    """
    A patching strategy for sampling random patches.

    The output of `get_patch_spec` will be deterministic, i.e. if the same index is
    given twice the two outputs will be the same.

    The number of patches in each sample is based on the number of patches that would
    fit if they were sampled sequentially.
    """

    def __init__(
        self,
        data_shapes: Sequence[Sequence[int]],
        patch_size: Sequence[int],
        seed: Optional[int] = None,
    ):
        """A patching strategy for sampling random patches.

        Parameters
        ----------
        data_shapes : sequence of (sequence of int)
            The shapes of the underlying data. Each element is the dimension of the
            axes SC(Z)YX.
        patch_size : sequence of int
            The size of the patch. The sequence will have length 2 or 3, for 2D and 3D
            data respectively.
        seed : int, optional
            An optional seed to ensure the reproducibility of the random pathces.
        """
        self.rng = np.random.default_rng(seed=seed)
        self.patch_size = patch_size
        self.data_shapes = data_shapes

        # simply generate all the patches at initialisation, so they will be fixed
        self.fixed_patch_specs: list[PatchSpecs] = []
        for data_idx, data_shape in enumerate(self.data_shapes):
            spatial_shape = data_shape[2:]
            n_patches = _n_patches(spatial_shape, self.patch_size)
            for sample_idx in range(data_shape[0]):
                for _ in range(n_patches):
                    random_coords = _random_coords(
                        spatial_shape, self.patch_size, self.rng
                    )
                    patch_specs: PatchSpecs = {
                        "data_idx": data_idx,
                        "sample_idx": sample_idx,
                        "coords": random_coords,
                        "patch_size": self.patch_size,
                    }
                    self.fixed_patch_specs.append(patch_specs)

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


def _random_coords(
    spatial_shape: Sequence[int], patch_size: Sequence[int], rng: np.random.Generator
) -> tuple[int, ...]:
    """Generate random patch coordinates for a given `spatial_shape` and `patch_size`.

    The coords are the top-left (and first z-slice for 3D data) of a patch. The
    sequence will have length 2 or 3, for 2D and 3D data respectively.

    Parameters
    ----------
    spatial_shape : sequence of int
        The dimension of the axes (Z)YX, a sequence of length 2 or 3, for 2D and 3D
        data respectively.
    patch_size : sequence of int
        The size of the patch. The sequence will have length 2 or 3, for 2D and 3D
        data respectively.
    rng : numpy.random.Generator
        A numpy generator to ensure the reproducibility of the random patches.

    Returns
    -------
    coords: tuple of int
        The top-left (and first z-slice for 3D data) coords of a patch. The tuple will
        have length 2 or 3, for 2D and 3D data respectively.

    Raises
    ------
    ValueError
        Raises if the number of spatial dimensions do not match the number of patch
        dimensions.
    """
    if len(patch_size) != len(spatial_shape):
        raise ValueError(
            "Number of patch dimension do not match the number of spatial "
            "dimensions."
        )
    return tuple(
        rng.integers(
            np.zeros(len(patch_size), dtype=int),
            np.array(spatial_shape) - np.array(patch_size),
            endpoint=False,
            dtype=int,
        ).tolist()
    )


def _n_patches(spatial_shape: Sequence[int], patch_size: Sequence[int]) -> int:
    """
    Calculates the number of patches for a given `spatial_shape` and `patch_size`.

    This is based on the number of patches that would fit if they were sampled
    sequentially.

    Parameters
    ----------
    spatial_shape : sequence of int
        The dimension of the axes (Z)YX, a sequence of length 2 or 3, for 2D and 3D
        data respectively.
    patch_size : sequence of int
        The size of the patch. The sequence will have length 2 or 3, for 2D and 3D
        data respectively.

    Returns
    -------
    int
        The number of patches.
    """
    if len(patch_size) != len(spatial_shape):
        raise ValueError(
            "Number of patch dimension do not match the number of spatial "
            "dimensions."
        )
    return int(np.ceil(np.prod(spatial_shape) / np.prod(patch_size)))
