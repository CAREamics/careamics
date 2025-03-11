from collections.abc import Sequence
from typing import Optional

import numpy as np

from .patching_strategy_types import PatchSpecs


class RandomPatchingStrategy:

    def __init__(
        self,
        data_shapes: Sequence[Sequence[int]],
        patch_size: Sequence[int],
        seed: Optional[int] = None,
    ):
        self.rng = np.random.default_rng(seed=seed)
        self.patch_size = patch_size
        self.data_shapes = data_shapes

        self.image_stack_index_bins, self.sample_index_bins = self._calc_patch_bins(
            self.data_shapes, self.patch_size
        )

    @property
    def n_patches(self) -> int:
        # last bin boundary will be total patches
        return self.image_stack_index_bins[-1]

    def get_patch_spec(self, index: int):
        """Return the patch specs at a given instance."""
        data_index = np.digitize(index, bins=self.image_stack_index_bins)
        sample_index = np.digitize(index, bins=self.sample_index_bins)
        patch_specs = self._generate_random_patch_specs(
            # TODO: type checking doesn't accept numpy scalar, is there a better way?
            data_index.item(),
            sample_index.item(),
        )
        return patch_specs

    def _generate_random_patch_specs(
        self, data_index: int, sample_index: int
    ) -> PatchSpecs:
        """Generate a random patch for an image at `data_index`, `sample_index`"""
        data_shape = self.data_shapes[data_index]
        spatial_shape = data_shape[2:]
        coords = _random_coords(spatial_shape, self.patch_size, self.rng)
        return {
            "data_idx": data_index,
            "sample_idx": sample_index,
            "coords": coords,
            "patch_size": self.patch_size,
        }

    @staticmethod
    def _calc_patch_bins(
        data_shapes: Sequence[Sequence[int]], patch_size: Sequence[int]
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        patches_per_image_stack: list[int] = []
        patches_per_sample: list[int] = []
        for data_shape in data_shapes:
            spatial_shape = data_shape[2:]
            n_single_sample_patches = _n_patches(spatial_shape, patch_size)
            # multiply by number of samples in image_stack
            patches_per_image_stack.append(n_single_sample_patches * data_shape[0])
            # list of length sample filled with `n_single_sample_patches`
            patches_per_sample.extend([n_single_sample_patches] * data_shape[0])

        image_stack_index_bins = np.cumsum(patches_per_image_stack)
        sample_index_bins = np.cumsum(patches_per_sample)
        return tuple(image_stack_index_bins), tuple(sample_index_bins)


class FixedRandomPatchingStrategy:

    def __init__(
        self,
        data_shapes: Sequence[Sequence[int]],
        patch_size: Sequence[int],
        seed: Optional[int] = None,
    ):
        self.rng = np.random.default_rng(seed=seed)
        self.patch_size = patch_size
        self.data_shapes = data_shapes

        self.fixed_patch_specs: list[PatchSpecs] = []
        for data_idx, data_shape in enumerate(self.data_shapes):
            spatial_shape = data_shape[2:]
            for sample_idx in range(data_shape[0]):

                random_coords = _random_coords(spatial_shape, self.patch_size, self.rng)
                patch_specs: PatchSpecs = {
                    "data_idx": data_idx,
                    "sample_idx": sample_idx,
                    "coords": random_coords,
                    "patch_size": self.patch_size,
                }
                self.fixed_patch_specs.append(patch_specs)

    @property
    def n_patches(self):
        return len(self.fixed_patch_specs)

    def get_patch_spec(self, index: int) -> PatchSpecs:
        return self.fixed_patch_specs[index]


def _random_coords(
    spatial_shape: Sequence[int], patch_size: Sequence[int], rng: np.random.Generator
) -> tuple[int, ...]:
    if len(patch_size) != len(spatial_shape):
        raise ValueError(
            "Number of patch dimension do not match the number of spatial "
            "dimensions."
        )
    return tuple(
        rng.integers(
            np.zeros(len(patch_size), dtype=int),
            np.array(spatial_shape) - np.array(patch_size),
            endpoint=True,
            dtype=int,
        ).tolist()
    )


def _n_patches(spatial_shape: Sequence[int], patch_size: Sequence[int]) -> int:
    """
    Calculates the number of patches for a given `spatial_shape` and `patch_size`.
    """
    if len(patch_size) != len(spatial_shape):
        raise ValueError(
            "Number of patch dimension do not match the number of spatial "
            "dimensions."
        )
    return int(np.ceil(np.prod(spatial_shape) / np.prod(patch_size)))
