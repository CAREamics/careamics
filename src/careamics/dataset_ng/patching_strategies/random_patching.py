from collections.abc import Sequence
from typing import Optional

import numpy as np

from .patching_strategy_protocol import PatchSpecs


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

    def get_patch_specs(self, index: int):
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
        coords = tuple(
            self.rng.integers(
                np.zeros(len(self.patch_size), dtype=int),
                np.array(spatial_shape) - np.array(self.patch_size),
                endpoint=True,
                dtype=int,
            ).tolist()
        )
        return {
            "data_idx": data_index,
            "sample_idx": sample_index,
            "coords": coords,
            "patch_size": self.patch_size,
        }

    @staticmethod
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

    @staticmethod
    def _calc_patch_bins(
        data_shapes: Sequence[Sequence[int]], patch_size: Sequence[int]
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        patches_per_image_stack: list[int] = []
        patches_per_sample: list[int] = []
        for data_shape in data_shapes:
            spatial_shape = data_shape[2:]
            n_single_sample_patches = RandomPatchingStrategy._n_patches(
                spatial_shape, patch_size
            )
            # multiply by number of samples in image_stack
            patches_per_image_stack.append(n_single_sample_patches * data_shape[0])
            # list of length sample filled with `n_single_sample_patches`
            patches_per_sample.extend([n_single_sample_patches] * data_shape[0])

        image_stack_index_bins = np.cumsum(patches_per_image_stack)
        sample_index_bins = np.cumsum(patches_per_sample)
        return tuple(image_stack_index_bins), tuple(sample_index_bins)


if __name__ == "__main__":

    patching_strategy = RandomPatchingStrategy(
        data_shapes=((1, 1, 8, 8),), patch_size=(2, 2)
    )
    patch_spec = patching_strategy.get_patch_specs(1)
