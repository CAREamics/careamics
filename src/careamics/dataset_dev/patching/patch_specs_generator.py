from collections.abc import Sequence
from typing import Protocol, TypedDict

import numpy as np


class PatchSpecs(TypedDict):
    data_idx: int
    sample_idx: int
    coords: Sequence[int]
    patch_size: Sequence[int]


class PatchSpecsGenerator(Protocol):

    def generate(
        self, patch_size: Sequence[int], *args, **kwargs
    ) -> list[PatchSpecs]: ...

    # Should return the number of patches that will be produced for a set of args
    # Will be for mapped dataset length
    def n_patches(self, patch_size: Sequence[int], *args, **kwargs): ...


class RandomPatchSpecsGenerator:

    def __init__(self, data_shapes: Sequence[Sequence[int]]):
        self.data_shapes = data_shapes

    def generate(self, patch_size: Sequence[int], seed: int):
        rng = np.random.default_rng(seed=seed)
        patch_specs: list[PatchSpecs] = []
        for data_idx, data_shape in enumerate(self.data_shapes):

            # shape on which data is patched
            data_spatial_shape = data_shape[-len(patch_size) :]

            n_patches = self._n_patches_in_sample(patch_size, data_spatial_shape)
            data_patch_specs = [
                PatchSpecs(
                    data_idx=data_idx,
                    sample_idx=sample_idx,
                    coords=tuple(
                        rng.integers(
                            np.zeros(len(patch_size), dtype=int),
                            np.array(data_spatial_shape) - np.array(patch_size),
                            endpoint=True,
                        )
                    ),
                    patch_size=patch_size,
                )
                for sample_idx in range(data_shape[0])
                for _ in range(n_patches)
            ]
            patch_specs.extend(data_patch_specs)
        return patch_specs

    def n_patches(self, patch_size: Sequence[int], seed: int):
        n_sample_patches = np.array(
            [
                self._n_patches_in_sample(patch_size, data_shape[-len(patch_size) :])
                for data_shape in self.data_shapes
            ],
            dtype=int,
        )
        n_samples = np.array(
            [data_shape[0] for data_shape in self.data_shapes], dtype=int
        )
        n_data_patches = n_samples * n_sample_patches
        return n_data_patches.sum()

    @staticmethod
    def _n_patches_in_sample(patch_size: Sequence[int], spatial_shape: Sequence[int]):
        if len(patch_size) != len(spatial_shape):
            raise ValueError(
                "Number of patch dimension do not match the number of spatial "
                "dimensions."
            )
        return int(np.ceil(np.prod(spatial_shape) / np.prod(patch_size)))


if __name__ == "__main__":
    # testing mypy accepts protocol type
    patch_specs_generator: PatchSpecsGenerator = RandomPatchSpecsGenerator(
        [(1, 1, 6, 6), (1, 1, 4, 4)]
    )
