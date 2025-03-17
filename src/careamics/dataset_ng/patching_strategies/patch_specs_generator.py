import itertools
from collections.abc import Sequence
from typing import Optional, Protocol

import numpy as np
from numpy.typing import NDArray
from typing_extensions import ParamSpec

from ..patch_extractor import PatchSpecs

P = ParamSpec("P")


class PatchSpecsGenerator(Protocol[P]):

    def generate(
        self, data_shapes: Sequence[Sequence[int]], *args: P.args, **kwargs: P.kwargs
    ) -> list[PatchSpecs]: ...

    # Should return the number of patches that will be produced for a set of args
    # Will be for mapped dataset length
    def n_patches(
        self, data_shapes: Sequence[Sequence[int]], *args: P.args, **kwargs: P.kwargs
    ) -> int: ...


class RandomPatchSpecsGenerator:

    def __init__(self, patch_size: Sequence[int], random_seed: int):
        self.patch_size = patch_size
        self.random_seed = random_seed

    def generate(self, data_shapes: Sequence[Sequence[int]]) -> list[PatchSpecs]:
        rng = np.random.default_rng(seed=self.random_seed)
        patch_specs: list[PatchSpecs] = []
        for data_idx, data_shape in enumerate(data_shapes):

            # shape on which data is patched
            data_spatial_shape = data_shape[-len(self.patch_size) :]

            n_patches = self._n_patches_in_sample(self.patch_size, data_spatial_shape)
            data_patch_specs = [
                PatchSpecs(
                    data_idx=data_idx,
                    sample_idx=sample_idx,
                    coords=tuple(
                        rng.integers(
                            np.zeros(len(self.patch_size), dtype=int),
                            np.array(data_spatial_shape) - np.array(self.patch_size),
                            endpoint=True,
                        )
                    ),
                    patch_size=self.patch_size,
                )
                for sample_idx in range(data_shape[0])
                for _ in range(n_patches)
            ]
            patch_specs.extend(data_patch_specs)
        return patch_specs

    # NOTE: enerate and n_patches methods must have matching signatures
    #   as dictated by protocol
    def n_patches(self, data_shapes: Sequence[Sequence[int]]) -> int:
        n_sample_patches: NDArray[np.int_] = np.array(
            [
                self._n_patches_in_sample(
                    self.patch_size, data_shape[-len(self.patch_size) :]
                )
                for data_shape in data_shapes
            ],
            dtype=int,
        )
        n_samples = np.array([data_shape[0] for data_shape in data_shapes], dtype=int)
        n_data_patches = n_samples * n_sample_patches
        return int(n_data_patches.sum())

    @staticmethod
    def _n_patches_in_sample(
        patch_size: Sequence[int], spatial_shape: Sequence[int]
    ) -> int:
        if len(patch_size) != len(spatial_shape):
            raise ValueError(
                "Number of patch dimension do not match the number of spatial "
                "dimensions."
            )
        return int(np.ceil(np.prod(spatial_shape) / np.prod(patch_size)))


# TODO: this is an unfinished prototype based on current tiling implementation
#  not guaranteed to work!
class TiledPatchSpecsGenerator:
    def __init__(self, patch_size: Sequence[int], overlap: Optional[list[int]] = None):
        self.patch_size = patch_size
        if overlap is None:
            overlap = [0] * len(patch_size)
        self.overlap = np.asarray(overlap)

    def _compute_coords_1d(
        self, patch_size: int, spatial_shape: int, overlap: int
    ) -> list[tuple[int, int]]:
        step = patch_size - overlap
        crop_coords = []

        current_pos = 0
        while current_pos <= spatial_shape - patch_size:
            crop_coords.append((current_pos, current_pos + patch_size))
            current_pos += step

        if crop_coords[-1][1] < spatial_shape:
            crop_coords.append((spatial_shape - patch_size, spatial_shape))

        return crop_coords

    def _n_patches_in_sample(
        self, patch_size: Sequence[int], spatial_shape: Sequence[int]
    ) -> int:
        """Calculate number of patches per sample."""
        if len(patch_size) != len(spatial_shape):
            raise ValueError(
                "Number of patch dimensions do not match spatial dimensions."
            )
        steps = np.ceil(
            (np.array(spatial_shape) - np.array(patch_size))
            / (patch_size - self.overlap)
            + 1
        )
        return int(np.prod(steps))

    def generate(self, data_shapes: Sequence[Sequence[int]]) -> list[PatchSpecs]:
        patch_specs: list[PatchSpecs] = []
        for data_idx, data_shape in enumerate(data_shapes):

            data_spatial_shape = data_shape[-len(self.patch_size) :]
            coords_list = [
                self._compute_coords_1d(
                    self.patch_size[i], data_spatial_shape[i], self.overlap[i]
                )
                for i in range(len(self.patch_size))
            ]
            for sample_idx in range(data_shape[0]):
                for crop_coord in itertools.product(*coords_list):
                    patch_specs.append(
                        PatchSpecs(
                            data_idx=data_idx,
                            sample_idx=sample_idx,
                            coords=tuple([coord[0] for coord in crop_coord]),
                            patch_size=self.patch_size,
                        )
                    )
        return patch_specs

    def n_patches(self, data_shapes: Sequence[Sequence[int]]) -> int:
        n_sample_patches: NDArray[np.int_] = np.array(
            [
                self._n_patches_in_sample(
                    self.patch_size, data_shape[-len(self.patch_size) :]
                )
                for data_shape in data_shapes
            ],
            dtype=int,
        )
        n_samples = np.array([data_shape[0] for data_shape in data_shapes], dtype=int)
        n_data_patches = n_samples * n_sample_patches
        return int(n_data_patches.sum())


if __name__ == "__main__":
    # testing mypy accepts protocol type
    patch_specs_generator: PatchSpecsGenerator = RandomPatchSpecsGenerator(
        patch_size=(2, 2), random_seed=42
    )
    patch_specs_generator.generate([(1, 1, 6, 6), (1, 1, 4, 4)])
