import itertools
from collections.abc import Sequence
from typing import Optional

import numpy as np
from typing_extensions import ParamSpec

from .patching_strategy_protocol import PatchSpecs

P = ParamSpec("P")


# TODO: this is an unfinished prototype based on current tiling implementation
#  not guaranteed to work!
class SequentialPatchingStrategy:
    # TODO: docs
    def __init__(
        self,
        data_shapes: Sequence[Sequence[int]],
        patch_size: Sequence[int],
        overlaps: Optional[Sequence[int]] = None,
    ):
        self.data_shapes = data_shapes
        self.patch_size = patch_size
        if overlaps is None:
            overlaps = [0] * len(patch_size)
        self.overlaps = np.asarray(overlaps)

        self.patch_specs: list[PatchSpecs] = self._initialize_patch_specs()

    @property
    def n_patches(self) -> int:
        return len(self.patch_specs)

    def get_patch_spec(self, index: int) -> PatchSpecs:
        return self.patch_specs[index]

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

    def _initialize_patch_specs(self) -> list[PatchSpecs]:
        patch_specs: list[PatchSpecs] = []
        for data_idx, data_shape in enumerate(self.data_shapes):

            data_spatial_shape = data_shape[-len(self.patch_size) :]
            coords_list = [
                self._compute_coords_1d(
                    self.patch_size[i], data_spatial_shape[i], self.overlaps[i]
                )
                for i in range(len(self.patch_size))
            ]
            for sample_idx in range(data_shape[0]):
                for crop_coord in itertools.product(*coords_list):
                    patch_specs.append(
                        PatchSpecs(
                            data_idx=data_idx,
                            sample_idx=sample_idx,
                            coords=tuple(coord[0] for coord in crop_coord),
                            patch_size=self.patch_size,
                        )
                    )
        return patch_specs
