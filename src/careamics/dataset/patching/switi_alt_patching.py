import itertools
from collections.abc import Sequence
from math import prod

import numpy as np

from careamics.dataset.patching import TileSpecs


class SwitiAltPatching:

    def __init__(
        self,
        data_shapes: Sequence[Sequence[int]],
        patch_size: Sequence[int],
        overlaps: Sequence[int],
        coverage: Sequence[int],
    ):
        self.data_shapes = data_shapes
        self.patch_size = patch_size
        self.coverage = coverage
        self.overlaps = overlaps
        self.tile_specs: list[TileSpecs] = self._generate_specs()

    @property
    def n_patches(self) -> int:
        """Total number of tile specs.

        Returns
        -------
        int
            Total number of patches.
        """
        return len(self.tile_specs)

    def get_patch_spec(self, index: int) -> TileSpecs:
        """Return the tile specs for a given index.

        Parameters
        ----------
        index : int
            A patch index.

        Returns
        -------
        TileSpecs
            A dictionary that specifies a single patch in a series of `ImageStacks`.
        """
        return self.tile_specs[index]

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
            i for i, spec in enumerate(self.tile_specs) if spec["data_idx"] == data_idx
        ]

    def _generate_specs(self) -> list[TileSpecs]:
        """Build the full list of tile specs.

        Returns
        -------
        list of TileSpecs
            Full list of tile specs.
        """
        tile_specs: list[TileSpecs] = []
        for data_idx, data_shape in enumerate(self.data_shapes):
            spatial_shape = data_shape[2:]

            axis_specs: list[tuple[list[int], list[int], list[int], list[int]]] = [
                self._compute_1d_coords(
                    axis_size,
                    self.patch_size[axis_idx],
                    self.coverage[axis_idx],
                    self.overlaps[axis_idx],
                )
                for axis_idx, axis_size in enumerate(spatial_shape)
            ]

            all_coords, all_stitch_coords, all_crop_coords, all_crop_size = zip(
                *axis_specs, strict=False
            )

            n_tiles = prod(len(dim) for dim in all_coords) * data_shape[0]

            for sample_idx in range(data_shape[0]):
                for coords, stitch_coords, crop_coords, crop_size in zip(
                    itertools.product(*all_coords),
                    itertools.product(*all_stitch_coords),
                    itertools.product(*all_crop_coords),
                    itertools.product(*all_crop_size),
                    strict=False,
                ):
                    tile_specs.append(
                        {
                            "data_idx": data_idx,
                            "sample_idx": sample_idx,
                            "coords": coords,
                            "patch_size": self.patch_size,
                            "crop_coords": crop_coords,
                            "crop_size": crop_size,
                            "stitch_coords": stitch_coords,
                            "total_tiles": n_tiles,
                        }
                    )

        return tile_specs

    @staticmethod
    def _compute_1d_coords(
        axis_size: int, patch_size: int, coverage: int, overlap: int
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        if axis_size <= patch_size:
            return (
                [0] * coverage,
                [0] * coverage,
                [0] * coverage,
                [axis_size] * coverage,
            )

        inner_region = patch_size - overlap
        size = (inner_region // coverage) * (coverage - 1)
        parition_starts = -np.round(np.linspace(0, size, coverage)).astype(int)[::-1]
        partition = np.arange(0, axis_size - parition_starts[0], inner_region)
        stitch_coords = np.concat([partition + start for start in parition_starts])
        stitch_coords[stitch_coords < axis_size]
        stitch_coords_end = stitch_coords + inner_region
        coords = stitch_coords - overlap // 2

        stitch_coords[stitch_coords < 0] = 0
        stitch_coords_end[stitch_coords_end > axis_size] = axis_size
        coords[coords < 0] = 0
        crop_coords = stitch_coords - coords
        crop_size = stitch_coords_end - stitch_coords

        return (
            coords.tolist(),
            stitch_coords.tolist(),
            crop_coords.tolist(),
            crop_size.tolist(),
        )
