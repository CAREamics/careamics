"""Module for the `TilingStrategy` class."""

import itertools
from collections.abc import Sequence

from .patching_strategy_protocol import TileSpecs


class TilingStrategy:
    """
    The tiling strategy should be used for prediction. The `get_patch_specs`
    method returns `TileSpec` dictionaries that contains information on how to
    stitch the tiles back together to create the full image.
    """

    def __init__(
        self,
        data_shapes: Sequence[Sequence[int]],
        tile_size: Sequence[int],
        overlaps: Sequence[int],
    ):
        """
        The tiling strategy should be used for prediction. The `get_patch_specs`
        method returns `TileSpec` dictionaries that contains information on how to
        stitch the tiles back together to create the full image.

        Parameters
        ----------
        data_shapes : sequence of (sequence of int)
            The shapes of the underlying data. Each element is the dimension of the
            axes SC(Z)YX.
        tile_size : sequence of int
            The size of the tile. The sequence will have length 2 or 3, for 2D and 3D
            data respectively.
        overlaps : sequence of int
            How much a tile will overlap with adjacent tiles in each spatial dimension.
        """
        self.data_shapes = data_shapes
        self.tile_size = tile_size
        self.overlaps = overlaps
        # tile_size and overlap should have same length validated in pydantic configs
        self.tile_specs: list[TileSpecs] = self._generate_specs()

    @property
    def n_patches(self) -> int:
        """
        The number of patches that this patching strategy will return.

        It also determines the maximum index that can be given to `get_patch_spec`.
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
            for i, patch_spec in enumerate(self.tile_specs)
            if patch_spec["data_idx"] == data_idx
        ]

    def _generate_specs(self) -> list[TileSpecs]:
        tile_specs: list[TileSpecs] = []
        for data_idx, data_shape in enumerate(self.data_shapes):
            spatial_shape = data_shape[2:]

            # spec info for each axis
            axis_specs: list[tuple[list[int], list[int], list[int], list[int]]] = [
                self._compute_1d_coords(
                    axis_size, self.tile_size[axis_idx], self.overlaps[axis_idx]
                )
                for axis_idx, axis_size in enumerate(spatial_shape)
            ]

            # combine by using zip
            all_coords, all_stitch_coords, all_crop_coords, all_crop_size = zip(
                *axis_specs, strict=False
            )
            # patches will be the same for each sample in a stack
            for sample_idx in range(data_shape[0]):
                # iterate through all combinations using itertools.product
                for coords, stitch_coords, crop_coords, crop_size in zip(
                    itertools.product(*all_coords),
                    itertools.product(*all_stitch_coords),
                    itertools.product(*all_crop_coords),
                    itertools.product(*all_crop_size),
                    strict=False,
                ):
                    tile_specs.append(
                        {
                            # PatchSpecs
                            "data_idx": data_idx,
                            "sample_idx": sample_idx,
                            "coords": coords,
                            "patch_size": self.tile_size,
                            # TileSpecs additional fields
                            "crop_coords": crop_coords,
                            "crop_size": crop_size,
                            "stitch_coords": stitch_coords,
                        }
                    )
        return tile_specs

    @staticmethod
    def _compute_1d_coords(
        axis_size: int, tile_size: int, overlap: int
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        """
        Computes the TileSpec information for a single axis.

        Parameters
        ----------
        axis_size : int
            The size of the axis.
        tile_size : int
            The tile size.
        overlap : int
            The tile overlap.

        Returns
        -------
        coords: list of int
            The top-left (and first z-slice for 3D data) of a tile, in coords relative
            to the image.
        stitch_coords: list of int
            Where the tile will be stitched back into an image, taking into account
            that the tile will be cropped, in coords relative to the image.
        crop_coords: list of int
            The top-left side of where the tile will be cropped, in coordinates relative
            to the tile.
        crop_size: list of int
            The size of the cropped tile.
        """
        coords: list[int] = []
        stitch_coords: list[int] = []
        crop_coords: list[int] = []
        crop_size: list[int] = []

        step = tile_size - overlap
        for i in range(0, max(1, axis_size - overlap), step):
            if i == 0:
                coords.append(i)
                crop_coords.append(0)
                stitch_coords.append(0)
                if axis_size <= tile_size:
                    crop_size.append(axis_size)
                else:
                    crop_size.append(tile_size - overlap // 2)
            elif (0 < i) and (i + tile_size < axis_size):
                coords.append(i)
                crop_coords.append(overlap // 2)
                stitch_coords.append(coords[-1] + crop_coords[-1])
                crop_size.append(tile_size - overlap)
            else:
                previous_crop_size = crop_size[-1] if crop_size else 1
                previous_stitch_coord = stitch_coords[-1] if stitch_coords else 0
                previous_tile_end = previous_stitch_coord + previous_crop_size

                coords.append(max(0, axis_size - tile_size))
                stitch_coords.append(previous_tile_end)
                crop_coords.append(stitch_coords[-1] - coords[-1])
                crop_size.append(axis_size - stitch_coords[-1])

        return (
            coords,
            stitch_coords,
            crop_coords,
            crop_size,
        )
