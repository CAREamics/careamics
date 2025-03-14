from collections.abc import Sequence

from .patching_strategy_types import TileSpecs


class TilingStrategy:

    def __init__(
        self,
        data_shapes: Sequence[Sequence[int]],
        tile_size: Sequence[int],
        overlaps: Sequence[int],
    ):
        self.data_shapes = data_shapes
        self.tile_size = tile_size
        self.overlaps = overlaps

        self.tile_specs: list[TileSpecs] = self._generate_specs()

    @property
    def n_patches(self) -> int:
        return len(self.tile_specs)

    def get_patch_spec(self, index: int) -> TileSpecs:
        return self.tile_specs[index]

    def _generate_specs(self) -> list[TileSpecs]:
        tile_specs: list[TileSpecs] = []
        for i, data_shape in enumerate(self.data_shapes):
            spatial_shape = data_shape[2:]
            axis_specs: list[dict[str, list[int]]] = []
            for axis_idx, axis_size in enumerate(spatial_shape):
                axis_specs.append(
                    self._compute_1d_coords(
                        axis_size, self.tile_size[axis_idx], self.overlaps[axis_idx]
                    )
                )
            n_specs = len(axis_specs[0]["coords"])
            for sample_idx in range(data_shape[0]):
                collated: list[TileSpecs] = [
                    {
                        "data_idx": i,
                        "sample_idx": sample_idx,
                        "coords": tuple(
                            axis_specs[k]["coords"][j] for k in range(len(axis_specs))
                        ),
                        "patch_size": self.tile_size,
                        "crop_coords": tuple(
                            axis_specs[k]["crop_coords"][j]
                            for k in range(len(axis_specs))
                        ),
                        "crop_size": tuple(
                            axis_specs[k]["crop_size"][j]
                            for k in range(len(axis_specs))
                        ),
                        "stitch_coords": tuple(
                            axis_specs[k]["stitch_coords"][j]
                            for k in range(len(axis_specs))
                        ),
                    }
                    for j in range(n_specs)
                ]
                tile_specs.extend(collated)
        return tile_specs

    @staticmethod
    def _compute_1d_coords(axis_size: int, tile_size: int, overlap: int):
        coords: list[int] = []
        stitch_coords: list[int] = []
        crop_coords: list[int] = []
        crop_size: list[int] = []

        step = tile_size - overlap
        for i in range(0, max(1, axis_size - overlap), step):
            if i + tile_size < axis_size:
                coords.append(i)
                crop_coords.append(overlap // 2 if i > 0 else 0)
                stitch_coords.append(coords[-1] + crop_coords[-1])
                crop_size.append(
                    tile_size - overlap if i > 0 else tile_size - overlap // 2
                )
            else:
                previous_crop_size = crop_size[-1] if crop_size else 1
                previous_stitch_coord = stitch_coords[-1] if stitch_coords else 0
                previous_tile_end = previous_stitch_coord + previous_crop_size
                coords.append(max(0, axis_size - tile_size - 1))
                stitch_coords.append(previous_tile_end)
                crop_coords.append(stitch_coords[-1] - coords[-1])
                crop_size.append(axis_size - stitch_coords[-1])

        return {
            "coords": coords,
            "stitch_coords": stitch_coords,
            "crop_coords": crop_coords,
            "crop_size": crop_size,
        }
