"""Tile information dataclass."""

from __future__ import annotations

from typing import Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated

# Updated to support 1D data (minimum 2 dimensions: C, X)
DimTuple = Annotated[
    tuple[int, ...], Field(min_length=2, max_length=5)  # Changed from min_length=3 to 2
]


class TileInformation(BaseModel):
    """
    Tile information.

    Contains information about a tile, such as its coordinates in the original array
    and whether it is the last tile in the array.

    Parameters
    ----------
    array_shape : tuple of int
        Shape of the tile array.
    last_tile : bool
        Whether the tile is the last tile in the array.
    overlap_crop_coords : tuple of tuple of int
        Coordinates for cropping overlaps.
    stitch_coords : tuple of tuple of int
        Coordinates for stitching the tile back into the original array.
    sample_id : int
        Sample id.
    """

    array_shape: DimTuple  # TODO: find a way to add custom error message?
    """Shape of the original (untiled) array."""

    last_tile: bool = False
    """Whether this tile is the last one of the array."""

    overlap_crop_coords: tuple[tuple[int, int], ...]
    """Inner coordinates of the tile where to crop the prediction in order to stitch
    it back into the original image."""

    stitch_coords: tuple[tuple[int, int], ...]
    """Coordinates in the original image where to stitch the cropped tile back."""

    sample_id: int
    """Sample id."""

    def compatible_with(self, other_tile: TileInformation) -> bool:
        """
        Check if two tiles are compatible for stitching.

        Two tiles are compatible if they have the same array shape.

        Parameters
        ----------
        other_tile : TileInformation
            Other tile to check compatibility with.

        Returns
        -------
        bool
            Whether the two tiles are compatible.
        """
        return (
            self.array_shape == other_tile.array_shape
            and self.sample_id == other_tile.sample_id
        )
