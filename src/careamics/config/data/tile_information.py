"""Pydantic model representing the metadata of a prediction tile."""

from __future__ import annotations

from typing import Annotated

from annotated_types import Len
from pydantic import BaseModel, ConfigDict

DimTuple = Annotated[tuple, Len(min_length=3, max_length=4)]


class TileInformation(BaseModel):
    """
    Pydantic model containing tile information.

    This model is used to represent the information required to stitch back a tile into
    a larger image. It is used throughout the prediction pipeline of CAREamics.

    Array shape should be C(Z)YX, where Z is an optional dimensions.
    """

    model_config = ConfigDict(validate_default=True)

    array_shape: DimTuple  # TODO: find a way to add custom error message?
    """Shape of the original (untiled) array."""

    last_tile: bool = False
    """Whether this tile is the last one of the array."""

    overlap_crop_coords: tuple[tuple[int, ...], ...]
    """Inner coordinates of the tile where to crop the prediction in order to stitch
    it back into the original image."""

    stitch_coords: tuple[tuple[int, ...], ...]
    """Coordinates in the original image where to stitch the cropped tile back."""

    sample_id: int
    """Sample ID of the tile."""

    # TODO: Test that ZYX axes are not singleton ?

    def __eq__(self, other_tile: object):
        """Check if two tile information objects are equal.

        Parameters
        ----------
        other_tile : object
            Tile information object to compare with.

        Returns
        -------
        bool
            Whether the two tile information objects are equal.
        """
        if not isinstance(other_tile, TileInformation):
            return NotImplemented

        return (
            self.array_shape == other_tile.array_shape
            and self.last_tile == other_tile.last_tile
            and self.overlap_crop_coords == other_tile.overlap_crop_coords
            and self.stitch_coords == other_tile.stitch_coords
            and self.sample_id == other_tile.sample_id
        )
