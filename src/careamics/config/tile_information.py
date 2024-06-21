"""Pydantic model representing the metadata of a prediction tile."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator


class TileInformation(BaseModel):
    """
    Pydantic model containing tile information.

    This model is used to represent the information required to stitch back a tile into
    a larger image. It is used throughout the prediction pipeline of CAREamics.

    Array shape should be (C)(Z)YX, where C and Z are optional dimensions, and must not
    contain singleton dimensions.
    """

    model_config = ConfigDict(validate_default=True)

    array_shape: tuple[int, ...]
    last_tile: bool = False
    overlap_crop_coords: tuple[tuple[int, ...], ...]
    stitch_coords: tuple[tuple[int, ...], ...]
    sample_id: int

    @field_validator("array_shape")
    @classmethod
    def no_singleton_dimensions(cls, v: tuple[int, ...]):
        """
        Check that the array shape does not have any singleton dimensions.

        Parameters
        ----------
        v : tuple of int
            Array shape to check.

        Returns
        -------
        tuple of int
            The array shape if it does not contain singleton dimensions.

        Raises
        ------
        ValueError
            If the array shape contains singleton dimensions.
        """
        if any(dim == 1 for dim in v):
            raise ValueError("Array shape must not contain singleton dimensions.")
        return v

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
