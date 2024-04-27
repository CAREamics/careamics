from __future__ import annotations

from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class TileInformation(BaseModel):
    """
    Pydantic model containing tile information.

    This model is used to represent the information required to stitch back a tile into
    a larger image. It is used throughout the prediction pipeline of CAREamics.
    """

    model_config = ConfigDict(validate_default=True)

    array_shape: Tuple[int, ...]
    tiled: bool = False
    last_tile: bool = False
    overlap_crop_coords: Optional[Tuple[Tuple[int, ...], ...]] = Field(default=None)
    stitch_coords: Optional[Tuple[Tuple[int, ...], ...]] = Field(default=None)

    @field_validator("array_shape")
    @classmethod
    def no_singleton_dimensions(cls, v: Tuple[int, ...]):
        """
        Check that the array shape does not have any singleton dimensions.

        Parameters
        ----------
        v : Tuple[int, ...]
            Array shape to check.

        Returns
        -------
        Tuple[int, ...]
            The array shape if it does not contain singleton dimensions.

        Raises
        ------
        ValueError
            If the array shape contains singleton dimensions.
        """
        if any(dim == 1 for dim in v):
            raise ValueError("Array shape must not contain singleton dimensions.")
        return v

    @field_validator("last_tile")
    @classmethod
    def only_if_tiled(cls, v: bool, values: ValidationInfo):
        """
        Check that the last tile flag is only set if tiling is enabled.

        Parameters
        ----------
        v : bool
            Last tile flag.
        values : ValidationInfo
            Validation information.

        Returns
        -------
        bool
            The last tile flag.
        """
        if not values.data["tiled"]:
            return False
        return v

    @field_validator("overlap_crop_coords", "stitch_coords")
    @classmethod
    def mandatory_if_tiled(
        cls, v: Optional[Tuple[int, ...]], values: ValidationInfo
    ) -> Optional[Tuple[int, ...]]:
        """
        Check that the coordinates are not `None` if tiling is enabled.

        The method also return `None` if tiling is not enabled.

        Parameters
        ----------
        v : Optional[Tuple[int, ...]]
            Coordinates to check.
        values : ValidationInfo
            Validation information.

        Returns
        -------
        Optional[Tuple[int, ...]]
            The coordinates if tiling is enabled, otherwise `None`.

        Raises
        ------
        ValueError
            If the coordinates are `None` and tiling is enabled.
        """
        if values.data["tiled"]:
            if v is None:
                raise ValueError("Value must be specified if tiling is enabled.")

            return v
        else:
            return None
