from typing import List

from pydantic import BaseModel, Field, FieldValidationInfo, field_validator


class Prediction(BaseModel):
    """Prediction configuration.

    Tile and overlap shapes must be divisible by 2, 2D or 3D. Overlaps must be of same
    dimensions than tile shape and each dimension must be smaller than the corresponding
    tile shape.

    Attributes
    ----------
    tile_shape : List[int]
        2D or 3D shape of the tiles to be predicted.
    overlaps : List[int]
        2D or 3D verlaps between tiles.
    """

    # Mandatory parameters
    tile_shape: List[int] = Field(..., min_length=2, max_length=3)
    overlaps: List[int] = Field(..., min_length=2, max_length=3)

    @field_validator("tile_shape", "overlaps")
    def check_divisible_by_2(cls, dims_list: List[int]) -> List[int]:
        """Validate tile shape and overlaps.

        Both must be positive and divisible by 2.
        """
        for dim in dims_list:
            if dim < 1:
                raise ValueError(f"Entry must be non-null positive (got {dim}).")

            if dim % 2 != 0:
                raise ValueError(f"Entry must be divisible by 2 (got {dim}).")

        return dims_list

    @field_validator("overlaps")
    def check_smaller_than_tile(
        cls, overlaps: List[int], values: FieldValidationInfo
    ) -> List[int]:
        """Validate overlaps.

        Overlaps must be smaller than tile shape.
        """
        if "tile_shape" not in values.data:
            raise ValueError(
                "Cannot validate overlaps without `tile_shape`, make sure it has "
                "correctly been specified."
            )

        # retrieve tile shape
        tile_shape = values.data["tile_shape"]

        if len(overlaps) != len(tile_shape):
            raise ValueError(
                f"Overlaps ({len(overlaps)}) and tile shape ({len(tile_shape)}) must "
                f"have the same number of dimensions."
            )

        for overlap, tile_dim in zip(overlaps, tile_shape):
            if overlap >= tile_dim:
                raise ValueError(
                    f"Overlap ({overlap}) must be smaller than tile shape ({tile_dim})."
                )

        return overlaps
