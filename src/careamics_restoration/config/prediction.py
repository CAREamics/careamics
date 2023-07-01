from pydantic import BaseModel, FieldValidationInfo, conlist, field_validator


class Prediction(BaseModel):
    """Prediction configuration.

    Tile and overlap shapes must be divisible by 2, and overlaps must be smaller than
    the tile shapes.

    Attributes
    ----------
    tile_shape : conlist(int, min_length=2, max_length=3)
        Shape of the tiles to be predicted.
    overlaps : conlist(int, min_length=2, max_length=3)
        Overlaps between tiles.
    """

    # Mandatory parameters
    tile_shape: conlist(int, min_length=2, max_length=3)
    overlaps: conlist(int, min_length=2, max_length=3)

    @field_validator("tile_shape", "overlaps")
    def check_divisible_by_2(cls, dims_list: conlist) -> conlist:
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
        cls, overlaps: conlist, values: FieldValidationInfo
    ) -> conlist:
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
