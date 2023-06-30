from pydantic import conlist, validator

from .data import Data
from .stage import Stage


class Prediction(Stage):
    # Mandatory parameters
    tile_shape: conlist(int, min_items=2, max_items=3)

    # Optional parameter
    overlaps: conlist(int, min_items=2, max_items=3) = [48, 48]

    # Sub-configuration
    prediction_data: Data

    @validator("tile_shape, overlaps")
    def check_divisible_by_2(cls, dims_list: conlist) -> conlist:
        """Validate tile shape and overlaps.

        Both must be positive and divisible by 2.
        """
        for dim in dims_list:
            if dim < 0:
                raise ValueError(f"Entry must be positive (got {dim}).")

            if dim % 2 != 0:
                raise ValueError(f"Entry must be divisible by 2 (got {dim}).")

        return dims_list

    @validator("overlaps")
    def check_smaller_than_tile(cls, overlaps: conlist, values: dict) -> conlist:
        """Validate overlaps.

        Overlaps must be smaller than tile shape.
        """
        tile_shape = values["tile_shape"]

        for overlap, tile_dim in zip(overlaps, tile_shape):
            if overlap >= tile_dim:
                raise ValueError(
                    f"Overlap ({overlap}) must be smaller than tile shape ({tile_dim})."
                )

        return overlaps

    class Config:
        allow_mutation = False
