from typing import List, Optional

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

    # Optional parameters
    tile_shape: Optional[List[int]] = Field(default=None, min_length=2, max_length=3)
    overlaps: Optional[List[int]] = Field(default=None, min_length=2, max_length=3)

    # Mandatory parameter
    # defined after the optional ones to allow checking for their presence
    use_tiling: bool

    @field_validator("tile_shape", "overlaps")
    def all_elements_non_zero_divisible_by_two(cls, dims_list: List[int]) -> List[int]:
        """Validate tile shape and overlaps.

        Both must be positive and divisible by 2.
        """
        if dims_list is None:
            raise ValueError("Entry cannot be None.")

        for dim in dims_list:
            if dim < 1:
                raise ValueError(f"Entry must be non-null positive (got {dim}).")

            if dim % 2 != 0:
                raise ValueError(f"Entry must be divisible by 2 (got {dim}).")

        return dims_list

    @field_validator("overlaps")
    def overlaps_smaller_than_tiles(
        cls, overlaps: List[int], values: FieldValidationInfo
    ) -> List[int]:
        """Validate overlaps.

        Overlaps must be smaller than tile shape.
        """
        if overlaps is None:
            raise ValueError("Overlaps cannot be None.")

        if "tile_shape" not in values.data or values.data["tile_shape"] is None:
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

    @field_validator("use_tiling")
    def optional_parameters_if_tiling_true(
        cls, use_tiling: bool, values: FieldValidationInfo
    ) -> bool:
        """ TODO Joran fix me 
        Validate `use_tiling` if False, or only when optional parameters are
        specified if True.
        """
        if use_tiling:
            if (
                "tile_shape" not in values.data or values.data["tile_shape"] is None
            ) or ("overlaps" not in values.data or values.data["overlaps"] is None):
                raise ValueError(
                    "Cannot use tiling without specifying `tile_shape` and `overlaps`, "
                    "make sure they have correctly been specified."
                )

        return use_tiling

    def model_dump(self, *args, **kwargs) -> dict:
        """Override model_dump method.

        The purpose is to ensure export smooth import to yaml. It includes:
            - remove entries with None value

        Returns
        -------
        dict
            Dictionary containing the model parameters
        """
        return super().model_dump(exclude_none=True)
