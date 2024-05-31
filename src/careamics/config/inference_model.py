"""Pydantic model representing CAREamics prediction configuration."""

from __future__ import annotations

from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from .validators import check_axes_validity, patch_size_ge_than_8_power_of_2


class InferenceConfig(BaseModel):
    """Configuration class for the prediction model."""

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    # Mandatory fields
    data_type: Literal["array", "tiff", "custom"]  # As defined in SupportedData
    tile_size: Optional[Union[List[int]]] = Field(
        default=None, min_length=2, max_length=3
    )
    tile_overlap: Optional[Union[List[int]]] = Field(
        default=None, min_length=2, max_length=3
    )

    axes: str

    mean: float
    std: float = Field(..., ge=0.0)

    # only default TTAs are supported for now
    tta_transforms: bool = Field(default=True)

    # Dataloader parameters
    batch_size: int = Field(default=1, ge=1)

    @field_validator("tile_overlap")
    @classmethod
    def all_elements_non_zero_even(
        cls, tile_overlap: Optional[Union[List[int]]]
    ) -> Optional[Union[List[int]]]:
        """
        Validate tile overlap.

        Overlaps must be non-zero, positive and even.

        Parameters
        ----------
        tile_overlap : Optional[Union[List[int]]]
            Patch size.

        Returns
        -------
        Optional[Union[List[int]]]
            Validated tile overlap.

        Raises
        ------
        ValueError
            If the patch size is 0.
        ValueError
            If the patch size is not even.
        """
        if tile_overlap is not None:
            for dim in tile_overlap:
                if dim < 1:
                    raise ValueError(
                        f"Patch size must be non-zero positive (got {dim})."
                    )

                if dim % 2 != 0:
                    raise ValueError(f"Patch size must be even (got {dim}).")

        return tile_overlap

    @field_validator("tile_size")
    @classmethod
    def tile_min_8_power_of_2(
        cls, tile_list: Optional[Union[List[int]]]
    ) -> Optional[Union[List[int]]]:
        """
        Validate that each entry is greater or equal than 8 and a power of 2.

        Parameters
        ----------
        tile_list : List[int]
            Patch size.

        Returns
        -------
        List[int]
            Validated patch size.

        Raises
        ------
        ValueError
            If the patch size if smaller than 8.
        ValueError
            If the patch size is not a power of 2.
        """
        patch_size_ge_than_8_power_of_2(tile_list)

        return tile_list

    @field_validator("axes")
    @classmethod
    def axes_valid(cls, axes: str) -> str:
        """
        Validate axes.

        Axes must:
        - be a combination of 'STCZYX'
        - not contain duplicates
        - contain at least 2 contiguous axes: X and Y
        - contain at most 4 axes
        - not contain both S and T axes

        Parameters
        ----------
        axes : str
            Axes to validate.

        Returns
        -------
        str
            Validated axes.

        Raises
        ------
        ValueError
            If axes are not valid.
        """
        # Validate axes
        check_axes_validity(axes)

        return axes

    @model_validator(mode="after")
    def validate_dimensions(self: Self) -> Self:
        """
        Validate 2D/3D dimensions between axes and tile size.

        Returns
        -------
        Self
            Validated prediction model.
        """
        expected_len = 3 if "Z" in self.axes else 2

        if self.tile_size is not None and self.tile_overlap is not None:
            if len(self.tile_size) != expected_len:
                raise ValueError(
                    f"Tile size must have {expected_len} dimensions given axes "
                    f"{self.axes} (got {self.tile_size})."
                )

            if len(self.tile_overlap) != expected_len:
                raise ValueError(
                    f"Tile overlap must have {expected_len} dimensions given axes "
                    f"{self.axes} (got {self.tile_overlap})."
                )

            if any((i >= j) for i, j in zip(self.tile_overlap, self.tile_size)):
                raise ValueError("Tile overlap must be smaller than tile size.")

        return self

    @model_validator(mode="after")
    def std_only_with_mean(self: Self) -> Self:
        """
        Check that mean and std are either both None, or both specified.

        Returns
        -------
        Self
            Validated prediction model.

        Raises
        ------
        ValueError
            If std is not None and mean is None.
        """
        # check that mean and std are either both None, or both specified
        if (self.mean is None) != (self.std is None):
            raise ValueError(
                "Mean and std must be either both None, or both specified."
            )

        return self

    def _update(self, **kwargs: Any) -> None:
        """
        Update multiple arguments at once.

        Parameters
        ----------
        **kwargs : Any
            Key-value pairs of arguments to update.
        """
        self.__dict__.update(kwargs)
        self.__class__.model_validate(self.__dict__)

    def set_3D(self, axes: str, tile_size: List[int], tile_overlap: List[int]) -> None:
        """
        Set 3D parameters.

        Parameters
        ----------
        axes : str
            Axes.
        tile_size : List[int]
            Tile size.
        tile_overlap : List[int]
            Tile overlap.
        """
        self._update(axes=axes, tile_size=tile_size, tile_overlap=tile_overlap)
