"""Pydantic model representing CAREamics prediction configuration."""

from __future__ import annotations

from typing import Any, Literal, Self, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .validators import check_axes_validity, patch_size_ge_than_8_power_of_2


class InferenceConfig(BaseModel):
    """Configuration class for the prediction model."""

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    data_type: Literal["array", "tiff", "czi", "custom"]  # As defined in SupportedData
    """Type of input data: numpy.ndarray (array) or path (tiff, czi, or custom)."""

    tile_size: Union[list[int]] | None = Field(default=None, min_length=2, max_length=3)
    """Tile size of prediction, only effective if `tile_overlap` is specified."""

    tile_overlap: Union[list[int]] | None = Field(
        default=None, min_length=2, max_length=3
    )
    """Overlap between tiles, only effective if `tile_size` is specified."""

    axes: str
    """Data axes (TSCZYX) in the order of the input data."""

    image_means: list = Field(..., min_length=0, max_length=32)
    """Mean values for each input channel."""

    image_stds: list = Field(..., min_length=0, max_length=32)
    """Standard deviation values for each input channel."""

    # TODO only default TTAs are supported for now
    tta_transforms: bool = Field(default=True)
    """Whether to apply test-time augmentation (all 90 degrees rotations and flips)."""

    # Dataloader parameters
    batch_size: int = Field(default=1, ge=1)
    """Batch size for prediction."""

    @field_validator("tile_overlap")
    @classmethod
    def all_elements_non_zero_even(
        cls, tile_overlap: list[int] | None
    ) -> list[int] | None:
        """
        Validate tile overlap.

        Overlaps must be non-zero, positive and even.

        Parameters
        ----------
        tile_overlap : list[int] or None
            Patch size.

        Returns
        -------
        list[int] or None
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
    def tile_min_8_power_of_2(cls, tile_list: list[int] | None) -> list[int] | None:
        """
        Validate that each entry is greater or equal than 8 and a power of 2.

        Parameters
        ----------
        tile_list : list of int
            Patch size.

        Returns
        -------
        list of int
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

            if any(
                (i >= j)
                for i, j in zip(self.tile_overlap, self.tile_size, strict=False)
            ):
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
        if not self.image_means and not self.image_stds:
            raise ValueError("Mean and std must be specified during inference.")

        if (self.image_means and not self.image_stds) or (
            self.image_stds and not self.image_means
        ):
            raise ValueError(
                "Mean and std must be either both None, or both specified."
            )

        elif (self.image_means is not None and self.image_stds is not None) and (
            len(self.image_means) != len(self.image_stds)
        ):
            raise ValueError(
                "Mean and std must be specified for each " "input channel."
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

    def set_3D(self, axes: str, tile_size: list[int], tile_overlap: list[int]) -> None:
        """
        Set 3D parameters.

        Parameters
        ----------
        axes : str
            Axes.
        tile_size : list of int
            Tile size.
        tile_overlap : list of int
            Tile overlap.
        """
        self._update(axes=axes, tile_size=tile_size, tile_overlap=tile_overlap)
