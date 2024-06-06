"""Pydantic model representing CAREamics prediction configuration."""

from __future__ import annotations

from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from .support import SupportedTransform
from .transformations.normalize_model import NormalizeModel
from .validators import check_axes_validity, patch_size_ge_than_8_power_of_2

TRANSFORMS_UNION = Union[NormalizeModel]


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

    image_mean: Optional[List] = Field(default=[], min_length=0, max_length=32)
    image_std: Optional[List] = Field(default=[], min_length=0, max_length=32)

    transforms: List[TRANSFORMS_UNION] = Field(
        default=[
            {
                "name": SupportedTransform.NORMALIZE.value,
            },
        ],
        validate_default=True,
    )

    # only default TTAs are supported for now
    tta_transforms: bool = Field(default=True)

    # Dataloader parameters
    batch_size: int = Field(default=1, ge=1)

    @field_validator("tile_overlap")
    @classmethod
    def all_elements_non_zero_even(
        cls, patch_list: Optional[Union[List[int]]]
    ) -> Optional[Union[List[int]]]:
        """
        Validate patch size.

        Patch size must be non-zero, positive and even.

        Parameters
        ----------
        patch_list : Optional[Union[List[int]]]
            Patch size.

        Returns
        -------
        Optional[Union[List[int]]]
            Validated patch size.

        Raises
        ------
        ValueError
            If the patch size is 0.
        ValueError
            If the patch size is not even.
        """
        if patch_list is not None:
            for dim in patch_list:
                if dim < 1:
                    raise ValueError(
                        f"Patch size must be non-zero positive (got {dim})."
                    )

                if dim % 2 != 0:
                    raise ValueError(f"Patch size must be even (got {dim}).")

        return patch_list

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

    @field_validator("transforms")
    @classmethod
    def validate_transforms(
        cls, transforms: List[TRANSFORMS_UNION]
    ) -> List[TRANSFORMS_UNION]:
        """
        Validate that transforms do not have N2V pixel manipulate transforms.

        Parameters
        ----------
        transforms : List[TRANSFORMS_UNION]
            Transforms.

        Returns
        -------
        List[TRANSFORMS_UNION]
            Validated transforms.

        Raises
        ------
        ValueError
            If transforms contain N2V pixel manipulate transforms.
        """
        if transforms is not None:
            for transform in transforms:
                if transform.name == SupportedTransform.N2V_MANIPULATE.value:
                    raise ValueError(
                        "N2V_Manipulate transform is not allowed in "
                        "prediction transforms."
                    )

        return transforms

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
        if not self.image_mean and not self.image_std:
            raise ValueError("Mean and std must be specified during inference.")

        if (self.image_mean and not self.image_std) or (
            self.image_std and not self.image_mean
        ):
            raise ValueError(
                "Mean and std must be either both None, or both specified."
            )

        elif (self.image_mean is not None and self.image_std is not None) and (
            len(self.image_mean) != len(self.image_std)
        ):
            raise ValueError(
                "Mean and std must be specified for each " "input channel."
            )

        return self

    @model_validator(mode="after")
    def add_std_and_mean_to_normalize(self: Self) -> Self:
        """
        Add mean and std to the Normalize transform if it is present.

        Returns
        -------
        Self
            Inference model with mean and std added to the Normalize transform.
        """
        if self.image_mean and self.image_std:
            # search in the transforms for Normalize and update parameters
            for transform in self.transforms:
                if transform.name == SupportedTransform.NORMALIZE.value:
                    transform.image_means = self.image_mean
                    transform.image_stds = self.image_std

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
