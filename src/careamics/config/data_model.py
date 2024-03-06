"""Data configuration."""
from __future__ import annotations

from typing import List, Literal, Optional, Union

from albumentations import Compose
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from careamics.utils import check_axes_validity

from .support import SupportedTransform
from .transform_model import TransformModel


class DataModel(BaseModel):
    """
    Data configuration.

    If std is specified, mean must be specified as well. Note that setting the std first
    and then the mean (if they were both `None` before) will raise a validation error.
    Prefer instead the following:
    >>> set_mean_and_std(mean, std)
    """

    # Pydantic class configuration
    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    # Dataset configuration
    # Mandatory fields
    data_type: Literal["array", "tiff", "custom"]
    patch_size: List[int] = Field(..., min_length=2, max_length=3)

    axes: str

    # Optional fields
    mean: Optional[float] = None
    std: Optional[float] = None

    transforms: Union[List[TransformModel], Compose] = Field(
        default=[
            {
                "name": SupportedTransform.NDFLIP.value,
            },
            {
                "name": SupportedTransform.XY_RANDOM_ROTATE90.value,
            },
            {
                "name": SupportedTransform.NORMALIZE.value,
            },
            {
                "name": SupportedTransform.N2V_MANIPULATE.value,
            },
        ],
        validate_default=True,
    )

    tta_transforms: Union[List[TransformModel], Compose] = Field(
        default=[
            {
                "name": SupportedTransform.NORMALIZE.value,
            },
        ],
        validate_default=True,
    )

    # Dataloader configuration
    batch_size: int = Field(default=1, ge=1, validate_default=True)
    num_workers: int = Field(default=0, ge=0, validate_default=True)
    pin_memory: bool = Field(default=False, validate_default=True)

    @field_validator("patch_size")
    @classmethod
    def all_elements_non_zero_even(cls, patch_list: List[int]) -> List[int]:
        """
        Validate patch size.

        Patch size must be non-zero, positive and even.

        Parameters
        ----------
        patch_list : List[int]
            Patch size.

        Returns
        -------
        List[int]
            Validated patch size.

        Raises
        ------
        ValueError
            If the patch size is 0.
        ValueError
            If the patch size is not even.
        """
        for dim in patch_list:
            if dim < 1:
                raise ValueError(f"Patch size must be non-zero positive (got {dim}).")

            if dim % 2 != 0:
                raise ValueError(f"Patch size must be even (got {dim}).")

        return patch_list

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
    
    @field_validator("tta_transforms")
    @classmethod
    def validate_tta_transforms(
        cls, 
        tta_transforms: Union[List[TransformModel], Compose]
    ) -> Union[List[TransformModel], Compose]:
        """Validate that tta transforms do not have N2V pixel manipulate transforms.

        Parameters
        ----------
        tta_transforms : Union[List[TransformModel], Compose]
            tta transforms.

        Returns
        -------
        Union[List[TransformModel], Compose]
            Validated tta transforms.

        Raises
        ------
        ValueError
            If tta transforms contain N2V pixel manipulate transforms.
        """
        if not isinstance(tta_transforms, Compose):
            for transform in tta_transforms:
                if transform.name == SupportedTransform.N2V_MANIPULATE.value:
                    raise ValueError(
                        f"N2V pixel manipulate transforms are not allowed in "
                        f"tta transforms."
                    )
                
        return tta_transforms


    @model_validator(mode="after")
    def std_only_with_mean(cls, data_model: DataModel) -> DataModel:
        """
        Check that mean and std are either both None, or both specified.

        If we enforce both None or both specified, we cannot set the values one by one
        due to the ConfDict enforcing the validation on assignment. Therefore, we check
        only when the std is not None and the mean is None.

        Parameters
        ----------
        data_model : Data
            Data model.

        Returns
        -------
        Data
            Validated data model.

        Raises
        ------
        ValueError
            If std is not None and mean is None.
        """
        if data_model.std is not None and data_model.mean is None:
            raise ValueError("Cannot have `std` field if `mean` is None.")

        return data_model

    @model_validator(mode="after")
    def validate_transforms_and_axes(cls, data_model: DataModel) -> DataModel:
        """
        Validate the transforms with respect to the axes.

        Parameters
        ----------
        data_model : DataModel
            Data model.

        Returns
        -------
        DataModel
            Validated data model.

        Raises
        ------
        ValueError
            If the transforms are not valid.
        """
        if "Z" in data_model.axes:
            if data_model.has_transform_list():
                for transform in data_model.transforms:
                    if transform.name == SupportedTransform.NDFLIP:
                        transform.parameters["is_3D"] = True
                    elif transform.name == SupportedTransform.XY_RANDOM_ROTATE90:
                        transform.parameters["is_3D"] = True

            if data_model.has_tta_transform_list():
                for transform in data_model.tta_transforms:
                    if transform.name == SupportedTransform.NDFLIP:
                        transform.parameters["is_3D"] = True
                    elif transform.name == SupportedTransform.XY_RANDOM_ROTATE90:
                        transform.parameters["is_3D"] = True
        else:
            if data_model.has_transform_list():
                for transform in data_model.transforms:
                    if transform.name == SupportedTransform.NDFLIP:
                        transform.parameters["is_3D"] = False
                    elif transform.name == SupportedTransform.XY_RANDOM_ROTATE90:
                        transform.parameters["is_3D"] = False

            if data_model.has_tta_transform_list():
                for transform in data_model.tta_transforms:
                    if transform.name == SupportedTransform.NDFLIP:
                        transform.parameters["is_3D"] = False
                    elif transform.name == SupportedTransform.XY_RANDOM_ROTATE90:
                        transform.parameters["is_3D"] = False


        return data_model

    def has_transform_list(self) -> bool:
        """
        Check if the transforms are a list, as opposed to a Compose object.

        Returns
        -------
        bool
            True if the transforms are a list, False otherwise.
        """
        return isinstance(self.transforms, list)

    def has_tta_transform_list(self) -> bool:
        """
        Check if the tta transforms are a list, as opposed to a Compose object.

        Returns
        -------
        bool
            True if the transforms are a list, False otherwise.
        """
        return isinstance(self.tta_transforms, list)

    def set_mean_and_std(self, mean: float, std: float) -> None:
        """
        Set mean and standard deviation of the data.

        This method is preferred to setting the fields directly, as it ensures that the
        mean is set first, then the std; thus avoiding a validation error to be thrown.

        Parameters
        ----------
        mean : float
            Mean of the data.
        std : float
            Standard deviation of the data.
        """
        self.mean = mean
        self.std = std

        # search in the transforms for Normalize and update parameters
        if not isinstance(self.transforms, Compose):
            for transform in self.transforms:
                if transform.name == SupportedTransform.NORMALIZE.value:
                    transform.parameters["mean"] = mean
                    transform.parameters["std"] = std
                    transform.parameters["max_pixel_value"] = 1.0
        else:
            raise ValueError(
                f"Setting mean and std with Compose transforms is not allowed. Add "
                f"mean and std parameters directly to the transform in the Compose."
            )
        
        # search in the tta transforms for Normalize and update parameters
        if not isinstance(self.tta_transforms, Compose):
            for transform in self.tta_transforms:
                if transform.name == SupportedTransform.NORMALIZE.value:
                    transform.parameters["mean"] = mean
                    transform.parameters["std"] = std
                    transform.parameters["max_pixel_value"] = 1.0
        else:
            raise ValueError(
                f"Setting mean and std with Compose tta transforms is not allowed. Add "
                f"mean and std parameters directly to the transform in the Compose."
            )
        