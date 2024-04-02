from __future__ import annotations

from typing import Any, List, Literal, Optional, Union

from albumentations import Compose
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from careamics.utils import check_axes_validity

from .support import SupportedTransform
from .transformations.normalize_model import NormalizeModel
from .transformations.transform_model import TransformModel

TRANSFORMS_UNION = Union[NormalizeModel, TransformModel]


class InferenceModel(BaseModel):
    """Configuration class for the prediction model."""

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True
    )

    # Mandatory fields
    data_type: Literal["array", "tiff", "custom"] # As defined in SupportedData
    tile_size: List[int] = Field(..., min_items=2, max_items=3)
    # TODO Overlaps?

    axes: str

    # Optional fields
    tile_overlap: Optional[List[int]] = Field(default=[48, 48])
    mean: Optional[float] = (None,)
    std: Optional[float] = (None,)

    transforms: Optional[Union[List[TRANSFORMS_UNION], Compose]] = Field(
        default=[
            {
                "name": SupportedTransform.NORMALIZE.value,
            },
        ],
        validate_default=True,
    )

    # only default TTAs are supported for now
    tta_transforms: bool = Field(default=True)
    # extension_filter: str = ""

    # Dataloader parameters
    batch_size: int = Field(default=1, ge=1)

    @field_validator("tile_size")
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

    @field_validator("transforms")
    @classmethod
    def validate_transforms(
        cls, transforms: Union[List[TRANSFORMS_UNION], Compose]
    ) -> Union[List[TRANSFORMS_UNION], Compose]:
        """Validate that transforms do not have N2V pixel manipulate transforms.

        Parameters
        ----------
        transforms : Union[List[TransformModel], Compose]
            Transforms.

        Returns
        -------
        Union[List[Transformations_Union], Compose]
            Validated transforms.

        Raises
        ------
        ValueError
            If transforms contain N2V pixel manipulate transforms.
        """
        if not isinstance(transforms, Compose) and transforms is not None:
            for transform in transforms:
                if transform.name == SupportedTransform.N2V_MANIPULATE.value:
                    raise ValueError(
                        "N2V_Manipulate transform is not allowed in "
                        "prediction transforms."
                    )

        return transforms

    @model_validator(mode="after")
    def validate_dimensions(cls, pred_model: InferenceModel) -> InferenceModel:
        """
        Validate 2D/3D dimensions between axes and tile size.

        Parameters
        ----------
        pred_model : PredictionModel
            Prediction model.

        Returns
        -------
        PredictionModel
            Validated prediction model.
        """
        expected_len = 3 if "Z" in pred_model.axes else 2

        if len(pred_model.tile_size) != expected_len:
            raise ValueError(
                f"Tile size must have {expected_len} dimensions given axes "
                f"{pred_model.axes}."
            )

        return pred_model

    @model_validator(mode="after")
    def std_only_with_mean(cls, pred_model: InferenceModel) -> InferenceModel:
        """
        Check that mean and std are either both None, or both specified.

        Parameters
        ----------
        pred_model : Data
            Data model.

        Returns
        -------
        Data
            Validated prediction model.

        Raises
        ------
        ValueError
            If std is not None and mean is None.
        """
        # check that mean and std are either both None, or both specified
        if (pred_model.mean is None) != (pred_model.std is None):
            raise ValueError(
                "Mean and std must be either both None, or both specified."
            )

        return pred_model

    def _update(self, **kwargs: Any):
        """Update multiple arguments at once."""
        self.__dict__.update(kwargs)
        self.__class__.model_validate(self.__dict__)

    def set_mean_and_std(self, mean: float, std: float) -> None:
        """
        Set mean and standard deviation of the data.

        This method should be used instead setting the fields directly, as it would
        otherwise trigger a validation error.

        Parameters
        ----------
        mean : float
            Mean of the data.
        std : float
            Standard deviation of the data.
        """
        self._update(mean=mean, std=std)

        # search in the transforms for Normalize and update parameters
        if not isinstance(self.transforms, Compose):
            for transform in self.transforms:
                if transform.name == SupportedTransform.NORMALIZE.value:
                    transform.parameters.mean = mean
                    transform.parameters.std = std
        else:
            raise ValueError(
                "Setting mean and std with Compose transforms is not allowed. Add "
                "mean and std parameters directly to the transform in the Compose."
            )

    def set_3D(self, axes: str, tile_size: List[int]) -> None:
        """
        Set 3D parameters.

        Parameters
        ----------
        axes : str
            Axes.
        tile_size : List[int]
            Tile size.
        """
        self._update(axes=axes, tile_size=tile_size)
