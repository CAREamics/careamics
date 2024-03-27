from __future__ import annotations

from typing import List, Literal, Optional, Union

from albumentations import Compose
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .support import SupportedTransform
from .transformations.normalize_model import NormalizeModel
from .transformations.transform_model import TransformModel

TRANSFORMS_UNION = Union[NormalizeModel, TransformModel]


class InferenceConfiguration(BaseModel):
    """Configuration class for the prediction model."""

    model_config = ConfigDict(
        validate_assignment=True, 
        arbitrary_types_allowed=True
    )

    # Mandatory fields
    data_type: Literal["array", "tiff", "custom"]
    tile_size: List[int]
    axes: str

    # Optional fields
    tile_overlap: Optional[List[int]] = Field(default=[48, 48])
    mean: Optional[float] = (None,)
    std: Optional[float] = (None,)

    transforms: Union[List[TRANSFORMS_UNION], Compose] = Field(
        default=[
            {
                "name": SupportedTransform.NORMALIZE.value,
            },
        ],
        validate_default=True,
    )

    # only default TTAs are supported fro now
    tta_transforms: bool = Field(default=True)
    extension_filter: str = ""

    # Dataloader parameters
    batch_size: int

    @field_validator("transforms")
    @classmethod
    def validate_transforms(
        cls, transforms: Union[List[TRANSFORMS_UNION], Compose]
    ) -> Union[List[TRANSFORMS_UNION], Compose]:
        """Validate that transforms do not have N2V pixel manipulate transforms.

        Parameters
        ----------
        tta_transforms : Union[List[TransformModel], Compose]
            tta transforms.

        Returns
        -------
        Union[List[Transformations_Union], Compose]
            Validated tta transforms.

        Raises
        ------
        ValueError
            If tta transforms contain N2V pixel manipulate transforms.
        """
        if not isinstance(transforms, Compose):
            for transform in transforms:
                if transform.name == SupportedTransform.N2V_MANIPULATE.value:
                    raise ValueError(
                        "N2V pixel manipulate transforms are not allowed in "
                        "prediction transforms."
                    )

        return transforms

    def has_transform_list(self) -> bool:
        """
        Check if the transforms are a list, as opposed to a Compose object.

        Returns
        -------
        bool
            True if the transforms are a list, False otherwise.
        """
        return isinstance(self.transforms, list)

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

        # search in the tta transforms for Normalize and update parameters
        if not isinstance(self.transforms, Compose):
            for transform in self.transforms:
                if transform.name == SupportedTransform.NORMALIZE.value:
                    transform.parameters.mean = mean
                    transform.parameters.std = std
        else:
            raise ValueError(
                "Setting mean and std with Compose prediction transforms is not"
                "allowed. Add mean and std parameters directly to the transform in the"
                "Compose."
            )
