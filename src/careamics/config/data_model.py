"""Data configuration."""
from __future__ import annotations

from pprint import pformat
from typing import Any, List, Literal, Optional, Union

from albumentations import Compose
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from careamics.utils import check_axes_validity

from .support import SupportedTransform
from .transformations.n2v_manipulate_model import N2VManipulationModel
from .transformations.nd_flip_model import NDFlipModel
from .transformations.normalize_model import NormalizeModel
from .transformations.transform_model import TransformModel
from .transformations.xy_random_rotate90_model import XYRandomRotate90Model

TRANSFORMS_UNION = Union[
    NDFlipModel,
    XYRandomRotate90Model,
    NormalizeModel,
    N2VManipulationModel,
    TransformModel,
]

# TODO can we check whether N2V manipulate is in a Compose?
# TODO does patches need to be multiple of 8 with UNet?
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
        arbitrary_types_allowed=True,  # Allow Compose declaration
    )

    # Dataset configuration
    data_type: Literal["array", "tiff", "custom"]  # As defined in SupportedData
    patch_size: List[int] = Field(..., min_length=2, max_length=3)
    batch_size: int = Field(default=1, ge=1, validate_default=True)
    axes: str

    # Optional fields
    mean: Optional[float] = None
    std: Optional[float] = None

    transforms: Union[List[TRANSFORMS_UNION], Compose] = Field(
        default=[
            {
                "name": SupportedTransform.NORMALIZE.value,
            },
            {
                "name": SupportedTransform.NDFLIP.value,
            },
            {
                "name": SupportedTransform.XY_RANDOM_ROTATE90.value,
            },
            {
                "name": SupportedTransform.N2V_MANIPULATE.value,
            },
        ],
        validate_default=True,
    )

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

    @field_validator("transforms")
    @classmethod
    def validate_prediction_transforms(
        cls, transforms: Union[List[TRANSFORMS_UNION], Compose]
    ) -> Union[List[TRANSFORMS_UNION], Compose]:
        """Validate N2VManipulate transform position in the transform list.

        Parameters
        ----------
        transforms : Union[List[Transformations_Union], Compose]
            Transforms.

        Returns
        -------
        Union[List[Transformations_Union], Compose]
            Validated transforms.

        Raises
        ------
        ValueError
            If multiple instances of N2VManipulate are found.
        """
        if not isinstance(transforms, Compose):
            transform_list = [t.name for t in transforms]

            if SupportedTransform.N2V_MANIPULATE in transform_list:
                # multiple N2V_MANIPULATE
                if transform_list.count(SupportedTransform.N2V_MANIPULATE) > 1:
                    raise ValueError(
                        f"Multiple instances of "
                        f"{SupportedTransform.N2V_MANIPULATE} transforms "
                        f"are not allowed."
                    )

                # N2V_MANIPULATE not the last transform
                elif transform_list[-1] != SupportedTransform.N2V_MANIPULATE:
                    index = transform_list.index(SupportedTransform.N2V_MANIPULATE)
                    transform = transforms.pop(index)
                    transforms.append(transform)

        return transforms

    @model_validator(mode="after")
    def std_only_with_mean(cls, data_model: DataModel) -> DataModel:
        """
        Check that mean and std are either both None, or both specified.

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
        # check that mean and std are either both None, or both specified
        if (data_model.mean is None) != (data_model.std is None):
            raise ValueError(
                "Mean and std must be either both None, or both specified."
            )

        return data_model
    

    @model_validator(mode="after")
    def add_std_and_mean_to_normalize(
        cls, data_model: DataModel
    ) -> DataModel:
        """
        Add mean and std to the Normalize transform if it is present.

        Parameters
        ----------
        data_model : DataModel
            Data model.

        Returns
        -------
        DataModel
            Data model with mean and std added to the Normalize transform.
        """
        if data_model.mean is not None or data_model.std is not None:
            # search in the transforms for Normalize and update parameters
            if data_model.has_transform_list():
                for transform in data_model.transforms:
                    if transform.name == SupportedTransform.NORMALIZE.value:
                        transform.parameters.mean = data_model.mean
                        transform.parameters.std = data_model.std

        return data_model


    @model_validator(mode="after")
    def validate_dimensions(cls, data_model: DataModel) -> DataModel:
        """
        Validate 2D/3D dimensions between axes, patch size and transforms.

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
            if len(data_model.patch_size) != 3:
                raise ValueError(
                    f"Patch size must have 3 dimensions if the data is 3D "
                    f"({data_model.axes})."
                )

            if data_model.has_transform_list():
                for transform in data_model.transforms:
                    if transform.name == SupportedTransform.NDFLIP:
                        transform.parameters.is_3D = True
                    elif transform.name == SupportedTransform.XY_RANDOM_ROTATE90:
                        transform.parameters.is_3D = True

        else:
            if len(data_model.patch_size) != 2:
                raise ValueError(
                    f"Patch size must have 3 dimensions if the data is 3D "
                    f"({data_model.axes})."
                )

            if data_model.has_transform_list():
                for transform in data_model.transforms:
                    if transform.name == SupportedTransform.NDFLIP:
                        transform.parameters.is_3D = False
                    elif transform.name == SupportedTransform.XY_RANDOM_ROTATE90:
                        transform.parameters.is_3D = False

        return data_model

    def __str__(self) -> str:
        """Pretty string reprensenting the configuration.

        Returns
        -------
        str
            Pretty string.
        """
        return pformat(self.model_dump())

    def _update(self, **kwargs: Any) -> None:
        """Update multiple arguments at once."""
        self.__dict__.update(kwargs)
        self.__class__.model_validate(self.__dict__)

    def has_transform_list(self) -> bool:
        """
        Check if the transforms are a list, as opposed to a Compose object.

        Returns
        -------
        bool
            True if the transforms are a list, False otherwise.
        """
        return isinstance(self.transforms, list)
    
    def has_n2v_manipulate(self) -> bool:
        """
        Check if the transforms contain N2VManipulate.

        Use `has_transform_list` to check if the transforms are a list.

        Returns
        -------
        bool
            True if the transforms contain N2VManipulate, False otherwise.

        Raises
        ------
        ValueError
            If the transforms are a Compose object.
        """
        if self.has_transform_list():
            return any(
                transform.name == SupportedTransform.N2V_MANIPULATE.value
                for transform in self.transforms
            )
        else:
            raise ValueError(
                "Checking for N2VManipulate with Compose transforms is not allowed. "
                "Check directly in the Compose."
            )
        
    def add_n2v_manipulate(self) -> None:
        """
        Add N2VManipulate to the transforms.

        Use `has_transform_list` to check if the transforms are a list.
        
        Raises
        ------
        ValueError
            If the transforms are a Compose object.
        """
        if self.has_transform_list(): 
            if not self.has_n2v_manipulate():
                self.transforms.append(
                    N2VManipulationModel(name=SupportedTransform.N2V_MANIPULATE.value)
                )    
        else:
            raise ValueError(
                "Adding N2VManipulate with Compose transforms is not allowed. Add "
                "N2VManipulate directly to the transform in the Compose."
            )
        
    def remove_n2v_manipulate(self) -> None:
        """
        Remove N2VManipulate from the transforms.

        Use `has_transform_list` to check if the transforms are a list.
        
        Raises
        ------
        ValueError
            If the transforms are a Compose object.
        """
        if self.has_transform_list() and self.has_n2v_manipulate():
            self.transforms.pop(-1)
        else:
            raise ValueError(
                "Removing N2VManipulate with Compose transforms is not allowed. Remove "
                "N2VManipulate directly from the transform in the Compose."
            )

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
        if self.has_transform_list():
            for transform in self.transforms:
                if transform.name == SupportedTransform.NORMALIZE.value:
                    transform.parameters.mean = mean
                    transform.parameters.std = std
        else:
            raise ValueError(
                "Setting mean and std with Compose transforms is not allowed. Add "
                "mean and std parameters directly to the transform in the Compose."
            )

    def set_3D(self, axes: str, patch_size: List[int]) -> None:
        """
        Set 3D parameters.

        Parameters
        ----------
        axes : str
            Axes.
        patch_size : List[int]
            Patch size.
        """
        self._update(axes=axes, patch_size=patch_size)


    def set_N2V2(self, use_n2v2: bool) -> None:
        """Set N2V2.

        Parameters
        ----------
        use_n2v2 : bool
            Whether to use N2V2.

        Raises
        ------
        ValueError
            If the N2V pixel manipulate transform is not found in the transforms.
        ValueError
            If the transforms are a Compose object.
        """
        if use_n2v2:
            self.set_N2V2_strategy("median")
        else:
            self.set_N2V2_strategy("uniform")


    def set_N2V2_strategy(self, strategy: Literal["uniform", "median"]) -> None:
        """Set N2V2 strategy.

        Parameters
        ----------
        strategy : Literal["uniform", "median"]
            Strategy to use for N2V2.

        Raises
        ------
        ValueError
            If the N2V pixel manipulate transform is not found in the transforms.
        ValueError
            If the transforms are a Compose object.
        """
        if isinstance(self.transforms, list):
            found_n2v = False

            for transform in self.transforms:
                if transform.name == SupportedTransform.N2V_MANIPULATE.value:
                    transform.parameters.strategy = strategy
                    found_n2v = True

            if not found_n2v:
                transforms = [t.name for t in self.transforms]
                raise ValueError(
                    f"N2V_Manipulate transform not found in the transforms list "
                    f"({transforms})."
                )

        else:
            raise ValueError(
                "Setting N2V2 strategy with Compose transforms is not allowed. Add "
                "N2V2 strategy parameters directly to the transform in the Compose."
            )

    def set_structN2V_mask(
        self, mask_axis: Literal["horizontal", "vertical", "none"], mask_span: int
    ) -> None:
        """Set structN2V mask parameters.

        Setting `mask_axis` to `none` will disable structN2V.

        Parameters
        ----------
        mask_axis : Literal["horizontal", "vertical", "none"]
            Axis along which to apply the mask. `none` will disable structN2V.
        mask_span : int
            Total span of the mask in pixels.

        Raises
        ------
        ValueError
            If the N2V pixel manipulate transform is not found in the transforms.
        ValueError
            If the transforms are a Compose object.
        """
        if isinstance(self.transforms, list):
            found_n2v = False

            for transform in self.transforms:
                if transform.name == SupportedTransform.N2V_MANIPULATE.value:
                    transform.parameters.struct_mask_axis = mask_axis
                    transform.parameters.struct_mask_span = mask_span
                    found_n2v = True

            if not found_n2v:
                transforms = [t.name for t in self.transforms]
                raise ValueError(
                    f"N2V pixel manipulate transform not found in the transforms "
                    f"({transforms})."
                )

        else:
            raise ValueError(
                "Setting structN2VMask with Compose transforms is not allowed. Add "
                "structN2VMask parameters directly to the transform in the Compose."
            )
