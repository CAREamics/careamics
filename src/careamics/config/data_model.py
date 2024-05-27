"""Data configuration."""

from __future__ import annotations

from pprint import pformat
from typing import Any, List, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    field_validator,
    model_validator,
)
from typing_extensions import Annotated, Self

from .support import SupportedTransform
from .transformations.n2v_manipulate_model import N2VManipulateModel
from .transformations.nd_flip_model import NDFlipModel
from .transformations.normalize_model import NormalizeModel
from .transformations.xy_random_rotate90_model import XYRandomRotate90Model
from .validators import check_axes_validity, patch_size_ge_than_8_power_of_2

TRANSFORMS_UNION = Annotated[
    Union[
        NDFlipModel,
        XYRandomRotate90Model,
        NormalizeModel,
        N2VManipulateModel,
    ],
    Discriminator("name"),  # used to tell the different transform models apart
]


class DataConfig(BaseModel):
    """
    Data configuration.

    If std is specified, mean must be specified as well. Note that setting the std first
    and then the mean (if they were both `None` before) will raise a validation error.
    Prefer instead `set_mean_and_std` to set both at once.

    Examples
    --------
    Minimum example:

    >>> data = DataConfig(
    ...     data_type="array", # defined in SupportedData
    ...     patch_size=[128, 128],
    ...     batch_size=4,
    ...     axes="YX"
    ... )

    To change the mean and std of the data:
    >>> data.set_mean_and_std(mean=214.3, std=84.5)

    One can pass also a list of transformations, by keyword, using the
    SupportedTransform or the name of an Albumentation transform:
    >>> from careamics.config.support import SupportedTransform
    >>> data = DataConfig(
    ...     data_type="tiff",
    ...     patch_size=[128, 128],
    ...     batch_size=4,
    ...     axes="YX",
    ...     transforms=[
    ...         {
    ...             "name": SupportedTransform.NORMALIZE.value,
    ...             "mean": 167.6,
    ...             "std": 47.2,
    ...         },
    ...         {
    ...             "name": "NDFlip",
    ...         }
    ...     ]
    ... )
    """

    # Pydantic class configuration
    model_config = ConfigDict(
        validate_assignment=True,
    )

    # Dataset configuration
    data_type: Literal["array", "tiff", "custom"]  # As defined in SupportedData
    patch_size: Union[List[int]] = Field(..., min_length=2, max_length=3)
    batch_size: int = Field(default=1, ge=1, validate_default=True)
    axes: str

    # Optional fields
    mean: Optional[float] = None
    std: Optional[float] = None

    transforms: List[TRANSFORMS_UNION] = Field(
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

    dataloader_params: Optional[dict] = None

    @field_validator("patch_size")
    @classmethod
    def all_elements_power_of_2_minimum_8(
        cls, patch_list: Union[List[int]]
    ) -> Union[List[int]]:
        """
        Validate patch size.

        Patch size must be powers of 2 and minimum 8.

        Parameters
        ----------
        patch_list : Union[List[int]]
            Patch size.

        Returns
        -------
        Union[List[int]]
            Validated patch size.

        Raises
        ------
        ValueError
            If the patch size is smaller than 8.
        ValueError
            If the patch size is not a power of 2.
        """
        patch_size_ge_than_8_power_of_2(patch_list)

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
        cls, transforms: List[TRANSFORMS_UNION]
    ) -> List[TRANSFORMS_UNION]:
        """
        Validate N2VManipulate transform position in the transform list.

        Parameters
        ----------
        transforms : List[Transformations_Union]
            Transforms.

        Returns
        -------
        List[TRANSFORMS_UNION]
            Validated transforms.

        Raises
        ------
        ValueError
            If multiple instances of N2VManipulate are found.
        """
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
    def std_only_with_mean(self: Self) -> Self:
        """
        Check that mean and std are either both None, or both specified.

        Returns
        -------
        Self
            Validated data model.

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

    @model_validator(mode="after")
    def add_std_and_mean_to_normalize(self: Self) -> Self:
        """
        Add mean and std to the Normalize transform if it is present.

        Returns
        -------
        Self
            Data model with mean and std added to the Normalize transform.
        """
        if self.mean is not None or self.std is not None:
            # search in the transforms for Normalize and update parameters
            for transform in self.transforms:
                if transform.name == SupportedTransform.NORMALIZE.value:
                    transform.mean = self.mean
                    transform.std = self.std

        return self

    @model_validator(mode="after")
    def validate_dimensions(self: Self) -> Self:
        """
        Validate 2D/3D dimensions between axes, patch size and transforms.

        Returns
        -------
        Self
            Validated data model.

        Raises
        ------
        ValueError
            If the transforms are not valid.
        """
        if "Z" in self.axes:
            if len(self.patch_size) != 3:
                raise ValueError(
                    f"Patch size must have 3 dimensions if the data is 3D "
                    f"({self.axes})."
                )

        else:
            if len(self.patch_size) != 2:
                raise ValueError(
                    f"Patch size must have 3 dimensions if the data is 3D "
                    f"({self.axes})."
                )

        return self

    def __str__(self) -> str:
        """
        Pretty string reprensenting the configuration.

        Returns
        -------
        str
            Pretty string.
        """
        return pformat(self.model_dump())

    def _update(self, **kwargs: Any) -> None:
        """
        Update multiple arguments at once.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments to update.
        """
        self.__dict__.update(kwargs)
        self.__class__.model_validate(self.__dict__)

    def has_n2v_manipulate(self) -> bool:
        """
        Check if the transforms contain N2VManipulate.

        Returns
        -------
        bool
            True if the transforms contain N2VManipulate, False otherwise.
        """
        return any(
            transform.name == SupportedTransform.N2V_MANIPULATE.value
            for transform in self.transforms
        )

    def add_n2v_manipulate(self) -> None:
        """Add N2VManipulate to the transforms."""
        if not self.has_n2v_manipulate():
            self.transforms.append(
                N2VManipulateModel(name=SupportedTransform.N2V_MANIPULATE.value)
            )

    def remove_n2v_manipulate(self) -> None:
        """Remove N2VManipulate from the transforms."""
        if self.has_n2v_manipulate():
            self.transforms.pop(-1)

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
        for transform in self.transforms:
            if transform.name == SupportedTransform.NORMALIZE.value:
                transform.mean = mean
                transform.std = std

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
        """
        Set N2V2.

        Parameters
        ----------
        use_n2v2 : bool
            Whether to use N2V2.

        Raises
        ------
        ValueError
            If the N2V pixel manipulate transform is not found in the transforms.
        """
        if use_n2v2:
            self.set_N2V2_strategy("median")
        else:
            self.set_N2V2_strategy("uniform")

    def set_N2V2_strategy(self, strategy: Literal["uniform", "median"]) -> None:
        """
        Set N2V2 strategy.

        Parameters
        ----------
        strategy : Literal["uniform", "median"]
            Strategy to use for N2V2.

        Raises
        ------
        ValueError
            If the N2V pixel manipulate transform is not found in the transforms.
        """
        found_n2v = False

        for transform in self.transforms:
            if transform.name == SupportedTransform.N2V_MANIPULATE.value:
                transform.strategy = strategy
                found_n2v = True

        if not found_n2v:
            transforms = [t.name for t in self.transforms]
            raise ValueError(
                f"N2V_Manipulate transform not found in the transforms list "
                f"({transforms})."
            )

    def set_structN2V_mask(
        self, mask_axis: Literal["horizontal", "vertical", "none"], mask_span: int
    ) -> None:
        """
        Set structN2V mask parameters.

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
        """
        found_n2v = False

        for transform in self.transforms:
            if transform.name == SupportedTransform.N2V_MANIPULATE.value:
                transform.struct_mask_axis = mask_axis
                transform.struct_mask_span = mask_span
                found_n2v = True

        if not found_n2v:
            transforms = [t.name for t in self.transforms]
            raise ValueError(
                f"N2V pixel manipulate transform not found in the transforms "
                f"({transforms})."
            )
