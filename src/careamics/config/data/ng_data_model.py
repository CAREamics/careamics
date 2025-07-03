"""Data configuration."""

from __future__ import annotations

from collections.abc import Sequence
from pprint import pformat
from typing import Annotated, Any, Literal, Optional, Union
from warnings import warn

import numpy as np
from numpy.typing import NDArray
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from ..transformations import XYFlipModel, XYRandomRotate90Model
from ..validators import check_axes_validity
from .patching_strategies import (
    RandomPatchingModel,
    TiledPatchingModel,
    WholePatchingModel,
)

# TODO: Validate the specific sizes of tiles and overlaps given UNet constraints
#   - needs to be done in the Configuration
#   - patches and overlaps sizes must also be checked against dimensionality

# TODO: is 3D updated anywhere in the code in CAREamist/downstream?
#       - this will be important when swapping the data config in Configuration
#       - `set_3D` currently not implemented here
# TODO: we can't tell that the patching strategy is correct
#       - or is the responsibility of the creator (e.g. conveneince functions)


def np_float_to_scientific_str(x: float) -> str:
    """Return a string scientific representation of a float.

    In particular, this method is used to serialize floats to strings, allowing
    numpy.float32 to be passed in the Pydantic model and written to a yaml file as str.

    Parameters
    ----------
    x : float
        Input value.

    Returns
    -------
    str
        Scientific string representation of the input value.
    """
    return np.format_float_scientific(x, precision=7)


Float = Annotated[float, PlainSerializer(np_float_to_scientific_str, return_type=str)]
"""Annotated float type, used to serialize floats to strings."""

PatchingStrategies = Union[
    RandomPatchingModel,
    # SequentialPatchingModel, # not supported yet
    TiledPatchingModel,
    WholePatchingModel,
]
"""Patching strategies."""


class NGDataConfig(BaseModel):
    """Next-Generation Dataset configuration.

    NGDataConfig are used for both training and prediction, with the patching strategy
    determining how the data is processed. Note that `random` is the only patching
    strategy compatible with training, while `tiled` and `whole` are only used for
    prediction.

    If std is specified, mean must be specified as well. Note that setting the std first
    and then the mean (if they were both `None` before) will raise a validation error.
    Prefer instead `set_means_and_stds` to set both at once. Means and stds are expected
    to be lists of floats, one for each channel. For supervised tasks, the mean and std
    of the target could be different from the input data.

    All supported transforms are defined in the SupportedTransform enum.
    """

    # Pydantic class configuration
    model_config = ConfigDict(
        validate_assignment=True,
    )

    # Dataset configuration
    data_type: Literal["array", "tiff", "zarr", "custom"]
    """Type of input data."""

    axes: str
    """Axes of the data, as defined in SupportedAxes."""

    patching: PatchingStrategies = Field(..., discriminator="name")
    """Patching strategy to use. Note that `random` is the only supported strategy for
    training, while `tiled` and `whole` are only used for prediction."""

    # Optional fields
    batch_size: int = Field(default=1, ge=1, validate_default=True)
    """Batch size for training."""

    image_means: Optional[list[Float]] = Field(
        default=None, min_length=0, max_length=32
    )
    """Means of the data across channels, used for normalization."""

    image_stds: Optional[list[Float]] = Field(default=None, min_length=0, max_length=32)
    """Standard deviations of the data across channels, used for normalization."""

    target_means: Optional[list[Float]] = Field(
        default=None, min_length=0, max_length=32
    )
    """Means of the target data across channels, used for normalization."""

    target_stds: Optional[list[Float]] = Field(
        default=None, min_length=0, max_length=32
    )
    """Standard deviations of the target data across channels, used for
    normalization."""

    transforms: Sequence[Union[XYFlipModel, XYRandomRotate90Model]] = Field(
        default=(
            XYFlipModel(),
            XYRandomRotate90Model(),
        ),
        validate_default=True,
    )
    """List of transformations to apply to the data, available transforms are defined
    in SupportedTransform."""

    train_dataloader_params: dict[str, Any] = Field(
        default={"shuffle": True}, validate_default=True
    )
    """Dictionary of PyTorch training dataloader parameters. The dataloader parameters,
    should include the `shuffle` key, which is set to `True` by default. We strongly
    recommend to keep it as `True` to ensure the best training results."""

    val_dataloader_params: dict[str, Any] = Field(default={})
    """Dictionary of PyTorch validation dataloader parameters."""

    test_dataloader_params: dict[str, Any] = Field(default={})
    """Dictionary of PyTorch test dataloader parameters."""

    seed: Optional[int] = Field(default=None, gt=0)
    """Random seed for reproducibility."""

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

    @field_validator("train_dataloader_params")
    @classmethod
    def shuffle_train_dataloader(
        cls, train_dataloader_params: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Validate that "shuffle" is included in the training dataloader params.

        A warning will be raised if `shuffle=False`.

        Parameters
        ----------
        train_dataloader_params : dict of {str: Any}
            The training dataloader parameters.

        Returns
        -------
        dict of {str: Any}
            The validated training dataloader parameters.

        Raises
        ------
        ValueError
            If "shuffle" is not included in the training dataloader params.
        """
        if "shuffle" not in train_dataloader_params:
            raise ValueError(
                "Value for 'shuffle' was not included in the `train_dataloader_params`."
            )
        elif ("shuffle" in train_dataloader_params) and (
            not train_dataloader_params["shuffle"]
        ):
            warn(
                "Dataloader parameters include `shuffle=False`, this will be passed to "
                "the training dataloader and may lead to lower quality results.",
                stacklevel=1,
            )
        return train_dataloader_params

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
        if (self.image_means and not self.image_stds) or (
            self.image_stds and not self.image_means
        ):
            raise ValueError(
                "Mean and std must be either both None, or both specified."
            )

        elif (self.image_means is not None and self.image_stds is not None) and (
            len(self.image_means) != len(self.image_stds)
        ):
            raise ValueError("Mean and std must be specified for each input channel.")

        if (self.target_means and not self.target_stds) or (
            self.target_stds and not self.target_means
        ):
            raise ValueError(
                "Mean and std must be either both None, or both specified "
            )

        elif self.target_means is not None and self.target_stds is not None:
            if len(self.target_means) != len(self.target_stds):
                raise ValueError(
                    "Mean and std must be either both None, or both specified for each "
                    "target channel."
                )

        return self

    @model_validator(mode="after")
    def validate_dimensions(self: Self) -> Self:
        """
        Validate 2D/3D dimensions between axes and patch size.

        Returns
        -------
        Self
            Validated data model.

        Raises
        ------
        ValueError
            If the patch size dimension is not compatible with the axes.
        """
        if "Z" in self.axes:
            if (
                hasattr(self.patching, "patch_size")
                and len(self.patching.patch_size) != 3
            ):
                raise ValueError(
                    f"`patch_size` in `patching` must have 3 dimensions if the data is"
                    f" 3D, got axes {self.axes})."
                )
        else:
            if (
                hasattr(self.patching, "patch_size")
                and len(self.patching.patch_size) != 2
            ):
                raise ValueError(
                    f"`patch_size` in `patching` must have 2 dimensions if the data is"
                    f" 3D, got axes {self.axes})."
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

    def set_means_and_stds(
        self,
        image_means: Union[NDArray, tuple, list, None],
        image_stds: Union[NDArray, tuple, list, None],
        target_means: Optional[Union[NDArray, tuple, list, None]] = None,
        target_stds: Optional[Union[NDArray, tuple, list, None]] = None,
    ) -> None:
        """
        Set mean and standard deviation of the data across channels.

        This method should be used instead setting the fields directly, as it would
        otherwise trigger a validation error.

        Parameters
        ----------
        image_means : numpy.ndarray, tuple or list
            Mean values for normalization.
        image_stds : numpy.ndarray, tuple or list
            Standard deviation values for normalization.
        target_means : numpy.ndarray, tuple or list, optional
            Target mean values for normalization, by default ().
        target_stds : numpy.ndarray, tuple or list, optional
            Target standard deviation values for normalization, by default ().
        """
        # make sure we pass a list
        if image_means is not None:
            image_means = list(image_means)
        if image_stds is not None:
            image_stds = list(image_stds)
        if target_means is not None:
            target_means = list(target_means)
        if target_stds is not None:
            target_stds = list(target_stds)

        self._update(
            image_means=image_means,
            image_stds=image_stds,
            target_means=target_means,
            target_stds=target_stds,
        )

    # def set_3D(self, axes: str, patch_size: list[int]) -> None:
    #     """
    #     Set 3D parameters.

    #     Parameters
    #     ----------
    #     axes : str
    #         Axes.
    #     patch_size : list of int
    #         Patch size.
    #     """
    #     self._update(axes=axes, patch_size=patch_size)
