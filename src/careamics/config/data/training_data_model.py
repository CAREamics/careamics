"""Data configuration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Union
from warnings import warn

from pydantic import (
    Field,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from ..transformations import XYFlipModel, XYRandomRotate90Model
from ..validators import patch_size_ge_than_8_power_of_2
from .general_data_model import DataConfig


class TrainingDataConfig(DataConfig):
    """Training data configuration.

    If std is specified, mean must be specified as well. Note that setting the std first
    and then the mean (if they were both `None` before) will raise a validation error.
    Prefer instead `set_mean_and_std` to set both at once. Means and stds are expected
    to be lists of floats, one for each channel. For supervised tasks, the mean and std
    of the target could be different from the input data.

    All supported transforms are defined in the SupportedTransform enum.

    Examples
    --------
    Minimum example:

    >>> data = DataConfig(
    ...     data_type="array", # defined in SupportedData
    ...     patch_size=[128, 128],
    ...     batch_size=4,
    ...     axes="YX"
    ... )

    To change the image_means and image_stds of the data:
    >>> data.set_means_and_stds(image_means=[214.3], image_stds=[84.5])

    One can pass also a list of transformations, by keyword, using the
    SupportedTransform value:
    >>> from careamics.config.support import SupportedTransform
    >>> data = DataConfig(
    ...     data_type="tiff",
    ...     patch_size=[128, 128],
    ...     batch_size=4,
    ...     axes="YX",
    ...     transforms=[
    ...         {
    ...             "name": "XYFlip",
    ...         }
    ...     ]
    ... )
    """

    patch_size: Union[list[int]] = Field(..., min_length=2, max_length=3)
    """Patch size used during training."""

    transforms: Sequence[Union[XYFlipModel, XYRandomRotate90Model]] = Field(
        default=[
            XYFlipModel(),
            XYRandomRotate90Model(),
        ],
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

    @field_validator("patch_size")
    @classmethod
    def all_elements_power_of_2_minimum_8(
        cls, patch_list: Union[list[int]]
    ) -> Union[list[int]]:
        """
        Validate patch size.

        Patch size must be powers of 2 and minimum 8.

        Parameters
        ----------
        patch_list : list of int
            Patch size.

        Returns
        -------
        list of int
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
            If the patch size does not match axes.
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
