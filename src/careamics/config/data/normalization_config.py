"""Pydantic models for normalization strategies."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StandardizeModel(BaseModel):
    """
    Standardization normalization configuration.

    Standardizes each channel to zero mean and unit variance using the provided
    statistics. Statistics can be given for both input and, optionally, target arrays.

    Attributes
    ----------
    name : Literal["standard"]
        Name of the normalization strategy ("standard").
    input_means : list
        Per-channel mean values for the input array.
    input_stds : list
        Per-channel standard deviation values for the input array.
    target_means : list | None
        Per-channel mean values for the target array, if provided.
    target_stds : list | None
        Per-channel standard deviation values for the target array, if provided.
    """

    model_config = ConfigDict(validate_assignment=True)

    name: Literal["standardize"] = "standardize"
    input_means: list = Field(..., min_length=0, max_length=32)
    input_stds: list = Field(..., min_length=0, max_length=32)
    target_means: list | None = Field(default=None, min_length=0, max_length=32)
    target_stds: list | None = Field(default=None, min_length=0, max_length=32)

    @model_validator(mode="after")
    def validate_means_stds(self: Self) -> Self:
        """
        Validate that input and target means and stds have the same length,
        and are consistently specified.

        Returns
        -------
        Self
            The validated StandardizeModel instance.

        Raises
        ------
        ValueError
            If provided mean/std lists do not have equal length, or target stats are
            inconsistently provided.
        """
        if len(self.input_means) != len(self.input_stds):
            raise ValueError("The number of input means and stds must be the same.")

        if (self.target_means is None) != (self.target_stds is None):
            raise ValueError(
                "Both target means and stds must be provided together, or both None."
            )

        if self.target_means is not None and self.target_stds is not None:
            if len(self.target_means) != len(self.target_stds):
                raise ValueError(
                    "The number of target means and stds must be the same."
                )

        return self


class NoNormModel(BaseModel):
    """
    No normalization configuration.

    Indicates that no normalization should be performed.

    Attributes
    ----------
    name : Literal["none"]
        Name of the normalization strategy ("none").
    """

    name: Literal["none"] = "none"


class QuantileModel(BaseModel):
    """
    Quantile normalization configuration.

    Normalizes each channel to a specified quantile range using the provided quantile values.
    Statistics can be given for both input and, optionally, target arrays.

    Attributes
    ----------
    name : Literal["quantile"]
        Name of the normalization strategy ("quantile").
    input_lower_quantiles : list
        Lower quantile for each input channel.
    input_upper_quantiles : list
        Upper quantile for each input channel.
    target_lower_quantiles : list | None
        Lower quantile for each target channel, if provided.
    target_upper_quantiles : list | None
        Upper quantile for each target channel, if provided.
    """

    name: Literal["quantile"] = "quantile"
    input_lower_quantiles: list = Field(..., min_length=0, max_length=32)
    input_upper_quantiles: list = Field(..., min_length=0, max_length=32)
    target_lower_quantiles: list | None = Field(
        default=None, min_length=0, max_length=32
    )
    target_upper_quantiles: list | None = Field(
        default=None, min_length=0, max_length=32
    )

    @model_validator(mode="after")
    def validate_quantiles(self: Self) -> Self:
        """
        Validate that the lower and upper quantile lists have the same length,
        and are consistently specified for input and target.

        Returns
        -------
        Self
            The validated QuantileModel instance.

        Raises
        ------
        ValueError
            If provided quantile lists do not have equal length, or target quantiles
            are inconsistently provided.
        """
        if len(self.input_lower_quantiles) != len(self.input_upper_quantiles):
            raise ValueError(
                "The number of lower and upper quantiles must be the same."
            )
        if (self.target_lower_quantiles is None) != (
            self.target_upper_quantiles is None
        ):
            raise ValueError(
                "Both target lower and upper quantiles must be provided together, or both None."
            )
        if (
            self.target_lower_quantiles is not None
            and self.target_upper_quantiles is not None
        ):
            if len(self.target_lower_quantiles) != len(self.target_upper_quantiles):
                raise ValueError(
                    "The number of target lower and upper quantiles must be the same."
                )
        return self


class MinMaxModel(BaseModel):
    """
    Min-max normalization configuration.

    Scales each channel into the [min, max] range given. Statistics can be given for
    both input and, optionally, target arrays.

    Attributes
    ----------
    name : Literal["minmax"]
        Name of the normalization strategy ("minmax").
    input_mins : list
        Minimum value for each input channel.
    input_maxes : list
        Maximum value for each input channel.
    target_mins : list | None
        Minimum value for each target channel, if provided.
    target_maxes : list | None
        Maximum value for each target channel, if provided.
    """

    name: Literal["minmax"] = "minmax"
    input_mins: list = Field(..., min_length=0, max_length=32)
    input_maxes: list = Field(..., min_length=0, max_length=32)
    target_mins: list | None = Field(default=None, min_length=0, max_length=32)
    target_maxes: list | None = Field(default=None, min_length=0, max_length=32)

    @model_validator(mode="after")
    def validate_mins_maxs(self: Self) -> Self:
        """
        Validate that the min and max lists have the same length,
        and are consistently specified for input and target.

        Returns
        -------
        Self
            The validated MinMaxModel instance.

        Raises
        ------
        ValueError
            If provided min/max lists do not have equal length, or target mins/maxes
            are inconsistently provided.
        """
        if len(self.input_mins) != len(self.input_maxes):
            raise ValueError("The number of input mins and maxes must be the same.")
        if (self.target_mins is None) != (self.target_maxes is None):
            raise ValueError(
                "Both target mins and maxes must be provided together, or both None."
            )
        if self.target_mins is not None and self.target_maxes is not None:
            if len(self.target_mins) != len(self.target_maxes):
                raise ValueError(
                    "The number of target mins and maxes must be the same."
                )
        return self
