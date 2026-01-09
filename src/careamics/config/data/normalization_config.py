"""Pydantic models for normalization strategies."""

from typing import Annotated, Literal, Self, Union

from pydantic import BaseModel, ConfigDict, Discriminator, Field, model_validator


class MeanStdConfig(BaseModel):
    """
    Mean and standard deviation normalization configuration.

    Holds mean and standard deviation statistics for input and target, used to
    normalize data. If not provided, statistics can be computed later.

    Attributes
    ----------
    name : Literal["mean_std"]
        Identifier for the mean-std normalization scheme.
    input_means : list[float] | None
        Means of input channels/features, or None for automatic computation.
    input_stds : list[float] | None
        Standard deviations of input channels/features,
        or None for automatic computation.
    target_means : list[float] | None
        Means of target channels/features, or None for automatic computation.
    target_stds : list[float] | None
        Standard deviations of target channels/features,
        or None for automatic computation.
    """

    model_config = ConfigDict(validate_assignment=True)

    name: Literal["mean_std"] = "mean_std"
    input_means: list[float] | None = Field(default=None, min_length=0, max_length=32)
    input_stds: list[float] | None = Field(default=None, min_length=0, max_length=32)
    target_means: list[float] | None = Field(default=None, min_length=0, max_length=32)
    target_stds: list[float] | None = Field(default=None, min_length=0, max_length=32)

    @model_validator(mode="after")
    def validate_means_stds(self: Self) -> Self:
        """Validate that means and stds are provided in pairs or set to None.

        Returns
        -------
        Self
            The validated model instance.

        Raises
        ------
        ValueError
            If only one of means or stds is provided for input or target,
            or if each pair has mismatched lengths.
        """
        if (self.input_means is None) != (self.input_stds is None):
            raise ValueError(
                "input_means and input_stds must be both provided or both None."
            )
        if self.input_means is not None and self.input_stds is not None:
            if len(self.input_means) != len(self.input_stds):
                raise ValueError("input_means and input_stds must have same length.")

        if (self.target_means is None) != (self.target_stds is None):
            raise ValueError(
                "target_means and target_stds must be both provided or both None."
            )
        if self.target_means is not None and self.target_stds is not None:
            if len(self.target_means) != len(self.target_stds):
                raise ValueError("target_means and target_stds must have same length.")

        return self

    def needs_computation(self) -> bool:
        """
        Check if statistics need to be computed.

        Returns
        -------
        bool
            True if input statistics are missing, False otherwise.
        """
        return self.input_means is None or self.input_stds is None

    def set_input_stats(self, means: list[float], stds: list[float]) -> None:
        """
        Set input means and stds together to avoid validation errors.

        Parameters
        ----------
        means : list[float]
            Mean values per channel.
        stds : list[float]
            Standard deviation values per channel.
        """
        self.__dict__["input_means"] = means
        self.__dict__["input_stds"] = stds
        self.__class__.model_validate(self)

    def set_target_stats(self, means: list[float], stds: list[float]) -> None:
        """
        Set target means and stds together to avoid validation errors.

        Parameters
        ----------
        means : list[float]
            Mean values per channel.
        stds : list[float]
            Standard deviation values per channel.
        """
        self.__dict__["target_means"] = means
        self.__dict__["target_stds"] = stds
        self.__class__.model_validate(self)


class QuantileConfig(BaseModel):
    """
    Quantile normalization configuration.

    Normalizes data using quantile-based range scaling. Quantile levels can be
    specified as a single value (applied to all channels) or a list (one per channel).

    Attributes
    ----------
    name : Literal["quantile"]
        Identifier for quantile normalization.
    lower_quantile : float | list[float]
        Lower quantile level(s). Single float applies to all channels,
        or list for per-channel levels. Values must be in [0, 1).
    upper_quantile : float | list[float]
        Upper quantile level(s). Single float applies to all channels,
        or list for per-channel levels. Values must be in (0, 1].
    input_lower_quantile_values : list[float] | None
        Computed lower quantile values for each input channel.
    input_upper_quantile_values : list[float] | None
        Computed upper quantile values for each input channel.
    target_lower_quantile_values : list[float] | None
        Computed lower quantile values for each target channel.
    target_upper_quantile_values : list[float] | None
        Computed upper quantile values for each target channel.
    """

    model_config = ConfigDict(validate_assignment=True)

    name: Literal["quantile"] = "quantile"

    lower_quantile: float | list[float] = Field(default=0.01)
    upper_quantile: float | list[float] = Field(default=0.99)

    input_lower_quantile_values: list[float] | None = Field(default=None)
    input_upper_quantile_values: list[float] | None = Field(default=None)
    target_lower_quantile_values: list[float] | None = Field(default=None)
    target_upper_quantile_values: list[float] | None = Field(default=None)

    @model_validator(mode="after")
    def validate_quantile_levels(self: Self) -> Self:
        """Validate quantile levels are in valid range and properly ordered.

        Returns
        -------
        Self
            The validated model instance.
        """
        lower = (
            self.lower_quantile
            if isinstance(self.lower_quantile, list)
            else [self.lower_quantile]
        )
        upper = (
            self.upper_quantile
            if isinstance(self.upper_quantile, list)
            else [self.upper_quantile]
        )

        for lq in lower:
            if not (0.0 <= lq < 1.0):
                raise ValueError(f"lower_quantile values must be in [0, 1), got {lq}")
        for uq in upper:
            if not (0.0 < uq <= 1.0):
                raise ValueError(f"upper_quantile values must be in (0, 1], got {uq}")

        if len(lower) != len(upper):
            raise ValueError(
                f"lower_quantile and upper_quantile lists must have same length, "
                f"got {len(lower)} and {len(upper)}"
            )
        for i, (lq, uq) in enumerate(zip(lower, upper, strict=True)):
            if lq >= uq:
                raise ValueError(
                    f"lower_quantile[{i}] ({lq}) must be less than "
                    f"upper_quantile[{i}] ({uq})"
                )
        return self

    @model_validator(mode="after")
    def validate_quantile_values(self: Self) -> Self:
        """Validate that computed quantile value lists are provided in pairs.

        Returns
        -------
        Self
            The validated model instance.
        """
        if (self.input_lower_quantile_values is None) != (
            self.input_upper_quantile_values is None
        ):
            raise ValueError(
                "input_lower_quantile_values and input_upper_quantile_values "
                "must be both provided or both None."
            )
        if (
            self.input_lower_quantile_values is not None
            and self.input_upper_quantile_values is not None
        ):
            if len(self.input_lower_quantile_values) != len(
                self.input_upper_quantile_values
            ):
                raise ValueError(
                    "input_lower_quantile_values and input_upper_quantile_values "
                    "must have same length."
                )
            for i, (lower, upper) in enumerate(
                zip(
                    self.input_lower_quantile_values,
                    self.input_upper_quantile_values,
                    strict=True,
                )
            ):
                if lower >= upper:
                    raise ValueError(
                        f"input_lower_quantile_values[{i}] ({lower}) must be less than "
                        f"input_upper_quantile_values[{i}] ({upper})"
                    )

        if (self.target_lower_quantile_values is None) != (
            self.target_upper_quantile_values is None
        ):
            raise ValueError(
                "target_lower_quantile_values and target_upper_quantile_values "
                "must be both provided or both None."
            )
        if (
            self.target_lower_quantile_values is not None
            and self.target_upper_quantile_values is not None
        ):
            if len(self.target_lower_quantile_values) != len(
                self.target_upper_quantile_values
            ):
                raise ValueError(
                    "target_lower_quantile_values and target_upper_quantile_values "
                    "must have same length."
                )
            for i, (lower, upper) in enumerate(
                zip(
                    self.target_lower_quantile_values,
                    self.target_upper_quantile_values,
                    strict=True,
                )
            ):
                if lower >= upper:
                    raise ValueError(
                        f"target_lower_quantile_values[{i}] ({lower}) must be less "
                        f"than target_upper_quantile_values[{i}] ({upper})"
                    )
        return self

    def get_lower_quantiles_for_channels(self, n_channels: int) -> list[float]:
        """Get lower quantile levels expanded to n_channels.

        Parameters
        ----------
        n_channels : int
            Number of channels in the data.

        Returns
        -------
        list[float]
            Lower quantile levels for each channel.
        """
        if isinstance(self.lower_quantile, list):
            if len(self.lower_quantile) != n_channels:
                raise ValueError(
                    f"lower_quantile has {len(self.lower_quantile)} values but "
                    f"data has {n_channels} channels"
                )
            return self.lower_quantile
        return [self.lower_quantile] * n_channels

    def get_upper_quantiles_for_channels(self, n_channels: int) -> list[float]:
        """Get upper quantile levels expanded to n_channels.

        Parameters
        ----------
        n_channels : int
            Number of channels in the data.

        Returns
        -------
        list[float]
            Upper quantile levels for each channel.
        """
        if isinstance(self.upper_quantile, list):
            if len(self.upper_quantile) != n_channels:
                raise ValueError(
                    f"upper_quantile has {len(self.upper_quantile)} values but "
                    f"data has {n_channels} channels"
                )
            return self.upper_quantile
        return [self.upper_quantile] * n_channels

    def needs_computation(self) -> bool:
        """Check if quantile values need to be computed.

        Returns
        -------
        bool
            True if quantile values need to be computed.
        """
        return (
            self.input_lower_quantile_values is None
            or self.input_upper_quantile_values is None
        )

    def set_input_quantile_values(self, lower: list[float], upper: list[float]) -> None:
        """
        Set input quantile values together to avoid validation errors.

        Parameters
        ----------
        lower : list[float]
            Lower quantile values per channel.
        upper : list[float]
            Upper quantile values per channel.
        """
        self.__dict__["input_lower_quantile_values"] = lower
        self.__dict__["input_upper_quantile_values"] = upper
        self.__class__.model_validate(self)

    def set_target_quantile_values(
        self, lower: list[float], upper: list[float]
    ) -> None:
        """
        Set target quantile values together to avoid validation errors.

        Parameters
        ----------
        lower : list[float]
            Lower quantile values per channel.
        upper : list[float]
            Upper quantile values per channel.
        """
        self.__dict__["target_lower_quantile_values"] = lower
        self.__dict__["target_upper_quantile_values"] = upper
        self.__class__.model_validate(self)


class MinMaxConfig(BaseModel):
    """
    Min-max normalization configuration.

    Stores minimum and maximum statistics for scaling data into a desired range.
    If not provided, statistics can be computed from the data.

    Attributes
    ----------
    name : Literal["minmax"]
        Identifier for min-max normalization.
    input_mins : list[float] | None
        Minimum values for input channels/features.
    input_maxes : list[float] | None
        Maximum values for input channels/features.
    target_mins : list[float] | None
        Minimum values for target channels/features.
    target_maxes : list[float] | None
        Maximum values for target channels/features.
    """

    model_config = ConfigDict(validate_assignment=True)

    name: Literal["minmax"] = "minmax"
    input_mins: list[float] | None = Field(default=None)
    input_maxes: list[float] | None = Field(default=None)
    target_mins: list[float] | None = Field(default=None)
    target_maxes: list[float] | None = Field(default=None)

    @model_validator(mode="after")
    def validate_mins_maxes(self: Self) -> Self:
        """Validate that mins and maxes are provided in pairs or both None.

        Returns
        -------
        Self
            The validated model instance.
        """
        if (self.input_mins is None) != (self.input_maxes is None):
            raise ValueError(
                "input_mins and input_maxes must be both provided or both None."
            )
        if self.input_mins is not None and self.input_maxes is not None:
            if len(self.input_mins) != len(self.input_maxes):
                raise ValueError("input_mins and input_maxes must have same length.")

        if (self.target_mins is None) != (self.target_maxes is None):
            raise ValueError(
                "target_mins and target_maxes must be both provided or both None."
            )
        if self.target_mins is not None and self.target_maxes is not None:
            if len(self.target_mins) != len(self.target_maxes):
                raise ValueError("target_mins and target_maxes must have same length.")

        return self

    def needs_computation(self) -> bool:
        """
        Check if min/max values need to be computed.

        Returns
        -------
        bool
            True if input statistics are missing, False otherwise.
        """
        return self.input_mins is None or self.input_maxes is None

    def set_input_range(self, mins: list[float], maxes: list[float]) -> None:
        """
        Set input mins and maxes together to avoid validation errors.

        Parameters
        ----------
        mins : list[float]
            Minimum values per channel.
        maxes : list[float]
            Maximum values per channel.
        """
        self.__dict__["input_mins"] = mins
        self.__dict__["input_maxes"] = maxes
        self.__class__.model_validate(self)

    def set_target_range(self, mins: list[float], maxes: list[float]) -> None:
        """
        Set target mins and maxes together to avoid validation errors.

        Parameters
        ----------
        mins : list[float]
            Minimum values per channel.
        maxes : list[float]
            Maximum values per channel.
        """
        self.__dict__["target_mins"] = mins
        self.__dict__["target_maxes"] = maxes
        self.__class__.model_validate(self)


class NoNormConfig(BaseModel):
    """
    No normalization configuration.

    Indicates that no normalization should be applied.

    Attributes
    ----------
    name : Literal["none"]
        Identifier for no normalization scheme.
    """

    name: Literal["none"] = "none"

    def needs_computation(self) -> bool:
        """Check if statistics need to be computed.

        Returns
        -------
        bool
            Always False, as no statistics are required.
        """
        return False


NormalizationConfig = Annotated[
    Union[MeanStdConfig, NoNormConfig, QuantileConfig, MinMaxConfig],
    Discriminator("name"),
]
