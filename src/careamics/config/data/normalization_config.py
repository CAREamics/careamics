"""Pydantic models for normalization strategies."""

from typing import Annotated, Any, Literal, Self, Union

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Discriminator,
    ValidationInfo,
    field_validator,
    model_validator,
)


def _wrap_scalar(v: Any) -> Any:
    """
    Wrap scalar values as a list of floats.

    Parameters
    ----------
    v : Any
        Value to convert (scalar or list).

    Returns
    -------
    Any
        List containing the float value if input is scalar, otherwise input as-is.
    """
    if isinstance(v, (int, float)):
        return [float(v)]
    return v


FloatStats = Annotated[list[float], BeforeValidator(_wrap_scalar)]
OptionalFloatStats = Annotated[list[float] | None, BeforeValidator(_wrap_scalar)]


def _validate_global_stats_single_element(
    v: list[float] | None, info: ValidationInfo
) -> list[float] | None:
    """Validate that global statistics are provided as single-element lists.

    Parameters
    ----------
    v : list[float] | None
        The value to validate.
    info : ValidationInfo
        Information about the validation context.

    Returns
    -------
    list[float] | None
        The validated value.

    Raises
    ------
    ValueError
        If ``per_channel`` is False and the list does not have exactly one element.
    """
    if "per_channel" not in info.data:
        raise ValueError("per_channel parameter must be provided for validation.")

    per_channel = info.data["per_channel"]
    if not per_channel and v is not None and len(v) != 1:
        raise ValueError(
            f'Global statistics (`per_channel="False"`) must be provided as '
            f"single-element lists (got {v})."
        )

    return v


class MeanStdConfig(BaseModel):
    """
    Mean and standard deviation normalization configuration.

    Holds mean and standard deviation statistics for input and target, used to
    normalize data. Each statistic can be a single float (applied globally to
    all channels) or a list of floats (one per channel). If not provided,
    statistics can be computed automatically.

    Attributes
    ----------
    name : Literal["mean_std"]
        Identifier for the mean-std normalization scheme.
    input_means : float | list[float] | None
        Means for input normalization. None for automatic computation.
    input_stds : float | list[float] | None
        Standard deviations for input normalization. None for automatic
        computation.
    target_means : float | list[float] | None
        Means for target normalization. None for automatic computation.
    target_stds : float | list[float] | None
        Standard deviations for target normalization. None for automatic
        computation.
    per_channel : bool
        When True (default), statistics are computed independently for each
        channel. When False, a single statistic is computed across all channels.
    """

    model_config = ConfigDict(validate_assignment=True)

    name: Literal["mean_std"] = "mean_std"
    per_channel: bool = True
    input_means: OptionalFloatStats = None
    input_stds: OptionalFloatStats = None
    target_means: OptionalFloatStats = None
    target_stds: OptionalFloatStats = None

    @field_validator(
        "input_means", "input_stds", "target_means", "target_stds", mode="after"
    )
    @classmethod
    def validate_global_stats_single_element(
        cls,
        v: OptionalFloatStats,
        info: ValidationInfo,
    ) -> OptionalFloatStats:
        """Validate stats length against the `per_channel` parameter.

        Parameters
        ----------
        v : OptionalFloatStats
            Value to validate.
        info : ValidationInfo
            Validated values.

        Returns
        -------
        OptionalFloatStats
            Validate value.
        """
        return _validate_global_stats_single_element(v, info)

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
            or if paired lists have mismatched lengths.
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
        object.__setattr__(self, "input_means", means)
        object.__setattr__(self, "input_stds", stds)
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
        object.__setattr__(self, "target_means", means)
        object.__setattr__(self, "target_stds", stds)
        self.__class__.model_validate(self)

    def _validate_input_stats_size(self: Self, n_input_channels: int) -> None:
        """Validate that the input statistics have the correct length.

        Parameters
        ----------
        n_input_channels : int
            The number of input channels to validate against.

        Raises
        ------
        ValueError
            If the length of input_means does not match the number of input channels.
        """
        # we assume that means and stds are already validated against each other, and
        # against per-channel parameter
        if self.per_channel:
            if self.input_means is not None:
                if len(self.input_means) != n_input_channels:
                    raise ValueError(
                        f"input_means length {len(self.input_means)} does not match "
                        f"number of input channels {n_input_channels}."
                    )

    def _validate_target_stats_size(self: Self, n_target_channels: int) -> None:
        """Validate that the target statistics have the correct length.

        Parameters
        ----------
        n_target_channels : int
            The number of target channels to validate against.

        Raises
        ------
        ValueError
            If the length of target_means does not match the number of target channels.
        """
        # we assume that means and stds are already validated against each other, and
        # against per-channel parameter
        if self.per_channel:
            if self.target_means is not None:
                if len(self.target_means) != n_target_channels:
                    raise ValueError(
                        f"target_means length {len(self.target_means)} does not match "
                        f"number of target channels {n_target_channels}."
                    )

    def validate_size(
        self: Self, n_input_channels: int, n_output_channels: int
    ) -> None:
        """Validate that statistics sizes match the number of channels.

        Parameters
        ----------
        n_input_channels : int
            The number of input channels to validate against.
        n_output_channels : int
            The number of output channels to validate against.

        Raises
        ------
        ValueError
            If any provided statistics list does not match the expected size.
        """
        self._validate_input_stats_size(n_input_channels)
        self._validate_target_stats_size(n_output_channels)


class QuantileConfig(BaseModel):
    """
    Quantile normalization configuration.

    Normalizes data using quantile-based range scaling. Quantile levels can be
    specified as a single value (applied to all channels) or a list (one per
    channel). If not provided, quantile values can be computed automatically.

    Attributes
    ----------
    name : Literal["quantile"]
        Identifier for quantile normalization.
    lower_quantiles : float | list[float]
        Lower quantile level(s). Values must be in [0, 1).
    upper_quantiles : float | list[float]
        Upper quantile level(s). Values must be in (0, 1].
    input_lower_quantile_values : float | list[float] | None
        Computed lower quantile values for input.
    input_upper_quantile_values : float | list[float] | None
        Computed upper quantile values for input.
    target_lower_quantile_values : float | list[float] | None
        Computed lower quantile values for target.
    target_upper_quantile_values : float | list[float] | None
        Computed upper quantile values for target.
    per_channel : bool
        When True (default), quantile values are computed independently for
        each channel. When False, a single quantile is computed across all
        channels.
    """

    model_config = ConfigDict(validate_assignment=True)

    name: Literal["quantile"] = "quantile"
    per_channel: bool = True
    lower_quantiles: FloatStats = [0.01]
    upper_quantiles: FloatStats = [0.99]
    input_lower_quantile_values: OptionalFloatStats = None
    input_upper_quantile_values: OptionalFloatStats = None
    target_lower_quantile_values: OptionalFloatStats = None
    target_upper_quantile_values: OptionalFloatStats = None

    @field_validator(
        "lower_quantiles",
        "upper_quantiles",
        "input_lower_quantile_values",
        "input_upper_quantile_values",
        "target_lower_quantile_values",
        "target_upper_quantile_values",
        mode="after",
    )
    @classmethod
    def validate_global_stats_single_element(
        cls,
        v: OptionalFloatStats,
        info: ValidationInfo,
    ) -> OptionalFloatStats:
        """Validate stats length against the `per_channel` parameter.

        Parameters
        ----------
        v : OptionalFloatStats
            Value to validate.
        info : ValidationInfo
            Validated values.

        Returns
        -------
        OptionalFloatStats
            Validate value.
        """
        return _validate_global_stats_single_element(v, info)

    @model_validator(mode="after")
    def validate_quantile_levels(self: Self) -> Self:
        """Validate quantile levels are in valid range and properly ordered.

        Returns
        -------
        Self
            The validated model instance.
        """
        for lq in self.lower_quantiles:
            if not (0.0 <= lq < 1.0):
                raise ValueError(f"lower_quantile values must be in [0, 1), got {lq}")
        for uq in self.upper_quantiles:
            if not (0.0 < uq <= 1.0):
                raise ValueError(f"upper_quantile values must be in (0, 1], got {uq}")

        if len(self.lower_quantiles) != len(self.upper_quantiles):
            raise ValueError(
                f"lower_quantile and upper_quantile lists must have same length, "
                f"got {len(self.lower_quantiles)} and {len(self.upper_quantiles)}"
            )
        for i, (lq, uq) in enumerate(
            zip(self.lower_quantiles, self.upper_quantiles, strict=True)
        ):
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
        object.__setattr__(self, "input_lower_quantile_values", lower)
        object.__setattr__(self, "input_upper_quantile_values", upper)
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
        object.__setattr__(self, "target_lower_quantile_values", lower)
        object.__setattr__(self, "target_upper_quantile_values", upper)
        self.__class__.model_validate(self)

    def _validate_input_quantile_size(self: Self, n_input_channels: int) -> None:
        """Validate that the input quantile values have the correct length.

        Parameters
        ----------
        n_input_channels : int
            The number of input channels to validate against.

        Raises
        ------
        ValueError
            If the length of input_lower_quantile_values does not match the number of
            input channels.
        """
        # we assume that lower and upper quantile values are already validated against
        # each other, and against per-channel parameter
        if self.per_channel:
            if self.input_lower_quantile_values is not None:
                if len(self.input_lower_quantile_values) != n_input_channels:
                    raise ValueError(
                        f"input_lower_quantile_values length "
                        f"{len(self.input_lower_quantile_values)} does not match "
                        f"number of input channels {n_input_channels}."
                    )

    def _validate_target_quantile_size(self: Self, n_target_channels: int) -> None:
        """Validate that the target quantile values have the correct length.

        Parameters
        ----------
        n_target_channels : int
            The number of target channels to validate against.

        Raises
        ------
        ValueError
            If the length of target_lower_quantile_values does not match the number
            of target channels.
        """
        # we assume that if lower and upper quantile values are already validated
        # against each other, and
        # against per-channel
        if self.per_channel:
            if self.target_lower_quantile_values is not None:
                if len(self.target_lower_quantile_values) != n_target_channels:
                    raise ValueError(
                        f"target_lower_quantile_values length "
                        f"{len(self.target_lower_quantile_values)} does not match "
                        f"number of target channels {n_target_channels}."
                    )

    def _validate_quantiles_size(self: Self, n_input_channels: int) -> None:
        """Validate that the quantiles values have the correct length.

        Parameters
        ----------
        n_input_channels : int
            The number of target channels to validate against.

        Raises
        ------
        ValueError
            If the length of lower_quantiles does not match the number
            of input channels.
        """
        # we assume that if lower and upper quantiles are already validated
        # against each other, and against per-channel
        if self.per_channel:
            if len(self.lower_quantiles) != n_input_channels:
                raise ValueError(
                    f"lower_quantiles length "
                    f"{len(self.lower_quantiles)} does not match "
                    f"number of input channels {n_input_channels}."
                )

    def validate_size(
        self: Self, n_input_channels: int, n_output_channels: int
    ) -> None:
        """Validate that statistics sizes match the number of channels.

        Parameters
        ----------
        n_input_channels : int
            The number of input channels to validate against.
        n_output_channels : int
            The number of output channels to validate against.

        Raises
        ------
        ValueError
            If any provided statistics list does not match the expected size.
        """
        if self.per_channel:
            if n_input_channels != n_output_channels:
                raise ValueError(
                    f"Quantile normalization per channel is only compatible with "
                    f"matching number of input and output channels. Got "
                    f"{n_input_channels} input channels and {n_output_channels} output "
                    f"channels."
                )

        self._validate_input_quantile_size(n_input_channels)
        self._validate_input_quantile_size(n_input_channels)
        self._validate_target_quantile_size(n_output_channels)


class MinMaxConfig(BaseModel):
    """
    Min-max normalization configuration.

    Stores minimum and maximum statistics for scaling data into a desired range.
    Each statistic can be a single float (applied globally to all channels) or a
    list of floats (one per channel). If not provided, statistics can be computed
    automatically.

    Attributes
    ----------
    name : Literal["min_max"]
        Identifier for min-max normalization.
    input_mins : float | list[float] | None
        Minimum values for input normalization. None for automatic computation.
    input_maxes : float | list[float] | None
        Maximum values for input normalization. None for automatic computation.
    target_mins : float | list[float] | None
        Minimum values for target normalization. None for automatic computation.
    target_maxes : float | list[float] | None
        Maximum values for target normalization. None for automatic computation.
    per_channel : bool
        When True (default), statistics are computed independently for each
        channel. When False, a single statistic is computed across all channels.
    """

    model_config = ConfigDict(validate_assignment=True)

    name: Literal["min_max"] = "min_max"
    per_channel: bool = True
    input_mins: OptionalFloatStats = None
    input_maxes: OptionalFloatStats = None
    target_mins: OptionalFloatStats = None
    target_maxes: OptionalFloatStats = None

    @field_validator(
        "input_mins", "input_maxes", "target_mins", "target_maxes", mode="after"
    )
    @classmethod
    def validate_global_stats_single_element(
        cls,
        v: OptionalFloatStats,
        info: ValidationInfo,
    ) -> OptionalFloatStats:
        """Validate stats length against the `per_channel` parameter.

        Parameters
        ----------
        v : OptionalFloatStats
            Value to validate.
        info : ValidationInfo
            Validated values.

        Returns
        -------
        OptionalFloatStats
            Validate value.
        """
        return _validate_global_stats_single_element(v, info)

    @model_validator(mode="after")
    def validate_mins_maxes(self: Self) -> Self:
        """Validate that mins and maxes are provided in pairs or both None.

        Returns
        -------
        Self
            The validated model instance.

        Raises
        ------
        ValueError
            If only one of mins or maxes is provided for input or target,
            or if paired lists have mismatched lengths.
        """
        if (self.input_mins is None) != (self.input_maxes is None):
            raise ValueError(
                "input_mins and input_maxes must be both provided or both None."
            )
        if self.input_mins is not None and self.input_maxes is not None:
            if len(self.input_mins) != len(self.input_maxes):
                raise ValueError("input_mins and input_maxes must have same length.")

            for i, (min_val, max_val) in enumerate(
                zip(self.input_mins, self.input_maxes, strict=True)
            ):
                if min_val >= max_val:
                    raise ValueError(
                        f"input_mins[{i}] ({min_val}) must be less than "
                        f"input_maxes[{i}] ({max_val})"
                    )

        if (self.target_mins is None) != (self.target_maxes is None):
            raise ValueError(
                "target_mins and target_maxes must be both provided or both None."
            )
        if self.target_mins is not None and self.target_maxes is not None:
            if len(self.target_mins) != len(self.target_maxes):
                raise ValueError("target_mins and target_maxes must have same length.")

            for i, (min_val, max_val) in enumerate(
                zip(self.target_mins, self.target_maxes, strict=True)
            ):
                if min_val >= max_val:
                    raise ValueError(
                        f"target_mins[{i}] ({min_val}) must be less than "
                        f"target_maxes[{i}] ({max_val})"
                    )

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
        object.__setattr__(self, "input_mins", mins)
        object.__setattr__(self, "input_maxes", maxes)
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
        object.__setattr__(self, "target_mins", mins)
        object.__setattr__(self, "target_maxes", maxes)
        self.__class__.model_validate(self)

    def _validate_input_stats_size(self: Self, n_input_channels: int) -> None:
        """Validate that the input statistics have the correct length.

        Parameters
        ----------
        n_input_channels : int
            The number of input channels to validate against.

        Raises
        ------
        ValueError
            If the length of input_mins does not match the number of input channels.
        """
        # we assume that mins and maxes are already validated against each other, and
        # against per-channel parameter
        if self.per_channel:
            if self.input_mins is not None:
                if len(self.input_mins) != n_input_channels:
                    raise ValueError(
                        f"input_mins length {len(self.input_mins)} does not match "
                        f"number of input channels {n_input_channels}."
                    )

    def _validate_target_stats_size(self: Self, n_target_channels: int) -> None:
        """Validate that the target statistics have the correct length.

        Parameters
        ----------
        n_target_channels : int
            The number of target channels to validate against.

        Raises
        ------
        ValueError
            If the length of target_mins does not match the number of target channels.
        """
        # we assume that mins and maxes are already validated against each other, and
        # against per-channel parameter
        if self.per_channel:
            if self.target_mins is not None:
                if len(self.target_mins) != n_target_channels:
                    raise ValueError(
                        f"target_mins length {len(self.target_mins)} does not match "
                        f"number of target channels {n_target_channels}."
                    )

    def validate_size(
        self: Self, n_input_channels: int, n_output_channels: int
    ) -> None:
        """Validate that statistics sizes match the number of channels.

        Parameters
        ----------
        n_input_channels : int
            The number of input channels to validate against.
        n_output_channels : int
            The number of output channels to validate against.

        Raises
        ------
        ValueError
            If any provided statistics list does not match the expected size.
        """
        self._validate_input_stats_size(n_input_channels)
        self._validate_target_stats_size(n_output_channels)


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

    def validate_size(self: Self, *args: Any, **kwargs: Any) -> None:
        """No validation needed.

        Parameters
        ----------
        *args : Any
            Parameters will be ignored.
        **kwargs : Any
            Parameters will be ignored.
        """
        pass


NormalizationConfig = Annotated[
    Union[MeanStdConfig, NoNormConfig, QuantileConfig, MinMaxConfig],
    Discriminator("name"),
]
