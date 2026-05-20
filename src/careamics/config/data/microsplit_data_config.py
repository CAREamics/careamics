"""MicroSplit data configuration."""

from collections.abc import Sequence
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from .data_config import DataConfig


class MicroSplitDataConfig(DataConfig):
    """Dataset configuration for MicroSplit."""

    multiscale_count: int = Field(default=1, ge=1)
    """Number of lateral-context scales to construct for MicroSplit inputs."""

    padding_mode: Literal["reflect", "wrap"] = "reflect"
    """Padding mode used when lateral-context patches extend beyond image borders."""

    alpha_ranges: Sequence[tuple[float, float]] | None = None
    """Ranges used to sample channel mixing weights for synthetic training inputs.

    If `None`, the MicroSplit dataset factory will use equal fixed weights for each
    target channel.
    """

    uncorrelated_channel_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    """Probability of sampling uncorrelated channels for synthetic training inputs."""

    def convert_mode(
        self: Self,
        new_mode: Literal["validating", "predicting"],
        new_patch_size: Sequence[int] | None = None,
        overlap_size: Sequence[int] | None = None,
        new_batch_size: int | None = None,
        new_data_type: Literal["array", "tiff", "zarr", "czi", "custom"] | None = None,
        new_axes: str | None = None,
        new_channels: Sequence[int] | Literal["all"] | None = None,
        new_in_memory: bool | None = None,
        new_dataloader_params: dict[str, Any] | None = None,
    ) -> "MicroSplitDataConfig":
        """Convert mode while preserving MicroSplit-specific fields.

        Parameters
        ----------
        new_mode : Literal["validating", "predicting"]
            The new dataset mode, one of `validating` or `predicting`.
        new_patch_size : Sequence[int] or None, default=None
            New patch size. If `None` for `predicting`, uses whole image prediction.
        overlap_size : Sequence[int] or None, default=None
            New overlap size. Required when switching to tiled prediction with
            `new_patch_size`.
        new_batch_size : int or None, default=None
            New batch size. If `None`, keeps the current batch size.
        new_data_type : {"array", "tiff", "zarr", "czi", "custom"} or None, default=None
            New data type. If `None`, keeps the current data type.
        new_axes : str or None, default=None
            New axes. If `None`, keeps the current axes.
        new_channels : Sequence[int], "all" or None, default=None
            New channel selection. If `None`, keeps the current channel selection. If
            "all", selects all channels.
        new_in_memory : bool or None, default=None
            New in-memory loading setting. If `None`, keeps the current setting.
        new_dataloader_params : dict[str, Any] or None, default=None
            New dataloader parameters for the converted mode.

        Returns
        -------
        MicroSplitDataConfig
            Converted configuration with relevant MicroSplit-specific fields preserved.
        """
        converted = super().convert_mode(
            new_mode=new_mode,
            new_patch_size=new_patch_size,
            overlap_size=overlap_size,
            new_batch_size=new_batch_size,
            new_data_type=new_data_type,
            new_axes=new_axes,
            new_channels=new_channels,
            new_in_memory=new_in_memory,
            new_dataloader_params=new_dataloader_params,
        )
        model_dict = converted.model_dump()
        model_dict.update(
            {
                "multiscale_count": self.multiscale_count,
                "padding_mode": self.padding_mode,
            }
        )
        return MicroSplitDataConfig(**model_dict)

    @model_validator(mode="after")
    def validate_microsplit_params_against_mode(self):
        """Validate certain parameters are not set for prediction.

        Returns
        -------
        Self
            Validated config.
        """
        if self.mode == "predicting":
            if self.uncorrelated_channel_prob > 0:
                raise ValueError(
                    "Spatially uncorrelated channels are not supported for prediction."
                )
            if self.alpha_ranges is not None:
                raise ValueError("Alpha ranges cannot be set for prediction.")

        return self

    @model_validator(mode="after")
    def raise_unsupported_features(self):
        """Raise error for features not supported by MicroSplit.

        Returns
        -------
        Self
            Validated config.
        """
        if self.patch_filter is not None:
            raise NotImplementedError(
                # temporary
                "Patch filtering is currently not implemented for MicroSplit."
            )
        return self
