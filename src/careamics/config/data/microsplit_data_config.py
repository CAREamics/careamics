"""MicroSplit data configuration."""

from collections.abc import Sequence
from typing import Any, Literal, Self

from pydantic import Field

from .data_config import DataConfig


class MicroSplitDataConfig(DataConfig):
    """Dataset configuration for MicroSplit."""

    multiscale_count: int = Field(default=1, ge=1)
    """Number of lateral-context scales to construct for MicroSplit inputs."""

    padding_mode: Literal["reflect", "wrap"] = "reflect"
    """Padding mode used when lateral-context patches extend beyond image borders."""

    alpha_ranges: Sequence[tuple[float, float]] | None = None
    """Ranges used to sample channel mixing weights for synthetic inputs.

    If `None`, the MicroSplit dataset factory will use equal fixed weights for each
    target channel.
    """

    uncorrelated_channel_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    """Probability of sampling uncorrelated channels for supported constructors."""

    def convert_mode(self: Self, *args: Any, **kwargs: Any) -> Self:
        """Convert mode while preserving MicroSplit-specific fields."""
        converted = super().convert_mode(*args, **kwargs)
        model_dict = converted.model_dump()
        model_dict.update(
            {
                "multiscale_count": self.multiscale_count,
                "padding_mode": self.padding_mode,
                "alpha_ranges": self.alpha_ranges,
                "uncorrelated_channel_prob": self.uncorrelated_channel_prob,
            }
        )
        return self.__class__(**model_dict)
