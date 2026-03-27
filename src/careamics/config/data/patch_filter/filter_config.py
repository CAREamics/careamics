"""Base class for patch and coordinate filtering models."""

from pydantic import BaseModel, Field


class FilterConfig(BaseModel):
    """Base class for patch and coordinate filtering models."""

    name: str
    """Name of the filter."""

    ref_channel: int = 0
    """The channel to use as reference for filtering."""

    filtered_patch_prob: float = Field(default=0.1, ge=0.0, le=1.0)
    """The probability that each patch classed as background will be selected each epoch
    during training."""
