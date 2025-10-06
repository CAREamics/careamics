"""Base class for patch and coordinate filtering models."""

from pydantic import BaseModel, Field


class FilterModel(BaseModel):
    """Base class for patch and coordinate filtering models."""

    name: str
    """Name of the filter."""

    p: float = Field(1.0, ge=0.0, le=1.0)
    """Probability of applying the filter to a patch or coordinate."""

    seed: int | None = Field(default=None, gt=0)
    """Seed for the random number generator for reproducibility."""
