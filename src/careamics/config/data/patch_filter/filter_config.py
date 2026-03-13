"""Base class for patch and coordinate filtering models."""

from pydantic import BaseModel


class FilterConfig(BaseModel):
    """Base class for patch and coordinate filtering models."""

    name: str
    """Name of the filter."""

    filter_ref_channel: int = 0
    """The channel to use as reference for filtering."""
