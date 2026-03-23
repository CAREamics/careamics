"""Base class for patch and coordinate filtering models."""

from pydantic import BaseModel


class FilterConfig(BaseModel):
    """Base class for patch and coordinate filtering models."""

    name: str
    """Name of the filter."""
