"""Whole image patching Pydantic model."""

from typing import Literal

from pydantic import BaseModel


class WholePatchingConfig(BaseModel):
    """Whole image patching Pydantic model."""

    name: Literal["whole"] = "whole"
    """The name of the patching strategy."""
