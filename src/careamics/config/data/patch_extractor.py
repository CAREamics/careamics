"""Patch extractor configuration."""

from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class PatchExtractorConfig(BaseModel):
    """Patch extractor configuration."""

    model_config = ConfigDict(validate_assignment=True)

    multiscale_count: int = Field(default=1)
    """Number of lateral context levels."""

    collocate_patch_region: bool = Field(default=True)
    """Whether to extract patches from same spatial location for all channels."""

    artificial_input: bool = Field(default=False)
    """Whether to create artificial input from target channels."""

    uniform_mixing: bool = Field(default=True)
    """Whether to apply different weights to different channels dynamically during training."""

    # TODO add validators, e.g. artificial_input must be true if no real input
