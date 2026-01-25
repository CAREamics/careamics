"""Configuration classes for LVAE losses."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class KLLossConfig(BaseModel):
    """KL loss configuration."""

    model_config = ConfigDict(validate_assignment=True, validate_default=True)

    current_epoch: int = 0
    """Current epoch in the training loop."""


class LVAELossConfig(BaseModel):
    """LVAE loss configuration."""

    model_config = ConfigDict(
        validate_assignment=True, validate_default=True, arbitrary_types_allowed=True
    )

    loss_type: Literal[
        "hdn",
        "microsplit",
    ]
    """Type of loss to use for LVAE"""

    reconstruction_weight: float = 1.0
    """Weight for the reconstruction loss in the total net loss
    (i.e., `net_loss = reconstruction_weight * rec_loss + kl_weight * kl_loss`)."""
    kl_weight: float = 1.0
    """Weight for the KL loss in the total net loss.
    (i.e., `net_loss = reconstruction_weight * rec_loss + kl_weight * kl_loss`)."""
    musplit_weight: float = 0.0
    """Weight for the Gaussian likelihood (muSplit). Set to 0 to disable."""
    denoisplit_weight: float = 1.0
    """Weight for the noise model likelihood (denoiSplit). Set to 0 to disable."""
    predict_logvar: bool = True
    """Whether to predict log-variance (pixelwise uncertainty)."""
    logvar_lowerbound: float | None = -5.0
    """Lower bound for predicted log-variance. None means no bound."""
    kl_params: KLLossConfig = KLLossConfig()
    """KL loss configuration."""
