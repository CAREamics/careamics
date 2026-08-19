"""Configuration classes for LVAE losses."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LVAELossConfig(BaseModel):
    """Base LVAE loss configuration.

    Holds the loss terms shared by all LVAE-based algorithms (HDN and MicroSplit).
    Algorithm-specific terms live on the subclasses `HDNLossConfig` and
    `MicroSplitLossConfig`.
    """

    model_config = ConfigDict(
        validate_assignment=True, validate_default=True, arbitrary_types_allowed=True
    )

    reconstruction_weight: float = Field(default=1.0, ge=0.0)
    """Weight for the reconstruction loss in the total net loss
    (i.e., `net_loss = reconstruction_weight * rec_loss + kl_weight * kl_loss`)."""
    kl_weight: float = Field(default=1.0, ge=0.0)
    """Weight for the KL loss in the total net loss.
    (i.e., `net_loss = reconstruction_weight * rec_loss + kl_weight * kl_loss`)."""
    predict_logvar: bool = True
    """Whether to predict log-variance (pixelwise uncertainty)."""
    logvar_lowerbound: float | None = -5.0
    """Lower bound for predicted log-variance. None means no bound."""


class HDNLossConfig(LVAELossConfig):
    """HDN loss configuration.

    HDN uses a reconstruction term (Gaussian or noise-model likelihood) plus a KL
    term. It has no additional weights beyond the shared ones.
    """

    loss_type: Literal["hdn"] = "hdn"
    """Type of loss to use for LVAE."""


class MicroSplitLossConfig(LVAELossConfig):
    """MicroSplit loss configuration.

    The reconstruction likelihood is a weighted sum of a Gaussian likelihood term
    (muSplit) and a noise-model likelihood term (denoiSplit).
    """

    loss_type: Literal["microsplit"] = "microsplit"
    """Type of loss to use for LVAE."""

    gaussian_likelihood_weight: float = Field(default=0.1, ge=0.0)
    """Weight for the Gaussian likelihood term. Set to 0 to disable."""
    noise_model_likelihood_weight: float = Field(default=0.9, ge=0.0)
    """Weight for the noise model likelihood term. Set to 0 to disable."""
