"""Configuration classes for LVAE losses."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


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

    @model_validator(mode="after")
    def at_least_one_term(self: Self) -> Self:
        """Validate that at least one likelihood term is active.

        The reconstruction likelihood is a weighted sum of the Gaussian (muSplit) and
        noise-model (denoiSplit) terms; if both weights are 0 there is no data term.

        Returns
        -------
        Self
            The validated loss configuration.

        Raises
        ------
        ValueError
            If both `gaussian_likelihood_weight` and `noise_model_likelihood_weight`
            are 0.
        """
        if (
            self.gaussian_likelihood_weight == 0
            and self.noise_model_likelihood_weight == 0
        ):
            raise ValueError(
                "At least one of `gaussian_likelihood_weight` or "
                "`noise_model_likelihood_weight` must be greater than 0; both are 0 so "
                "no likelihood term is active."
            )
        return self

    @model_validator(mode="after")
    def predict_logvar_required_for_musplit(self: Self) -> Self:
        """Validate that `predict_logvar` is on when the muSplit likelihood is active.

        The Gaussian (muSplit) likelihood consumes the pixelwise predicted log-variance,
        so `predict_logvar` must be ``True`` whenever `gaussian_likelihood_weight` > 0.
        This mirrors the runtime check in the LVAE loss, surfaced here at configuration
        time. Pure denoiSplit (`gaussian_likelihood_weight` == 0) may use either value.

        Returns
        -------
        Self
            The validated loss configuration.

        Raises
        ------
        ValueError
            If `gaussian_likelihood_weight` > 0 but `predict_logvar` is False.
        """
        if self.gaussian_likelihood_weight > 0 and not self.predict_logvar:
            raise ValueError(
                "`predict_logvar` must be True when the muSplit Gaussian likelihood is "
                f"active (`gaussian_likelihood_weight` > 0, got "
                f"{self.gaussian_likelihood_weight})."
            )
        return self
