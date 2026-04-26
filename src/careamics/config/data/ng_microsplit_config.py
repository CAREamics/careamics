"""MicroSplit-specific data configuration extending NGDataConfig."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from .ng_data_config import NGDataConfig


class MicroSplitDataConfig(NGDataConfig):
    """Data configuration for MicroSplit (LVAE) channel-separation models.

    Extends :class:`NGDataConfig` with parameters specific to the MicroSplit
    algorithm:

    - **Lateral context (LC)**: multi-scale context patches centred on the
      primary crop, used as input to the hierarchical VAE encoder.
    - **Channel mixing**: synthetic creation of the input by alpha-weighted
      superposition of multiple fluorescence channels.
    - **Uncorrelated-channel augmentation**: with a given probability, channels
      other than channel 0 are drawn from random spatial locations, making them
      spatially uncorrelated with channel 0.
    - **Empty-patch mixing**: optionally force one channel to contain only
      background while another contains signal, mimicking the partially-labelled
      regime described in the MicroSplit paper.

    Parameters
    ----------
    lateral_context : bool
        If ``True``, the input extractor is built with
        ``lateral_context_patch_constr`` so that each extracted patch has the
        shape ``(C, L, (Z), Y, X)`` where ``L == multiscale_count``.
    multiscale_count : int
        Number of LC levels including the full-resolution level (level 0).
        Only used when ``lateral_context=True``. Must satisfy
        ``multiscale_count >= 1``.
    padding_mode : {"reflect", "wrap"}
        Padding mode passed to ``lateral_context_patch_constr`` for handling
        patch boundaries. Corresponds to legacy ``padding_kwargs["mode"]``.
    input_is_sum : bool
        If ``True`` the alpha-weighted channel average is multiplied by the
        number of channels to obtain the sum rather than the average.
        Corresponds to the legacy ``input_is_sum`` flag.
    alpha_range : tuple of (float, float) or None
        Per-channel alpha sampling range ``(start, end)``.  At each
        ``__getitem__`` call each channel's weight is drawn uniformly from this
        range.  ``None`` means equal weights ``1 / num_channels``.
    mix_uncorrelated_channels : bool
        If ``True``, apply uncorrelated-channel augmentation during training.
        Channels 1…C-1 are drawn from random spatial locations with probability
        ``uncorrelated_channel_probab``.
    uncorrelated_channel_probab : float
        Probability of applying the uncorrelated-channel swap on any given
        sample.  Only relevant when ``mix_uncorrelated_channels=True``.
    empty_patch_mixing : bool
        If ``True``, use ``get_empty_channel_patches`` to enforce that selected
        channels either contain signal or background (controlled by
        ``empty_signal_channels`` / ``empty_background_channels``).
    empty_signal_channels : list of int or None
        Channel indices that must contain signal patches (not filtered out).
        Only used when ``empty_patch_mixing=True``.
    empty_background_channels : list of int or None
        Channel indices that must be background (empty) patches.
        Only used when ``empty_patch_mixing=True``.
    empty_patch_patience : int
        Number of random patch candidates to try before giving up when looking
        for a patch satisfying the empty/signal criterion.

    Notes
    -----
    Model-side cross-validation (e.g. confirming ``multiscale_count`` matches
    the number of LVAE encoder levels) is delegated to the configuration
    factory (``create_ng_microsplit_configuration``); this Pydantic model only
    performs internal self-consistency checks.
    """

    # ── LC parameters ─────────────────────────────────────────────────────────
    lateral_context: bool = Field(default=True)
    """Enable lateral-context multi-scale patch construction."""

    multiscale_count: int = Field(default=3, ge=1)
    """Number of LC levels (including full-resolution level 0)."""

    padding_mode: Literal["reflect", "wrap"] = Field(default="reflect")
    """Boundary padding mode for the LC patch constructor."""

    # ── Alpha / input synthesis ────────────────────────────────────────────────
    input_is_sum: bool = Field(default=False)
    """Multiply the average-combined input by C to get the sum."""

    alpha_range: tuple[float, float] | None = Field(default=None)
    """Per-channel alpha sampling range (start, end). None → equal weights."""

    # ── Uncorrelated-channel augmentation ─────────────────────────────────────
    mix_uncorrelated_channels: bool = Field(default=False)
    """Draw channels 1…C-1 from random locations with given probability."""

    uncorrelated_channel_probab: float = Field(default=0.5, ge=0.0, le=1.0)
    """Probability of applying the uncorrelated-channel swap per sample."""

    # ── Empty-patch mixing ────────────────────────────────────────────────────
    empty_patch_mixing: bool = Field(default=False)
    """Force channels to contain signal or background (requires filters)."""

    empty_signal_channels: list[int] | None = Field(default=None)
    """Channel indices that must contain signal."""

    empty_background_channels: list[int] | None = Field(default=None)
    """Channel indices that must be background (empty)."""

    empty_patch_patience: int = Field(default=200, ge=1)
    """Max candidates per channel when enforcing empty/signal criterion."""

    # ── Validators ────────────────────────────────────────────────────────────

    @model_validator(mode="after")
    def _validate_alpha_range(self) -> MicroSplitDataConfig:
        """Validate alpha_range is ordered and within [0, 1]."""
        if self.alpha_range is not None:
            start, end = self.alpha_range
            if start < 0.0 or end > 1.0:
                raise ValueError(
                    f"alpha_range values must be in [0, 1], got {self.alpha_range}."
                )
            if start > end:
                raise ValueError(f"alpha_range start ({start}) must be <= end ({end}).")
        return self

    @model_validator(mode="after")
    def _validate_empty_patch_channels(self) -> MicroSplitDataConfig:
        """Validate that signal and background channel sets do not overlap."""
        if not self.empty_patch_mixing:
            return self
        signal = set(self.empty_signal_channels or [])
        background = set(self.empty_background_channels or [])
        overlap = signal & background
        if overlap:
            raise ValueError(
                f"Channels {overlap} appear in both empty_signal_channels and "
                f"empty_background_channels. A channel cannot be both."
            )
        return self

    @model_validator(mode="after")
    def _validate_lateral_context_needs_multiscale(self) -> MicroSplitDataConfig:
        """Warn (or fail) if lateral_context is False but multiscale_count > 1."""
        if not self.lateral_context and self.multiscale_count > 1:
            raise ValueError(
                f"lateral_context=False but multiscale_count={self.multiscale_count} "
                f"> 1. Set multiscale_count=1 when lateral_context is disabled."
            )
        return self

    def sample_alphas(self, n_channels: int, rng: Any | None = None) -> list[float]:
        """Sample per-channel alpha weights.

        Parameters
        ----------
        n_channels : int
            Number of channels.
        rng : numpy.random.Generator or None
            Optional RNG for reproducibility. Uses ``numpy.random`` module-level
            functions when ``None``.

        Returns
        -------
        list of float
            Alpha weight for each channel.
        """
        import numpy as np

        if self.alpha_range is None:
            return [1.0 / n_channels] * n_channels

        start, end = self.alpha_range
        if rng is not None:
            return [float(rng.uniform(start, end)) for _ in range(n_channels)]
        return [float(np.random.uniform(start, end)) for _ in range(n_channels)]
