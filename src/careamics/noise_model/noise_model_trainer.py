"""Noise model trainer for MicroSplit."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from careamics.config.noise_model import GaussianMixtureNMConfig
from careamics.config.noise_model.noise_model_config import MultiChannelNMConfig
from careamics.models.lvae.noise_models import (
    GaussianMixtureNoiseModel,
    MultiChannelNoiseModel,
    create_histogram,
)
from careamics.utils.reshape_array import reshape_array


class NoiseModelTrainer:
    """Train Gaussian Mixture noise models from signal-observation pairs.

    Fits one GMM noise model per data channel given a "clean" signal array
    (typically the output of N2V or another denoiser) and the corresponding
    noisy observations.

    Parameters
    ----------
    n_gaussian : int, default=3
        Number of Gaussian components in the mixture.
    n_coeff : int, default=3
        Number of polynomial coefficients for signal-dependent parameters.
    min_sigma : float, default=125.0
        Minimum standard deviation for GMM components.
    global_signal_range : bool, default=False
        When True, ``min_signal`` and ``max_signal`` used to initialise each
        per-channel GMM are computed as the global extrema over *all* channels
        (matching the behaviour of the original MicroSplit training loop).
        When False (default) per-channel extrema are used.

    Attributes
    ----------
    noise_models : list[GaussianMixtureNoiseModel] | None
        Trained noise models, one per channel. None before training.
    histograms : list[NDArray] | None
        2D histograms of signal vs observation for each channel.
    channel_indices : list[int] | None
        Ordered channel indices. ``channel_indices[i]`` is the data channel
        that ``noise_models[i]`` was trained on.  Set after training.
    train_losses : list[list[float]] | None
        Per-channel training loss curves.  Set after training.

    Examples
    --------
    Run N2V first (or use any other denoiser), then train noise models:

    >>> trainer = NoiseModelTrainer(n_gaussian=3, n_coeff=3)
    >>> noise_models = trainer.train_from_pairs( # doctest: +SKIP
    ...     signal=n2v_predictions,   # clean / denoised  (S, C, Y, X)
    ...     observation=noisy_data,   # original noisy    (S, C, Y, X)
    ...     signal_axes="SCYX",
    ...     observation_axes="SCYX",
    ...     n_epochs=2000,
    ... )

    Get a MultiChannelNMConfig for use in create_microsplit_configuration:

    >>> nm_config = trainer.get_config() # doctest: +SKIP
    """

    def __init__(
        self,
        n_gaussian: int = 3,
        n_coeff: int = 3,
        min_sigma: float = 125.0,
        global_signal_range: bool = False,
    ) -> None:
        self.n_gaussian = n_gaussian
        self.n_coeff = n_coeff
        self.min_sigma = min_sigma
        self.global_signal_range = global_signal_range

        self.noise_models: list[GaussianMixtureNoiseModel] | None = None
        self.histograms: list[NDArray] | None = None
        self.channel_indices: list[int] | None = None
        self.train_losses: list[list[float]] | None = None

    def train_from_pairs(
        self,
        signal: NDArray,
        observation: NDArray,
        signal_axes: str | None = None,
        observation_axes: str | None = None,
        n_epochs: int = 2000,
        learning_rate: float = 1e-1,
        batch_size: int = 250000,
    ) -> list[GaussianMixtureNoiseModel]:
        """Train noise models from pre-computed signal-observation pairs.

        Fits one GMM noise model per channel.  ``signal`` is the clean /
        denoised data — typically the output of a Noise2Void prediction on
        ``observation``.

        Parameters
        ----------
        signal : NDArray
            Clean/denoised signal data.
            Shape: (S, C, [Z], Y, X) or (S, [Z], Y, X) for single channel.
        observation : NDArray
            Noisy observation data. Same shape as signal.
        signal_axes : str | None, default=None
            Axes describing ``signal``.  If provided, signal is reshaped to
            CAREamics' canonical ``SC(Z)YX`` order before training.  If None,
            signal is assumed to already be in ``SC(Z)YX`` order, or ``S(Z)YX``
            for single-channel data.
        observation_axes : str | None, default=None
            Axes describing ``observation``.  If provided, observation is
            reshaped to CAREamics' canonical ``SC(Z)YX`` order before training.
            If None, observation is assumed to already match the signal layout.
        n_epochs : int, default=2000
            Number of training epochs.
        learning_rate : float, default=1e-1
            Learning rate for optimization.
        batch_size : int, default=250000
            Batch size for training.

        Returns
        -------
        list[GaussianMixtureNoiseModel]
            Trained noise models, one per channel.

        Raises
        ------
        ValueError
            If signal and observation shapes do not match after optional axes
            normalization.
        """
        if signal_axes is not None:
            signal = reshape_array(signal, signal_axes)
        if observation_axes is not None:
            observation = reshape_array(observation, observation_axes)

        if signal.shape != observation.shape:
            raise ValueError(
                f"Signal and observation shapes must match after axes "
                f"normalization. "
                f"Got signal: {signal.shape}, observation: {observation.shape}"
            )

        if signal.ndim == 3:
            signal = signal[:, np.newaxis, ...]
            observation = observation[:, np.newaxis, ...]

        n_channels = signal.shape[1]

        global_min: float | None = None
        global_max: float | None = None
        if self.global_signal_range:
            global_min = float(signal.min())
            global_max = float(signal.max())

        self.noise_models = []
        self.histograms = []
        self.channel_indices = []
        self.train_losses = []

        for channel_idx in range(n_channels):
            print(f"Training noise model for channel {channel_idx + 1}/{n_channels}")

            channel_signal = signal[:, channel_idx]
            channel_obs = observation[:, channel_idx]

            noise_model, losses = self._train_single_channel(
                signal=channel_signal,
                observation=channel_obs,
                n_epochs=n_epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                min_signal=global_min,
                max_signal=global_max,
            )

            histogram = create_histogram(
                bins=100,
                min_val=float(channel_signal.min()),
                max_val=float(channel_signal.max()),
                signal=channel_signal.reshape(-1, *channel_signal.shape[-2:]),
                observation=channel_obs.reshape(-1, *channel_obs.shape[-2:]),
            )

            self.noise_models.append(noise_model)
            self.histograms.append(histogram)
            self.channel_indices.append(channel_idx)
            self.train_losses.append(losses)

        self._run_diagnostics_after_training()
        return self.noise_models

    def save(self, path: Path | str, prefix: str = "noise_model") -> list[Path]:
        """Save trained noise models to disk.

        Channel index metadata is embedded in each ``.npz`` file so that
        ordering can be validated on reload.

        Parameters
        ----------
        path : Path | str
            Directory to save noise models.
        prefix : str, default="noise_model"
            Prefix for saved file names.

        Returns
        -------
        list[Path]
            Paths to saved noise model files.

        Raises
        ------
        ValueError
            If no noise models have been trained.
        """
        if self.noise_models is None:
            raise ValueError("No noise models to save. Call train_from_pairs() first.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for idx, nm in enumerate(self.noise_models):
            filename = f"{prefix}_ch{idx}.npz"
            channel_index = (
                self.channel_indices[idx] if self.channel_indices is not None else None
            )
            nm.save(str(path), filename, channel_index=channel_index)
            saved_paths.append(path / filename)

        return saved_paths

    @classmethod
    def load(cls, paths: list[Path | str]) -> list[GaussianMixtureNoiseModel]:
        """Load noise models from disk.

        Parameters
        ----------
        paths : list[Path | str]
            Paths to noise model ``.npz`` files.

        Returns
        -------
        list[GaussianMixtureNoiseModel]
            Loaded noise models.
        """
        noise_models = []
        for path in paths:
            config = GaussianMixtureNMConfig.from_npz(path)
            noise_models.append(GaussianMixtureNoiseModel(config))
        return noise_models

    @classmethod
    def config_from_paths(
        cls,
        paths: Sequence[Path | str],
    ) -> MultiChannelNMConfig:
        """Build a ``MultiChannelNMConfig`` from saved ``.npz`` files.

        Parameters
        ----------
        paths : Sequence[Path | str]
            Paths to per-channel GaussianMixture noise model files.

        Returns
        -------
        MultiChannelNMConfig
            Multi-channel configuration built from disk-stored models.

        Raises
        ------
        ValueError
            If ``paths`` is empty.
        """
        if len(paths) == 0:
            raise ValueError("No noise model paths provided.")

        gmm_configs = [GaussianMixtureNMConfig.from_npz(path) for path in paths]

        channel_indices: list[int] | None = None
        if all(cfg.channel_index is not None for cfg in gmm_configs):
            channel_indices = [int(cfg.channel_index) for cfg in gmm_configs]

        return MultiChannelNMConfig(
            noise_models=gmm_configs,
            channel_indices=channel_indices,
        )

    def get_config(self) -> MultiChannelNMConfig:
        """Build a ``MultiChannelNMConfig`` from the trained models.

        Extracts weights and signal bounds directly from the in-memory
        ``GaussianMixtureNoiseModel`` objects without requiring a save/load
        round-trip (closes issue #850).

        Returns
        -------
        MultiChannelNMConfig
            Configuration object ready to pass to
            ``create_microsplit_configuration`` or
            ``VAEModule.set_noise_model``.

        Raises
        ------
        ValueError
            If no noise models have been trained yet.
        """
        if self.noise_models is None:
            raise ValueError(
                "No noise models available. Call train_from_pairs() first."
            )

        gmm_configs = []
        for idx, nm in enumerate(self.noise_models):
            channel_idx = (
                self.channel_indices[idx] if self.channel_indices is not None else idx
            )
            config = GaussianMixtureNMConfig(
                weight=nm.weight.detach().cpu().numpy(),
                min_signal=float(nm.min_signal.item()),
                max_signal=float(nm.max_signal.item()),
                min_sigma=float(nm.min_sigma.item()),
                n_gaussian=nm.n_gaussian,
                n_coeff=nm.n_coeff,
                channel_index=channel_idx,
            )
            gmm_configs.append(config)

        channel_indices = self.channel_indices or list(range(len(self.noise_models)))
        return MultiChannelNMConfig(
            noise_models=gmm_configs,
            channel_indices=channel_indices,
        )

    def get_multichannel_model(self) -> MultiChannelNoiseModel:
        """Get a ``MultiChannelNoiseModel`` wrapping the trained models.

        Returns
        -------
        MultiChannelNoiseModel
            Multi-channel noise model for direct use in Lightning module.

        Raises
        ------
        ValueError
            If no noise models have been trained.
        """
        if self.noise_models is None:
            raise ValueError(
                "No noise models available. Call train_from_pairs() first."
            )
        return MultiChannelNoiseModel(self.noise_models)

    def diagnose(
        self,
        signal: NDArray,
        observation: NDArray,
        wasserstein: bool = True,
    ) -> list[dict]:
        """Report per-channel diagnostics for the trained noise models.

        This method is called automatically (without Wasserstein) after every
        ``train_from_pairs`` call and issues ``UserWarning`` for suspicious
        conditions.  Call manually to obtain the full diagnostic dictionary.

        Parameters
        ----------
        signal : NDArray
            Clean/denoised signal used during training.
            Shape: (S, C, [Z], Y, X) or (S, [Z], Y, X).
        observation : NDArray
            Noisy observation used during training.  Same shape as signal.
        wasserstein : bool, default=True
            Whether to compute the Wasserstein distance between real residuals
            and sampled residuals (slower but informative).

        Returns
        -------
        list[dict]
            One dict per channel with keys:
            ``channel_index``, ``final_loss``, ``loss_trend``,
            ``has_nan_weights``, ``has_inf_weights``, ``learned_sigma_mean``,
            ``signal_range_coverage``, and optionally
            ``wasserstein_distance``.
        """
        import torch
        from scipy.stats import wasserstein_distance as _wd

        if self.noise_models is None:
            raise ValueError(
                "No noise models available. Call train_from_pairs() first."
            )

        if signal.ndim == 3:
            signal = signal[:, np.newaxis, ...]
            observation = observation[:, np.newaxis, ...]

        results = []
        for idx, nm in enumerate(self.noise_models):
            channel_idx = (
                self.channel_indices[idx] if self.channel_indices is not None else idx
            )
            ch_signal = signal[:, channel_idx]
            ch_obs = observation[:, channel_idx]

            weight_arr = nm.weight.detach().cpu()
            has_nan = bool(weight_arr.isnan().any().item())
            has_inf = bool(weight_arr.isinf().any().item())

            sig_tensor = torch.from_numpy(ch_signal.astype(np.float32))
            gp = nm.get_gaussian_parameters(sig_tensor)
            n_g = nm.n_gaussian
            sigmas = gp[n_g : 2 * n_g]
            learned_sigma_mean = float(
                sum(s.mean().item() for s in sigmas) / len(sigmas)
            )

            sig_min = float(nm.min_signal.item())
            sig_max = float(nm.max_signal.item())
            coverage = float(np.mean((ch_signal >= sig_min) & (ch_signal <= sig_max)))

            losses = self.train_losses[idx] if self.train_losses is not None else None
            final_loss = losses[-1] if losses else float("nan")
            loss_trend: str
            if losses and len(losses) >= 20:
                early = float(np.mean(losses[:10]))
                late = float(np.mean(losses[-10:]))
                if late < early * 0.9:
                    loss_trend = "decreasing"
                elif late > early * 1.1:
                    loss_trend = "increasing"
                else:
                    loss_trend = "flat"
            else:
                loss_trend = "unknown"

            diag: dict = {
                "channel_index": channel_idx,
                "final_loss": final_loss,
                "loss_trend": loss_trend,
                "has_nan_weights": has_nan,
                "has_inf_weights": has_inf,
                "learned_sigma_mean": learned_sigma_mean,
                "signal_range_coverage": coverage,
            }

            if wasserstein:
                sampled = nm.sample_observation_from_signal(ch_signal)
                real_noise = (ch_obs - ch_signal).ravel()
                synth_noise = (sampled - ch_signal).ravel()
                scale = sig_max - sig_min if sig_max > sig_min else 1.0
                wd = _wd(real_noise / scale, synth_noise / scale)
                diag["wasserstein_distance"] = float(wd)

            if has_nan or has_inf:
                warnings.warn(
                    f"Channel {channel_idx}: NaN or Inf detected in noise model "
                    "weights. The model did not converge properly.",
                    UserWarning,
                    stacklevel=2,
                )
            if loss_trend == "increasing":
                warnings.warn(
                    f"Channel {channel_idx}: Training loss is increasing. "
                    "Consider reducing the learning rate.",
                    UserWarning,
                    stacklevel=2,
                )
            if loss_trend == "flat" and final_loss > 10.0:
                warnings.warn(
                    f"Channel {channel_idx}: Training loss did not decrease "
                    f"(final_loss={final_loss:.4f}). "
                    "Consider more epochs or a different learning rate.",
                    UserWarning,
                    stacklevel=2,
                )
            if coverage < 0.95:
                warnings.warn(
                    f"Channel {channel_idx}: Only {coverage * 100:.1f}% of signal "
                    "values fall within [min_signal, max_signal]. Signal range may "
                    "be too narrow.",
                    UserWarning,
                    stacklevel=2,
                )
            if wasserstein and diag.get("wasserstein_distance", 0.0) > 0.2:
                warnings.warn(
                    f"Channel {channel_idx}: Wasserstein distance between real and "
                    f"sampled noise is {diag['wasserstein_distance']:.4f} (> 0.2). "
                    "The noise model may not accurately capture the noise "
                    "distribution.",
                    UserWarning,
                    stacklevel=2,
                )

            results.append(diag)

        return results

    def sample_observation(self, signal: NDArray) -> NDArray:
        """Sample noisy observations from the trained noise models.

        For each pixel in the input signal, samples a corresponding noisy
        pixel from the learned Gaussian Mixture Model.

        Parameters
        ----------
        signal : NDArray
            Clean signal data. Shape should match the training data:
            - Single channel: (S, [Z], Y, X)
            - Multi-channel: (S, C, [Z], Y, X)

        Returns
        -------
        NDArray
            Sampled noisy observation with same shape as input signal.

        Raises
        ------
        ValueError
            If no noise models have been trained.
        """
        if self.noise_models is None:
            raise ValueError(
                "No noise models available. Call train_from_pairs() first."
            )

        if len(self.noise_models) == 1:
            return self.noise_models[0].sample_observation_from_signal(signal)

        multichannel = self.get_multichannel_model()
        return multichannel.sample_observation(signal)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_diagnostics_after_training(self) -> None:
        """Auto-run lightweight (no Wasserstein) diagnostics after training."""
        if self.noise_models is None:
            return
        for idx, nm in enumerate(self.noise_models):
            channel_idx = (
                self.channel_indices[idx] if self.channel_indices is not None else idx
            )
            weight_arr = nm.weight.detach().cpu()
            if bool(weight_arr.isnan().any().item()):
                warnings.warn(
                    f"Channel {channel_idx}: NaN detected in noise model weights "
                    "after training. The model did not converge.",
                    UserWarning,
                    stacklevel=3,
                )
            if bool(weight_arr.isinf().any().item()):
                warnings.warn(
                    f"Channel {channel_idx}: Inf detected in noise model weights "
                    "after training. The model did not converge.",
                    UserWarning,
                    stacklevel=3,
                )
            if self.train_losses is not None:
                losses = self.train_losses[idx]
                if losses and len(losses) >= 20:
                    early = float(np.mean(losses[:10]))
                    late = float(np.mean(losses[-10:]))
                    if late > early * 1.1:
                        warnings.warn(
                            f"Channel {channel_idx}: Training loss increased during "
                            "training. Consider reducing the learning rate.",
                            UserWarning,
                            stacklevel=3,
                        )

    def _train_single_channel(
        self,
        signal: NDArray,
        observation: NDArray,
        n_epochs: int,
        learning_rate: float,
        batch_size: int,
        min_signal: float | None = None,
        max_signal: float | None = None,
    ) -> tuple[GaussianMixtureNoiseModel, list[float]]:
        """Train a single-channel noise model."""
        config = GaussianMixtureNMConfig(
            model_type="GaussianMixtureNoiseModel",
            min_signal=min_signal if min_signal is not None else float(signal.min()),
            max_signal=max_signal if max_signal is not None else float(signal.max()),
            n_gaussian=self.n_gaussian,
            n_coeff=self.n_coeff,
            min_sigma=self.min_sigma,
        )

        noise_model = GaussianMixtureNoiseModel(config)

        signal_flat = signal.reshape(-1, *signal.shape[-2:])
        obs_flat = observation.reshape(-1, *observation.shape[-2:])

        losses = noise_model.fit(
            signal=signal_flat,
            observation=obs_flat,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_epochs=n_epochs,
        )

        return noise_model, losses
