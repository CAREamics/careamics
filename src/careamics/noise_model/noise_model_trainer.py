"""Noise model trainer for MicroSplit."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from careamics.config import create_n2v_configuration
from careamics.config.noise_model import GaussianMixtureNMConfig
from careamics.models.lvae.noise_models import (
    GaussianMixtureNoiseModel,
    MultiChannelNoiseModel,
    create_histogram,
)

if TYPE_CHECKING:
    from careamics.careamist import CAREamist


class NoiseModelTrainer:
    """Train Gaussian Mixture noise models.

    This class handles the complete noise model training workflow:
    1. Train a denoising model (N2V) on noisy data if clean data if not provided
    2. Predict "clean" signal from noisy data
    3. Train GMM noise models from signal-observation pairs

    Parameters
    ----------
    n_gaussian : int, default=3
        Number of Gaussian components in the mixture.
    n_coeff : int, default=3
        Number of polynomial coefficients for signal-dependent parameters.
    min_sigma : float, default=125.0
        Minimum standard deviation for GMM components.

    Attributes
    ----------
    noise_models : list[GaussianMixtureNoiseModel] | None
        Trained noise models, one per channel. None before training.
    histograms : list[NDArray] | None
        2D histograms of signal vs observation for each channel.

    Examples
    --------
    Train noise models from noisy data (full workflow):

    >>> trainer = NoiseModelTrainer(n_gaussian=3, n_coeff=3)
    >>> noise_models = trainer.train(
    ...     noisy_data=my_noisy_images,
    ...     axes="SCYX",
    ...     patch_size=[64, 64],
    ...     n2v_epochs=100,
    ...     nm_epochs=2000,
    ... )

    Train noise models with explicitly provided clean data (skip N2V):

    >>> trainer = NoiseModelTrainer(n_gaussian=3, n_coeff=3)
    >>> noise_models = trainer.train(
    ...     noisy_data=my_noisy_images,
    ...     axes="SCYX",
    ...     patch_size=[64, 64],
    ...     clean_data=my_clean_predictions,
    ...     nm_epochs=2000,
    ... )

    Get a MultiChannelNoiseModel for use in training:

    >>> multichannel_nm = trainer.get_multichannel_model()
    """

    def __init__(
        self,
        n_gaussian: int = 3,
        n_coeff: int = 3,
        min_sigma: float = 125.0,
    ) -> None:
        self.n_gaussian = n_gaussian
        self.n_coeff = n_coeff
        self.min_sigma = min_sigma

        self.noise_models: list[GaussianMixtureNoiseModel] | None = None
        self.histograms: list[NDArray] | None = None
        self._n2v_model: CAREamist | None = None

    def train(
        self,
        noisy_data: NDArray,
        axes: str,
        patch_size: Sequence[int],
        clean_data: NDArray | None = None,
        batch_size: int = 64,
        n2v_epochs: int = 100,
        nm_epochs: int = 2000,
        nm_learning_rate: float = 1e-1,
        nm_batch_size: int = 250000,
        val_data: NDArray | None = None,
        work_dir: Path | str | None = None,
    ) -> list[GaussianMixtureNoiseModel]:
        """Train noise models from noisy data.

        If clean_data is provided, it is used directly as the signal.
        Otherwise, N2V is trained to predict clean signal from noisy data.

        Parameters
        ----------
        noisy_data : NDArray
            Noisy observation data. Shape: (S, C, [Z], Y, X) where C is channels,
            or (S, [Z], Y, X) for single channel.
        axes : str
            Data axes string (e.g., "SCYX", "SCZYX", "SYX").
        patch_size : Sequence[int]
            Patch size for N2V training.
        clean_data : NDArray | None, optional
            Clean signal data. If provided, skips N2V training and uses this
            directly. Must have same shape as noisy_data.
        batch_size : int, default=64
            Batch size for N2V training.
        n2v_epochs : int, default=100
            Number of epochs for N2V training.
        nm_epochs : int, default=2000
            Number of epochs for noise model training.
        nm_learning_rate : float, default=1e-1
            Learning rate for noise model training.
        nm_batch_size : int, default=250000
            Batch size for noise model training.
        val_data : NDArray | None, optional
            Validation data for N2V training.
        work_dir : Path | str | None, optional
            Working directory for saving intermediate results.

        Returns
        -------
        list[GaussianMixtureNoiseModel]
            Trained noise models, one per channel.
        """
        if clean_data is not None and clean_data.shape != noisy_data.shape:
            raise ValueError(
                f"clean_data shape must match noisy_data shape. "
                f"Got clean_data: {clean_data.shape}, noisy_data: {noisy_data.shape}"
            )

        n_channels = self._get_n_channels(noisy_data, axes)

        self.noise_models = []
        self.histograms = []

        for channel_idx in range(n_channels):
            print(f"Training noise model for channel {channel_idx + 1}/{n_channels}")

            channel_obs = self._extract_channel(noisy_data, channel_idx, axes)

            if clean_data is not None:
                channel_signal = self._extract_channel(clean_data, channel_idx, axes)
            else:
                channel_val = (
                    self._extract_channel(val_data, channel_idx, axes)
                    if val_data is not None
                    else None
                )
                channel_axes = self._remove_channel_axis(axes)

                channel_signal = self._get_clean_signal(
                    noisy_data=channel_obs,
                    axes=channel_axes,
                    patch_size=patch_size,
                    batch_size=batch_size,
                    n_epochs=n2v_epochs,
                    val_data=channel_val,
                    work_dir=work_dir,
                )

            noise_model = self._train_single_channel(
                signal=channel_signal,
                observation=channel_obs,
                n_epochs=nm_epochs,
                learning_rate=nm_learning_rate,
                batch_size=nm_batch_size,
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

        return self.noise_models

    def train_from_pairs(
        self,
        signal: NDArray,
        observation: NDArray,
        n_epochs: int = 2000,
        learning_rate: float = 1e-1,
        batch_size: int = 250000,
    ) -> list[GaussianMixtureNoiseModel]:
        """Train noise models from pre-computed signal-observation pairs.

        Use this when you already have "clean" predictions (signal) and
        noisy observations. Skips the N2V training step.

        Parameters
        ----------
        signal : NDArray
            Clean signal data. Shape: (S, C, [Z], Y, X) or (S, [Z], Y, X).
        observation : NDArray
            Noisy observation data. Same shape as signal.
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
            If signal and observation shapes do not match.
        """
        if signal.shape != observation.shape:
            raise ValueError(
                f"Signal and observation shapes must match. "
                f"Got signal: {signal.shape}, observation: {observation.shape}"
            )

        if signal.ndim == 3:
            signal = signal[:, np.newaxis, ...]
            observation = observation[:, np.newaxis, ...]

        n_channels = signal.shape[1]

        self.noise_models = []
        self.histograms = []

        for channel_idx in range(n_channels):
            print(f"Training noise model for channel {channel_idx + 1}/{n_channels}")

            channel_signal = signal[:, channel_idx]
            channel_obs = observation[:, channel_idx]

            noise_model = self._train_single_channel(
                signal=channel_signal,
                observation=channel_obs,
                n_epochs=n_epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
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

        return self.noise_models

    def save(self, path: Path | str, prefix: str = "noise_model") -> list[Path]:
        """Save trained noise models to disk.

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
            raise ValueError("No noise models to save. Call train() first.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for idx, nm in enumerate(self.noise_models):
            filename = f"{prefix}_ch{idx}.npz"
            nm.save(str(path), filename)
            saved_paths.append(path / filename)

        return saved_paths

    @classmethod
    def load(cls, paths: list[Path | str]) -> list[GaussianMixtureNoiseModel]:
        """Load noise models from disk.

        Parameters
        ----------
        paths : list[Path | str]
            Paths to noise model .npz files.

        Returns
        -------
        list[GaussianMixtureNoiseModel]
            Loaded noise models.
        """
        noise_models = []
        for path in paths:
            config = GaussianMixtureNMConfig(path=str(path))
            noise_models.append(GaussianMixtureNoiseModel(config))
        return noise_models

    def get_multichannel_model(self) -> MultiChannelNoiseModel:
        """Get a MultiChannelNoiseModel wrapping the trained models.

        Returns
        -------
        MultiChannelNoiseModel
            Multi-channel noise model for use in training.

        Raises
        ------
        ValueError
            If no noise models have been trained.
        """
        if self.noise_models is None:
            raise ValueError("No noise models available. Call train() first.")
        return MultiChannelNoiseModel(self.noise_models)

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

        Examples
        --------
        >>> trainer = NoiseModelTrainer()
        >>> trainer.train_from_pairs(signal, observation)
        >>> synthetic_noisy = trainer.sample_observation(clean_signal)
        >>> real_residuals = observation - signal
        >>> synth_residuals = synthetic_noisy - clean_signal
        """
        if self.noise_models is None:
            raise ValueError("No noise models available. Call train() first.")

        if len(self.noise_models) == 1:
            return self.noise_models[0].sample_observation_from_signal(signal)

        multichannel = self.get_multichannel_model()
        return multichannel.sample_observation(signal)

    def _get_clean_signal(
        self,
        noisy_data: NDArray,
        axes: str,
        patch_size: Sequence[int],
        batch_size: int,
        n_epochs: int,
        val_data: NDArray | None,
        work_dir: Path | str | None,
    ) -> NDArray:
        """Train N2V and predict clean signal."""
        from careamics.careamist import CAREamist

        config = create_n2v_configuration(
            experiment_name="noise_model_n2v",
            data_type="array",
            axes=axes,
            patch_size=list(patch_size),
            batch_size=batch_size,
            num_epochs=n_epochs,
        )

        careamist = CAREamist(source=config, work_dir=work_dir)
        careamist.train(train_source=noisy_data, val_source=val_data)

        predictions = careamist.predict(source=noisy_data, data_type="array")

        self._n2v_model = careamist

        return np.concatenate(list(predictions), axis=0)

    def _train_single_channel(
        self,
        signal: NDArray,
        observation: NDArray,
        n_epochs: int,
        learning_rate: float,
        batch_size: int,
    ) -> GaussianMixtureNoiseModel:
        """Train a single noise model."""
        config = GaussianMixtureNMConfig(
            model_type="GaussianMixtureNoiseModel",
            min_signal=float(signal.min()),
            max_signal=float(signal.max()),
            n_gaussian=self.n_gaussian,
            n_coeff=self.n_coeff,
            min_sigma=self.min_sigma,
        )

        noise_model = GaussianMixtureNoiseModel(config)

        signal_flat = signal.reshape(-1, *signal.shape[-2:])
        obs_flat = observation.reshape(-1, *observation.shape[-2:])

        noise_model.fit(
            signal=signal_flat,
            observation=obs_flat,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_epochs=n_epochs,
        )

        return noise_model

    @staticmethod
    def _get_n_channels(data: NDArray, axes: str) -> int:
        """Get number of channels from data and axes."""
        if "C" not in axes:
            return 1
        return data.shape[axes.index("C")]

    @staticmethod
    def _extract_channel(data: NDArray, channel_idx: int, axes: str) -> NDArray:
        """Extract a single channel from data."""
        if "C" not in axes:
            return data
        c_idx = axes.index("C")
        return np.take(data, channel_idx, axis=c_idx)

    @staticmethod
    def _remove_channel_axis(axes: str) -> str:
        """Remove C from axes string."""
        return axes.replace("C", "")

