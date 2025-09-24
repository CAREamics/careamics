import logging
from pathlib import Path
from typing import Union

from numpy.typing import NDArray
from pytorch_lightning.callbacks import Callback

from careamics.config.configuration import Configuration
from careamics.config.nm_model import GaussianMixtureNMConfig
from careamics.models.lvae.noise_models import (
    GaussianMixtureNoiseModel,
    MultiChannelNoiseModel,
)

logger = logging.getLogger(__name__)


# In src/careamics/careamist.py (newly added section only)
def __init__(
    self,
    source: Union[Path, str, Configuration],
    work_dir: Union[Path, str] | None = None,
    callbacks: list[Callback] | None = None,
    enable_progress_bar: bool = True,
) -> None:
    # ... existing initialization code ...

    # Initialize untrained noise models if needed
    self.untrained_noise_models: list[GaussianMixtureNoiseModel] | None = None
    if (
        hasattr(self.cfg.algorithm_config, "train_noise_model")
        and self.cfg.algorithm_config.train_noise_model_from_data
    ):
        self._initialize_noise_models_for_training()


# In src/careamics/careamist.py
def train_noise_model(
    self,
    clean_data: Union[Path, str, NDArray],  # data is provided by the user
    noisy_data: Union[Path, str, NDArray],  # called signal/observation in the nm fit
    learning_rate: float = 1e-1,
    batch_size: int = 250000,
    n_epochs: int = 2000,
    lower_clip: float = 0.0,
    upper_clip: float = 100.0,
    save_noise_models: bool = True,
) -> None:
    """Train noise models from clean/noisy data pairs.

    Parameters
    ----------
    clean_data : Union[Path, str, NDArray]
        Clean (signal) data for training noise models.
    noisy_data : Union[Path, str, NDArray]
        Noisy (observation) data for training noise models.
    learning_rate : float, default=1e-1
        Learning rate for noise model training.
    batch_size : int, default=250000
        Batch size for noise model training.
    n_epochs : int, default=2000
        Number of epochs for noise model training.
    lower_clip : float, default=0.0
        Lower percentile for clipping training data.
    upper_clip : float, default=100.0
        Upper percentile for clipping training data.
    save_models : bool, default=True
        Whether to save trained noise models to disk.

    Raises
    ------
    ValueError
        If noise models are not initialized for training.
    ValueError
        If data shapes don't match expectations.
    """
    # Check if noise model is initialized (config should have MultiChannelNMConfig)
    if self.cfg.algorithm_config.noise_model is None:
        raise ValueError(
            "No untrained noise models found. Set `train_noise_model=True` "
            "in configuration."
        )

    # Load data if paths provided (currently NM expects only numpy)
    if isinstance(clean_data, str | Path):
        clean_data = self._load_data(clean_data)
    if isinstance(noisy_data, str | Path):
        noisy_data = self._load_data(noisy_data)

    # Validate data shapes
    if clean_data.shape != noisy_data.shape:
        raise ValueError(
            f"Clean and noisy data shapes must match. "
            f"Got clean: {clean_data.shape}, noisy: {noisy_data.shape}"
        )
    # TODO other data shape checks

    # parameter controlling the number of channels to split for MS, for HDN it's 1
    output_channels = self.cfg.algorithm_config.model.output_channels

    # Train noise model for each channel
    trained_noise_models = []
    for channel_idx in range(output_channels):
        logger.info(
            f"Training noise model for channel {channel_idx + 1}/{output_channels}"
        )

        # Extract single channel data
        clean_channel = clean_data[:, channel_idx]  # (N, H, W)
        noisy_channel = noisy_data[:, channel_idx]  # (N, H, W)

        # Train noise model for this channel
        noise_model = self.untrained_noise_models[channel_idx]
        noise_model.fit(
            signal=clean_channel,
            observation=noisy_channel,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_epochs=n_epochs,
            lower_clip=lower_clip,
            upper_clip=upper_clip,
        )

        trained_noise_models.append(noise_model)

        # Save individual noise model if requested
        if save_noise_models:
            save_path = self.work_dir / "noise_models"
            noise_model.save(str(save_path), f"noise_model_ch{channel_idx}.npz")
            logger.info(f"Saved noise model for channel {channel_idx} to {save_path}")

    # Update the algorithm configuration with trained noise models
    self._update_config_with_trained_noise_models(trained_noise_models)

    logger.info("Noise model training completed successfully")


def _update_config_with_trained_noise_models(
    self, trained_models: list[GaussianMixtureNoiseModel]
) -> None:
    """Update algorithm config with trained noise models."""

    # Currently the model is initialized in the __init__ of CAREamist
    # noise_model_factory inside VAEModule expects paths to noise models
    # Ideally, we change that and call noise_model_factory here after the model init
    # And update the parameters of noise models right in the MultiChannelNoiseModel

def _load_data(self, data_path: Union[Path, str]) -> NDArray:
    """Load data from file path."""
    pass
