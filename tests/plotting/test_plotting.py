from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from careamics.config import GaussianMixtureNMConfig
from careamics.models.lvae.noise_models import (
    GaussianMixtureNoiseModel,
    create_histogram,
)
from careamics.plotting import (
    plot_autocorrelation,
    plot_loss,
    plot_noise_model_distribution,
    plot_noise_residuals,
)


def train_noise_model(image_size, max_value, noise_scale):
    """Train a noise model."""
    gen = np.random.default_rng(42)
    signal_normalized = gen.uniform(0, 1, image_size)
    noise = gen.normal(0, noise_scale, image_size)
    observation_normalized = signal_normalized + noise
    signal = signal_normalized * max_value
    observation = observation_normalized * max_value

    nm_config = GaussianMixtureNMConfig(
        model_type="GaussianMixtureNoiseModel",
        n_gaussian=1,
        min_signal=signal.min(),
        max_signal=signal.max(),
    )
    noise_model = GaussianMixtureNoiseModel(nm_config)
    training_losses = noise_model.fit(
        signal=signal, observation=observation, n_epochs=500
    )
    initial_loss = training_losses[0]
    last_loss = training_losses[-1]
    # Check if model is training
    assert initial_loss > last_loss

    hist = create_histogram(100, signal.min(), signal.max(), observation, signal)

    return noise_model, hist


def test_plot_loss_empty_dict():
    """Test plotting an empty loss dict."""
    loss_dict = {}
    with pytest.raises(ValueError) as ex_info:
        plot_loss(loss_dict)
    assert ex_info.type is ValueError


def test_plot_loss(tmp_path: Path):
    """Test plotting the loss dict."""
    loss_dict = {
        "train_loss": np.random.rand(10).tolist(),
        "val_loss": np.random.rand(10).tolist(),
    }
    with patch("matplotlib.pyplot.show") as show_patch:
        plot_loss(loss_dict, save_path=tmp_path)
    assert show_patch.called
    assert (tmp_path / "losses.png").exists()


def test_plot_noise_model(tmp_path: Path):
    """Test plotting noise model distribution."""
    noise_model, hist = train_noise_model([3, 64, 64], 255, 0.1)

    with patch("matplotlib.pyplot.show") as show_patch:
        plot_noise_model_distribution(
            noise_model,
            signal_bin_index=50,
            histogram=hist[0],
            number_of_bins=100,
            save_path=tmp_path,
        )
    assert show_patch.called
    assert (tmp_path / "noise_model.png").exists()


def test_plot_residuals(tmp_path: Path):
    """Test plotting noise residuals."""
    image = np.random.rand(64, 64)
    noisy = image + np.random.randn(64, 64)

    with patch("matplotlib.pyplot.show") as show_patch:
        plot_noise_residuals(image, noisy, save_path=tmp_path)
    assert show_patch.called
    assert (tmp_path / "noise_residuals.png").exists()


def test_plot_autocorrelation(tmp_path: Path):
    """Test plotting autocorrelation image."""
    image = np.random.rand(64, 64)
    noisy = image + np.random.randn(64, 64)

    with patch("matplotlib.pyplot.show") as show_patch:
        plot_autocorrelation(noisy, save_path=tmp_path)
    assert show_patch.called
    assert (tmp_path / "autocorrelation.png").exists()
