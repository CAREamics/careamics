"""Noise Model plotting utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray

from careamics.models.lvae.noise_models import GaussianMixtureNoiseModel

from .utils import get_plot_file_path


def plot_noise_model_distribution(
    noise_model: GaussianMixtureNoiseModel,
    signal_bin_index: int,
    histogram: NDArray,
    channel: str | None = None,
    number_of_bins: int = 100,
    save_path: Path | str | None = None,
) -> None:
    """Plot probability distribution P(x|s) for a certain ground truth signal.

    Predictions from both Histogram and GMM-based
    Noise models are displayed for comparison.

    Parameters
    ----------
    noise_model : GaussianMixtureNoiseModel
        Trained GaussianMixtureNoiseModel.
    signal_bin_index : int
        Index of signal bin. Values go from 0 to number of bins (`n_bin`).
    histogram : NDArray
        Histogram based noise model.
    channel : Optional[str], optional
        Channel name used for plotting. Default is None.
    number_of_bins : int, optional
        Number of bins in the resulting histogram. Default is 100.
    save_path : Path | str | None, optional
        Path to save the figure. If None, the figure will not be saved. Default is None.
    """
    # TODO: this should be adapted for multi-channel case with a list of GMMs.
    min_signal = noise_model.min_signal.item()
    max_signal = noise_model.max_signal.item()
    bin_size = (max_signal - min_signal) / number_of_bins

    query_signal_normalized = signal_bin_index / number_of_bins
    query_signal = query_signal_normalized * (max_signal - min_signal) + min_signal
    query_signal += bin_size / 2
    query_signal = torch.tensor(query_signal)

    query_observations = torch.arange(min_signal, max_signal, bin_size)
    query_observations += bin_size / 2

    likelihoods = noise_model.likelihood(
        observations=query_observations, signals=query_signal
    ).numpy()

    # plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if channel:
        fig.suptitle(f"Noise model for channel {channel}")
    else:
        fig.suptitle("Noise model")

    axes[0].imshow(histogram**0.25, cmap="gray")
    axes[0].axhline(y=signal_bin_index + 0.5, linewidth=5, color="blue", alpha=0.5)
    axes[0].set_xlabel("Observation Bin")
    axes[0].set_ylabel("Signal Bin")

    axes[1].plot(
        query_observations,
        likelihoods,
        label="GMM : " + " signal = " + str(np.round(query_signal, 2)),
        marker=".",
        color="red",
        linewidth=2,
    )
    axes[1].set_xlabel("Observations (x) for signal s = " + str(query_signal))
    axes[1].set_ylabel("Probability Density")
    axes[1].set_title("Probability Distribution P(x|s) at signal =" + str(query_signal))

    # check for saving the figure as an image
    if save_path is not None:
        _save_file = get_plot_file_path(save_path, "noise_model.png")
        fig.savefig(_save_file, format="png", dpi=300)
        print(f"The figure was saved at {_save_file.resolve()}")

    plt.legend()
    plt.show()
