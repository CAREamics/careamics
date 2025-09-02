"""Plotting utilities."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray

from careamics.models.lvae.noise_models import GaussianMixtureNoiseModel


def plot_noise_model_probability_distribution(
    noise_model: GaussianMixtureNoiseModel,
    signalBinIndex: int,
    histogram: NDArray,
    channel: str | None = None,
    number_of_bins: int = 100,
) -> None:
    """Plot probability distribution P(x|s) for a certain ground truth signal.

    Predictions from both Histogram and GMM-based
    Noise models are displayed for comparison.

    Parameters
    ----------
    noise_model : GaussianMixtureNoiseModel
        Trained GaussianMixtureNoiseModel.
    signalBinIndex : int
        Index of signal bin. Values go from 0 to number of bins (`n_bin`).
    histogram : NDArray
        Histogram based noise model.
    channel : Optional[str], optional
        Channel name used for plotting. Default is None.
    number_of_bins : int, optional
        Number of bins in the resulting histogram. Default is 100.
    """
    min_signal = noise_model.min_signal.item()
    max_signal = noise_model.max_signal.item()
    bin_size = (max_signal - min_signal) / number_of_bins

    query_signal_normalized = signalBinIndex / number_of_bins
    query_signal = query_signal_normalized * (max_signal - min_signal) + min_signal
    query_signal += bin_size / 2
    query_signal = torch.tensor(query_signal)

    query_observations = torch.arange(min_signal, max_signal, bin_size)
    query_observations += bin_size / 2

    likelihoods = noise_model.likelihood(
        observations=query_observations, signals=query_signal
    ).numpy()

    plt.figure(figsize=(12, 5))
    if channel:
        plt.suptitle(f"Noise model for channel {channel}")
    else:
        plt.suptitle("Noise model")

    plt.subplot(1, 2, 1)
    plt.xlabel("Observation Bin")
    plt.ylabel("Signal Bin")
    plt.imshow(histogram**0.25, cmap="gray")
    plt.axhline(y=signalBinIndex + 0.5, linewidth=5, color="blue", alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(
        query_observations,
        likelihoods,
        label="GMM : " + " signal = " + str(np.round(query_signal, 2)),
        marker=".",
        color="red",
        linewidth=2,
    )
    plt.xlabel("Observations (x) for signal s = " + str(query_signal))
    plt.ylabel("Probability Density")
    plt.title("Probability Distribution P(x|s) at signal =" + str(query_signal))
    plt.legend()
