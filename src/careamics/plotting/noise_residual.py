"""Plotting noise residuals."""

from pathlib import Path

import matplotlib.pyplot as plt
from numpy.typing import NDArray

from .utils import get_plot_file_path


def plot_noise_residuals(
    image: NDArray,
    prediction: NDArray,
    save_path: Path | str | None = None,
) -> None:
    """Plot noise residuals.

    Parameters
    ----------
    image : NDArray
        The input image.
    prediction : NDArray
        The predicted image.
    save_path : Path | str | None, optional
        Path to save the figure. If None, the figure will not be saved. Default is None.
    """
    residuals = image - prediction

    fig, axes = plt.subplots(1, 3, figsize=(12, 5), layout="constrained")
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(prediction, cmap="gray")
    axes[1].set_title("Predicted Image")
    axes[1].axis("off")

    axes[2].imshow(residuals, cmap="gray")
    axes[2].set_title("Residuals")
    axes[2].axis("off")

    # check for saving the figure as an image
    if save_path is not None:
        _save_file = get_plot_file_path(save_path, "noise_residuals.png")
        fig.savefig(_save_file, format="png", dpi=300)
        print(f"The figure was saved at {_save_file.resolve()}")

    plt.show()
