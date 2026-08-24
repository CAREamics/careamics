"""Plotting image autocorrelation."""

from pathlib import Path

import matplotlib.pyplot as plt
from numpy.typing import NDArray

from careamics.utils import autocorrelation

from .utils import get_plot_file_path


def plot_autocorrelation(
    image: NDArray,
    crop_center: bool = True,
    fig_size: tuple[float, float] = (8, 5.5),
    save_path: Path | str | None = None,
) -> None:
    """Plot training and validation losses.

    Parameters
    ----------
    image : NDArray
        Input image.
    crop_center : bool
        Whether to crop the autocorrelation image around the center pixel or not.
    fig_size : tuple[float, float], optional
        Size of the figure to be plotted. Default is (8, 5.5).
    save_path : Path | str | None, optional
        Path to save the figure. If None, the figure will not be saved. Default is None.
    """
    if image.ndim > 2:
        raise ValueError(
            f"The input image must be 2D (YX), but it has {image.ndim} dimensions."
        )

    autocorr_image = autocorrelation(image)
    if crop_center:
        crop_size = 15
        mid_point = autocorr_image.shape[0] // 2
        autocorr_image = autocorr_image[
            mid_point - crop_size : mid_point + crop_size,
            mid_point - crop_size : mid_point + crop_size,
        ]

    # plotting
    fig, ax = plt.subplots(figsize=fig_size)
    ax.imshow(autocorr_image, cmap="gray")
    ax.set_axis_off()
    ax.set_title("Autocorrelation")

    # check for saving the figure as an image
    if save_path is not None:
        _save_file = get_plot_file_path(save_path, "autocorrelation.png")
        fig.savefig(_save_file, format="png", dpi=300)
        print(f"The figure was saved at {_save_file.resolve()}")

    plt.show()
