"""Plotting Losses."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from .utils import get_plot_file_path


def plot_loss(
    loss_dict: dict[str, list],
    fig_size: tuple[float, float] = (8, 5.5),
    save_path: Path | str | None = None,
) -> None:
    """Plot training and validation losses.

    Parameters
    ----------
    loss_dict : dict[str, list]
        Dictionary containing the training and validation losses. Must contain the key
        "train_loss" and optionally "val_loss".
        The values should be lists of loss values.
    fig_size : tuple[float, float], optional
        Size of the figure to be plotted. Default is (8, 5.5).
    save_path : Path | str | None, optional
        Path to save the figure. If None, the figure will not be saved. Default is None.
    """
    if "train_loss" not in loss_dict:
        raise ValueError("train_loss key is missing in losses_dict.")

    train_loss = loss_dict["train_loss"]
    train_epoch = loss_dict.get("train_epoch", np.arange(len(train_loss)))
    val_loss = loss_dict.get("val_loss")
    val_epoch = loss_dict.get("val_epoch")

    # plotting the losses
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(train_epoch, train_loss, color="dodgerblue", label="Train Loss")
    if val_loss and val_epoch:
        ax.plot(val_epoch, val_loss, color="coral", label="Val Loss")

    ax.legend()
    ax.grid(alpha=0.35)
    ax.set_title("Losses")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # check for saving the figure as an image
    if save_path is not None:
        _save_file = get_plot_file_path(save_path, "losses.png")
        fig.savefig(_save_file, format="png", dpi=300)
        print(f"The figure was saved at {_save_file.resolve()}")

    plt.show()
