"""Plotting Losses."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from careamics.lightning.utils import TrainingReport
from careamics.utils import get_logger

from .utils import get_plot_file_path

logger = get_logger("Plot losses")


def plot_loss(
    training_report: TrainingReport,
    fig_size: tuple[float, float] = (8, 5.5),
    save_path: Path | str | None = None,
    plot_metrics: bool = True,
    plot_learning_rate: bool = True,
) -> None:
    """Plot training and validation losses.

    Parameters
    ----------
    training_report : TrainingReport
        Dataclass containing train and validation loss, learning rate, and any
        discovered validation metrics.
    fig_size : tuple[float, float], default=None
        Size of the figure to be plotted. If `None`, then default settings are applied
        depending on `plot_metrics` and `plot_learning_rate`.
    save_path : Path | str | None, default=None
        Path to save the figure. If None, the figure will not be saved.
    plot_metrics : bool, default=True
        Whether to plot the metrics.
    plot_learning_rate : bool, default=True
        Whether to plot the learning rate.
    """
    train_loss = training_report.train_loss
    val_loss = training_report.val_loss
    learning_rate = training_report.learning_rate
    metrics_dict = training_report.metrics

    # warn if no metrics found
    if plot_metrics and not metrics_dict:
        logger.warning(msg="No metrics found, ignoring `plot_metrics`.", stacklevel=2)
        plot_metrics = False

    # compute fig size
    n_plots = 1 + plot_metrics + plot_learning_rate
    fig_size = (4 * n_plots, 3.5)

    # plot figure
    fig, ax = plt.subplots(1, n_plots, figsize=fig_size)

    plot_idx = 0
    if n_plots == 1:
        axis = ax
    else:
        axis = ax[plot_idx]
    axis.grid(alpha=0.35)
    axis.plot(
        train_loss.epoch, train_loss.value, color="dodgerblue", label="Train loss"
    )
    axis.plot(val_loss.epoch, val_loss.value, color="coral", label="Val loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("Losses")
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.legend(frameon=False)

    if plot_metrics:
        plot_idx += 1

        # process metrics name to remove underscores and add capitalization
        metric_names = {
            metric_name: " ".join(word.capitalize() for word in metric_name.split("_"))
            for metric_name in metrics_dict.keys()
        }

        for k in metrics_dict.keys():
            metric = metrics_dict[k]
            ax[plot_idx].plot(metric.epoch, metric.value, label=metric_names[k])

        ax[plot_idx].grid(alpha=0.35)
        ax[plot_idx].set_xlabel("Epoch")
        ax[plot_idx].set_ylabel("Score")
        ax[plot_idx].set_title("Metrics")
        ax[plot_idx].xaxis.set_major_locator(MaxNLocator(integer=True))
        ax[plot_idx].legend(frameon=False)

    if plot_learning_rate:
        plot_idx += 1
        ax[plot_idx].grid(alpha=0.35)
        ax[plot_idx].plot(
            learning_rate.epoch, learning_rate.value, label="Learning rate"
        )
        ax[plot_idx].set_xlabel("Epoch")
        ax[plot_idx].set_ylabel("Value")
        ax[plot_idx].set_title("Learning rate")
        ax[plot_idx].xaxis.set_major_locator(MaxNLocator(integer=True))
        ax[plot_idx].legend(frameon=False)

    fig.tight_layout()

    # check for saving the figure as an image
    if save_path is not None:
        _save_file = get_plot_file_path(save_path, "losses.png")
        fig.savefig(_save_file, format="png", dpi=300)
        print(f"The figure was saved at {_save_file.resolve()}")

    plt.show()
