"""Validators shared by the Noise2Void family of configurations (N2V and PN2V)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from careamics.config.data import DataConfig


def check_no_target_axes(data_config: DataConfig, algorithm_name: str) -> None:
    """Check that the data configuration has no separate target axes.

    Parameters
    ----------
    data_config : DataConfig
        Data configuration.
    algorithm_name : str
        Algorithm name used in the error message.

    Raises
    ------
    ValueError
        If `target_axes` is not `None`.
    """
    if data_config.target_axes is not None:
        raise ValueError(f"{algorithm_name} does not support non-None `target_axes`.")


def check_masked_pixel_probability(
    data_config: DataConfig, masked_pixel_percentage: float, is_3D: bool
) -> None:
    """Check that every training patch is expected to have a blind-spot pixel.

    The probability of creating a blind-spot pixel is a function of the masked pixel
    percentage and of the patch size.

    Parameters
    ----------
    data_config : DataConfig
        Data configuration.
    masked_pixel_percentage : float
        Percentage of masked pixels per patch.
    is_3D : bool
        Whether the model is 3D.

    Raises
    ------
    ValueError
        If the probability of masking a pixel within a patch is less than 1 for the
        chosen masked pixel percentage and patch size.
    """
    # runtime import to avoid a circular import with `careamics.config.data`
    from careamics.config.data.patching_strategies import WholePatchingConfig

    if data_config.mode != "training":
        return

    assert not isinstance(data_config.patching, WholePatchingConfig)

    patch_size = data_config.patching.patch_size
    expected_area_per_pixel = 1 / (masked_pixel_percentage / 100)

    n_dims = 3 if is_3D else 2
    patch_size_lower_bound = int(np.ceil(expected_area_per_pixel ** (1 / n_dims)))
    required_patch_size = tuple(
        2 ** int(np.ceil(np.log2(patch_size_lower_bound))) for _ in range(n_dims)
    )
    required_mask_pixel_perc = (1 / np.prod(patch_size)) * 100

    if expected_area_per_pixel > np.prod(patch_size):
        raise ValueError(
            "The probability of creating a blind-spot pixel within a patch is "
            f"below 1, for a patch size of {patch_size} with a masked pixel "
            f"percentage of {masked_pixel_percentage}%. Either increase the patch size "
            f"to {required_patch_size} or increase the masked pixel percentage "
            f"to at least {required_mask_pixel_perc}%."
        )


def check_monitor_without_validation(
    data_config: DataConfig, monitor_metric: str
) -> None:
    """Check that a training metric is monitored when no validation data is used.

    Parameters
    ----------
    data_config : DataConfig
        Data configuration.
    monitor_metric : str
        Metric monitored by the learning rate scheduler.

    Raises
    ------
    ValueError
        If no validation data is used and the monitored metric is not a training
        metric.
    """
    if data_config.mode == "training" and data_config.n_val_patches == 0:
        if monitor_metric not in ["train_loss", "train_loss_epoch"]:
            raise ValueError(
                f"No validation data is used (`n_val_patches=0`), but the monitored"
                f" metric ({monitor_metric}) is not a training metric. Please set"
                f" `algorithm_config.monitor_metric` to `train_loss` or "
                f"`train_loss_epoch` to monitor the training loss during training. "
                f"Note that the `n_val_patches` parameter is ignored if passing "
                f"validation data. In which case, keep the default value."
            )
