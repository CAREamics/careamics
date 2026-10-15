"""Utilities for Lightning modules."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import lightning.pytorch as L
import torch
from torch import nn
from torchmetrics import MetricCollection

from careamics.config.support import SupportedOptimizer, SupportedScheduler
from careamics.models.lvae.noise_models import MultiChannelNoiseModel
from careamics.utils.logging import get_logger

logger = get_logger(__name__)


def resolve_noise_model(
    noise_model: MultiChannelNoiseModel | Sequence[str | Path],
) -> MultiChannelNoiseModel:
    """Resolve a noise model argument into a runtime ``MultiChannelNoiseModel``.

    Accepts either an already-built ``MultiChannelNoiseModel`` (returned as-is) or a
    sequence of per-channel ``.npz`` paths (loaded via ``MultiChannelNoiseModel.from_npz``).

    Parameters
    ----------
    noise_model : MultiChannelNoiseModel or sequence of str or Path
        The noise model object, or the per-channel ``.npz`` paths to load it from.

    Returns
    -------
    MultiChannelNoiseModel
        The resolved runtime noise model.
    """
    if isinstance(noise_model, MultiChannelNoiseModel):
        return noise_model
    return MultiChannelNoiseModel.from_npz(list(noise_model))


def check_noise_model_channels(
    noise_model: MultiChannelNoiseModel, output_channels: int
) -> None:
    """Validate that the noise model covers exactly ``output_channels`` channels.

    Parameters
    ----------
    noise_model : MultiChannelNoiseModel
        The runtime noise model.
    output_channels : int
        The number of output channels of the LVAE model.

    Raises
    ------
    ValueError
        If the noise model channel count does not match ``output_channels``.
    """
    if len(noise_model) != output_channels:
        raise ValueError(
            f"Noise model has {len(noise_model)} channel(s) but the model has "
            f"{output_channels} output channel(s); they must match."
        )


# Dedicated checkpoint key for the (raw-space) noise model, kept separate from
# `hyper_parameters` so it is persisted with the checkpoint but not via the
# training `Configuration`.
NOISE_MODEL_CKPT_KEY = "noise_model"


def save_noise_model_to_checkpoint(
    raw_noise_model: MultiChannelNoiseModel | None, checkpoint: dict[str, Any]
) -> None:
    """Persist the raw-space noise model into the checkpoint dict.

    The noise model is a frozen, loss-side artifact and is intentionally kept out of
    the module ``state_dict``; it is stored here under a dedicated key so continued
    training (resume / fine-tune) can restore it without re-passing it.

    Parameters
    ----------
    raw_noise_model : MultiChannelNoiseModel or None
        The raw-space noise model to persist, or ``None`` to persist nothing.
    checkpoint : dict
        The checkpoint dictionary to write into.
    """
    if raw_noise_model is not None:
        checkpoint[NOISE_MODEL_CKPT_KEY] = raw_noise_model.to_config().model_dump()


def load_noise_model_from_checkpoint(
    checkpoint: dict[str, Any],
) -> MultiChannelNoiseModel | None:
    """Rebuild the raw-space noise model from a checkpoint dict.

    Parameters
    ----------
    checkpoint : dict
        The checkpoint dictionary previously written by
        ``save_noise_model_to_checkpoint``.

    Returns
    -------
    MultiChannelNoiseModel or None
        The restored raw-space noise model, or ``None`` if the checkpoint carries none.
    """
    payload = checkpoint.get(NOISE_MODEL_CKPT_KEY)
    if payload is None:
        return None
    # local imports to avoid importing config at module import time
    from careamics.config.noise_model.noise_model_config import MultiChannelNMConfig
    from careamics.models.lvae.noise_models import multichannel_noise_model_factory

    return multichannel_noise_model_factory(MultiChannelNMConfig(**payload))


def log_training_stats(module: L.LightningModule, loss: Any, batch_size: int) -> None:
    """Log training loss and learning rate.

    Parameters
    ----------
    module : L.LightningModule
        The Lightning module to log stats for.
    loss : Any
        The loss value for the current training step.
    batch_size : int
        The size of the batch used in the current training step.
    """
    module.log(
        "train_loss",
        loss,
        on_step=True,
        on_epoch=True,
        prog_bar=True,
        logger=True,
        batch_size=batch_size,
    )

    optimizer = module.optimizers()
    if isinstance(optimizer, list):
        current_lr = optimizer[0].param_groups[0]["lr"]
    else:
        current_lr = optimizer.param_groups[0]["lr"]
    module.log(
        "learning_rate",
        current_lr,
        on_step=False,
        on_epoch=True,
        logger=True,
        batch_size=batch_size,
    )


def log_validation_stats(
    module: L.LightningModule,
    loss: Any,
    batch_size: int,
    metrics: MetricCollection,
) -> None:
    """Log validation loss and metrics.

    Parameters
    ----------
    module : L.LightningModule
        The Lightning module to log stats for.
    loss : Any
        The loss value for the current validation step.
    batch_size : int
        The size of the batch used in the current validation step.
    metrics : MetricCollection
        The metrics collection to log.
    """
    module.log(
        "val_loss",
        loss,
        on_step=False,
        on_epoch=True,
        prog_bar=True,
        logger=True,
        batch_size=batch_size,
    )
    module.log_dict(metrics, on_step=False, on_epoch=True, batch_size=batch_size)


def get_optimizer(name: str) -> type[torch.optim.Optimizer]:
    """
    Return the optimizer class given its name.

    Parameters
    ----------
    name : str
        Optimizer name.

    Returns
    -------
    torch.nn.Optimizer
        Optimizer class.
    """
    try:
        SupportedOptimizer(name)
    except ValueError as e:
        raise NotImplementedError(f"Optimizer {name} is not yet supported.") from e

    return getattr(torch.optim, name)


def get_scheduler(
    name: str,
) -> type[torch.optim.lr_scheduler.ReduceLROnPlateau]:
    """
    Return the scheduler class given its name.

    Parameters
    ----------
    name : str
        Scheduler name.

    Returns
    -------
    Union
        Scheduler class.
    """
    try:
        SupportedScheduler(name)
    except ValueError as e:
        raise NotImplementedError(f"Scheduler {name} is not yet supported.") from e

    return getattr(torch.optim.lr_scheduler, name)


def configure_optimizers(
    model: nn.Module,
    optimizer_name: str,
    optimizer_parameters: dict[str, Any],
    lr_scheduler_name: str,
    lr_scheduler_parameters: dict[str, Any],
    monitor: str = "val_loss",
) -> dict[str, Any]:
    """Configure optimizer and learning rate scheduler.

    Parameters
    ----------
    model : nn.Module
        The model whose parameters will be optimized.
    optimizer_name : str
        The name of the optimizer to use.
    optimizer_parameters : dict[str, Any]
        Parameters to pass to the optimizer constructor.
    lr_scheduler_name : str
        The name of the learning rate scheduler to use.
    lr_scheduler_parameters : dict[str, Any]
        Parameters to pass to the learning rate scheduler constructor.
    monitor : str, optional
        The metric to monitor for the learning rate scheduler, by default "val_loss".

    Returns
    -------
    dict[str, Any]
        A dictionary containing the optimizer and learning rate scheduler configuration.
    """
    optimizer_func = get_optimizer(optimizer_name)
    optimizer = optimizer_func(  # type: ignore[operator]
        model.parameters(), **optimizer_parameters
    )

    scheduler_func = get_scheduler(lr_scheduler_name)
    scheduler = scheduler_func(optimizer, **lr_scheduler_parameters)  # type: ignore[operator]

    return {
        "optimizer": optimizer,
        "lr_scheduler": scheduler,
        "monitor": monitor,
    }


def mmse_and_sample_std(
    sample_prediction: Callable[[torch.Tensor], torch.Tensor],
    x_data: torch.Tensor,
    n_samples: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Draw several stochastic predictions and reduce them to the MMSE and its spread.

    The samples are accumulated one at a time with Welford's algorithm
    so the memory footprint does not grow with `n_samples`.

    Parameters
    ----------
    sample_prediction : callable
        Draws a single sample from `x_data`, shape (B, C, [Z], Y, X).
    x_data : torch.Tensor
        Input passed to `sample_prediction` on every draw.
    n_samples : int
        Number of samples to draw.

    Returns
    -------
    tuple of (torch.Tensor, torch.Tensor or None)
        The MMSE estimate and the standard deviation across the samples, both of
        the shape returned by `sample_prediction`. The standard deviation is `None`
        when `n_samples == 1`.
    """
    # TODO: The standard deviation across samples excludes the likelihood variance
    # and is a per-pixel marginal, so it under-estimates the total predictive
    # uncertainty.

    mean = sample_prediction(x_data)
    if n_samples == 1:
        return mean, None

    sum_squared_deviations = torch.zeros_like(mean)
    for count in range(2, n_samples + 1):
        sample = sample_prediction(x_data)
        deviation = sample - mean
        mean = mean + deviation / count
        sum_squared_deviations = sum_squared_deviations + deviation * (sample - mean)

    std = (sum_squared_deviations / (n_samples - 1)).sqrt()

    return mean, std
