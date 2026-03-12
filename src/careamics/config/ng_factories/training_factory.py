"""Convenience functions to create training configurations."""

from typing import Any, Literal

from careamics.config.ng_configs.ng_training_configuration import (
    NGTrainingConfig,
    default_training_dict,
)


def create_training_configuration(
    algorithm: Literal["care", "n2n", "n2v"],
    trainer_params: dict,
    logger: Literal["wandb", "tensorboard", "none"],
    checkpoint_params: dict[str, Any] | None = None,
    monitor_metric: str = "val_loss",
) -> NGTrainingConfig:
    """
    Create a dictionary with the parameters of the training model.

    Parameters
    ----------
    algorithm : {"care", "n2n", "n2v"}
        Algorithm type, used to select the default checkpointing preset.
    trainer_params : dict
        Parameters for Lightning Trainer class, see PyTorch Lightning documentation.
    logger : {"wandb", "tensorboard", "none"}
        Logger to use.
    checkpoint_params : dict, default=None
        Parameters for the checkpoint callback, see PyTorch Lightning documentation
        (`ModelCheckpoint`) for the list of available parameters. If `None`, then
        default parameters are applied.
    monitor_metric : str, default="val_loss"
        Metric to monitor for early stopping.

    Returns
    -------
    NGTrainingConfig
        Training configuration with the specified parameters.
    """
    return NGTrainingConfig(
        **default_training_dict(
            algorithm=algorithm,
            trainer_params=trainer_params,
            logger=logger,
            checkpoint_params=checkpoint_params,
            monitor_metric=monitor_metric,
        )
    )


def update_trainer_params(
    trainer_params: dict[str, Any] | None = None,
    num_epochs: int | None = None,
    num_steps: int | None = None,
) -> dict[str, Any]:
    """
    Update trainer parameters with num_epochs and num_steps.

    Parameters
    ----------
    trainer_params : dict, optional
        Parameters for Lightning Trainer class, by default None.
    num_epochs : int, optional
        Number of epochs to train for. If provided, this will be added as max_epochs
        to trainer_params, by default None.
    num_steps : int, optional
        Number of batches in 1 epoch. If provided, this will be added as
        limit_train_batches to trainer_params, by default None.

    Returns
    -------
    dict
        Updated trainer parameters dictionary.
    """
    final_trainer_params = {} if trainer_params is None else trainer_params.copy()

    if num_epochs is not None:
        final_trainer_params["max_epochs"] = num_epochs
    if num_steps is not None:
        final_trainer_params["limit_train_batches"] = num_steps

    return final_trainer_params
