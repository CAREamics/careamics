"""Convenience functions to create training configurations."""

from typing import Any, Literal

from careamics.config.lightning.training_config import TrainingConfig


def create_training_configuration(
    trainer_params: dict,
    logger: Literal["wandb", "tensorboard", "none"],
    checkpoint_params: dict[str, Any] | None = None,
) -> TrainingConfig:
    """
    Create a dictionary with the parameters of the training model.

    Parameters
    ----------
    trainer_params : dict
        Parameters for Lightning Trainer class, see PyTorch Lightning documentation.
    logger : {"wandb", "tensorboard", "none"}
        Logger to use.
    checkpoint_params : dict, default=None
        Parameters for the checkpoint callback, see PyTorch Lightning documentation
        (`ModelCheckpoint`) for the list of available parameters.

    Returns
    -------
    TrainingConfig
        Training model with the specified parameters.
    """
    return TrainingConfig(
        lightning_trainer_config=trainer_params,
        logger=None if logger == "none" else logger,
        checkpoint_callback={} if checkpoint_params is None else checkpoint_params,
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
