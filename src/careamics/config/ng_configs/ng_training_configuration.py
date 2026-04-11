"""Training configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pprint import pformat
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


@dataclass
class SupervisedCheckpointing:
    """Presets for checkpointing CARE.

    This preset saves the top 3 best performing checkpoints based on `val_loss`, as well
    as the last one.
    """

    monitor: str = "val_loss"
    """Monitor `val_loss`."""

    mode: str = "min"
    """Top checkpoints are selected by minimum `val_loss`."""

    save_top_k: int = 3
    """Save the top 3 best performing checkpoints."""

    save_last: bool = True
    """Save the last checkpoint."""

    auto_insert_metric_name: bool = False
    """Do not insert the monitored value in the checkpoint name."""


@dataclass
class SelfSupervisedCheckpointing:
    """Presets for checkpointing Noise2Noise and Noise2Void.

    Because self-supervised algorithms are evaluating the loss against noisy pixels,
    its value is not a good measure of performances after a few epochs. Therefore, it
    cannot be used to evaluate the best performing models.

    This presets saves checkpoints every 10 epochs, as well as the last one.
    """

    every_n_epochs: int = 10
    """Save a checkpoint every 10 epochs."""

    save_top_k: int = -1
    """Save all checkpoints. Checkpoints are checked `every_n_epochs`."""

    save_last: bool = True
    """Save the last checkpoint."""

    auto_insert_metric_name: bool = False
    """Do not insert the monitored value in the checkpoint name."""


def default_training_dict(
    algorithm: Literal["care", "n2n", "n2v"],
    trainer_params: dict[str, Any] | None = None,
    logger: Literal["wandb", "tensorboard", "none"] = "none",
    checkpoint_params: dict[str, Any] | None = None,
    early_stopping_params: dict[str, Any] | None = None,
    monitor_metric: str = "val_loss",
) -> dict:
    """Default training configuration constructor.

    This function sets default training parameters based on the algorithm configuration.
    If the user provides any of the parameters, they will take precedence over the
    defaults.

    Parameters
    ----------
    algorithm : {"care", "n2n", "n2v"}
        Algorithm type, used to select the default checkpointing preset.
    trainer_params : dict, optional
        Parameters for Lightning Trainer class, by default None.
    logger : {"wandb", "tensorboard", "none"}, optional
        Logger to use, by default "none".
    checkpoint_params : dict, optional
        Parameters for the checkpoint callback, by default None. If None, then default
        parameters are applied based on the algorithm.
    early_stopping_params : dict, optional
        Parameters for the early stopping callback, by default None. If None, then
        default parameters are applied based on the algorithm.
    monitor_metric : str, optional
        Metric to monitor for early stopping, by default "val_loss".

    Returns
    -------
    dict
        Training configuration dictionary with the specified parameters.
    """
    # user parameters take precedence over defaults
    # since resulting checkpointing behaviour depends on complex interactions between
    # parameters, we keep either user defined or the defaults
    if checkpoint_params is None:
        # select default checkpointing preset based on algorithm
        default_ckpt_preset = (
            SupervisedCheckpointing
            if algorithm == "care"
            # since Noise2Noise is comparing noisy pixels to other noisy pixels, it
            # cannot be monitored based on a metric, we use the self-supervised preset
            else SelfSupervisedCheckpointing
        )
        default_checkpoint = asdict(default_ckpt_preset())
        checkpoint_params = default_checkpoint

    if early_stopping_params is None:
        # early stopping is only relevant for supervised algorithms, we set it to None
        # for self-supervised ones
        early_stopping_params = (
            {
                "monitor": monitor_metric,
                "mode": "min",
            }
            if algorithm == "care"
            else None
        )

    return {
        "trainer_params": {} if trainer_params is None else trainer_params,
        "logger": None if logger == "none" else logger,
        "checkpoint_params": checkpoint_params,
        "early_stopping_params": early_stopping_params,
    }


def default_training_factory(validated_dict: dict[str, Any]) -> NGTrainingConfig:
    """Default training configuration constructor.

    Parameters
    ----------
    validated_dict : dict
        Validated configuration dictionary, used to set default training parameters
        based on the algorithm configuration. This is expected to be passed by Pydantic
        when calling the default constructor.

    Returns
    -------
    NGTrainingConfig
        Training configuration with the specified parameters.
    """
    key = "algorithm_config"

    if key not in validated_dict:
        raise ValueError(
            "Algorithm configuration is required to set default training parameters, "
            "but the algorithm configuration was not found during validation. The most "
            "likely cause is that the validation of the algorithm configuration failed."
            " Try validating it seprately, for instance with "
            "`instantiate_algorithm_config`."
        )
    algorithm = validated_dict[key].algorithm

    # N2V algorithm has a monitor_metric parameter
    monitor_metric = getattr(validated_dict[key], "monitor_metric", "val_loss")

    return NGTrainingConfig(
        **default_training_dict(
            algorithm=algorithm,
            monitor_metric=monitor_metric,
        )
    )


class NGTrainingConfig(BaseModel):
    """
    Parameters related to the training.

    By default, `checkpoint_params` and `early_stopping_params` have presets based on
    whether the algorithm is supervised (CARE) or not (Noise2Void and by extension
    Noise2Noise). In the case of CARE, the top 3 checkpoints are saved based on
    `val_loss`. For the self-supervised algorithms, checkpoints are saved every 10
    epochs. In both cases, the last checkpoint is saved. Early stopping is disabled
    for self-supervised algorithms.

    Attributes
    ----------
    trainer_params : dict
        Parameters passed to the PyTorch Lightning Trainer class.
    logger : Literal["wandb", "tensorboard"] | None
        Additional Logger to use during training. If None, no logger will be used.
        Note that the `CAREamist` uses the `csv` logger regardless of the value of this
        field.
    checkpoint_params : dict[str, Any]
        Checkpoint callback parameters, following PyTorch Lightning Checkpoint
        callback.
    early_stopping_params : dict[str, Any] | None
        Early stopping callback parameters, following PyTorch Lightning Checkpoint
        callback.
    """

    # Pydantic class configuration
    model_config = ConfigDict(
        validate_assignment=True,
    )

    trainer_params: dict = Field(default={})
    """Parameters passed to the PyTorch Lightning Trainer class"""

    logger: Literal["wandb", "tensorboard"] | None = None
    """Logger to use during training. If None, no logger will be used. Available
    loggers are defined in SupportedLogger."""

    # Only basic callbacks - they may have different defaults for different algorithms
    checkpoint_params: dict[str, Any] = Field(default_factory=dict)
    """Checkpoint callback parameters, following PyTorch Lightning Checkpoint
    callback."""

    early_stopping_params: dict[str, Any] | None = Field(default_factory=dict)
    """Early stopping callback parameters, following PyTorch Lightning Checkpoint
    callback."""

    def __str__(self) -> str:
        """Pretty string reprensenting the configuration.

        Returns
        -------
        str
            Pretty string.
        """
        return pformat(self.model_dump())
