"""Validation helpers for checkpointing parameters."""

from careamics.config.lightning.training_configuration import TrainingConfig
from careamics.utils import get_logger
from careamics.utils.checkpointing import get_default_checkpoint_type

logger = get_logger(__name__)


def warn_self_supervised_checkpointing_params(
    training_config: TrainingConfig,
) -> TrainingConfig:
    """Warn when best checkpointing is enabled for self-supervised training.

    Parameters
    ----------
    training_config : TrainingConfig
        Training configuration to validate.

    Returns
    -------
    TrainingConfig
        Validated training configuration.
    """
    if get_default_checkpoint_type(training_config.checkpoint_params) == "best":
        logger.warning(
            "Best checkpointing is enabled for a self-supervised algorithm, but its "
            "validation loss may not correlate with model performance. The last "
            "checkpoint is recommended.",
            stacklevel=2,
        )
    return training_config


def warn_supervised_checkpointing_params(
    training_config: TrainingConfig,
) -> TrainingConfig:
    """Warn when best checkpointing is disabled for supervised training.

    Parameters
    ----------
    training_config : TrainingConfig
        Training configuration to validate.

    Returns
    -------
    TrainingConfig
        Validated training configuration.
    """
    if get_default_checkpoint_type(training_config.checkpoint_params) == "last":
        logger.warning(
            "Best checkpointing is disabled for a supervised algorithm. It is "
            "recommended to monitor the best checkpoint according to a metric on the "
            "validation data.",
            stacklevel=2,
        )
    return training_config
