"""Lightning callback for storing CAREamics configuration in checkpoints."""

from typing import Any

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

from careamics.config import TrainingConfig


class CareamicsCheckpointInfo(Callback):
    """
    Callback to save CAREamics configuration in Lightning checkpoints.

    This callback automatically stores CAREamics version, experiment name,
    and training configuration in the checkpoint file for reproducibility.

    Parameters
    ----------
    careamics_version : str
        Version of CAREamics used for training.
    experiment_name : str
        Name of the experiment.
    training_config : TrainingConfig
        Training configuration to store in checkpoint.

    Attributes
    ----------
    careamics_version : str
        Version of CAREamics used for training.
    experiment_name : str
        Name of the experiment.
    training_config : TrainingConfig
        Training configuration to store in checkpoint.
    """

    def __init__(
        self,
        careamics_version: str,
        experiment_name: str,
        training_config: TrainingConfig,
    ):
        """
        Initialize the callback.

        Parameters
        ----------
        careamics_version : str
            Version of CAREamics used for training.
        experiment_name : str
            Name of the experiment.
        training_config : TrainingConfig
            Training configuration to store in checkpoint.
        """
        super().__init__()
        self.careamics_version = careamics_version
        self.experiment_name = experiment_name
        self.training_config = training_config

    def on_save_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict[str, Any]
    ) -> None:
        """
        Lightning hook called when saving a checkpoint.

        Adds CAREamics configuration to the checkpoint dictionary.

        Parameters
        ----------
        trainer : Trainer
            Lightning trainer instance.
        pl_module : LightningModule
            Lightning module being trained.
        checkpoint : dict[str, Any]
            Checkpoint dictionary to modify.
        """
        checkpoint["careamics_info"] = {
            "version": self.careamics_version,
            "experiment_name": self.experiment_name,
            "training_config": self.training_config.model_dump(mode="json"),
        }
