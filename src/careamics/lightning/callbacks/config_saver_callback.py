"""Lightning callback for storing CAREamics configuration in checkpoints."""

from typing import Any

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from careamics.config.data.data_config import DataConfig
from careamics.config.lightning.training_configuration import TrainingConfig


class ConfigSaverCallback(Callback):
    """
    Callback to save CAREamics configuration in Lightning checkpoints.

    This callback automatically stores CAREamics version, experiment name,
    and training configuration in the checkpoint file for reproducibility.
    If a training data configuration is provided, it is also persisted into the
    checkpoint independently of the datamodule currently active on the trainer.

    Parameters
    ----------
    careamics_version : str
        Version of CAREamics used for training.
    experiment_name : str
        Name of the experiment.
    training_config : TrainingConfig
        Training configuration to store in checkpoint.
    data_config : DataConfig | None, default=None
        Training data configuration to store in checkpoint. If provided, it is
        written into the checkpoint regardless of which datamodule is active on
        the trainer, ensuring the checkpoint keeps the training data config even
        when saved after prediction.

    Attributes
    ----------
    careamics_version : str
        Version of CAREamics used for training.
    experiment_name : str
        Name of the experiment.
    training_config : TrainingConfig
        Training configuration to store in checkpoint.
    data_config : DataConfig | None
        Training data configuration to store in checkpoint.
    """

    def __init__(
        self,
        careamics_version: str,
        experiment_name: str,
        training_config: TrainingConfig,
        data_config: DataConfig | None = None,
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
        data_config : DataConfig | None, default=None
            Training data configuration to store in checkpoint. If provided, it is
            written into the checkpoint regardless of which datamodule is active on
            the trainer.
        """
        super().__init__()
        self.careamics_version = careamics_version
        self.experiment_name = experiment_name
        self.training_config = training_config
        self.data_config = data_config

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

        # Persist the training data config if provided
        if self.data_config is not None:
            checkpoint.setdefault("datamodule_hyper_parameters", {})
            checkpoint["datamodule_hyper_parameters"]["data_config"] = (
                self.data_config.model_dump(mode="json")
            )
