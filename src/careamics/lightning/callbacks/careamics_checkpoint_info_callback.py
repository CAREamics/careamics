from typing import Any

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

from careamics.config import TrainingConfig


class CareamicsCheckpointInfo(Callback):

    def __init__(
        self,
        careamics_version: str,
        experiment_name: str,
        training_config: TrainingConfig,
    ):
        super().__init__()
        self.careamics_version = careamics_version
        self.experiment_name = experiment_name
        self.training_config = training_config

    def on_save_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict[str, Any]
    ) -> None:
        checkpoint["careamics_info"] = {
            "version": self.careamics_version,
            "experiment_name": self.experiment_name,
            "training_config": self.training_config.model_dump(mode="json"),
        }
