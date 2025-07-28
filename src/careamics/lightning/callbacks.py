"""Lightning callbacks for training."""

import pytorch_lightning as L
from pytorch_lightning.callbacks import Callback


class DataStatsCallback(Callback):
    """Callback to update model's data statistics from datamodule.

    This callback ensures that the model has access to the data statistics (mean and std)
    calculated by the datamodule before training starts.
    """

    def setup(self, trainer: L.Trainer, module: L.LightningModule, stage: str) -> None:
        """Called when trainer is setting up."""
        if stage == "fit":
            # Get data statistics from datamodule
            data_mean, data_std = trainer.datamodule.get_data_stats()

            # Set data statistics in the model's likelihood module
            module.model.likelihood.data_mean = data_mean
            module.model.likelihood.data_std = data_std
