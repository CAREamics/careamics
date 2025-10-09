"""Data statistics callback."""

import pytorch_lightning as L
from pytorch_lightning.callbacks import Callback


class DataStatsCallback(Callback):
    """Callback to update model's data statistics from datamodule.

    This callback ensures that the model has access to the data statistics (mean and
    std) calculated by the datamodule before training starts.
    """

    def setup(self, trainer: L.Trainer, module: L.LightningModule, stage: str) -> None:
        """Called when trainer is setting up.

        Parameters
        ----------
        trainer : Lightning.Trainer
            The trainer instance.
        module : Lightning.LightningModule
            The model being trained.
        stage : str
            The current stage of training (e.g., 'fit', 'validate', 'test', 'predict').
        """
        if stage == "fit":
            # Get data statistics from datamodule
            (data_mean, data_std), _ = trainer.datamodule.get_data_stats()

            # Set data statistics in the model's likelihood module
            module.noise_model_likelihood.set_data_stats(
                data_mean=data_mean["target"], data_std=data_std["target"]
            )
