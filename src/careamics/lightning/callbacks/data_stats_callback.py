"""Data statistics callback."""

import lightning as L
from lightning.pytorch.callbacks import Callback


class DataStatsCallback(Callback):
    """Callback to update model's data statistics from datamodule.

    This callback ensures that the model has access to the data statistics (mean, std)
    calculated by the datamodule before training starts.
    """

    def setup(self, trainer: L.Trainer, module: L.LightningModule, stage: str) -> None:
        """Called when trainer is setting up.

        Parameters
        ----------
        trainer : Lightning.Trainer
            PyTorch Lightning trainer.
        module : Lightning.LightningModule
            Lightning module.
        stage : str
            Current stage (fit, validate, test, or predict).
        """
        if stage == "fit":
            # Get data statistics from datamodule
            datamodule = getattr(trainer, "datamodule", None)
            if datamodule is None:
                raise RuntimeError("Trainer has no datamodule attached.")
            (data_mean, data_std), _ = datamodule.get_data_stats()

            # Set data statistics in the model's likelihood module
            likelihood = getattr(module, "noise_model_likelihood", None)
            if likelihood is None:
                raise RuntimeError(
                    "Lightning module has no `noise_model_likelihood`; "
                    "DataStatsCallback only supports noise-model VAE modules."
                )
            likelihood.set_data_stats(
                data_mean=data_mean["target"], data_std=data_std["target"]
            )
