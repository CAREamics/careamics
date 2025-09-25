"""Callback for dataset reshuffling during training.

This callback handles reshuffling operations after each epoch to ensure
proper data randomization between epochs.
"""

import pytorch_lightning as L

from careamics.dataset_ng.dataset import CareamicsDataset
from careamics.dataset_ng.patching_strategies import FixedRandomPatchingStrategy


class DatasetReshuffleCallback(L.Callback):
    """Callback for reshuffling datasets after every epoch."""

    def __init__(self) -> None:
        """Initialize the dataset reshuffle callback."""
        super().__init__()

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """
        Reshuffle the dataset at the end of each training epoch.

        Parameters
        ----------
        trainer : L.Trainer
            The Lightning trainer instance.
        pl_module : L.LightningModule
            The Lightning module being trained.
        """
        if trainer.datamodule is None or not hasattr(
            trainer.datamodule, "train_dataset"
        ):
            return

        training_dataset = trainer.datamodule.train_dataset

        # Handle CareamicsDataset with FixedRandomPatchingStrategy
        if isinstance(training_dataset, CareamicsDataset):
            if isinstance(
                training_dataset.patching_strategy, FixedRandomPatchingStrategy
            ):
                training_dataset.patching_strategy.resample_patches()
