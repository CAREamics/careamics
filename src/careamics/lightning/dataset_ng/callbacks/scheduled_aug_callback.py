"""Lightning callback for updating epoch on scheduled augmentation transforms."""

from typing import Any

import pytorch_lightning as L


class ScheduledAugCallback(L.Callback):
    """Update the epoch counter on any scheduled-augmentation transforms.

    At the start of each training epoch this callback iterates over all
    transforms in ``trainer.datamodule.train_dataset.transforms`` and calls
    ``set_epoch(current_epoch)`` on any that expose that method.

    This is the epoch-propagation half of the deterministic complementary
    augmentation feature.  The other half (sample-index propagation) happens
    inside :meth:`CareamicsDataset.__getitem__`.

    Usage
    -----
    Add this callback to your ``Trainer``::

        from careamics.lightning.dataset_ng.callbacks import ScheduledAugCallback

        trainer = L.Trainer(callbacks=[ScheduledAugCallback()])

    The callback is a no-op when no scheduled augmentation transform is present
    in the training dataset, so it is safe to include unconditionally.
    """

    def on_train_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Set the current epoch on any scheduled augmentation transforms.

        Parameters
        ----------
        trainer : pytorch_lightning.Trainer
            The current trainer instance.
        pl_module : pytorch_lightning.LightningModule
            The current Lightning module (unused).
        """
        datamodule = trainer.datamodule
        if datamodule is None:
            return

        train_dataset = getattr(datamodule, "train_dataset", None)
        if train_dataset is None:
            return

        transforms = getattr(train_dataset, "transforms", None)
        if transforms is None:
            return

        transform_list = getattr(transforms, "transforms", [])
        for t in transform_list:
            if hasattr(t, "set_epoch"):
                t.set_epoch(trainer.current_epoch)

    # ------------------------------------------------------------------
    # Type annotation helpers (unused at runtime, silences mypy)
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        """Constructor."""
        super().__init__()

    def state_dict(self) -> dict[str, Any]:
        """Return an empty state dict (no persistent state)."""
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """No-op: this callback has no persistent state."""
