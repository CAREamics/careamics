"""Module containing the protocol that defines the WriteStrategy interface."""

from pathlib import Path
from typing import Any, Optional, Protocol, Sequence

from pytorch_lightning import LightningModule, Trainer


class WriteStrategy(Protocol):
    """Protocol for write strategy classes."""

    def write_batch(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: Any,  # TODO: change to expected type
        batch_indices: Optional[Sequence[int]],
        batch: Any,  # TODO: change to expected type
        batch_idx: int,
        dataloader_idx: int,
        dirpath: Path,
    ) -> None:
        """
        WriteStrategy subclasses must contain this function to write a batch.

        Parameters
        ----------
        trainer : Trainer
            PyTorch Lightning Trainer.
        pl_module : LightningModule
            PyTorch Lightning LightningModule.
        prediction : Any
            Predictions on `batch`.
        batch_indices : sequence of int
            Indices identifying the samples in the batch.
        batch : Any
            Input batch.
        batch_idx : int
            Batch index.
        dataloader_idx : int
            Dataloader index.
        dirpath : Path
            Path to directory to save predictions to.
        """

    def set_file_data(
        self, write_filenames: list[str], n_samples_per_file: list[int]
    ) -> None: ...

    def reset(self) -> None:
        """
        Reset internal attributes of a `WriteStrategy` instance.

        This is to unexpected behaviour if a `WriteStrategy` instance is used twice.
        """
