"""Module containing the protocol that defines the WriteStrategy interface."""

from pathlib import Path
from typing import Any, Optional, Protocol, Sequence

from pytorch_lightning import LightningModule, Trainer


class WriteStrategy(Protocol):
    """
    Protocol for write strategy classes.

    A `WriteStrategy` is an object that will be an attribute in the
    `PredictionWriterCallback`; it will determine how predictions will be saved.
    `WriteStrategy`s must be interchangeable so they must follow the interface set out
    in this `Protocol` class.
    """

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
    ) -> None:
        """
        Set file information after the `WriteStrategy` has been initialized.

        Parameters
        ----------
        write_filenames : list[str]
            A list of filenames to save to.
        n_samples_per_file : list[int]
            The number of samples that will be saved within each file. Each element in
            the list will correspond to the equivelant file in `write_filenames`.
            (Should most likely mirror the input file structure).
        """

    def reset(self) -> None:
        """
        Reset internal state (attributes) of a `WriteStrategy` instance.

        This is to unexpected behaviour if a `WriteStrategy` instance is used twice.
        """
