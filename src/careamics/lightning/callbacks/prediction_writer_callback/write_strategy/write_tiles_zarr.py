"""Module containing a write strategy for writing tiles directly to zarr datasets."""

from pathlib import Path
from typing import Any, Optional, Sequence

from pytorch_lightning import LightningModule, Trainer

from .protocol import WriteStrategy


class WriteTilesZarr(WriteStrategy):
    """Strategy to write tiles to Zarr file."""

    def write_batch(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
        dirpath: Path,
    ) -> None:
        """
        Write tiles to zarr file.

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

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset internal attributes."""
        raise NotImplementedError
