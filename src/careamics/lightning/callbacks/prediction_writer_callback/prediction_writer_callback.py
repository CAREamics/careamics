"""Module containing `PredictionWriterCallback` class."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Union

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
from torch.utils.data import DataLoader

from careamics.dataset import (
    IterablePredDataset,
    IterableTiledPredDataset,
)
from careamics.file_io import SupportedWriteType, WriteFunc
from careamics.utils import get_logger

from .write_strategy import WriteStrategy
from .write_strategy_factory import create_write_strategy

logger = get_logger(__name__)

ValidPredDatasets = Union[IterablePredDataset, IterableTiledPredDataset]


class PredictionWriterCallback(BasePredictionWriter):
    """
    A PyTorch Lightning callback to save predictions.

    Parameters
    ----------
    write_strategy : WriteStrategy
        A strategy for writing predictions.
    dirpath : Path or str, default="predictions"
        The path to the directory where prediction outputs will be saved. If
        `dirpath` is not absolute it is assumed to be relative to current working
        directory.

    Attributes
    ----------
    write_strategy : WriteStrategy
        A strategy for writing predictions.
    dirpath : pathlib.Path, default="predictions"
        The path to the directory where prediction outputs will be saved. If
        `dirpath` is not absolute it is assumed to be relative to current working
        directory.
    writing_predictions : bool
        If writing predictions is turned on or off.
    """

    def __init__(
        self,
        write_strategy: WriteStrategy,
        dirpath: Union[Path, str] = "predictions",
    ):
        """
        A PyTorch Lightning callback to save predictions.

        Parameters
        ----------
        write_strategy : WriteStrategy
            A strategy for writing predictions.
        dirpath : pathlib.Path or str, default="predictions"
            The path to the directory where prediction outputs will be saved. If
            `dirpath` is not absolute it is assumed to be relative to current working
            directory.
        """
        super().__init__(write_interval="batch")

        # Toggle for CAREamist to switch off saving if desired
        self.writing_predictions: bool = True

        self.write_strategy: WriteStrategy = write_strategy

        # forward declaration
        self.dirpath: Path
        # attribute initialisation
        self._init_dirpath(dirpath)

    @classmethod
    def from_write_func_params(
        cls,
        write_type: SupportedWriteType,
        tiled: bool,
        write_func: WriteFunc | None = None,
        write_extension: str | None = None,
        write_func_kwargs: dict[str, Any] | None = None,
        dirpath: Union[Path, str] = "predictions",
    ) -> PredictionWriterCallback:  # TODO: change type hint to self (find out how)
        """
        Initialize a `PredictionWriterCallback` from write function parameters.

        This will automatically create a `WriteStrategy` to be passed to the
        initialization of `PredictionWriterCallback`.

        Parameters
        ----------
        write_type : {"tiff", "custom"}
            The data type to save as, includes custom.
        tiled : bool
            Whether the prediction will be tiled or not.
        write_func : WriteFunc, optional
            If a known `write_type` is selected this argument is ignored. For a custom
            `write_type` a function to save the data must be passed. See notes below.
        write_extension : str, optional
            If a known `write_type` is selected this argument is ignored. For a custom
            `write_type` an extension to save the data with must be passed.
        write_func_kwargs : dict of {{str: any}}, optional
            Additional keyword arguments to be passed to the save function.
        dirpath : pathlib.Path or str, default="predictions"
            The path to the directory where prediction outputs will be saved. If
            `dirpath` is not absolute it is assumed to be relative to current working
            directory.

        Returns
        -------
        PredictionWriterCallback
            Callback for writing predictions.
        """
        write_strategy = create_write_strategy(
            write_type=write_type,
            tiled=tiled,
            write_func=write_func,
            write_extension=write_extension,
            write_func_kwargs=write_func_kwargs,
        )
        return cls(write_strategy=write_strategy, dirpath=dirpath)

    def _init_dirpath(self, dirpath):
        """
        Initialize directory path. Should only be called from `__init__`.

        Parameters
        ----------
        dirpath : pathlib.Path
            See `__init__` description.
        """
        dirpath = Path(dirpath)
        if not dirpath.is_absolute():
            dirpath = Path.cwd() / dirpath
            logger.warning(
                "Prediction output directory is not absolute, absolute path assumed to"
                f"be '{dirpath}'"
            )
        self.dirpath = dirpath

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """
        Create the prediction output directory when predict begins.

        Called when fit, validate, test, predict, or tune begins.

        Parameters
        ----------
        trainer : Trainer
            PyTorch Lightning trainer.
        pl_module : LightningModule
            PyTorch Lightning module.
        stage : str
            Stage of training e.g. 'predict', 'fit', 'validate'.
        """
        super().setup(trainer, pl_module, stage)
        if stage == "predict":
            # make prediction output directory
            logger.info("Making prediction output directory.")
            self.dirpath.mkdir(parents=True, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: Any,  # TODO: change to expected type
        batch_indices: Sequence[int] | None,
        batch: Any,  # TODO: change to expected type
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """
        Write predictions at the end of a batch.

        The method of prediction is determined by the attribute `write_strategy`.

        Parameters
        ----------
        trainer : Trainer
            PyTorch Lightning trainer.
        pl_module : LightningModule
            PyTorch Lightning module.
        prediction : Any
            Prediction outputs of `batch`.
        batch_indices : sequence of Any, optional
            Batch indices.
        batch : Any
            Input batch.
        batch_idx : int
            Batch index.
        dataloader_idx : int
            Dataloader index.
        """
        # if writing prediction is turned off
        if not self.writing_predictions:
            return

        dataloaders: Union[DataLoader, list[DataLoader]] = trainer.predict_dataloaders
        dataloader: DataLoader = (
            dataloaders[dataloader_idx]
            if isinstance(dataloaders, list)
            else dataloaders
        )
        dataset: ValidPredDatasets = dataloader.dataset
        if not (
            isinstance(dataset, IterablePredDataset)
            or isinstance(dataset, IterableTiledPredDataset)
        ):
            # Note: Error will be raised before here from the source type
            # This is for extra redundancy of errors.
            raise TypeError(
                "Prediction dataset has to be `IterableTiledPredDataset` or "
                "`IterablePredDataset`. Cannot be `InMemoryPredDataset` because "
                "filenames are taken from the original file."
            )

        self.write_strategy.write_batch(
            trainer=trainer,
            pl_module=pl_module,
            prediction=prediction,
            batch_indices=batch_indices,
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
            dirpath=self.dirpath,
        )
