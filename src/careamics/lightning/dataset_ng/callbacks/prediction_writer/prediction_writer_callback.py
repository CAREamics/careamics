"""Module containing `PredictionWriterCallback` class."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import BasePredictionWriter

from careamics.dataset_ng.dataset import ImageRegionData
from careamics.file_io.write.get_func import SupportedWriteType, WriteFunc
from careamics.lightning.dataset_ng.prediction import decollate_image_region_data
from careamics.utils import get_logger

from .write_strategy import WriteStrategy
from .write_strategy_factory import create_write_strategy

logger = get_logger(__name__)


class PredictionWriterCallback(BasePredictionWriter):
    """
    PyTorch Lightning callback to save predictions.

    A `WriteStrategy` must be provided at instantiation or later via
    `set_writing_strategy`.

    Parameters
    ----------
    dirpath : Path or str, default="predictions"
        The path to the directory where prediction outputs will be saved. If
        `dirpath` is not absolute it is assumed to be relative to current working
        directory.
    write_strategy : WriteStrategy or None, default=None
        A strategy for writing predictions.

    Attributes
    ----------
    writing_predictions : bool
        If writing predictions is turned on or off.
    dirpath : pathlib.Path, default=""
        The path to the directory where prediction outputs will be saved. If
        `dirpath` is not absolute it is assumed to be relative to current working
        directory.
    write_strategy : WriteStrategy or None
            A strategy for writing predictions.
    """

    def __init__(
        self,
        dirpath: Path | str = "",
        write_strategy: WriteStrategy | None = None,
    ):
        """
        Constructor.

        A `WriteStrategy` must be provided at instantiation or later via
        `set_writing_strategy`.

        Parameters
        ----------
        dirpath : pathlib.Path or str, default="predictions"
            The path to the directory where prediction outputs will be saved. If
            `dirpath` is not absolute it is assumed to be relative to current working
            directory.
        write_strategy : WriteStrategy or None, default=None
            A strategy for writing predictions.
        """
        super().__init__(write_interval="batch")

        self.writing_predictions = True  # flag to turn off predictions

        # forward declaration
        self.write_strategy: WriteStrategy
        if write_strategy is not None:  # avoid `WriteStrategy | None` type
            self.write_strategy = write_strategy

        self.dirpath: Path

        # if a dirpath is provided, initialize it
        # in some cases (e.g. zarr), destination is provided by the zarr store path
        if dirpath != "":
            self._init_dirpath(dirpath)

    def disable_writing(self, disable_writing: bool) -> None:
        """Disable writing.

        Parameters
        ----------
        disable_writing : bool
            If writing predictions should be disabled.
        """
        self.writing_predictions = disable_writing

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
            if self.dirpath is not None:
                # make prediction output directory
                logger.info("Making prediction output directory.")
                self.dirpath.mkdir(parents=True, exist_ok=True)

    def set_writing_strategy(
        self,
        write_type: SupportedWriteType,
        tiled: bool,
        write_func: WriteFunc | None = None,
        write_extension: str | None = None,
        write_func_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Set the writing strategy.

        Must be called before writing predictions.

        Parameters
        ----------
        write_type : SupportedWriteType
            The type of writing to perform.
        tiled : bool
            Whether to write in tiled format.
        write_func : WriteFunc or None, default=None
            A custom writing function.
        write_extension : str or None, default=None
            The file extension to use when writing files.
        write_func_kwargs : dict of str to Any, default=None
            Additional keyword arguments to pass to `write_func`.
        """
        self.write_strategy = create_write_strategy(
            write_type=write_type,
            tiled=tiled,
            write_func=write_func,
            write_extension=write_extension,
            write_func_kwargs=write_func_kwargs,
        )

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: ImageRegionData,
        batch_indices: Sequence[int] | None,
        batch: ImageRegionData,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """
        Write predictions at the end of a batch.

        Writing method is determined by the attribute `write_strategy`.

        Parameters
        ----------
        trainer : Trainer
            PyTorch Lightning trainer.
        pl_module : LightningModule
            PyTorch Lightning module.
        prediction : ImageRegionData
            Prediction outputs of `batch`.
        batch_indices : sequence of Any, optional
            Batch indices.
        batch : ImageRegionData
            Input batch.
        batch_idx : int
            Batch index.
        dataloader_idx : int
            Dataloader index.
        """
        # if writing prediction is turned off
        if not self.writing_predictions:
            return

        if self.write_strategy is not None:
            assert prediction is not None
            predictions = decollate_image_region_data(prediction)

            self.write_strategy.write_batch(
                dirpath=self.dirpath,
                predictions=predictions,
            )
        else:
            raise RuntimeError(
                "No write strategy defined for `PredictionWriterCallback`, cannot write"
                " predictions. Call `set_writing_strategy` to pass a write strategy."
            )
