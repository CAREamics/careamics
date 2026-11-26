"""Module containing `PredictionWriterCallback` class."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter

from careamics.dataset.image_region_data import ImageRegionData
from careamics.image_io.write.get_func import SupportedWriteType, WriteFunc
from careamics.lightning.prediction import (
    decollate_image_region_data,
    prediction_region,
    uncertainty_region,
)
from careamics.utils import get_logger

from .file_path_utils import common_source_base
from .write_strategy import WriteStrategy
from .write_strategy_factory import create_write_strategy

logger = get_logger(__name__)


class PredictionWriterCallback(BasePredictionWriter):
    """
    PyTorch Lightning callback to save predictions.

    A `WriteStrategy` must be provided at instantiation or later via
    `set_writing_strategy`. This allows passing the callback to the Lightning `Trainer`
    before knowing what writing strategy (e.g. tiling or file type) will be used.

    By default the prediction writer is enabled, but it can be disabled by passing
    `enable_writing=False` at instantiation or by calling `enable_writing(False)`.
    Subsequently, writing can be re-enabled by calling `enable_writing(True)`.

    Parameters
    ----------
    dirpath : Path or str, default="predictions"
        The path to the directory where prediction outputs will be saved. If
        `dirpath` is not absolute it is assumed to be relative to current working
        directory.
    write_strategy : WriteStrategy or None, default=None
        A strategy for writing predictions.
    enable_writing : bool, default=True
        If writing predictions should be enabled by default.
    uncertainty_write_strategy : WriteStrategy or None, default=None
        A strategy for writing uncertainty estimates. If None, uncertainty is not
        written.

    Attributes
    ----------
    is_enabled : bool
        Whether writing predictions is enabled.
    dirpath : pathlib.Path, default=""
        The path to the directory where prediction outputs will be saved. If
        `dirpath` is not absolute it is assumed to be relative to current working
        directory.
    write_strategy : WriteStrategy or None
            A strategy for writing predictions.
    uncertainty_write_strategy : WriteStrategy or None
        A strategy for writing uncertainty estimates.
    """

    def __init__(
        self,
        dirpath: Path | str = "",
        write_strategy: WriteStrategy | None = None,
        enable_writing: bool = True,
        uncertainty_write_strategy: WriteStrategy | None = None,
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
        enable_writing : bool, default=True
            If writing predictions should be enabled by default.
        uncertainty_write_strategy : WriteStrategy or None, default=None
            A strategy for writing uncertainty estimates. If None, uncertainty is not
            written.
        """
        super().__init__(write_interval="batch")

        self.is_enabled = enable_writing
        self.write_strategy = write_strategy
        self.uncertainty_write_strategy = uncertainty_write_strategy

        self.dirpath: Path

        # if a dirpath is provided, initialize it
        # in some cases (e.g. zarr), destination is provided by the zarr store path
        if dirpath != "":
            self._init_dirpath(dirpath)

    def enable_writing(self, enable_writing: bool) -> None:
        """Enable or disable writing.

        Parameters
        ----------
        enable_writing : bool
            If writing predictions should be enabled.
        """
        self.is_enabled = enable_writing

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
                "Prediction output directory is not absolute, absolute path assumed to "
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
                logger.info(f"Creating prediction output directory: '{self.dirpath}'")
                self.dirpath.mkdir(parents=True, exist_ok=True)

            # preserve the source directory structure in the output, so that
            # identically named files in different directories do not collide
            if self.is_enabled and self.write_strategy is not None:
                source_paths = getattr(trainer.datamodule, "predict_source_paths", [])
                source_base = common_source_base(source_paths)
                self.write_strategy.set_source_base(source_base)
                if self.uncertainty_write_strategy is not None:
                    self.uncertainty_write_strategy.set_source_base(source_base)

    def set_writing_strategy(
        self,
        write_type: SupportedWriteType,
        tiled: bool,
        write_func: WriteFunc | None = None,
        write_extension: str | None = None,
        write_func_kwargs: dict[str, Any] | None = None,
        write_uncertainty: bool = False,
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
        write_uncertainty : bool, default=False
            If True, also create a strategy that writes uncertainty estimates.

        Raises
        ------
        NotImplementedError
            If `write_uncertainty` is True and `write_type` is `"zarr"`.
        """
        if write_uncertainty and write_type == "zarr":
            raise NotImplementedError(
                'Writing uncertainty to Zarr is not supported. Use write_type="tiff".'
            )

        self.write_strategy = create_write_strategy(
            write_type=write_type,
            tiled=tiled,
            write_func=write_func,
            write_extension=write_extension,
            write_func_kwargs=write_func_kwargs,
        )
        if write_uncertainty:
            self.uncertainty_write_strategy = create_write_strategy(
                write_type=write_type,
                tiled=tiled,
                write_func=write_func,
                write_extension=write_extension,
                write_func_kwargs=write_func_kwargs,
                filename_postfix="_uncertainty",
            )
        else:
            self.uncertainty_write_strategy = None

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: ImageRegionData | tuple[ImageRegionData, ImageRegionData | None],
        batch_indices: Sequence[int] | None,
        batch: ImageRegionData,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """
        Write predictions at the end of a batch.

        Writing method is determined by the attribute `write_strategy`. If
        `uncertainty_write_strategy` is set and the prediction has an uncertainty
        estimate, that estimate is written to a separate file.

        Parameters
        ----------
        trainer : Trainer
            PyTorch Lightning trainer.
        pl_module : LightningModule
            PyTorch Lightning module.
        prediction : ImageRegionData or tuple of ImageRegionData
            Prediction outputs of `batch`, optionally paired with the uncertainty.
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
        if not self.is_enabled:
            return

        if self.write_strategy is not None:
            assert prediction is not None
            predictions = decollate_image_region_data(prediction_region(prediction))

            self.write_strategy.write_batch(
                dirpath=self.dirpath,
                predictions=predictions,
            )

            if self.uncertainty_write_strategy is not None:
                uncertainty = uncertainty_region(prediction)
                if uncertainty is not None:
                    uncertainties = decollate_image_region_data(uncertainty)
                    self.uncertainty_write_strategy.write_batch(
                        dirpath=self.dirpath,
                        predictions=uncertainties,
                    )
        else:
            raise RuntimeError(
                "No write strategy defined for `PredictionWriterCallback`, cannot write"
                " predictions. Call `set_writing_strategy` to pass a write strategy."
            )
