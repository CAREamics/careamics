"""Module containing PredictionWriterCallback class."""

from pathlib import Path
from typing import Any, Optional, Sequence, Union

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import BasePredictionWriter

from careamics.config.support import SupportedData
from careamics.file_io import WriteFunc, get_write_func
from careamics.utils import get_logger

logger = get_logger(__name__)


class PredictionWriterCallback(BasePredictionWriter):
    """
    A PyTorch Lightning callback to save predictions.

    Parameters
    ----------
    save_type : SupportedData or str, default="tiff"
        The data type to save as, includes custom.
    save_func : WriteFunc, optional
        If a known `save_type` is selected this argument is ignored. For a custom
        `save_type` a function to save the data must be passed. See notes below.
    save_extension : str, optional
        If a known `save_type` is selected this argument is ignored. For a custom
        `save_type` an extension to save the data with must be passed.
    save_func_kwargs : dict of {str: any}, optional
        Additional keyword arguments to be passed to the save function.
    dirpath : pathlib.Path or str, default="predictions"
        Directory to save outputs to. If `dirpath is not absolute it is assumed to
        be relative to current working directory. Nested directories will not be
        automatically created.

    Attributes
    ----------
    save_predictions : bool
        A toggle to optionally switch off prediction saving.
    save_type : SupportedData
        The data type of the saved data.
    save_func : WriteFunc
        The function for saving data.
    save_extension: str
        The extension that will be added to the save paths.
    save_func_kwargs : dict of {str, any}
        Additional keyword arguments that will be passed to the save function.
    dirpath : pathlib.Path
        The path to the directory where prediction outputs will be saved.
    """

    def __init__(
        self,
        save_type: Union[SupportedData, str] = "tiff",
        save_func: Optional[WriteFunc] = None,
        save_extension: Optional[str] = None,
        save_func_kwargs: Optional[dict[str, Any]] = None,
        dirpath: Union[Path, str] = "predictions",
    ):
        """
        A PyTorch Lightning callback to save predictions.

        Parameters
        ----------
        save_type : SupportedData or str, default="tiff"
            The data type to save as, includes custom.
        save_func : WriteFunc, optional
            If a known `save_type` is selected this argument is ignored. For a custom
            `save_type` a function to save the data must be passed. See notes below.
        save_extension : str, optional
            If a known `save_type` is selected this argument is ignored. For a custom
            `save_type` an extension to save the data with must be passed.
        save_func_kwargs : dict of {str: any}, optional
            Additional keyword arguments to be passed to the save function.
        dirpath : pathlib.Path or str, default="predictions"
            Directory to save outputs to. If `dirpath is not absolute it is assumed to
            be relative to current working directory. Nested directories will not be
            automatically created.

        Raises
        ------
        ValueError
            If `save_type="custom"` but `save_func` has not been given.
        ValueError
            If `save_type="custom"` but `save_extension` has not been given.

        Notes
        -----
        The `save_func` function signature must match that of the example below
            ```
            save_func(file_path: Path, img: NDArray, *args, **kwargs) -> None: ...
            ```

        The `save_func_kwargs` will be passed to the `save_func` doing the following:
            ```
            save_func(file_path=file_path, img=img, **kwargs)
            ```
        """
        super().__init__(write_interval="batch")

        # Toggle for CAREamist to switch off saving if desired
        self.save_predictions: bool = True

        self.save_type: SupportedData = SupportedData(save_type)
        self.save_func_kwargs = save_func_kwargs

        # forward declarations
        self.dirpath: Path
        self.save_func: WriteFunc
        self.save_extension: str
        # attribute initialisation
        self._init_dirpath(dirpath)
        self._init_save_func(save_func)
        self._init_save_extension(save_extension)

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

    def _init_save_func(self, save_func: Optional[WriteFunc]):
        """
        Initialize save function. Should only be called from `__init__`.

        Parameters
        ----------
        save_func : WriteFunc, optional
            See `__init__` description.

        Raises
        ------
        ValueError
            If `self.save_type="custom"` but `save_func` has not been given.
        """
        if self.save_type == SupportedData.CUSTOM:
            if save_func is None:
                raise ValueError(
                    "A save function must be provided for custom data types."
                    # TODO: link to how save functions should be implemented
                )
            else:
                self.save_func = save_func
        else:
            self.save_func = get_write_func(self.save_type)

    def _init_save_extension(self, save_extension: Optional[str]):
        """
        Initialize save extension. Should only be called from `__init__`.

        Parameters
        ----------
        save_extension : str, optional
            See `__init__` description.

        Raises
        ------
        ValueError
            If `self.save_type="custom"` but `save_extension` has not been given.
        """
        if self.save_type == SupportedData.CUSTOM:
            if save_extension is None:
                raise ValueError(
                    "A save extension must be provided for custom data types."
                )
            else:
                self.save_extension = save_extension
        else:
            # kind of a weird pattern -> reason to move get_extension from SupportedData
            self.save_extension = self.save_type.get_extension(self.save_type)

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
            if not self.dirpath.is_dir():
                logger.info("Making prediction output directory.")
                self.dirpath.mkdir()

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,  # TODO: change to expected type
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """
        Write predictions at the end of a batch.

        If tiff or custom, and tiled, tiles will be cached until a whole image is
        predicted and then the stictched prediction will be saved.

        If zarr, when implemented, tiles can be saved directly to disk.

        Parameters
        ----------
        trainer : Trainer
            PyTorch Lightning trainer.
        pl_module : LightningModule
            PyTorch Lightning module.
        prediction : Any
            Prediction outputs on batch.
        batch_indices : sequence of Any, optional
            Batch indices.
        batch : Any
            Input batch.
        batch_idx : int
            Batch index.
        dataloader_idx : int
            Dataloader index.
        """
        raise NotImplementedError
