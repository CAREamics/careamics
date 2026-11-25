"""CryoCARE-specific user interface."""

from pathlib import Path
from typing import Any, NoReturn, cast

from lightning.pytorch import Callback
from numpy.typing import NDArray

from .careamist import (
    CAREamist,
    DataOutputFormat,
    OutputType,
    OutputTypeWithUncertainty,
)
from .dataset.factory import ImageStackLoading, Loading, ReadFuncLoading
from .image_io import WriteFunc
from .lightning.checkpoint_record_logger import CheckpointRecord
from .lightning.data import InputVar
from .lightning.utils import TrainingReport


def _adapt_pair_validation_error(error: ValueError) -> NoReturn:
    """Adapt CAREamist's input/target length error to CryoCARE terminology.

    Parameters
    ----------
    error : ValueError
        Validation error raised by CAREamist.
    """
    if "same length" in str(error):
        raise ValueError(
            "Odd and even reconstructions must contain the same number of "
            "reconstructions."
        ) from error
    raise error


class CryoCAREamist:
    """High-level interface for CryoCARE training and prediction.

    Parameters
    ----------
    config : Configuration | Path | str, default=None
        CAREamics configuration, or a path to a configuration file. See
        `careamics.config.factories` for methods to build configurations. `config`
        is mutually exclusive with `checkpoint_path` and `bmz_path`.
    checkpoint_path : Path | str, default=None
        Path to a checkpoint file from which to load the model and configuration.
        `checkpoint_path` is mutually exclusive with `config` and `bmz_path`.
    bmz_path : Path | str, default=None
        Path to a BioImage Model Zoo archive from which to load the model and
        configuration. `bmz_path` is mutually exclusive with `config` and
        `checkpoint_path`.
    work_dir : Path | str, default=None
        Working directory in which to save training outputs. If None, the current
        working directory will be used.
    callbacks : list of PyTorch Lightning Callbacks, default=None
        List of callbacks to use during training. If None, no additional callbacks
        will be used. `ModelCheckpoint` and `EarlyStopping` callbacks are already
        defined by CAREamics and should be modified through the training
        configuration.
    enable_progress_bar : bool, default=True
        Whether to display the training progress bar.
    """

    def __init__(
        self,
        config: Any = None,
        *,
        checkpoint_path: Path | str | None = None,
        bmz_path: Path | str | None = None,
        work_dir: Path | str | None = None,
        callbacks: list[Callback] | None = None,
        enable_progress_bar: bool = True,
    ) -> None:
        """Initialize a CryoCARE user interface.

        Parameters
        ----------
        config : Configuration | Path | str, default=None
            CAREamics configuration, or a path to a configuration file.
        checkpoint_path : Path | str, default=None
            Path to a checkpoint file from which to load the model and configuration.
        bmz_path : Path | str, default=None
            Path to a BioImage Model Zoo archive from which to load the model and
            configuration.
        work_dir : Path | str, default=None
            Working directory in which to save training outputs. If None, the current
            working directory is used.
        callbacks : list of PyTorch Lightning Callbacks, default=None
            Additional callbacks to use during training.
        enable_progress_bar : bool, default=True
            Whether to display the progress bar.
        """
        self._careamist = CAREamist(
            config=config,
            checkpoint_path=checkpoint_path,
            bmz_path=bmz_path,
            work_dir=work_dir,
            callbacks=callbacks,
            enable_progress_bar=enable_progress_bar,
        )

    def train(
        self,
        odd_reconstructions: InputVar,
        even_reconstructions: InputVar,
        *,
        filtering_mask: InputVar | None = None,
        loading: Loading = None,
    ) -> None:
        """Train CryoCARE using paired odd and even reconstructions.

        The training data can be provided as paths, arrays, or sequences of paths or
        arrays. Validation data is intentionally not exposed by this interface;
        automatic validation remains controlled by the CryoCARE configuration.

        Parameters
        ----------
        odd_reconstructions : pathlib.Path, str, numpy.ndarray, or sequence of these
            Odd reconstructions used as the training input.
        even_reconstructions : pathlib.Path, str, numpy.ndarray, or sequence of these
            Even reconstructions paired with `odd_reconstructions` as training
            targets.
        filtering_mask : pathlib.Path, str, numpy.ndarray, or sequence of these
            Filtering mask for coordinate-based patch filtering, by default None.
        loading : Loading, default=None
            Loading strategy to use for the training data. May be a `ReadFuncLoading`
            or `ImageStackLoading`. If None, uses the loading strategy from the
            training configuration.

        Raises
        ------
        ValueError
            If the odd and even reconstruction sequences have different lengths.
        """
        try:
            self._careamist.train(
                train_data=odd_reconstructions,
                train_data_target=even_reconstructions,
                filtering_mask=filtering_mask,
                loading=loading,
            )
        except ValueError as error:
            _adapt_pair_validation_error(error)

    def predict(
        self,
        odd_reconstructions: InputVar,
        even_reconstructions: InputVar,
        tile_size: tuple[int, ...],
        *,
        batch_size: int | None = None,
        tile_overlap: tuple[int, ...] = (64, 64, 64),
        num_workers: int | None = None,
        in_memory: bool | None = None,
        loading: ReadFuncLoading | ImageStackLoading | None = None,
        checkpoint: str | Path | None = None,
    ) -> OutputType | OutputTypeWithUncertainty:
        """Predict on paired odd and even reconstructions.

        Input can be a path to a data file, a list of paths, a numpy array, or a
        list of numpy arrays. If `tile_size` is not provided, prediction will be
        performed on the whole image.

        Note that when using a UNet model and tiling, the tile size must be divisible
        in every dimension by 2**d, where d is the depth of the model. Images smaller
        than the tile size in any spatial dimension will be automatically zero-padded.

        Parameters
        ----------
        odd_reconstructions : pathlib.Path, str, numpy.ndarray, or sequence of these
            Odd reconstructions to predict on.
        even_reconstructions : pathlib.Path, str, numpy.ndarray, or sequence of these
            Even reconstructions paired with `odd_reconstructions`. CryoCARE averages
            predictions from both reconstructions.
        tile_size : tuple of int
            Size of the tiles to use for prediction. Must be divisible by 2**d, where d
            is the depth of the model, by default 8 (depth 3).
        batch_size : int, optional
            Batch size for prediction. If not provided, uses the training
            configuration batch size.

        tile_overlap : tuple of int, default=(64, 64, 64)
            Overlap between tiles. Must be larger or equal to the receptive field of the
            model (by default 64).
        num_workers : int, optional
            Number of workers for the dataloader, by default None.
        in_memory : bool, optional
            Whether to load all data into memory. If None, uses the training
            configuration setting, except for zarr/czi data which is never loaded
            in-memory.
        loading : Loading, default=None
            Loading strategy to use for the prediction data. May be a `ReadFuncLoading`
            or `ImageStackLoading`. If None, uses the loading strategy from the
            training configuration.
        checkpoint : str or Path, optional
            Checkpoint to load before making predictions. Can be "best", "last", or a
            path to a specific checkpoint. If None, reuses the checkpoint supplied
            when constructing `CAREamist`, then the checkpoint selected in the current
            run's checkpoint record. If neither is available, uses the in-memory model.
            Call `get_checkpoints` to inspect available checkpoints.

        Returns
        -------
        tuple of (list of NDArray, list of str)
            Predictions and their source identifiers.

        Raises
        ------
        ValueError
            If the odd and even reconstruction sequences have different lengths, if
            tile overlap is not specified when `tile_size` is provided.
        """
        try:
            return self._careamist.predict(
                pred_data=odd_reconstructions,
                pred_data_target=even_reconstructions,
                batch_size=batch_size,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                num_workers=num_workers,
                in_memory=in_memory,
                loading=loading,
                checkpoint=checkpoint,
            )
        except ValueError as error:
            _adapt_pair_validation_error(error)

    def predict_to_disk(
        self,
        odd_reconstructions: InputVar,
        even_reconstructions: InputVar,
        tile_size: tuple[int, ...],
        *,
        prediction_dir: Path | str = "predictions",
        batch_size: int | None = None,
        tile_overlap: tuple[int, ...] = (64, 64, 64),
        num_workers: int | None = None,
        in_memory: bool | None = None,
        loading: ReadFuncLoading | None = None,
        checkpoint: str | Path | None = None,
        write_type: DataOutputFormat | None = None,
        write_extension: str | None = None,
        write_func: WriteFunc | None = None,
        write_func_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Predict on paired reconstructions and save outputs to files.

        Parameters
        ----------
        odd_reconstructions : pathlib.Path, str, numpy.ndarray, or sequence of these
            Odd reconstructions to predict on.
        even_reconstructions : pathlib.Path, str, numpy.ndarray, or sequence of these
            Even reconstructions paired with `odd_reconstructions`.
        tile_size : tuple of int
            Size of the tiles to use for prediction. Must be divisible by 2**d, where d
            is the depth of the model, by default 8 (depth 3).
        prediction_dir : Path | str, default="predictions"
            Directory in which to save predictions. Relative paths are resolved
            against `work_dir`.
        batch_size : int, optional
            Batch size for prediction. If not provided, uses the training configuration
            batch size.
        tile_overlap : tuple of int, default=(64, 64, 64)
            Overlap between tiles. Must be larger or equal to the receptive field of the
            model (by default 64).
        num_workers : int, optional
            Number of workers for the dataloader.
        in_memory : bool, optional
            Whether to load all data into memory.
        loading : Loading, default=None
            Loading strategy for the prediction data.
        checkpoint : str or Path, optional
            Checkpoint to load before making predictions.
        write_type : DataOutputFormat, optional
            Output data type. Defaults to TIFF, or Zarr when the effective input type
            is Zarr.
        write_extension : str, optional
            Extension to use for a custom output type.
        write_func : WriteFunc, optional
            Function used to save a custom output type.
        write_func_kwargs : dict, optional
            Additional keyword arguments passed to `write_func`.

        Raises
        ------
        ValueError
            If the odd and even reconstruction sequences have different lengths or
            an output-writing option is invalid.
        """
        try:
            cast(Any, self._careamist).predict_to_disk(
                pred_data=odd_reconstructions,
                pred_data_target=even_reconstructions,
                prediction_dir=prediction_dir,
                batch_size=batch_size,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                num_workers=num_workers,
                in_memory=in_memory,
                loading=loading,
                checkpoint=checkpoint,
                write_type=write_type,
                write_extension=write_extension,
                write_func=write_func,
                write_func_kwargs=write_func_kwargs,
            )
        except ValueError as error:
            _adapt_pair_validation_error(error)

    def get_checkpoints(self) -> CheckpointRecord | None:
        """Return the checkpoint record for the current training run.

        `last` and `best` identify checkpoint files, `monitor` and `mode` describe
        ranking, `best_k` contains ranked checkpoints, `checkpoints` contains all
        saved checkpoints, and `selected` identifies the default checkpoint for
        inference.

        Returns
        -------
        CheckpointRecord or None
            Saved checkpoint record, or `None` if no record exists.
        """
        return self._careamist.get_checkpoints()

    def get_losses(self) -> TrainingReport:
        """Return plottable training curves from Lightning CSV logs.

        Returns
        -------
        TrainingReport
            Dataclass containing train and validation loss, learning rate, and any
            discovered validation metrics.
        """
        return self._careamist.get_losses()

    def export_to_bmz(
        self,
        path_to_archive: Path | str,
        friendly_model_name: str,
        input_array: NDArray,
        authors: list[dict],
        general_description: str,
        data_description: str,
        covers: list[Path | str] | None = None,
    ) -> None:
        """Export the model to the BioImage Model Zoo format.

        This method packages the current weights into a zip file that can be uploaded
        to the BioImage Model Zoo. `path_to_archive` should end with a `.zip` suffix.

        Parameters
        ----------
        path_to_archive : pathlib.Path or str
            Path in which to save the model, including file name.
        friendly_model_name : str
            Name of the model as used in the BMZ specifications.
        input_array : NDArray
            Input array used to validate the model and as an example.
        authors : list of dict
            List of authors of the model.
        general_description : str
            General description of the model used in the BMZ metadata.
        data_description : str
            Description of the data the model was trained on.
        covers : list of pathlib.Path or str, default=None
            Paths to the cover images.
        """
        self._careamist.export_to_bmz(
            path_to_archive=path_to_archive,
            friendly_model_name=friendly_model_name,
            input_array=input_array,
            authors=authors,
            general_description=general_description,
            data_description=data_description,
            covers=covers,
        )

    def stop_training(self) -> None:
        """Stop the training loop."""
        self._careamist.stop_training()
