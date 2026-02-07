from collections.abc import Callable, Sequence
from pathlib import Path
from typing import (
    Any,
    Literal,
    TypedDict,
    Union,
    Unpack,
    overload,
)

import numpy as np
import torch
from numpy.typing import NDArray
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger

from careamics.lightning.dataset_ng.lightning_modules.get_module import CAREamicsModule

from .config import load_configuration_ng
from .config.ng_configs import N2VConfiguration
from .config.support import SupportedData, SupportedLogger
from .dataset.dataset_utils import reshape_array
from .file_io import WriteFunc, get_write_func
from .lightning.callbacks import CareamicsCheckpointInfo, ProgressBarCallback
from .lightning.dataset_ng.callbacks.prediction_writer import PredictionWriterCallback
from .lightning.dataset_ng.lightning_modules import (
    CareamicsDataModule,
    create_module,
    load_module_from_checkpoint,
)
from .lightning.dataset_ng.prediction import convert_prediction
from .model_io import export_to_bmz
from .utils import get_logger
from .utils.lightning_utils import read_csv_logger

logger = get_logger(__name__)

ExperimentLogger = TensorBoardLogger | WandbLogger | CSVLogger
Configuration = N2VConfiguration


class UserContext(TypedDict, total=False):
    work_dir: Path | str | None
    callbacks: list[Callback] | None
    enable_progress_bar: bool


class CAREamistV2:
    def __init__(
        self,
        config: Configuration | Path | None = None,
        *,
        checkpoint_path: Path | None = None,
        bmz_path: Path | None = None,
        **user_context: Unpack[UserContext],
    ):
        self.checkpoint_path = checkpoint_path
        self.work_dir = self._resolve_work_dir(user_context.get("work_dir"))
        self.config, self.model = self._load_model(config, checkpoint_path, bmz_path)

        enable_progress_bar = user_context.get("enable_progress_bar", True)
        self.config.training_config.lightning_trainer_config["enable_progress_bar"] = (
            enable_progress_bar
        )
        callbacks = user_context.get("callbacks", None)
        self.callbacks = self._define_callbacks(callbacks, self.config, self.work_dir)

        # init callbacks
        self.prediction_writer = PredictionWriterCallback(
            self.work_dir, enable_writing=False
        )

        experiment_loggers = self._create_loggers(
            self.config.training_config.logger,
            self.config.experiment_name,
            self.work_dir,
        )

        self.trainer = Trainer(
            callbacks=[self.prediction_writer, *self.callbacks],
            default_root_dir=self.work_dir,
            logger=experiment_loggers,
            **self.config.training_config.lightning_trainer_config or {},
        )

    def _load_model(
        self,
        config: Configuration | Path | None,
        checkpoint_path: Path | None,
        bmz_path: Path | None,
    ) -> tuple[Configuration, CAREamicsModule]:
        n_inputs = sum(
            [config is not None, checkpoint_path is not None, bmz_path is not None]
        )
        if n_inputs != 1:
            raise ValueError(
                "Exactly one of `config`, `checkpoint_path`, or `bmz_path` "
                "must be provided."
            )
        if config is not None:
            return self._from_config(config)
        elif checkpoint_path is not None:
            return self._from_checkpoint(checkpoint_path)
        else:
            assert bmz_path is not None
            return self._from_bmz(bmz_path)

    @staticmethod
    def _from_config(
        config: Configuration | Path,
    ) -> tuple[Configuration, CAREamicsModule]:
        if isinstance(config, Path):
            config = load_configuration_ng(config)
        assert not isinstance(config, Path)

        model = create_module(config.algorithm_config)
        return config, model

    @staticmethod
    def _from_checkpoint(
        checkpoint_path: Path,
    ) -> tuple[Configuration, CAREamicsModule]:
        checkpoint: dict = torch.load(checkpoint_path, map_location="cpu")

        careamics_info = checkpoint.get("careamics_info", None)
        if careamics_info is None:
            raise ValueError(
                "Could not find CAREamics related information within the provided "
                "checkpoint. This means that it was saved without using the "
                "CAREamics callback `CareamicsCheckpointInfo`. "
                "Please use a checkpoint saved with CAREamics or initialize with "
                "a config instead."
            )

        try:
            algorithm_config: dict[str, Any] = checkpoint["hyper_parameters"][
                "algorithm_config"
            ]
        except (KeyError, IndexError) as e:
            raise ValueError(
                "Could not determine CAREamics supported algorithm from the provided "
                f"checkpoint at: {checkpoint_path!s}."
            ) from e

        data_hparams_key = checkpoint.get(
            "datamodule_hparams_name", "datamodule_hyper_parameters"
        )
        try:
            data_config: dict[str, Any] = checkpoint[data_hparams_key]["data_config"]
        except (KeyError, IndexError) as e:
            raise ValueError(
                "Could not determine the data configuration from the provided "
                f"checkpoint at: {checkpoint_path!s}."
            ) from e

        # TODO: will need to resolve this with type adapter once more configs are added
        config = Configuration.model_validate(
            {
                "algorithm_config": algorithm_config,
                "data_config": data_config,
                **careamics_info,
            }
        )

        module = load_module_from_checkpoint(checkpoint_path)
        return config, module

    @staticmethod
    def _from_bmz(
        bmz_path: Path,
    ) -> tuple[Configuration, CAREamicsModule]:
        raise NotImplementedError("Loading from BMZ is not implemented yet.")

    @staticmethod
    def _resolve_work_dir(work_dir: str | Path | None) -> Path:
        if work_dir is None:
            work_dir = Path.cwd().resolve()
            logger.warning(
                f"No working directory provided. Using current working directory: "
                f"{work_dir}."
            )
        else:
            work_dir = Path(work_dir).resolve()
        return work_dir

    @staticmethod
    def _define_callbacks(
        callbacks: list[Callback] | None,
        config: Configuration,
        work_dir: Path,
    ) -> list[Callback]:
        callbacks = [] if callbacks is None else callbacks
        for c in callbacks:
            if isinstance(c, (ModelCheckpoint, EarlyStopping)):
                raise ValueError(
                    "`ModelCheckpoint` and `EarlyStopping` callbacks are already "
                    "defined in CAREamics and should only be modified through the "
                    "training configuration (see TrainingConfig)."
                )

            if isinstance(c, (CareamicsCheckpointInfo, ProgressBarCallback)):
                raise ValueError(
                    "`CareamicsCheckpointInfo` and `ProgressBar` callbacks are defined "
                    "internally and should not be passed as callbacks."
                )

        internal_callbacks = [
            ModelCheckpoint(
                dirpath=work_dir / "checkpoints",
                filename=f"{config.experiment_name}_{{epoch:02d}}_step_{{step}}",
                **config.training_config.checkpoint_callback.model_dump(),
            ),
            CareamicsCheckpointInfo(
                config.version, config.experiment_name, config.training_config
            ),
        ]

        enable_progress_bar = config.training_config.lightning_trainer_config.get(
            "enable_progress_bar", True
        )
        if enable_progress_bar:
            internal_callbacks.append(ProgressBarCallback())

        if config.training_config.early_stopping_callback is not None:
            internal_callbacks.append(
                EarlyStopping(
                    **config.training_config.early_stopping_callback.model_dump()
                )
            )

        return internal_callbacks + callbacks

    @staticmethod
    def _create_loggers(
        logger: str | None, experiment_name: str, work_dir: Path
    ) -> list[ExperimentLogger]:
        csv_logger = CSVLogger(name=experiment_name, save_dir=work_dir / "csv_logs")

        if logger is not None:
            logger = SupportedLogger(logger)

        match logger:
            case SupportedLogger.WANDB:
                return [
                    WandbLogger(name=experiment_name, save_dir=work_dir / "wandb_logs"),
                    csv_logger,
                ]
            case SupportedLogger.TENSORBOARD:
                return [
                    TensorBoardLogger(save_dir=work_dir / "tb_logs"),
                    csv_logger,
                ]
            case _:
                return [csv_logger]

    def train(
        self,
        *,
        # BASIC PARAMS
        train_data: Any | None = None,
        train_data_target: Any | None = None,
        val_data: Any | None = None,
        val_data_target: Any | None = None,
        # val_percentage: float | None = None, # TODO: hidden till re-implemented
        # val_minimum_split: int = 5,
        # ADVANCED PARAMS
        filtering_mask: Any | None = None,
        read_source_func: Callable | None = None,
        read_kwargs: dict[str, Any] | None = None,
        extension_filter: str = "",
    ) -> None:
        # TODO: init datamodule
        # TODO: remember to pass self.checkpoint_path to Trainer.fit
        # ^ this will load optimizer and lr_schedular state dicts
        raise NotImplementedError("Training is not implemented yet.")

    @overload
    def predict(self, pred_data: CareamicsDataModule) -> list[NDArray]: ...

    @overload
    def predict(
        self,
        pred_data: Union[Path, str, Sequence[Union[Path, str]]],
        *,
        batch_size: int = 1,
        tile_size: tuple[int, ...] | None = None,
        tile_overlap: tuple[int, ...] | None = (48, 48),
        axes: str | None = None,
        data_type: Literal["tiff", "zarr", "custom"] | None = None,
        read_source_func: Callable | None = None,
        extension_filter: str = "",
    ) -> list[NDArray]: ...

    @overload
    def predict(
        self,
        pred_data: Union[NDArray, Sequence[NDArray]],
        *,
        batch_size: int = 1,
        tile_size: tuple[int, ...] | None = None,
        tile_overlap: tuple[int, ...] | None = (48, 48),
        axes: str | None = None,
        data_type: Literal["array"] | None = None,
    ) -> list[NDArray]: ...

    def predict(
        self,
        # BASIC PARAMS
        pred_data: (
            CareamicsDataModule
            | Path
            | str
            | NDArray
            | Sequence[Union[Path, str, NDArray]]
        ),
        *,
        batch_size: int | None = None,
        tile_size: tuple[int, ...] | None = None,
        tile_overlap: tuple[int, ...] | None = (48, 48),
        axes: str | None = None,
        data_type: Literal["array", "tiff", "zarr", "custom"] | None = None,
        # ADVANCED PARAMS
        num_workers: int | None = None,
        channels: Sequence[int] | Literal["all"] | None = None,
        in_memory: bool | None = None,
        read_source_func: Callable | None = None,
        read_kwargs: dict[str, Any] | None = None,
        extension_filter: str = "",
    ) -> list[NDArray]:
        """
        Make predictions on the provided data.

        Input can be a CareamicsDataModule instance, a path to a data file, a list of
        paths, a numpy array, or a list of numpy arrays.

        If `data_type` and `axes` are not provided, the training configuration
        parameters will be used. If `tile_size` is not provided prediction will be performed on the whole image.

        Note that if you are using a UNet model and tiling, the tile size must be
        divisible in every dimension by 2**d, where d is the depth of the model. This
        avoids artefacts arising from the broken shift invariance induced by the
        pooling layers of the UNet. Images smaller than the tile size in any spatial
        dimension will be automatically zero-padded.

        Parameters
        ----------
        pred_data : CareamicsDataModule, pathlib.Path, str, numpy.ndarray, or sequence
            Data to predict on. Can be a single item or a sequence of paths/arrays.
        batch_size : int, optional
            Batch size for prediction. If not provided, uses the training configuration
            batch size.
        tile_size : tuple of int, optional
            Size of the tiles to use for prediction. If not provided, prediction will be performed on the whole image.
        tile_overlap : tuple of int, default=(48, 48)
            Overlap between tiles, can be None.
        axes : str, optional
            Axes of the input data, by default None.
        data_type : {"array", "tiff", "zarr", "custom"}, optional
            Type of the input data.
        num_workers : int, optional
            Number of workers for the dataloader, by default None.
        channels : sequence of int or "all", optional
            Channels to use from the data. If None, uses the training configuration
            channels.
        in_memory : bool, optional
            Whether to load all data into memory. If None, uses the training
            configuration setting.
        read_source_func : Callable, optional
            Function to read the source data.
        read_kwargs : dict of {str: Any}, optional
            Additional keyword arguments to be passed to the read function.
        extension_filter : str, default=""
            Filter for the file extension.

        Returns
        -------
        list of NDArray
            Predictions made by the model, one array per input sample.

        Raises
        ------
        ValueError
            If tile overlap is not specified when tile_size is provided.
        """
        # If datamodule is provided directly, use it
        if isinstance(pred_data, CareamicsDataModule):
            datamodule = pred_data
        else:
            # Prepare dataloader params
            dataloader_params: dict[str, Any] | None = None
            if num_workers is not None:
                dataloader_params = {"num_workers": num_workers}

            # Create prediction data config using convert_mode
            pred_data_config = self.config.data_config.convert_mode(
                new_mode="predicting",
                new_patch_size=tile_size,
                overlap_size=tile_overlap,
                new_batch_size=batch_size,
                new_data_type=data_type,
                new_axes=axes,
                new_channels=channels,
                new_in_memory=in_memory,
            )

            # Create datamodule for prediction
            datamodule = CareamicsDataModule(
                data_config=pred_data_config,
                pred_data=pred_data,
                read_source_func=read_source_func,
                read_kwargs=read_kwargs,
                extension_filter=extension_filter,
            )

        # Predict
        predictions = self.trainer.predict(model=self.model, datamodule=datamodule)

        # Convert predictions using convert_prediction
        tiled = tile_size is not None
        predictions_output, sources = convert_prediction(predictions, tiled=tiled)

        return predictions_output, sources

    def predict_to_disk(
        self,
        # BASIC PARAMS
        pred_data: Any | None = None,
        pred_data_target: Any | None = None,
        prediction_dir: Path | str = "predictions",
        batch_size: int = 1,
        tile_size: tuple[int, ...] | None = None,
        tile_overlap: tuple[int, ...] | None = (48, 48),
        axes: str | None = None,
        data_type: Literal["array", "tiff", "zarr", "custom"] | None = None,
        # ADVANCED PARAMS
        num_workers: int | None = None,
        channels: Sequence[int] | Literal["all"] | None = None,
        in_memory: bool | None = None,
        read_source_func: Callable | None = None,
        read_kwargs: dict[str, Any] | None = None,
        extension_filter: str = "",
        # WRITE OPTIONS
        write_type: Literal["tiff", "zarr", "custom"] = "tiff",
        write_extension: str | None = None,
        write_func: WriteFunc | None = None,
        write_func_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Make predictions on the provided data and save outputs to files.

        The predictions will be saved in the specified `prediction_dir`. If
        `prediction_dir` is an absolute path, it will be used as-is. If it is a relative
        path, it will be relative to the pre-set `work_dir`. The directory structure
        within the `prediction_dir` will match that of the source directory.

        The file names of the predictions will match those of the source. If there is
        more than one sample within a file, the samples will be stacked along the sample
        dimension in the output file.

        If `data_type` and `axes` are not provided, the training configuration
        parameters will be used. If `tile_size` is not provided, the whole image
        strategy will be used for prediction.

        Note that if you are using a UNet model and tiling, the tile size must be
        divisible in every dimension by 2**d, where d is the depth of the model. This
        avoids artefacts arising from the broken shift invariance induced by the
        pooling layers of the UNet. Images smaller than the tile size in any spatial
        dimension will be automatically zero-padded.

        Parameters
        ----------
        pred_data : pathlib.Path, str, numpy.ndarray, sequence, or CareamicsDataModule
            Data to predict on. Can be a single item, a sequence of paths/arrays, or
            a CareamicsDataModule instance.
        pred_data_target : Any, optional
            Prediction data target, by default None.
        prediction_dir : Path | str, default="predictions"
            The path to save the prediction results to. If `prediction_dir` is an
            absolute path, it will be used as-is. If it is a relative path, it will
            be relative to the pre-set `work_dir`. If the directory does not exist it
            will be created.
        batch_size : int, default=1
            Batch size for prediction.
        tile_size : tuple of int, optional
            Size of the tiles to use for prediction. If not provided, uses whole image
            strategy.
        tile_overlap : tuple of int, default=(48, 48)
            Overlap between tiles.
        axes : str, optional
            Axes of the input data, by default None.
        data_type : {"array", "tiff", "zarr", "custom"}, optional
            Type of the input data.
        num_workers : int, optional
            Number of workers for the dataloader, by default None.
        channels : sequence of int or "all", optional
            Channels to use from the data. If None, uses the training configuration
            channels.
        in_memory : bool, optional
            Whether to load all data into memory. If None, uses the training
            configuration setting.
        read_source_func : Callable, optional
            Function to read the source data.
        read_kwargs : dict of {str: Any}, optional
            Additional keyword arguments to be passed to the read function.
        extension_filter : str, default=""
            Filter for the file extension.
        write_type : {"tiff", "zarr", "custom"}, default="tiff"
            The data type to save as, includes custom.
        write_extension : str, optional
            If a known `write_type` is selected this argument is ignored. For a custom
            `write_type` an extension to save the data with must be passed.
        write_func : WriteFunc, optional
            If a known `write_type` is selected this argument is ignored. For a custom
            `write_type` a function to save the data must be passed. See notes below.
        write_func_kwargs : dict of {str: any}, optional
            Additional keyword arguments to be passed to the save function.

        Raises
        ------
        ValueError
            If `write_type` is custom and `write_extension` is None.
        ValueError
            If `write_type` is custom and `write_func` is None.
        ValueError
            If `pred_data` is not provided.
        """
        if pred_data is None:
            raise ValueError("pred_data must be provided for predict_to_disk.")

        if write_func_kwargs is None:
            write_func_kwargs = {}

        if Path(prediction_dir).is_absolute():
            write_dir = Path(prediction_dir)
        else:
            write_dir = self.work_dir / prediction_dir

        # Set up the prediction writer callback
        self.prediction_writer.dirpath = write_dir

        # Guards for custom types
        if write_type == "custom":
            if write_extension is None:
                raise ValueError(
                    "A `write_extension` must be provided for custom write types."
                )
            if write_func is None:
                raise ValueError(
                    "A `write_func` must be provided for custom write types."
                )
        else:
            write_func = get_write_func(write_type)
            write_extension = SupportedData.get_extension(write_type)

        # Set writing strategy
        tiled = tile_size is not None
        self.prediction_writer.set_writing_strategy(
            write_type=write_type,
            tiled=tiled,
            write_func=write_func,
            write_extension=write_extension,
            write_func_kwargs=write_func_kwargs,
        )

        # Enable writing
        self.prediction_writer.enable_writing(True)

        try:
            # Create datamodule if not already provided
            if isinstance(pred_data, CareamicsDataModule):
                datamodule = pred_data
            else:
                # Prepare dataloader params
                dataloader_params: dict[str, Any] | None = None
                if num_workers is not None:
                    dataloader_params = {"num_workers": num_workers}

                # Create prediction data config using convert_mode
                pred_data_config = self.config.data_config.convert_mode(
                    new_mode="predicting",
                    new_patch_size=tile_size,
                    overlap_size=tile_overlap,
                    new_batch_size=batch_size,
                    new_data_type=data_type,
                    new_axes=axes,
                    new_channels=channels,
                    new_in_memory=in_memory,
                    new_dataloader_params=dataloader_params,
                )

                # Create datamodule for prediction
                datamodule = CareamicsDataModule(
                    data_config=pred_data_config,
                    pred_data=pred_data,
                    pred_data_target=pred_data_target,
                    read_source_func=read_source_func,
                    read_kwargs=read_kwargs,
                    extension_filter=extension_filter,
                )

            # Predict (writing will be handled by the callback)
            self.trainer.predict(
                model=self.model, datamodule=datamodule, return_predictions=False
            )

        finally:
            # Disable writing after prediction is complete
            self.prediction_writer.enable_writing(False)

    def export_to_bmz(
        self,
        path_to_archive: Path | str,
        friendly_model_name: str,
        input_array: NDArray,
        authors: list[dict],
        general_description: str,
        data_description: str,
        covers: list[Path | str] | None = None,
        channel_names: list[str] | None = None,
        model_version: str = "0.1.0",
    ) -> None:
        """Export the model to the BioImage Model Zoo format.

        This method packages the current weights into a zip file that can be uploaded
        to the BioImage Model Zoo. The archive consists of the model weights, the model
        specifications and various files (inputs, outputs, README, env.yaml etc.).

        `path_to_archive` should point to a file with a ".zip" extension.

        `friendly_model_name` is the name used for the model in the BMZ specs
        and website, it should consist of letters, numbers, dashes, underscores and
        parentheses only.

        Input array must be of the same dimensions as the axes recorded in the
        configuration of the `CAREamist`.

        Parameters
        ----------
        path_to_archive : pathlib.Path or str
            Path in which to save the model, including file name, which should end with
            ".zip".
        friendly_model_name : str
            Name of the model as used in the BMZ specs, it should consist of letters,
            numbers, dashes, underscores and parentheses only.
        input_array : NDArray
            Input array used to validate the model and as example.
        authors : list of dict
            List of authors of the model.
        general_description : str
            General description of the model used in the BMZ metadata.
        data_description : str
            Description of the data the model was trained on.
        covers : list of pathlib.Path or str, default=None
            Paths to the cover images.
        channel_names : list of str, default=None
            Channel names.
        model_version : str, default="0.1.0"
            Version of the model.
        """
        output_patch = self.predict(
            pred_data=input_array,
            data_type=SupportedData.ARRAY.value,
        )
        output = np.concatenate(output_patch, axis=0)
        input_array = reshape_array(input_array, self.config.data_config.axes)

        export_to_bmz(
            model=self.model,
            config=self.config,
            path_to_archive=path_to_archive,
            model_name=friendly_model_name,
            general_description=general_description,
            data_description=data_description,
            authors=authors,
            input_array=input_array,
            output_array=output,
            covers=covers,
            channel_names=channel_names,
            model_version=model_version,
        )

    def get_losses(self) -> dict[str, list]:
        """Return data that can be used to plot train and validation loss curves.

        Returns
        -------
        dict of str: list
            Dictionary containing losses for each epoch.
        """
        return read_csv_logger(self.config.experiment_name, self.work_dir / "csv_logs")

    def stop_training(self) -> None:
        """Stop the training loop."""
        self.trainer.should_stop = True
        self.trainer.limit_val_batches = 0  # skip validation
