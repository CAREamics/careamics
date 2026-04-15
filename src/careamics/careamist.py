"""Main interface for training and predicting with CAREamics."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, overload

from numpy.typing import NDArray
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger

from .config.algorithms import CAREAlgorithm, N2NAlgorithm, N2VAlgorithm
from .config.configuration import Configuration
from .config.support import SupportedLogger
from .config.utils.configuration_io import load_configuration
from .dataset.factory import ImageStackLoading, Loading, ReadFuncLoading
from .dataset.image_region_data import ImageRegionData
from .file_io import WriteFunc
from .lightning.callbacks import ConfigSaverCallback, ProgressBarCallback
from .lightning.callbacks.prediction import PredictionWriterCallback
from .lightning.data_module import CareamicsDataModule, InputVar
from .lightning.lightning_modules import (
    CAREamicsModule,
    create_module,
    get_model_constraints,
)
from .lightning.load_checkpoint import (
    load_config_from_checkpoint,
    load_module_from_checkpoint,
)
from .lightning.prediction import convert_prediction
from .utils import get_logger
from .utils.lightning_utils import read_csv_logger

logger = get_logger(__name__)

ArrayInput = NDArray[Any] | Sequence[NDArray[Any]]
PathInput = str | Path | Sequence[str | Path]
InputType = ArrayInput | PathInput

ConfigurationType = (
    Configuration[CAREAlgorithm]
    | Configuration[N2NAlgorithm]
    | Configuration[N2VAlgorithm]
)


class CAREamist:
    """Main interface for training and predicting with CAREamics.

    Attributes
    ----------
    workdir : Path
        Working directory in which to save training outputs.
    config : Configuration[AlgorithmConfig]
        CAREamics configuration.
    model : CAREamicsModule
        The PyTorch Lightning module to be trained and used for prediction.
    checkpoint_path : Path | None
        Path to a checkpoint file from which model and configuration may be loaded.
    trainer : Trainer
        The PyTorch Lightning Trainer used for training and prediction.
    callbacks : list[Callback]
        List of callbacks used during training.
    prediction_writer : PredictionWriterCallback
        Callback used to write predictions to disk during prediction.
    train_datamodule : CareamicsDataModule | None
        The datamodule used for training, set after calling `train()`.

    Parameters
    ----------
    config : Configuration | Path | str, default=None
        CAREamics configuration, or a path to a configuration file. See
        `careamics.config.ng_factories` for method to build configurations.
    checkpoint_path : Path | str, default=None
        Path to a checkpoint file from which to load the model and configuration.
    bmz_path : Path | str, default=None
        Path to a BioImage Model Zoo archive from which to load the model and
        configuration.
    work_dir : Path | str, default=None
        Working directory in which to save training outputs. If None, the current
        working directory will be used.
    callbacks : list of PyTorch Lightning Callbacks, default=None
        List of callbacks to use during training. If None, no additional callbacks
        will be used. Note that `ModelCheckpoint` and `EarlyStopping` callbacks are
        already defined in CAREamics and should only be modified through the
        training configuration (see Configuration and TrainingConfig).
    enable_progress_bar : bool, default=True
        Whether to show the progress bar during training.
    """

    def __init__(
        self,
        config: ConfigurationType | Path | str | None = None,
        *,
        checkpoint_path: Path | str | None = None,
        bmz_path: Path | str | None = None,
        work_dir: Path | str | None = None,
        callbacks: list[Callback] | None = None,
        enable_progress_bar: bool = True,
    ) -> None:
        """Constructor.

        Exactly one of `config`, `checkpoint_path`, or `bmz_path` must be provided.

        Parameters
        ----------
        config : Configuration | Path | str, default=None
            CAREamics configuration, or a path to a configuration file. See
            `careamics.config.ng_factories` for method to build configurations. `config`
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
            will be used. Note that `ModelCheckpoint` and `EarlyStopping` callbacks are
            already defined in CAREamics and should only be modified through the
            training configuration (see Configuration and TrainingConfig).
        enable_progress_bar : bool, default=True
            Whether to show the progress bar during training.
        """
        self.checkpoint_path = checkpoint_path
        self.work_dir = self._resolve_work_dir(work_dir)

        self.config: ConfigurationType
        self.config, self.model = self._load_model(config, checkpoint_path, bmz_path)

        self.config.training_config.trainer_params["enable_progress_bar"] = (
            enable_progress_bar
        )
        self.callbacks = self._define_callbacks(callbacks, self.config, self.work_dir)

        self.prediction_writer = PredictionWriterCallback(
            self.work_dir, enable_writing=False
        )

        experiment_loggers = self._create_loggers(
            self.config.training_config.logger,
            self.config.get_safe_experiment_name(),
            self.work_dir,
        )

        self.trainer = Trainer(
            callbacks=[self.prediction_writer, *self.callbacks],
            default_root_dir=self.work_dir,
            logger=experiment_loggers,
            **self.config.training_config.trainer_params or {},
        )

        self.train_datamodule: CareamicsDataModule | None = None

    def _load_model(
        self,
        config: ConfigurationType | Path | str | None,
        checkpoint_path: Path | str | None,
        bmz_path: Path | str | None,
    ) -> tuple[ConfigurationType, CAREamicsModule]:
        """Load model.

        Parameters
        ----------
        config : Configuration | Path | None
            CAREamics configuration, or a path to a configuration file.
        checkpoint_path : Path | None
            Path to a checkpoint file from which to load the model and configuration.
        bmz_path : Path | None
            Path to a BioImage Model Zoo archive from which to load the model and
            configuration.

        Returns
        -------
        Configuration
            The loaded configuration.
        CAREamicsModule
            The loaded model.

        Raises
        ------
        ValueError
            If not exactly one of `config`, `checkpoint_path`, or `bmz_path` is
            provided.
        """
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
        config: ConfigurationType | Path | str,
    ) -> tuple[ConfigurationType, CAREamicsModule]:
        """Create model from configuration.

        Parameters
        ----------
        config : Configuration | Path | str
            CAREamics configuration, or a path to a configuration file.

        Returns
        -------
        Configuration
            The loaded configuration if a path was provided, otherwise the original
            configuration.
        CAREamicsModule
            The created model.
        """
        if isinstance(config, (Path, str)):
            config = load_configuration(Path(config))
        assert not isinstance(config, (Path, str))

        model = create_module(config.algorithm_config)
        return config, model

    @staticmethod
    def _from_checkpoint(
        checkpoint_path: Path | str,
    ) -> tuple[ConfigurationType, CAREamicsModule]:
        """Load checkpoint and configuration from checkpoint file.

        Parameters
        ----------
        checkpoint_path : Path | str
            Path to a checkpoint file from which to load the model and configuration.

        Returns
        -------
        Configuration
            The loaded configuration.
        CAREamicsModule
            The loaded model.
        """
        checkpoint_path = Path(checkpoint_path)
        config = load_config_from_checkpoint(checkpoint_path)
        module = load_module_from_checkpoint(checkpoint_path)

        return config, module

    @staticmethod
    def _from_bmz(
        bmz_path: Path | str,
    ) -> tuple[ConfigurationType, CAREamicsModule]:
        """Load checkpoint and configuration from a BioImage Model Zoo archive.

        Parameters
        ----------
        bmz_path : Path | str
            Path to a BioImage Model Zoo archive from which to load the model and
            configuration.

        Returns
        -------
        Configuration
            The loaded configuration.
        CAREamicsModule
            The loaded model.

        Raises
        ------
        NotImplementedError
            Loading from BMZ is not implemented yet.
        """
        raise NotImplementedError("Loading from BMZ is not implemented yet.")

    @staticmethod
    def _resolve_work_dir(work_dir: str | Path | None) -> Path:
        """Resolve working directory.

        Parameters
        ----------
        work_dir : str | Path | None
            The working directory to resolve. If None, the current working directory
            will be used.

        Returns
        -------
        Path
            The resolved working directory.
        """
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
        config: ConfigurationType,
        work_dir: Path,
    ) -> list[Callback]:
        """Define callbacks for the training process.

        Parameters
        ----------
        callbacks : list[Callback] | None
            List of callbacks to use during training. If None, no additional callbacks
            will be used. Note that `ModelCheckpoint` and `EarlyStopping` callbacks are
            already defined in CAREamics and instantiated in this method.
        config : Configuration
            The CAREamics configuration, used to instantiate the callbacks.
        work_dir : Path
            The working directory, used as a parameter to the checkpointing callback.

        Returns
        -------
        list[Callback]
            The list of callbacks to use during training.

        Raises
        ------
        ValueError
            If `ModelCheckpoint` or `EarlyStopping` callbacks are included in the
            provided `callbacks` list, as these are already defined in CAREamics and
            should only be modified through the training configuration (see
            Configuration and TrainingConfig).
        """
        callback_lst: list[Callback] = [] if callbacks is None else callbacks
        for c in callback_lst:
            if isinstance(c, (ModelCheckpoint, EarlyStopping)):
                raise ValueError(
                    "`ModelCheckpoint` and `EarlyStopping` callbacks are already "
                    "defined in CAREamics and should only be modified through the "
                    "training configuration (see TrainingConfig)."
                )

            if isinstance(c, (ConfigSaverCallback, ProgressBarCallback)):
                raise ValueError(
                    "`CareamicsCheckpointInfo` and `ProgressBar` callbacks are defined "
                    "internally and should not be passed as callbacks."
                )

        checkpoint_callback = ModelCheckpoint(
            dirpath=work_dir / "checkpoints" / config.get_safe_experiment_name(),
            filename=(
                f"{config.get_safe_experiment_name()}_{{epoch:02d}}_step_{{step}}_"
                f"{{val_loss:.4f}}"
            ),
            **config.training_config.checkpoint_params,
        )
        checkpoint_callback.CHECKPOINT_NAME_LAST = (
            f"{config.get_safe_experiment_name()}_last"
        )
        internal_callbacks: list[Callback] = [
            checkpoint_callback,
            ConfigSaverCallback(
                config.version,
                config.get_safe_experiment_name(),
                config.training_config,
            ),
        ]

        enable_progress_bar = config.training_config.trainer_params.get(
            "enable_progress_bar", True
        )
        if enable_progress_bar:
            internal_callbacks.append(ProgressBarCallback())

        if config.training_config.early_stopping_params:
            internal_callbacks.append(
                EarlyStopping(**config.training_config.early_stopping_params)
            )

        return internal_callbacks + callback_lst

    @staticmethod
    def _create_loggers(
        logger: str | None, experiment_name: str, work_dir: Path
    ) -> list[TensorBoardLogger | WandbLogger | CSVLogger]:
        """Create loggers for the experiment.

        Parameters
        ----------
        logger : str | None
            Logger to use during training. If None, no logger will be used. Available
            loggers are defined in SupportedLogger.
        experiment_name : str
            Name of the experiment, used as a parameter to the loggers.
        work_dir : Path
            The working directory, used as a parameter to the loggers.

        Returns
        -------
        list[TensorBoardLogger | WandbLogger | CSVLogger]
            The list of loggers to use during training.
        """
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

    # Two overloads:
    # - 1st for supported data types & using ReadFuncLoading
    # - 2nd for ImageStackLoading
    # Why:
    #   ImageStackLoading supports any type as input, but we want to tell most users
    #   that they are only allowed Path, str, ndarray or a sequence of these.
    #   The first overload will be displaced first by most code editors, this is what
    #   most users will see.
    @overload
    def train(  # numpydoc ignore=GL08
        self,
        *,
        # BASIC PARAMS
        train_data: InputVar | None = None,
        train_data_target: InputVar | None = None,
        val_data: InputVar | None = None,
        val_data_target: InputVar | None = None,
        # ADVANCED PARAMS
        filtering_mask: InputVar | None = None,
        loading: ReadFuncLoading | None = None,
    ) -> None: ...

    @overload  # any data input is allowed for ImageStackLoading
    def train(  # numpydoc ignore=GL08
        self,
        *,
        # BASIC PARAMS
        train_data: Any | None = None,
        train_data_target: Any | None = None,
        val_data: Any | None = None,
        val_data_target: Any | None = None,
        # ADVANCED PARAMS
        filtering_mask: Any | None = None,
        loading: ImageStackLoading = ...,
    ) -> None: ...

    def train(
        self,
        *,
        # BASIC PARAMS
        train_data: Any | None = None,
        train_data_target: Any | None = None,
        val_data: Any | None = None,
        val_data_target: Any | None = None,
        # ADVANCED PARAMS
        filtering_mask: Any | None = None,
        loading: Loading = None,
    ) -> None:
        """Train the model on the provided data.

        The training data can be provided as arrays or paths.

        Parameters
        ----------
        train_data : pathlib.Path, str, numpy.ndarray, or sequence of these, optional
            Training data, by default None.
        train_data_target : pathlib.Path, str, numpy.ndarray, or sequence of these
            Training target data, by default None.
        val_data : pathlib.Path, str, numpy.ndarray, or sequence of these, optional
            Validation data. If not provided, `data_config.n_val_patches` patches will
            selected from the training data for validation.
        val_data_target : pathlib.Path, str, numpy.ndarray, or sequence of these
            Validation target data, by default None.
        filtering_mask : pathlib.Path, str, numpy.ndarray, or sequence of these
            Filtering mask for coordinate-based patch filtering, by default None.
        loading : Loading, default=None
            Loading strategy to use for the prediction data. May be a ReadFuncLoading or
            ImageStackLoading. If None, uses the loading strategy from the training
            configuration.

        Raises
        ------
        ValueError
            If train_data is not provided.
        """
        if train_data is None:
            raise ValueError("Training data must be provided. Provide `train_data`.")

        if self.config.is_supervised() and train_data_target is None:
            raise ValueError(
                f"Training target data must be provided for supervised training (got "
                f"{self.config.get_algorithm_friendly_name()} algorithm). Provide "
                f"`train_data_target`."
            )

        if (
            self.config.is_supervised()
            and val_data is not None
            and val_data_target is None
        ):
            raise ValueError(
                f"Validation target data must be provided for supervised training (got "
                f"{self.config.get_algorithm_friendly_name()} algorithm). Provide "
                f"`val_data_target`."
            )

        datamodule = CareamicsDataModule(  # type: ignore
            data_config=self.config.data_config,
            train_data=train_data,
            val_data=val_data,
            train_data_target=train_data_target,
            val_data_target=val_data_target,
            train_data_mask=filtering_mask,
            model_constraints=get_model_constraints(self.config.algorithm_config.model),
            loading=loading,  # type: ignore
        )

        self.train_datamodule = datamodule

        # set parameters back to defaults, this is a guard against `stop_training`
        # which changes them in order to interrupt training gracefully
        self.trainer.should_stop = False
        self.trainer.limit_val_batches = 1.0  # equivalent to all validation batches

        self.trainer.fit(
            self.model, datamodule=datamodule, ckpt_path=self.checkpoint_path
        )

    def _build_predict_datamodule(
        self,
        pred_data: Any,
        *,
        pred_data_target: Any | None = None,
        batch_size: int | None = None,
        tile_size: tuple[int, ...] | None = None,
        tile_overlap: tuple[int, ...] | None = (48, 48),
        axes: str | None = None,
        data_type: Literal["array", "tiff", "zarr", "czi", "custom"] | None = None,
        num_workers: int | None = None,
        channels: Sequence[int] | Literal["all"] | None = None,
        in_memory: bool | None = None,
        loading: Loading = None,
    ) -> CareamicsDataModule:
        """Create prediction data module.

        Parameters
        ----------
        pred_data : Any
            Prediction data.
        pred_data_target : Any | None, default=None
            Prediction target data, by default None. Can be used to compute metrics
            during prediction.
        batch_size : int | None, default=None
            Batch size for prediction. If None, uses the batch size from the training
            configuration.
        tile_size : tuple[int, ...] | None, default=None
            Tile size for prediction. If None, uses whole image prediction.
        tile_overlap : tuple[int, ...] | None, default=(48, 48)
            Tile overlap for prediction. If None, defaults to (48, 48).
        axes : str | None, default=None
            Axes for prediction. If None, uses training configuration axes.
        data_type : {"array", "tiff", "zarr", "czi", "custom"} | None, default=None
            Data type for prediction. If None, uses training configuration data type.
        num_workers : int | None, default=None
            Number of workers for data loading during prediction.
        channels : Sequence[int] | Literal["all"] | None, default=None
            Channels to use for prediction. If "all", uses all channels. If None, uses
            the channels from the training configuration.
        in_memory : bool | None, default=None
            Whether to load data into memory during prediction. If None, uses training
            configuration.
        loading : Loading, default=None
            Loading strategy for prediction data if data type (either from training
            configuration or specified) is `"custom"`.

        Returns
        -------
        CareamicsDataModule
            Prediction data module.
        """
        dataloader_params: dict[str, Any] | None = None
        if num_workers is not None:
            dataloader_params = {"num_workers": num_workers}

        pred_data_config = self.config.data_config.convert_mode(
            new_mode="predicting",
            new_patch_size=tile_size,
            overlap_size=tile_overlap,
            new_batch_size=batch_size,
            new_data_type=data_type,
            new_dataloader_params=dataloader_params,
            new_axes=axes,
            new_channels=channels,
            new_in_memory=in_memory,
        )

        # validate new data config against the rest of the configuration by triggering
        # the model level validation
        self.config.model_copy().data_config = pred_data_config

        return CareamicsDataModule(
            data_config=pred_data_config,
            pred_data=pred_data,
            pred_data_target=pred_data_target,
            model_constraints=get_model_constraints(self.config.algorithm_config.model),
            loading=loading,
        )

    # see comment on train func for a description of why we have these two overloads
    @overload  # constrained input data type for supported data or ReadFuncLoading
    def predict(  # numpydoc ignore=GL08
        self,
        # BASIC PARAMS
        pred_data: InputVar,
        *,
        batch_size: int | None = None,
        tile_size: tuple[int, ...] | None = None,
        tile_overlap: tuple[int, ...] | None = (48, 48),
        axes: str | None = None,
        data_type: Literal["array", "tiff", "zarr", "czi", "custom"] | None = None,
        # ADVANCED PARAMS
        num_workers: int | None = None,
        channels: Sequence[int] | Literal["all"] | None = None,
        in_memory: bool | None = None,
        loading: ReadFuncLoading | None = None,
    ) -> tuple[list[NDArray], list[str]]: ...

    @overload  # any data input is allowed for ImageStackLoading
    def predict(  # numpydoc ignore=GL08
        self,
        # BASIC PARAMS
        pred_data: Any,
        *,
        batch_size: int | None = None,
        tile_size: tuple[int, ...] | None = None,
        tile_overlap: tuple[int, ...] | None = (48, 48),
        axes: str | None = None,
        data_type: Literal["array", "tiff", "zarr", "czi", "custom"] | None = None,
        # ADVANCED PARAMS
        num_workers: int | None = None,
        channels: Sequence[int] | Literal["all"] | None = None,
        in_memory: bool | None = None,
        loading: ImageStackLoading = ...,
    ) -> tuple[list[NDArray], list[str]]: ...

    def predict(
        self,
        # BASIC PARAMS
        pred_data: InputVar,
        *,
        batch_size: int | None = None,
        tile_size: tuple[int, ...] | None = None,
        tile_overlap: tuple[int, ...] | None = (48, 48),
        axes: str | None = None,
        data_type: Literal["array", "tiff", "zarr", "czi", "custom"] | None = None,
        # ADVANCED PARAMS
        num_workers: int | None = None,
        channels: Sequence[int] | Literal["all"] | None = None,
        in_memory: bool | None = None,
        loading: Loading = None,
    ) -> tuple[list[NDArray], list[str]]:
        """
        Predict on data and return the predictions.

        Input can be a path to a data file, a list of paths, a numpy array, or a
        list of numpy arrays.

        If `data_type` and `axes` are not provided, the training configuration
        parameters will be used. If `tile_size` is not provided, prediction will
        be performed on the whole image.

        Note that if you are using a UNet model and tiling, the tile size must be
        divisible in every dimension by 2**d, where d is the depth of the model. This
        avoids artefacts arising from the broken shift invariance induced by the
        pooling layers of the UNet. Images smaller than the tile size in any spatial
        dimension will be automatically zero-padded.

        Parameters
        ----------
        pred_data : pathlib.Path, str, numpy.ndarray, or sequence of these
            Data to predict on. Can be a single item or a sequence of paths/arrays.
        batch_size : int, optional
            Batch size for prediction. If not provided, uses the training configuration
            batch size.
        tile_size : tuple of int, optional
            Size of the tiles to use for prediction. If not provided, prediction
            will be performed on the whole image.
        tile_overlap : tuple of int, default=(48, 48)
            Overlap between tiles, can be None.
        axes : str, optional
            Axes of the input data, by default None.
        data_type : {"array", "tiff", "czi", "zarr", "custom"}, optional
            Type of the input data.
        num_workers : int, optional
            Number of workers for the dataloader, by default None.
        channels : sequence of int or "all", optional
            Channels to use from the data. If None, uses the training configuration
            channels.
        in_memory : bool, optional
            Whether to load all data into memory. If None, uses the training
            configuration setting.
        loading : Loading, default=None
            Loading strategy to use for the prediction data. May be a ReadFuncLoading or
            ImageStackLoading. If None, uses the loading strategy from the training
            configuration.

        Returns
        -------
        tuple of (list of NDArray, list of str)
            Predictions made by the model and their source identifiers.

        Raises
        ------
        ValueError
            If tile overlap is not specified when tile_size is provided.
        """
        datamodule = self._build_predict_datamodule(
            pred_data,
            batch_size=batch_size,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            axes=axes,
            data_type=data_type,
            num_workers=num_workers,
            channels=channels,
            in_memory=in_memory,
            loading=loading,
        )

        predictions: list[ImageRegionData] = self.trainer.predict(
            model=self.model, datamodule=datamodule
        )  # type: ignore[assignment]
        tiled = tile_size is not None
        predictions_output, sources = convert_prediction(
            predictions, tiled=tiled, restore_shape=True
        )

        return predictions_output, sources

    # see comment on train func for a description of why we have these two overloads
    @overload  # constrained input data type for supported data or ReadFuncLoading
    def predict_to_disk(  # numpydoc ignore=GL08
        self,
        # BASIC PARAMS
        pred_data: InputVar,
        *,
        pred_data_target: InputVar | None = None,
        prediction_dir: Path | str = "predictions",
        batch_size: int | None = None,
        tile_size: tuple[int, ...] | None = None,
        tile_overlap: tuple[int, ...] | None = (48, 48),
        axes: str | None = None,
        data_type: Literal["array", "tiff", "zarr", "czi", "custom"] | None = None,
        # ADVANCED PARAMS
        num_workers: int | None = None,
        channels: Sequence[int] | Literal["all"] | None = None,
        in_memory: bool | None = None,
        loading: ReadFuncLoading | None = None,
        # WRITE OPTIONS
        write_type: Literal["tiff", "zarr", "custom"] = "tiff",
        write_extension: str | None = None,
        write_func: WriteFunc | None = None,
        write_func_kwargs: dict[str, Any] | None = None,
    ) -> None: ...

    @overload  # any data input is allowed for ImageStackLoading
    def predict_to_disk(  # numpydoc ignore=GL08
        self,
        # BASIC PARAMS
        pred_data: Any,
        *,
        pred_data_target: Any | None = None,
        prediction_dir: Path | str = "predictions",
        batch_size: int | None = None,
        tile_size: tuple[int, ...] | None = None,
        tile_overlap: tuple[int, ...] | None = (48, 48),
        axes: str | None = None,
        data_type: Literal["array", "tiff", "zarr", "czi", "custom"] | None = None,
        # ADVANCED PARAMS
        num_workers: int | None = None,
        channels: Sequence[int] | Literal["all"] | None = None,
        in_memory: bool | None = None,
        loading: ImageStackLoading = ...,
        # WRITE OPTIONS
        write_type: Literal["tiff", "zarr", "custom"] = "tiff",
        write_extension: str | None = None,
        write_func: WriteFunc | None = None,
        write_func_kwargs: dict[str, Any] | None = None,
    ) -> None: ...

    def predict_to_disk(
        self,
        # BASIC PARAMS
        pred_data: Any,
        *,
        pred_data_target: Any | None = None,
        prediction_dir: Path | str = "predictions",
        batch_size: int | None = None,
        tile_size: tuple[int, ...] | None = None,
        tile_overlap: tuple[int, ...] | None = (48, 48),
        axes: str | None = None,
        data_type: Literal["array", "tiff", "zarr", "czi", "custom"] | None = None,
        # ADVANCED PARAMS
        num_workers: int | None = None,
        channels: Sequence[int] | Literal["all"] | None = None,
        in_memory: bool | None = None,
        loading: Loading = None,
        # WRITE OPTIONS
        write_type: Literal["tiff", "zarr", "custom"] = "tiff",
        write_extension: str | None = None,
        write_func: WriteFunc | None = None,
        write_func_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Make predictions on the provided data and save outputs to files.

        Predictions are saved to `prediction_dir` (absolute paths are used as-is,
        relative paths are relative to `work_dir`). The directory structure matches
        the source directory.

        The file names of the predictions will match those of the source. If there is
        more than one sample within a file, the samples will be stacked along the sample
        dimension in the output file.

        If `data_type` and `axes` are not provided, the training configuration
        parameters will be used. If `tile_size` is not provided, prediction
        will be performed on whole images rather than in a tiled manner.

        Note that if you are using a UNet model and tiling, the tile size must be
        divisible in every dimension by 2**d, where d is the depth of the model. This
        avoids artefacts arising from the broken shift invariance induced by the
        pooling layers of the UNet. Images smaller than the tile size in any spatial
        dimension will be automatically zero-padded.

        Parameters
        ----------
        pred_data : pathlib.Path, str, numpy.ndarray, or sequence of these
            Data to predict on. Can be a single item or a sequence of paths/arrays.
        pred_data_target : pathlib.Path, str, numpy.ndarray, or sequence of these
            Prediction data target, by default None.
        prediction_dir : Path | str, default="predictions"
            The path to save the prediction results to. If `prediction_dir` is an
            absolute path, it will be used as-is. If it is a relative path, it will
            be relative to the pre-set `work_dir`. If the directory does not exist it
            will be created.
        batch_size : int, optional
            Batch size for prediction. If not provided, uses the training configuration
            batch size.
        tile_size : tuple of int, optional
            Size of the tiles to use for prediction. If not provided, uses whole image
            strategy.
        tile_overlap : tuple of int, default=(48, 48)
            Overlap between tiles.
        axes : str, optional
            Axes of the input data, by default None.
        data_type : {"array", "tiff", "czi", "zarr", "custom"}, optional
            Type of the input data.
        num_workers : int, optional
            Number of workers for the dataloader, by default None.
        channels : sequence of int or "all", optional
            Channels to use from the data. If None, uses the training configuration
            channels.
        in_memory : bool, optional
            Whether to load all data into memory. If None, uses the training
            configuration setting.
        loading : Loading, default=None
            Loading strategy to use for the prediction data. May be a ReadFuncLoading or
            ImageStackLoading. If None, uses the loading strategy from the training
            configuration.
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
        """
        if write_func_kwargs is None:
            write_func_kwargs = {}

        if Path(prediction_dir).is_absolute():
            write_dir = Path(prediction_dir)
        else:
            write_dir = self.work_dir / prediction_dir
        self.prediction_writer.dirpath = write_dir

        if write_type == "custom":
            if write_extension is None:
                raise ValueError(
                    "A `write_extension` must be provided for custom write types."
                )
            if write_func is None:
                raise ValueError(
                    "A `write_func` must be provided for custom write types."
                )
        if write_type == "zarr" and tile_size is None:
            raise ValueError(
                "Writing prediction to Zarr is only supported with tiling. Please "
                "provide a value for `tile_size`, and optionally `tile_overlap`."
            )

        tiled = tile_size is not None
        self.prediction_writer.set_writing_strategy(
            write_type=write_type,
            tiled=tiled,
            write_func=write_func,
            write_extension=write_extension,
            write_func_kwargs=write_func_kwargs,
        )

        self.prediction_writer.enable_writing(True)

        try:
            datamodule = self._build_predict_datamodule(
                pred_data,
                pred_data_target=pred_data_target,
                batch_size=batch_size,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                axes=axes,
                data_type=data_type,
                num_workers=num_workers,
                channels=channels,
                in_memory=in_memory,
                loading=loading,
            )

            self.trainer.predict(
                model=self.model, datamodule=datamodule, return_predictions=False
            )

        finally:
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
        model_version: str = "0.2.0",
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
        # from .model_io import export_to_bmz

        # output_patch = self.predict(
        #     pred_data=input_array,
        #     data_type=SupportedData.ARRAY.value,
        # )
        # output = np.concatenate(output_patch, axis=0)
        # input_array = reshape_array(input_array, self.config.data_config.axes)

        # export_to_bmz(
        #     model=self.model,
        #     config=self.config,
        #     path_to_archive=path_to_archive,
        #     model_name=friendly_model_name,
        #     general_description=general_description,
        #     data_description=data_description,
        #     authors=authors,
        #     input_array=input_array,
        #     output_array=output,
        #     covers=covers,
        #     channel_names=channel_names,
        #     model_version=model_version,
        # )
        raise NotImplementedError("Exporting to BMZ is not implemented yet.")

    def get_losses(self) -> dict[str, list]:
        """Return data that can be used to plot train and validation loss curves.

        Returns
        -------
        dict of str: list
            Dictionary containing losses for each epoch.
        """
        return read_csv_logger(
            self.config.get_safe_experiment_name(), self.work_dir / "csv_logs"
        )

    def stop_training(self) -> None:
        """Stop the training loop."""
        self.trainer.should_stop = True
        self.trainer.limit_val_batches = 0  # skip validation
