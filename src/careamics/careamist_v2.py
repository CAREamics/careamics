from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypedDict, Union, Unpack

import numpy as np
import torch
from numpy.typing import NDArray
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger

from .config import load_configuration_ng
from .config.ng_configs import N2VConfiguration
from .config.support import SupportedAlgorithm, SupportedData, SupportedLogger
from .dataset.dataset_utils import reshape_array
from .file_io import WriteFunc
from .lightning.callbacks import CareamicsCheckpointInfo, ProgressBarCallback
from .lightning.dataset_ng.callbacks.prediction_writer import PredictionWriterCallback
from .lightning.dataset_ng.data_module import CareamicsDataModule
from .lightning.dataset_ng.lightning_modules import (
    CAREamicsModule,
    create_module,
    load_module_from_checkpoint,
)
from .model_io import export_to_bmz
from .utils import check_path_exists, get_logger
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

        # Placeholder for the datamodule
        self.train_datamodule: CareamicsDataModule | None = None

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
                "Exactly one of `config`, `checkpoint_path`, or `bmz_path` must be provided."
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
                "Please use a checkpoint saved with CAREamics or initialize with a config instead."
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
        use_in_memory: bool = True,
    ) -> None:
        """Train the model on the provided data.

        The training data can be provided as arrays or paths. If `use_in_memory`
        is set to True, the source provided as Path or str will be loaded in memory
        if it fits. Otherwise, training will be performed by loading patches from
        the files one by one. Training on arrays is always performed in memory.

        If no validation data is provided, then the validation is extracted from
        the training data using default values (10% with minimum 5 patches/files).

        Parameters
        ----------
        train_data : pathlib.Path or str or numpy.ndarray, optional
            Training data, by default None.
        train_data_target : pathlib.Path or str or numpy.ndarray, optional
            Training target data, by default None.
        val_data : pathlib.Path or str or numpy.ndarray, optional
            Validation data, by default None.
        val_data_target : pathlib.Path or str or numpy.ndarray, optional
            Validation target data, by default None.
        filtering_mask : Any, optional
            Filtering mask for coordinate-based patch filtering, by default None.
        read_source_func : Callable, optional
            Function to read the source data, by default None.
        read_kwargs : dict[str, Any], optional
            Keyword arguments for the read function, by default None.
        extension_filter : str, optional
            Filter for file extensions, by default "".
        use_in_memory : bool, optional
            Use in memory dataset if possible, by default True.

        Raises
        ------
        ValueError
            If sources are not of the same type (e.g. train is an array and val is
            a Path).
        ValueError
            If the training target is provided to N2V.
        ValueError
            If neither train_data nor a datamodule is provided.
        """
        if train_data is None:
            raise ValueError(
                "Training data must be provided. Either provide `train_data` or "
                "use a datamodule (not yet supported)."
            )

        # Check that inputs are the same type
        source_types = {
            type(s)
            for s in (train_data, val_data, train_data_target, val_data_target)
            if s is not None
        }
        if len(source_types) > 1:
            raise ValueError("All sources should be of the same type.")

        # Raise error if target is provided to N2V
        if self.config.algorithm_config.algorithm == SupportedAlgorithm.N2V.value:
            if train_data_target is not None:
                raise ValueError("Training target not compatible with N2V training.")

        # Dispatch the training based on input type
        if isinstance(train_data, np.ndarray):
            # mypy checks
            assert isinstance(val_data, np.ndarray) or val_data is None
            assert (
                isinstance(train_data_target, np.ndarray) or train_data_target is None
            )
            assert isinstance(val_data_target, np.ndarray) or val_data_target is None

            self._train_on_array(
                train_data,
                val_data,
                train_data_target,
                val_data_target,
            )

        elif isinstance(train_data, (Path, str)):
            # mypy checks
            assert isinstance(val_data, (Path, str)) or val_data is None
            assert (
                isinstance(train_data_target, (Path, str)) or train_data_target is None
            )
            assert isinstance(val_data_target, (Path, str)) or val_data_target is None

            self._train_on_path(
                train_data,
                val_data,
                train_data_target,
                val_data_target,
                use_in_memory,
                read_source_func,
                read_kwargs,
                extension_filter,
            )

        else:
            raise ValueError(
                f"Invalid input, expected a str, Path, or array "
                f"(got {type(train_data)})."
            )

    def _train_on_datamodule(self, datamodule: CareamicsDataModule) -> None:
        """Train the model on the provided datamodule.

        Parameters
        ----------
        datamodule : CareamicsDataModule
            Datamodule to train on.
        """
        # Register datamodule
        self.train_datamodule = datamodule

        # Set defaults (in case `stop_training` was called before)
        self.trainer.should_stop = False
        self.trainer.limit_val_batches = 1.0  # 100%

        # Train
        ckpt_path = (
            str(self.checkpoint_path) if self.checkpoint_path is not None else None
        )
        self.trainer.fit(self.model, datamodule=datamodule, ckpt_path=ckpt_path)

    def _train_on_array(
        self,
        train_data: NDArray,
        val_data: NDArray | None = None,
        train_target: NDArray | None = None,
        val_target: NDArray | None = None,
    ) -> None:
        """Train the model on the provided data arrays.

        Parameters
        ----------
        train_data : NDArray
            Training data.
        val_data : NDArray, optional
            Validation data, by default None.
        train_target : NDArray, optional
            Train target data, by default None.
        val_target : NDArray, optional
            Validation target data, by default None.
        """
        # Create datamodule
        datamodule = CareamicsDataModule(
            data_config=self.config.data_config,
            train_data=train_data,
            val_data=val_data,
            train_data_target=train_target,
            val_data_target=val_target,
            read_source_func=None,
            read_kwargs=None,
            extension_filter="",
            val_percentage=0.1,
            val_minimum_split=5,
        )

        # Train
        self._train_on_datamodule(datamodule)

    def _train_on_path(
        self,
        path_to_train_data: Union[Path, str],
        path_to_val_data: Union[Path, str] | None = None,
        path_to_train_target: Union[Path, str] | None = None,
        path_to_val_target: Union[Path, str] | None = None,
        use_in_memory: bool = True,
        read_source_func: Callable | None = None,
        read_kwargs: dict[str, Any] | None = None,
        extension_filter: str = "",
    ) -> None:
        """Train the model on the provided data paths.

        Parameters
        ----------
        path_to_train_data : pathlib.Path or str
            Path to the training data.
        path_to_val_data : pathlib.Path or str, optional
            Path to validation data, by default None.
        path_to_train_target : pathlib.Path or str, optional
            Path to train target data, by default None.
        path_to_val_target : pathlib.Path or str, optional
            Path to validation target data, by default None.
        use_in_memory : bool, optional
            Use in memory dataset if possible, by default True.
        """
        # Sanity check on data (path exists)
        path_to_train_data = check_path_exists(path_to_train_data)

        if path_to_val_data is not None:
            path_to_val_data = check_path_exists(path_to_val_data)

        if path_to_train_target is not None:
            path_to_train_target = check_path_exists(path_to_train_target)

        if path_to_val_target is not None:
            path_to_val_target = check_path_exists(path_to_val_target)

        # Update in_memory setting in data_config
        original_in_memory = self.config.data_config.in_memory
        self.config.data_config.in_memory = use_in_memory

        try:
            # Create datamodule
            datamodule = CareamicsDataModule(
                data_config=self.config.data_config,
                train_data=path_to_train_data,
                val_data=path_to_val_data,
                train_data_target=path_to_train_target,
                val_data_target=path_to_val_target,
                read_source_func=read_source_func,
                read_kwargs=read_kwargs,
                extension_filter=extension_filter,
                val_percentage=0.1,
                val_minimum_split=5,
            )

            # Train
            self._train_on_datamodule(datamodule)
        finally:
            # Restore original in_memory setting
            self.config.data_config.in_memory = original_in_memory

    def predict(
        self,
        # BASIC PARAMS
        pred_data: Any | None = None,
        batch_size: int = 1,
        tile_size: tuple[int, ...] | None = None,
        tile_overlap: tuple[int, ...] | None = (48, 48),
        axes: str | None = None,
        data_type: Literal["array", "tiff", "custom"] | None = None,
        # ADVANCED PARAMS
        # tta_transforms: bool = False, # TODO: hidden till implemented
        num_workers: int | None = None,
        read_source_func: Callable | None = None,
        read_kwargs: dict[str, Any] | None = None,
        extension_filter: str = "",
    ) -> None:
        raise NotImplementedError("Predicting is not implemented yet.")

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
        data_type: Literal["array", "tiff", "custom"] | None = None,
        # ADVANCED PARAMS
        num_workers: int | None = None,
        read_source_func: Callable | None = None,
        read_kwargs: dict[str, Any] | None = None,
        extension_filter: str = "",
        # WRITE OPTIONS
        write_type: Literal["tiff", "zarr", "custom"] = "tiff",
        write_extension: str | None = None,
        write_func: WriteFunc | None = None,
        write_func_kwargs: dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError("Predicting to disk is not implemented yet.")

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
