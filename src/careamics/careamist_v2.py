from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypedDict, Unpack

import torch
from numpy.typing import NDArray
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger

from careamics.lightning.dataset_ng.lightning_modules.get_module import CAREamicsModule

from .config import load_configuration_ng
from .config.ng_configs import N2VConfiguration
from .config.support import SupportedLogger
from .file_io import WriteFunc
from .lightning.callbacks import CareamicsCheckpointInfo, ProgressBarCallback
from .lightning.dataset_ng.callbacks.prediction_writer import PredictionWriterCallback
from .lightning.dataset_ng.lightning_modules import (
    CAREamicsModule,
    create_module,
    load_module_from_checkpoint,
)
from .utils import get_logger

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

        self.prediction_writer = PredictionWriterCallback(self.work_dir)
        self.prediction_writer.disable_writing(True)

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
    ) -> None:
        # TODO: init datamodule
        # TODO: remember to pass self.checkpoint_path to Trainer.fit
        # ^ this will load optimizer and lr_schedular state dicts
        raise NotImplementedError("Training is not implemented yet.")

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
        raise NotImplementedError("Exporting to BMZ is not implemented yet.")

    def get_losses(self) -> dict[str, list]:
        raise NotImplementedError("Getting losses is not implemented yet.")

    def stop_training(self) -> None:
        raise NotImplementedError("Stopping training is not implemented yet.")
