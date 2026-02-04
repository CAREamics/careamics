from collections.abc import Callable
from pathlib import Path
from typing import (
    Any,
    Literal,
    Never,
    NotRequired,
    TypeAlias,
    TypedDict,
    Unpack,
    overload,
)

import torch
from numpy.typing import NDArray
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger

from .config import TrainingConfig, load_configuration_ng
from .config.ng_configs import N2VConfiguration
from .config.support import SupportedAlgorithm, SupportedLogger
from .file_io import WriteFunc
from .lightning.callbacks import CareamicsCheckpointInfo, ProgressBarCallback
from .lightning.dataset_ng.callbacks.prediction_writer import PredictionWriterCallback
from .lightning.dataset_ng.data_module import CareamicsDataModule
from .lightning.dataset_ng.lightning_modules import (
    CAREamicsModule,
    create_module,
    get_module_cls,
)
from .utils import get_logger

logger = get_logger(__name__)

ExperimentLogger: TypeAlias = TensorBoardLogger | WandbLogger | CSVLogger

Configuration = N2VConfiguration


class CareamicsInfo(
    TypedDict
):  # TODO: do we need this if we do not implement fine-tuning for now?
    version: NotRequired[str]
    experiment_name: str
    training_config: TrainingConfig | dict[str, Any]


class UserContext(
    TypedDict
):  # TODO: check that this is displayed correctly in the docs
    work_dir: Path | str | None
    callbacks: list[Callback] | None
    enable_progress_bar: bool


class CAREamistV2:

    # from configuration
    @overload
    def __init__(
        self,
        config: Configuration | Path,
        **user_context: Unpack[UserContext],
    ): ...

    # from checkpoint
    @overload
    def __init__(
        self,
        *,
        checkpoint_path: Path,
        **user_context: Unpack[UserContext],
    ): ...

    # from bmz
    @overload
    def __init__(
        self,
        *,
        bmz_path: Path,
        **user_context: Unpack[UserContext],
    ): ...

    def __init__(
        self,
        config: Configuration | Path | None = None,
        *,
        checkpoint_path: Path | None = None,
        bmz_path: Path | None = None,
        **user_context: Unpack[UserContext],
    ):
        # --- attributes
        self.config: Configuration
        self.model: CAREamicsModule
        self.data_module: CareamicsDataModule
        self.trainer: Trainer
        self.callbacks: list[Callback]
        # checkpoint path is saved to restore optimizer etc. state_dicts during training
        # only populated if loading from checkpoint.
        self.checkpoint_path = checkpoint_path
        self.work_dir = self._resolve_work_dir(user_context["work_dir"])
        # ---

        # --- init modules from config, checkpoint_path or bmz_path
        # guard against multiple types of input
        config_is_input = config is not None
        ckpt_is_input = checkpoint_path is not None
        bmz_is_input = bmz_path is not None
        # three way xor
        if (config_is_input ^ ckpt_is_input) or (ckpt_is_input ^ bmz_is_input):
            raise ValueError(
                "Only one of `config`, `checkpoint_path` or `bmz_path` can be used as "
                "input."
            )
        if config is not None:
            # TODO: raise errors if training config or experiment name are not None

            self.config, self.model = self._from_config(config)
        elif checkpoint_path is not None:
            self.config, self.model = self._from_checkpoint(checkpoint_path)
        elif bmz_path is not None:
            self.config, self.model = self._from_bmz(bmz_path)
        else:
            assert Never  # already covered by xor guard
        # ---

        # override progress bar choice
        self.config.training_config.lightning_trainer_config["enable_progress_bar"] = (
            user_context.get("enable_progress_bar", True)
        )

        # init callbacks
        self.prediction_writer = PredictionWriterCallback(self.work_dir)
        self.prediction_writer.enable_writing(True)
        self.callbacks = self._define_callbacks(
            user_context["callbacks"], self.config, self.work_dir
        )

        # init loggers
        if (self.config.training_config.logger) is not None:
            logger = SupportedLogger(self.config.training_config.logger)
        else:
            logger = None
        experiment_loggers = self._create_loggers(
            logger,
            self.config.experiment_name,
            self.work_dir,
        )

        # instantiate trainer
        self.trainer = Trainer(
            callbacks=[self.prediction_writer, *self.callbacks],
            default_root_dir=self.work_dir,
            logger=experiment_loggers,
            **self.config.training_config.lightning_trainer_config or {},
        )

    @staticmethod
    def _from_config(
        config: Configuration | Path,
    ) -> tuple[Configuration, CAREamicsModule]:
        if isinstance(config, Path):
            config = load_configuration_ng(config)
        assert not isinstance(config, Path)  # mypy not resolving

        model = create_module(config.algorithm_config)
        return config, model

    @staticmethod
    def _from_checkpoint(
        checkpoint_path: Path,
    ) -> tuple[Configuration, CAREamicsModule]:
        # map to cpu because we are only extracting the hyper-params here
        # loading weights will be handled by LightningModule.load_from_checkpoint
        checkpoint: dict = torch.load(checkpoint_path, map_location="cpu")

        # --- get careamics info, aka training_config, experiment_name and version,
        # (only loaded if not overridden)
        # (maybe version should get overwritten?)
        careamics_info = checkpoint.get("careamics_info", None)
        if careamics_info is None:
            raise ValueError(
                "Could not find CAREamics related information within the provided "
                "checkpoint. This means that it was saved without using the "
                "CAREamics callback `CAREamicsCheckpointInfo`. To load this "
                "checkpint with the CAREamist API please provide an "
                "`experiment_name` and `training_config`."
            )
        assert careamics_info is not None  # mypy not resolving
        # ---

        # --- get module hyperparams, aka algorithm config
        try:
            algorithm_config: dict[str, Any] = checkpoint["hyper_parameters"][
                "algorithm_config"
            ]
            # get algorithm type so that the correct module can be instantiated
            algorithm = algorithm_config["algorithm"]
            algorithm = SupportedAlgorithm(algorithm)
        except (IndexError, ValueError) as e:
            raise ValueError(
                "Could not determine CAREamics supported algorithm from the provided "
                f"checkpoint at: {checkpoint_path!s}."
            ) from e
        # ---

        # --- get datamodule_hyper_parameters aka, data_config
        data_hparams_key = checkpoint.get(
            "datamodule_hparams_name", "datamodule_hyper_parameters"
        )
        try:
            data_config: dict[str, Any] = checkpoint[data_hparams_key]["data_config"]
        except IndexError as e:
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
        # ---

        # instantiate correct algorithm module
        module_cls = get_module_cls(algorithm)
        module = module_cls.load_from_checkpoint(checkpoint_path)
        return config, module

    @staticmethod
    def _from_bmz(
        bmz_path: Path,
    ) -> tuple[Configuration, CAREamicsModule]: ...

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
        # check that user callbacks are not any of the CAREamics callbacks
        for c in callbacks:
            if isinstance(c, ModelCheckpoint) or isinstance(c, EarlyStopping):
                raise ValueError(
                    "`ModelCheckpoint` and `EarlyStopping` callbacks are already "
                    "defined in CAREamics and should only be modified through the "
                    "training configuration (see TrainingConfig)."
                )

            if isinstance(c, CareamicsCheckpointInfo) or isinstance(
                c, ProgressBarCallback
            ):
                raise ValueError(
                    "`CareamicsCheckpointInfo` and `ProgressBar` callbacks are defined "
                    "internally and should not be passed as callbacks."
                )

        interal_callbacks = [
            ModelCheckpoint(
                dirpath=work_dir / Path("checkpoints"),
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
            interal_callbacks.append(ProgressBarCallback())

        if config.training_config.early_stopping_callback is not None:
            interal_callbacks.append(
                EarlyStopping(
                    **config.training_config.early_stopping_callback.model_dump()
                )
            )

        return interal_callbacks + callbacks

    @staticmethod
    def _create_loggers(
        logger: SupportedLogger | None, experiment_name: str, work_dir: Path
    ) -> list[ExperimentLogger]:
        csv_logger = CSVLogger(
            name=experiment_name,
            save_dir=work_dir / "csv_logs",
        )
        match logger:
            case SupportedLogger.WANDB:
                experiment_loggers: list = [
                    WandbLogger(
                        name=experiment_name,
                        save_dir=work_dir / Path("wandb_logs"),
                    ),
                    csv_logger,
                ]
            case SupportedLogger.TENSORBOARD:
                experiment_loggers = [
                    TensorBoardLogger(
                        save_dir=work_dir / Path("tb_logs"),
                    ),
                    csv_logger,
                ]
            case _:
                experiment_loggers = [csv_logger]
        return experiment_loggers

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
    ) -> None: ...

    # TODO: init datamodule
    # TODO: remember to pass self.checkpoint_path to Trainer.fit
    # ^ this will load optimizer and lr_schedular state dicts

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
        return None

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
    ) -> None: ...

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
    ) -> None: ...

    def get_losses(self) -> dict[str, list]: ...

    def stop_training(self) -> None: ...
