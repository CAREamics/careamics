from collections.abc import Callable
from pathlib import Path
from typing import (
    Any,
    Literal,
    Never,
    TypedDict,
    Unpack,
    overload,
)

from numpy.typing import NDArray
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger

from .config import TrainingConfig
from .config.ng_configs import N2VConfiguration
from .config.support import SupportedLogger
from .dataset_ng.image_stack_loader import ImageStackLoader
from .file_io import WriteFunc
from .lightning.dataset_ng.data_module import CareamicsDataModule
from .lightning.dataset_ng.lightning_modules import (
    CAREamicsModule,
)
from .utils import get_logger

logger = get_logger(__name__)

LOGGER_TYPES = list[TensorBoardLogger | WandbLogger | CSVLogger]

# union of configurations
Configuration = N2VConfiguration


class UserContext(TypedDict):
    work_dir: Path | str | None
    callbacks: list[Callback] | None


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
        # allow overwriting experiment name & training config
        training_config: TrainingConfig | None,
        experiment_name: str | None = None,
        **user_context: Unpack[UserContext],
    ): ...

    # from bmz
    @overload
    def __init__(
        self,
        *,
        bmz_path: Path,
        # allow overwriting experiment name & training config
        training_config: TrainingConfig | None,
        experiment_name: str | None = None,
        **user_context: Unpack[UserContext],
    ): ...

    def __init__(
        self,
        config: Configuration | Path | None = None,
        *,
        checkpoint_path: Path | None = None,
        bmz_path: Path | None = None,
        training_config: TrainingConfig | None = None,
        experiment_name: str | None = None,
        **user_context: Unpack[UserContext],
    ):
        # --- attributes
        self.config: Configuration
        self.model: CAREamicsModule
        self.data_module: CareamicsDataModule
        self.trainer: Trainer
        self.callbacks: list[Callback] = []
        # checkpoint path is saved to restore optimizer etc. state_dicts during training
        # only populated if loading from checkpoint.
        self.checkpoint_path: Path | None
        # ---

        # --- set work_dir
        work_dir = user_context["work_dir"]
        if work_dir is None:
            self.work_dir = Path.cwd()
            logger.warning(
                f"No working directory provided. Using current working directory: "
                f"{self.work_dir}."
            )
        else:
            self.work_dir = Path(work_dir)
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

            self.config, self.model, self.data_module = self._from_config(config)
        elif checkpoint_path is not None:
            self.config, self.model, self.data_module = self._from_checkpoint(
                checkpoint_path, training_config, experiment_name
            )
            self.checkpoint_path = checkpoint_path
        elif bmz_path is not None:
            self.config, self.model, self.data_module = self._from_bmz(
                bmz_path, training_config, experiment_name
            )
        else:
            assert Never  # already covered by xor guard at start of init
        # ---

        # init callbacks
        self.callbacks = self._define_callbacks(
            user_context["callbacks"],
        )

        # TODO: separate into function
        # --- init logger
        csv_logger = CSVLogger(
            name=self.config.experiment_name,
            save_dir=self.work_dir / "csv_logs",
        )
        match self.config.training_config.logger:
            case SupportedLogger.WANDB:
                experiment_logger: LOGGER_TYPES = [
                    WandbLogger(
                        name=self.config.experiment_name,
                        save_dir=self.work_dir / Path("wandb_logs"),
                    ),
                    csv_logger,
                ]
            case SupportedLogger.TENSORBOARD:
                experiment_logger = [
                    TensorBoardLogger(
                        save_dir=self.work_dir / Path("tb_logs"),
                    ),
                    csv_logger,
                ]
            case _:
                experiment_logger = [csv_logger]
        # ---

        # instantiate trainer
        self.trainer = Trainer(
            callbacks=self.callbacks,
            default_root_dir=self.work_dir,
            logger=experiment_logger,
            **self.config.training_config.lightning_trainer_config or {},
        )

    @staticmethod
    def _from_config(
        config: Configuration | Path,
    ) -> tuple[Configuration, CAREamicsModule, CareamicsDataModule]: ...

    @staticmethod
    def _from_checkpoint(
        checkpoint_path: Path,
        # allow overwriting experiment name and training config
        training_config: TrainingConfig | None = None,
        experiment_name: str | None = None,
    ) -> tuple[Configuration, CAREamicsModule, CareamicsDataModule]: ...

    @staticmethod
    def _from_bmz(
        bmz_path: Path,
        # allow overwriting experiment name and training config
        training_config: TrainingConfig | None = None,
        experiment_name: str | None = None,
    ) -> tuple[Configuration, CAREamicsModule, CareamicsDataModule]: ...

    def _define_callbacks(self, callbacks: list[Callback] | None) -> list[Callback]: ...

    def train(
        self,
        *,
        # data init options
        train_data: Any | None = None,
        train_data_target: Any | None = None,
        train_data_mask: Any | None = None,
        val_data: Any | None = None,
        val_data_target: Any | None = None,
        val_percentage: float | None = None,
        val_minimum_split: int = 5,
        # custom loading options
        read_source_func: Callable | None = None,
        read_kwargs: dict[str, Any] | None = None,
        image_stack_loader: ImageStackLoader | None = None,
        image_stack_loader_kwargs: dict[str, Any] | None = None,
        extension_filter: str = "",
        # override max_epochs or max_steps in training config
        # TODO: consider allowing override of all trainer args?
        max_epochs: int | None,
        max_steps: int | None,
    ) -> None: ...

    def predict(
        self,
        # data init options
        pred_data: Any | None = None,
        pred_data_target: Any | None = None,
        # data config updates for prediction
        batch_size: int = 1,
        tile_size: tuple[int, ...] | None = None,
        tile_overlap: tuple[int, ...] | None = (48, 48),
        axes: str | None = None,
        data_type: Literal["array", "tiff", "custom"] | None = None,
        tta_transforms: bool = False,
        dataloader_params: dict | None = None,
        # custom loading options
        read_source_func: Callable | None = None,
        read_kwargs: dict[str, Any] | None = None,
        image_stack_loader: ImageStackLoader | None = None,
        image_stack_loader_kwargs: dict[str, Any] | None = None,
        extension_filter: str = "",
    ) -> list: ...

    def predict_to_disk(
        self,
        # data init options
        pred_data: Any | None = None,
        pred_data_target: Any | None = None,
        # data config updates for prediction
        batch_size: int = 1,
        tile_size: tuple[int, ...] | None = None,
        tile_overlap: tuple[int, ...] | None = (48, 48),
        axes: str | None = None,
        data_type: Literal["array", "tiff", "custom"] | None = None,
        tta_transforms: bool = False,
        dataloader_params: dict | None = None,
        # custom loading options
        read_source_func: Callable | None = None,
        read_kwargs: dict[str, Any] | None = None,
        image_stack_loader: ImageStackLoader | None = None,
        image_stack_loader_kwargs: dict[str, Any] | None = None,
        extension_filter: str = "",
        # write options
        write_type: Literal["tiff", "custom"] = "tiff",
        write_extension: str | None = None,
        write_func: WriteFunc | None = None,
        write_func_kwargs: dict[str, Any] | None = None,
        prediction_dir: Path | str = "predictions",
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
