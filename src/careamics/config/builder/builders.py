from collections.abc import Sequence
from dataclasses import asdict
from typing import Any, Literal, Self

from careamics.config.lightning.training_configuration import (
    SelfSupervisedCheckpointing,
    SupervisedCheckpointing,
)

from .config_builder import BaseConfigBuilder
from .mixins import (
    DataParamMixin,
    OptimizerParamMixin,
    TrainingParamMixin,
    UnetParamMixin,
)


class CAREConfigBuilder(
    TrainingParamMixin,
    DataParamMixin,
    UnetParamMixin,
    OptimizerParamMixin,
    BaseConfigBuilder,
):
    def __init__(
        self,
        experiment_name: str,
        data_type: Literal["array", "tiff", "zarr", "czi", "custom"],
        axes: str,
        patch_size: Sequence[int],
        batch_size: int,
        # optional
        num_epochs: int = 30,
        num_steps: int | None = None,
        n_channels_in: int | None = None,
        n_channels_out: int | None = None,
        seed: int | None = None,
    ):
        super().__init__(
            experiment_name,
            data_type,
            axes,
            patch_size,
            batch_size,
            num_epochs=num_epochs,
            num_steps=num_steps,
            n_channels_in=n_channels_in,
            n_channels_out=n_channels_out,
            seed=seed,
        )

        # set default checkpointing params
        # (can be overwritten with set_checkpoint_params from TrainingParamMixin)
        self.config_dict["training_config"]["checkpoint_params"] = asdict(
            SupervisedCheckpointing()
        )

        self.config_dict["training_config"]["early_stopping_params"] = {
            "monitor": "val_loss",
            "mode": "min",
        }

    def set_loss(self, loss: Literal["mae", "mse"]) -> Self:
        self.config_dict["algorithm_config"]["loss"] = loss
        return self


class N2NConfigBuilder(
    TrainingParamMixin,
    DataParamMixin,
    UnetParamMixin,
    OptimizerParamMixin,
    BaseConfigBuilder,
):
    def __init__(
        self,
        experiment_name: str,
        data_type: Literal["array", "tiff", "zarr", "czi", "custom"],
        axes: str,
        patch_size: Sequence[int],
        batch_size: int,
        # optional
        num_epochs: int = 30,
        num_steps: int | None = None,
        n_channels_in: int | None = None,
        n_channels_out: int | None = None,
        seed: int | None = None,
    ):
        super().__init__(
            experiment_name,
            data_type,
            axes,
            patch_size,
            batch_size,
            num_epochs=num_epochs,
            num_steps=num_steps,
            n_channels_in=n_channels_in,
            n_channels_out=n_channels_out,
            seed=seed,
        )

        # set default checkpointing params (n2n self supervised)
        # (can be overwritten with set_checkpoint_params from TrainingParamMixin)
        self.config_dict["training_config"]["checkpoint_params"] = asdict(
            SelfSupervisedCheckpointing()
        )

        # no early stopping by default
        self.config_dict["training_config"]["early_stopping_params"] = None

    def set_loss(self, loss: Literal["mae", "mse"]) -> Self:
        self.config_dict["algorithm_config"]["loss"] = loss
        return self


class N2VConfigBuilder(
    TrainingParamMixin,
    DataParamMixin,
    UnetParamMixin,
    OptimizerParamMixin,
    BaseConfigBuilder,
):
    def __init__(
        self,
        experiment_name: str,
        data_type: Literal["array", "tiff", "zarr", "czi", "custom"],
        axes: str,
        patch_size: Sequence[int],
        batch_size: int,
        # optional
        num_epochs: int = 30,
        num_steps: int | None = None,
        n_channels: int | None = None,
        seed: int | None = None,
    ):
        super().__init__(
            experiment_name,
            data_type,
            axes,
            patch_size,
            batch_size,
            num_epochs=num_epochs,
            num_steps=num_steps,
            n_channels_in=n_channels,
            n_channels_out=n_channels,
            seed=seed,
        )
        self.config_dict["algorithm_config"]["algorithm"] = "n2v"

        # this will be used to propagate the monitor metric before building the config
        # we have to wait for after set_checkpoint_params and set_early_stopping_params
        # it can be changed using the set_monitor_metric method
        self.monitor_metric: Literal["train_loss", "train_loss_epoch", "val_loss"] = (
            "val_loss"
        )

        # set default checkpointing params
        # (can be overwritten with set_checkpoint_params from TrainingParamMixin)
        self.config_dict["training_config"]["checkpoint_params"] = asdict(
            SelfSupervisedCheckpointing()
        )

        # no early stopping by default
        self.config_dict["training_config"]["early_stopping_params"] = None

        # propagate seed
        self.config_dict["algorithm_config"]["n2v_config"] = {}
        if self.seed is not None:
            self.config_dict["algorithm_config"]["n2v_config"]["seed"] = self.seed

    def set_n2v_params(
        self,
        use_n2v2: bool | None = None,
        roi_size: int | None = None,
        masked_pixel_percentage: float | None = None,
        # - structN2V specific
        struct_n2v_axis: Literal["horizontal", "vertical", "none"] | None = None,
        struct_n2v_span: int | None = None,
    ) -> Self:
        n2v_manipulate_config: dict[str, Any] = {}
        if roi_size is not None:
            n2v_manipulate_config["roi_size"] = roi_size

        if masked_pixel_percentage is not None:
            n2v_manipulate_config["masked_pixel_percentage"] = masked_pixel_percentage

        if struct_n2v_axis is not None:
            n2v_manipulate_config["struct_n2v_axis"] = struct_n2v_axis

        if struct_n2v_span is not None:
            n2v_manipulate_config["struct_n2v_span"] = struct_n2v_span

        if use_n2v2 is not None:
            # already added by UnetParamMixin
            assert isinstance(self.config_dict["algorithm_config"]["model"], dict)
            self.config_dict["algorithm_config"]["model"]["n2v2"] = use_n2v2

            n2v_manipulate_config["strategy"] = "median" if use_n2v2 else "uniform"

        assert isinstance(self.config_dict["algorithm_config"]["n2v_config"], dict)
        self.config_dict["algorithm_config"]["n2v_config"].update(n2v_manipulate_config)
        return self

    def set_monitor_metric(
        self, monitor_metric: Literal["train_loss", "train_loss_epoch", "val_loss"]
    ) -> Self:
        self.monitor_metric = monitor_metric
        self.config_dict["algorithm_config"]["monitor_metric"] = monitor_metric
        return self

    def _propagate_monitor_to_callbacks(self):
        # only overwrite monitor if it not explicitly set
        assert isinstance(
            self.config_dict["training_config"]["checkpoint_params"], dict
        )
        checkpoint_params = self.config_dict["training_config"]["checkpoint_params"]
        if "monitor" not in checkpoint_params:
            checkpoint_params["monitor"] = self.monitor_metric

        early_stopping_params = self.config_dict["training_config"][
            "early_stopping_params"
        ]
        has_early_stopping = early_stopping_params is not None
        if has_early_stopping and "monitor" not in early_stopping_params:
            assert isinstance(early_stopping_params, dict)
            early_stopping_params["monitor"] = self.monitor_metric

    def before_build(self) -> None:
        self._propagate_monitor_to_callbacks()
        return super().before_build()
