from collections.abc import Sequence
from dataclasses import asdict
from typing import Any, Literal, Self

from careamics.config.lightning.training_configuration import (
    SelfSupervisedCheckpointing,
)

from .config_builder import BaseConfigBuilder
from .mixins import (
    DataParamMixin,
    OptimizerParamMixin,
    TrainingParamMixin,
    UnetParamMixin,
)


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

        # set default checkpointing params
        # (can be overwritten with set_checkpoint_params from TrainingParamMixin)
        self.config_dict["training_config"]["checkpoint_params"] = asdict(
            SelfSupervisedCheckpointing()
        )

        # no early stopping by default
        self.config_dict["training_config"]["early_stopping_params"] = None

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

        if self.seed is not None:
            n2v_manipulate_config["seed"] = self.seed

        if use_n2v2 is not None:
            # already added by UnetParamMixin
            assert isinstance(self.config_dict["algorithm_config"]["model"], dict)
            self.config_dict["algorithm_config"]["model"]["n2v2"] = use_n2v2

            n2v_manipulate_config["strategy"] = "median" if use_n2v2 else "uniform"

        self.config_dict["algorithm_config"]["n2v_config"] = n2v_manipulate_config
        return self
