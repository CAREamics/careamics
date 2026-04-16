from collections.abc import Sequence
from typing import Any, Literal, TypeVar, overload

from careamics.config.augmentations import (
    XYFlipConfig,
    XYRandomRotate90Config,
)
from careamics.config.factories.data_factory import (
    list_spatial_augmentations,
)
from careamics.config.factories.training_factory import update_trainer_params

from .config_builder import ConfigBuilder

ConfigBuilderT = TypeVar("ConfigBuilderT", bound=ConfigBuilder)


class TrainingParamMixin:
    def __init__(self: ConfigBuilder, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.config_dict["training_config"].setdefault(
            "trainer_params", update_trainer_params({}, self.num_epochs, self.num_steps)
        )

    def set_trainer_params(self: ConfigBuilderT, **kwargs: Any) -> ConfigBuilderT:
        self.config_dict["training_config"].update(kwargs)
        return self

    def set_checkpoint_params(self: ConfigBuilderT, **kwargs: Any) -> ConfigBuilderT:
        self.config_dict["training_config"]["checkpoint_params"] = kwargs
        return self

    def early_stopping_params(self: ConfigBuilderT, **kwargs: Any) -> ConfigBuilderT:
        self.config_dict["training_config"]["early_stopping_params"] = kwargs
        return self

    def set_logger(
        self: ConfigBuilderT, name: Literal["wandb", "tensorboard"] | None = None
    ) -> ConfigBuilderT:
        self.config_dict["training_config"]["logger"] = name
        return self


class DataParamMixin:
    def __init__(self: ConfigBuilder, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        minimum_default_data_config: dict[str, Any] = {
            "mode": "training",
            "data_type": self.data_type,
            "axes": self.axes,
            "patching": {
                "name": "stratified",
                "patch_size": self.patch_size,
                "seed": self.seed,
            },
            "batch_size": self.batch_size,
        }
        for key, value in minimum_default_data_config.items():
            # just in case other mixins add things that we do not want to overwrite
            self.config_dict["data_config"].setdefault(key, value)

    def set_advanced_data_params(
        self: ConfigBuilderT,
        augmentations: Literal["x_flip", "y_flip", "rotate_90"] | None = None,
        in_memory: bool | None = None,
        channels: Sequence[int] | None = None,
        patching_strategy: Literal["random", "stratified"] | None = None,
        n_val_patches: int | None = None,
    ) -> ConfigBuilderT:
        augs: list[XYFlipConfig | XYRandomRotate90Config] | None = None
        if augmentations is not None:
            augs = []

            x_flip_present = "x_flip" in augmentations
            y_flip_present = "y_flip" in augmentations
            rotate_90_present = "rotate_90" in augmentations

            if x_flip_present or y_flip_present:
                augs.append(
                    XYFlipConfig(
                        flip_x=x_flip_present, flip_y=y_flip_present, seed=self.seed
                    )
                )
            if rotate_90_present:
                augs.append(XYRandomRotate90Config(seed=self.seed))
            spatial_transforms = list_spatial_augmentations(augs)
            self.config_dict["data_config"]["augmentations"] = spatial_transforms

        if in_memory is not None:
            self.config_dict["data_config"]["in_memory"] = in_memory

        if channels is not None:
            self.config_dict["data_config"]["channels"] = channels

        if patching_strategy is not None:
            self.config_dict["data_config"]["patching"] = {
                "name": patching_strategy,
                "patch_size": self.patch_size,
                "seed": self.seed,  # both random and stratified accept seed
            }

        if n_val_patches is not None:
            self.config_dict["data_config"]["n_val_patches"] = n_val_patches

        return self

    def set_data_loader_params(
        self: ConfigBuilderT,
        num_workers: int | None = None,
        train_dataloader_params: dict[str, Any] | None = None,
        val_dataloader_params: dict[str, Any] | None = None,
        pred_dataloader_params: dict[str, Any] | None = None,
    ) -> ConfigBuilderT:
        if num_workers is not None:
            self.config_dict["data_config"]["num_workers"] = num_workers
        if train_dataloader_params is not None:
            self.config_dict["data_config"][
                "train_dataloader_params"
            ] = train_dataloader_params
        if val_dataloader_params is not None:
            self.config_dict["data_config"][
                "val_dataloader_params"
            ] = val_dataloader_params
        if pred_dataloader_params is not None:
            self.config_dict["data_config"][
                "pred_dataloader_params"
            ] = pred_dataloader_params
        return self

    @overload
    def set_normalization(
        self: ConfigBuilderT,
        name: Literal["mean_std"],
        *,
        per_channel: bool | None = None,
        input_means: Sequence[float] | None = None,
        input_stds: Sequence[float] | None = None,
        target_means: Sequence[float] | None = None,
        target_stds: Sequence[float] | None = None,
    ) -> ConfigBuilderT: ...

    @overload
    def set_normalization(
        self: ConfigBuilderT,
        name: Literal["min_max"],
        *,
        per_channel: bool | None = None,
        input_mins: Sequence[float] | None = None,
        input_maxes: Sequence[float] | None = None,
        target_mins: Sequence[float] | None = None,
        target_maxes: Sequence[float] | None = None,
    ) -> ConfigBuilderT: ...

    @overload
    def set_normalization(
        self: ConfigBuilderT,
        name: Literal["quantile"],
        *,
        per_channel: bool = True,
        lower_quantiles: Sequence[float] | None = None,
        upper_quantiles: Sequence[float] | None = None,
        input_lower_quantile_values: Sequence[float] | None = None,
        input_upper_quantile_values: Sequence[float] | None = None,
        target_lower_quantile_values: Sequence[float] | None = None,
        target_upper_quantile_values: Sequence[float] | None = None,
    ) -> ConfigBuilderT: ...

    @overload
    def set_normalization(
        self: ConfigBuilderT,
        name: Literal["none"],
    ) -> ConfigBuilderT: ...

    def set_normalization(
        self: ConfigBuilderT,
        name: Literal["mean_std", "min_max", "quantile", "none"],
        **kwargs: Any,
    ) -> ConfigBuilderT:
        self.config_dict["data_config"]["normalization"] = {name: name, **kwargs}
        return self

    @overload
    def set_patch_filter(
        self: ConfigBuilderT,
        name: Literal["shannon"],
        ref_channel: int | None = None,
        filtered_patch_prob: float | None = None,
        *,
        mean_threshold: float,
        std_threshold: float | None = None,
    ) -> ConfigBuilderT: ...

    @overload
    def set_patch_filter(
        self: ConfigBuilderT,
        name: Literal["shannon"],
        ref_channel: int | None = None,
        filtered_patch_prob: float | None = None,
        *,
        threshold: float,
    ) -> ConfigBuilderT: ...

    @overload
    def set_patch_filter(
        self: ConfigBuilderT,
        name: Literal["max"],
        ref_channel: int | None = None,
        filtered_patch_prob: float | None = None,
        *,
        threshold: float,
        # TODO: Could calculate coverage default here (depends on is_3D)
        coverage: float | None = None,
    ) -> ConfigBuilderT: ...

    def set_patch_filter(
        self: ConfigBuilderT,
        name: Literal["max", "shannon", "mean_std"],
        ref_channel: int | None = None,
        filtered_patch_prob: float | None = None,
        **kwargs: Any,
    ) -> ConfigBuilderT:
        filter_config_dict: dict[str, Any] = {"name": name}
        if ref_channel is not None:
            filter_config_dict["ref_channel"] = ref_channel
        if filtered_patch_prob is not None:
            filter_config_dict["filtered_patch_prob"] = filtered_patch_prob

        self.config_dict["data_config"]["patch_filter"] = {
            **filter_config_dict,
            **kwargs,
        }
        return self


class OptimizerParamMixin:
    def __init__(self: ConfigBuilder, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def set_optimizer(
        self: ConfigBuilderT, name: Literal["Adam", "Adamax", "SGD"], **kwargs: Any
    ) -> ConfigBuilderT:
        self.config_dict["algorithm_config"]["optimizer"] = {
            "name": name,
            "parameters": kwargs,
        }
        return self

    def set_lr_scheduler(
        self: ConfigBuilderT,
        name: Literal["ReduceLROnPlateau", "StepLR"],
        **kwargs: Any,
    ) -> ConfigBuilderT:
        self.config_dict["algorithm_config"]["lr_scheduler"] = {
            "name": name,
            "parameters": kwargs,
        }
        return self


class UnetParamMixin:
    def __init__(self: ConfigBuilder, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        model_config: dict[str, Any] = {
            "conv_dims": 3 if self.is_3D else 2,
        }
        if self.n_channels_in is not None:
            model_config["in_channels"] = self.n_channels_in
        if self.n_channels_out is not None:
            model_config["num_classes"] = self.n_channels_out

        self.config_dict["algorithm_config"].setdefault("model", {})
        assert isinstance(self.config_dict["algorithm_config"]["model"], dict)

        # TODO: error if architecture already exists in data_config?
        self.config_dict["algorithm_config"]["model"]["architecture"] = "UNet"
        for key, value in model_config.items():
            self.config_dict["algorithm_config"]["model"].setdefault(key, value)

    def set_model_params(
        self: ConfigBuilderT,
        independent_channels: bool | None = None,
        depth: int | None = None,
        num_channels_init: int | None = None,
        residual: bool | None = None,
        final_activation: (
            Literal["None", "Sigmoid", "Softmax", "Tanh", "ReLU", "LeakyReLU"] | None
        ) = None,
        use_batch_norm: bool | None = None,
    ) -> ConfigBuilderT:
        assert isinstance(self.config_dict["algorithm_config"]["model"], dict)

        if independent_channels is not None:
            self.config_dict["algorithm_config"]["model"][
                "independent_channels"
            ] = independent_channels

        if depth is not None:
            self.config_dict["algorithm_config"]["model"]["depth"] = depth

        if num_channels_init is not None:
            self.config_dict["algorithm_config"]["model"][
                "num_channels_init"
            ] = num_channels_init

        if residual is not None:
            self.config_dict["algorithm_config"]["model"]["residual"] = residual

        if final_activation is not None:
            self.config_dict["algorithm_config"]["model"][
                "final_activation"
            ] = final_activation

        if use_batch_norm is not None:
            self.config_dict["algorithm_config"]["model"][
                "use_batch_norm"
            ] = use_batch_norm
        return self
