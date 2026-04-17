from collections.abc import Sequence
from typing import Any, Literal, overload

from careamics.config.augmentations import (
    XYFlipConfig,
    XYRandomRotate90Config,
)
from careamics.config.factories.data_factory import (
    list_spatial_augmentations,
)

from ..config_builder import ConfigBuilder, ConfigBuilderT


class DataParamsMixin:
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
        if self.seed is not None:
            minimum_default_data_config["seed"] = self.seed
        for key, value in minimum_default_data_config.items():
            # just in case other mixins add things that we do not want to overwrite
            self.config_dict["data_config"].setdefault(key, value)

    def set_advanced_data_params(
        self: ConfigBuilderT,
        augmentations: Sequence[Literal["x_flip", "y_flip", "rotate_90"]] | None = None,
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
        self.config_dict["data_config"]["normalization"] = {"name": name, **kwargs}
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
