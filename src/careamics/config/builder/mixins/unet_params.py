from typing import Any, Literal

from ..config_builder import ConfigBuilder, ConfigBuilderT


class UnetParamsMixin:
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
