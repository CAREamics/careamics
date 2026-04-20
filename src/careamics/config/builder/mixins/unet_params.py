from typing import Literal

from ..config_builder import ConfigBuilderT


class UnetParamsMixin:

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
