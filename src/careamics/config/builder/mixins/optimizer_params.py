from typing import Any, Literal

from ..config_builder import ConfigBuilderT


class OptimizerParamsMixin:

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
