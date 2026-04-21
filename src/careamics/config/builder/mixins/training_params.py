from typing import Any, Literal, overload

from ..config_builder import ConfigBuilderT


class TrainingParamsMixin:
    def set_trainer_params(self: ConfigBuilderT, **kwargs: Any) -> ConfigBuilderT:
        if "trainer_params" not in self.config_dict["training_config"]:
            self.config_dict["training_config"]["trainer_params"] = {}
        self.config_dict["training_config"]["trainer_params"].update(kwargs)
        return self

    def set_checkpoint_params(self: ConfigBuilderT, **kwargs: Any) -> ConfigBuilderT:
        if "checkpoint_params" not in self.config_dict["training_config"]:
            self.config_dict["training_config"]["checkpoint_params"] = {}
        self.config_dict["training_config"]["checkpoint_params"].update(kwargs)
        return self

    @overload
    def set_early_stopping(
        self: ConfigBuilderT, on: Literal[True], **kwargs: Any
    ) -> ConfigBuilderT: ...

    @overload
    def set_early_stopping(
        self: ConfigBuilderT, on: Literal[False]
    ) -> ConfigBuilderT: ...

    def set_early_stopping(
        self: ConfigBuilderT, on: bool, **kwargs: Any
    ) -> ConfigBuilderT:
        if on:
            if (
                "early_stopping_params" not in self.config_dict["training_config"]
                or self.config_dict["training_config"]["early_stopping_params"] is None
            ):
                self.config_dict["training_config"]["early_stopping_params"] = {}
            assert isinstance(  # mypy cannot resolve above
                self.config_dict["training_config"]["early_stopping_params"], dict
            )
            self.config_dict["training_config"]["early_stopping_params"].update(kwargs)
        else:
            self.config_dict["training_config"]["early_stopping_params"] = None

        return self

    def set_logger(
        self: ConfigBuilderT, name: Literal["wandb", "tensorboard"] | None = None
    ) -> ConfigBuilderT:
        self.config_dict["training_config"]["logger"] = name
        return self
