from typing import Any, Literal

from careamics.config.factories.training_factory import update_trainer_params

from ..config_builder import ConfigBuilder, ConfigBuilderT


class TrainingParamsMixin:
    def __init__(self: ConfigBuilder, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.config_dict["training_config"].setdefault(
            "trainer_params", update_trainer_params({}, self.num_epochs, self.num_steps)
        )

    def set_trainer_params(self: ConfigBuilderT, **kwargs: Any) -> ConfigBuilderT:
        assert isinstance(self.config_dict["training_config"]["trainer_params"], dict)
        self.config_dict["training_config"]["trainer_params"].update(kwargs)
        return self

    def set_checkpoint_params(self: ConfigBuilderT, **kwargs: Any) -> ConfigBuilderT:
        self.config_dict["training_config"]["checkpoint_params"] = kwargs
        return self

    def set_early_stopping_params(
        self: ConfigBuilderT, **kwargs: Any
    ) -> ConfigBuilderT:
        self.config_dict["training_config"]["early_stopping_params"] = kwargs
        return self

    def set_logger(
        self: ConfigBuilderT, name: Literal["wandb", "tensorboard"] | None = None
    ) -> ConfigBuilderT:
        self.config_dict["training_config"]["logger"] = name
        return self
