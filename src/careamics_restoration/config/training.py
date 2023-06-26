from typing import Optional
from pydantic import BaseModel, Field

from .stage import Stage
from .torch_optimizer import TorchOptimizer, TorchLRScheduler


class Optimizer(BaseModel):
    """Parameters related to the optimizer."""

    name: TorchOptimizer
    parameters: dict

    """
    @validator("parameters")
    def check_optim_parameters(cls, user_params, values):
        if "name" in values:
            optimizer_name = values["name"].value
            optimizer_class = getattr(optim, optimizer_name)

            return get_parameters(optimizer_class, user_params)
        else:
            raise ValueError("Cannot validate parameters without `name`.")
    """

    class Config:
        use_enum_values = True  # make sure that enum are exported as str


class LrScheduler(BaseModel):
    """Parameters related to the learning rate scheduler."""

    name: TorchLRScheduler
    parameters: dict
    """
    @validator("parameters")
    def check_parameters(cls, user_params, values):
        if "name" in values:
            lr_scheduler_name = values["name"].value
            lr_scheduler_class = getattr(optim.lr_scheduler, lr_scheduler_name)

            return get_parameters(lr_scheduler_class, user_params)
        else:
            raise ValueError("Cannot validate parameters without `name`.")
    """

    class Config:
        use_enum_values = True  # make sure that enum are exported as str


class Amp(BaseModel):
    use: bool = False
    init_scale: int  # TODO excessive ? <- what is that?


class Training(Stage):
    """Parameters related to the training."""

    num_epochs: int = Field(default=100, ge=1, le=1_000)
    num_steps: int = Field(default=100, ge=1, le=1_000)

    optimizer: Optional[Optimizer] = TorchOptimizer.Adam
    lr_scheduler: Optional[LrScheduler] = TorchLRScheduler.ReduceLROnPlateau

    amp: Optional[Amp] = None
    max_grad_norm: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    # learning_rate: float = Field(default=0.001, ge=0.0001, le=0.1)
    running_stats: bool = Field(default=False)

    class Config:
        use_enum_values = True  # enum are exported as str
        allow_mutation = False  # model is immutable
