from typing import Any, Optional, Union

import pytorch_lightning as L
import torch

from careamics.config.algorithm import Algorithm, Loss
from careamics.losses import create_loss_function
from careamics.models import create_model
from careamics.models.model_factory import model_registry

"""
CAREamist(configuration)
    DataLoader(cfg.data)
    Model(cfg.algorithm)
        get_model(cfg.algorithm)
        get_loss(cfg.algorithm.loss)
        get_noise_model(cfg.algorithm.noise_model)?
    Trainer(cfg.training)
"""

# TODO optimizer moves to Algorithm
# TODO not easy solution to be able to pass either Algorithm or its members
class CAREamicsModel(L.LightningModule):
    def __init__(
        self,
        *,
        algorithm_config: Algorithm,
        optimizer_name: str,
        optimizer_parameters: dict,
        lr_scheduler_name: str,
        lr_scheduler_parameters: dict,
    ) -> None:
        super().__init__()
        # TODO move config optim and scheduler to Algorithm

        # TODO: if the entry point is not with an Algorithm model, then we probably need to validate the parameters
        dims = 3 if algorithm_config.is_3D else 2
        self.model: torch.nn.Module = model_registry(algorithm_config.model.architecture, dims, algorithm_config.model.parameters)
        self.loss_func = create_loss_function(algorithm_config.loss)

        self.optimizer_name = optimizer_name
        self.optimizer_params = optimizer_parameters
        self.lr_scheduler_name = lr_scheduler_name
        self.lr_scheduler_params = lr_scheduler_parameters

    def forward(self, x: Any) -> Any:
        return self.model(x)

    def training_step(self, batch, batch_idx) -> Any:
        x, *aux = batch
        out = self.model(x)
        loss = self.loss_func(out, *aux)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, *aux = batch
        out = self.model(x)
        val_loss = self.loss_func(out, *aux)
        self.log("val_loss", val_loss)

    def predict_step(self, batch, batch_idx) -> Any:
        x, *aux = batch
        out = self.model(x)
        return out, aux

    def configure_optimizers(self) -> Any:
        # TODO what if they are None (mypy will fail)
        optimizer_func = getattr(torch.optim, self.optimizer_name)
        optimizer = optimizer_func(self.model.parameters(), **self.optimizer_params)

        scheduler_func = getattr(torch.optim.lr_scheduler, self.lr_scheduler_name)
        scheduler = scheduler_func(optimizer, **self.lr_scheduler_params)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss", # otherwise you get a MisconfigurationException
        }


class LUNet(L.LightningModule):
    """."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        (
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.scaler,
            self.cfg,
        ) = create_model(config=self.cfg)
        self.loss_func = create_loss_function(self.cfg.algorithm.loss)

    def forward(self, x: Any) -> Any:
        return self.model(x)

    def training_step(self, batch, batch_idx) -> Any:
        x, *aux = batch
        out = self.model(x)
        loss = self.loss_func(out, *aux)
        return loss

    def predict_step(self, batch, batch_idx) -> Any:
        x, *aux = batch
        out = self.model(x)
        return out, aux

    def configure_optimizers(self) -> Any:
        return self.optimizer
