from typing import Any

import pytorch_lightning as L
import torch

from careamics.config.algorithm import Algorithm
from careamics.losses import create_loss_function
from careamics.models.model_factory import model_registry


class CAREamicsModel(L.LightningModule):
    def __init__(
        self,
        algorithm_config: Algorithm
    ) -> None:
        super().__init__()

        # create model and loss function
        self.model: torch.nn.Module = model_registry(algorithm_config.model)
        self.loss_func = create_loss_function(algorithm_config.loss)

        # save optimizer and lr_scheduler names and parameters
        self.optimizer_name = algorithm_config.optimizer.name
        self.optimizer_params = algorithm_config.optimizer.parameters
        self.lr_scheduler_name = algorithm_config.lr_scheduler.name
        self.lr_scheduler_params = algorithm_config.lr_scheduler.parameters

    def forward(self, x: Any) -> Any:
        return self.model(x)

    def training_step(self, batch, batch_idx) -> Any:
        x, *aux = batch
        out = self.model(x)
        loss = self.loss_func(out, *aux)
        return loss

    def validation_step(self, batch, batch_idx):
        x, *aux = batch
        out = self.model(x)
        val_loss = self.loss_func(out, *aux)

        # log validation loss
        self.log("val_loss", val_loss)

    def predict_step(self, batch, batch_idx) -> Any:
        x, *aux = batch
        out = self.model(x)
        return out, aux

    def configure_optimizers(self) -> Any:
        # instantiate optimizer
        optimizer_func = getattr(torch.optim, self.optimizer_name)
        optimizer = optimizer_func(self.model.parameters(), **self.optimizer_params)

        # and scheduler
        scheduler_func = getattr(torch.optim.lr_scheduler, self.lr_scheduler_name)
        scheduler = scheduler_func(optimizer, **self.lr_scheduler_params)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss", # otherwise you get a MisconfigurationException
        }
    