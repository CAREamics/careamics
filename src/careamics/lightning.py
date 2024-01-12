from typing import Any

import pytorch_lightning as L

from careamics.losses import create_loss_function
from careamics.models import create_model


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
        self.loss_func = create_loss_function(self.cfg)

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


