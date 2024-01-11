from typing import Any

import pytorch_lightning as L
import torch

from careamics.config import load_configuration
from careamics.dataset.prepare_dataset import (
    get_train_dataset,
    get_validation_dataset,
)
from careamics.losses import create_loss_function
from careamics.models import create_model

cfg = load_configuration(
    "/home/igor.zubarev/projects/caremics/examples/2D/n2v/n2v_2D_BSD.yml"
)
train_path = "/home/igor.zubarev/projects/caremics/examples/2D/data/denoising-N2V_BSD68.unzip/BSD68_reproducibility_data/train"
val_path = "/home/igor.zubarev/projects/caremics/examples/2D/data/denoising-N2V_BSD68.unzip/BSD68_reproducibility_data/val"


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

    def configure_optimizers(self) -> Any:
        return self.optimizer


train_dataset = get_train_dataset(cfg, train_path)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=128,
    num_workers=0,
    pin_memory=False,
)

val_dataset = get_validation_dataset(cfg, val_path)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=128,
    num_workers=0,
    pin_memory=False,
)

model = LUNet(cfg)
trainer = L.Trainer(inference_mode=False, max_epochs=3)
trainer.fit(model, train_dataloader, val_dataloader)
