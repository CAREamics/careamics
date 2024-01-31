from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import pytorch_lightning as L
from torch import nn, optim
from torch.utils.data import DataLoader
from albumentations import Compose

from careamics.config import AlgorithmModel
from careamics.config.support import (
    SupportedAlgorithm,
    SupportedArchitecture,
    SupportedLoss,
    SupportedOptimizer,
    SupportedScheduler,
)
from careamics.losses import create_loss_function
from careamics.models.model_factory import model_registry


class CAREamicsKiln(L.LightningModule):
    def __init__(
        self,
        algorithm_config: AlgorithmModel
    ) -> None:
        super().__init__()

        # create model and loss function
        self.model: nn.Module = model_registry(algorithm_config.model)
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
        optimizer_func = getattr(optim, self.optimizer_name)
        optimizer = optimizer_func(self.model.parameters(), **self.optimizer_params)

        # and scheduler
        scheduler_func = getattr(optim.lr_scheduler, self.lr_scheduler_name)
        scheduler = scheduler_func(optimizer, **self.lr_scheduler_params)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss", # otherwise one gets a MisconfigurationException
        }


# TODO consider using a Literal[...] instead of the enums here?
class CAREamicsModule(CAREamicsKiln):

    def __init__(
        self,
        algorithm: Union[SupportedAlgorithm, str],
        loss: Union[SupportedLoss, str],
        architecture: Union[SupportedArchitecture, str],
        model_parameters: Optional[dict],
        optimizer: Optional[Union[SupportedOptimizer, str]] = None,
        lr_scheduler: Optional[Union[SupportedScheduler, str]] = None,
        optimizer_parameters: Optional[dict] = None,
        lr_scheduler_parameters: Optional[dict] = None,
    ) -> None:

        algorithm_configuration = {
            "algorithm": algorithm,
            "loss": loss,
            "model": {
                "architecture": architecture
            },
            "optimizer": {
                "name": optimizer,
                "parameters": optimizer_parameters
            } if optimizer is not None else {},
            "lr_scheduler": {
                "name": lr_scheduler,
                "parameters": lr_scheduler_parameters
            } if lr_scheduler is not None else {}
        }

        # add model parameters
        algorithm_configuration["model"].update(model_parameters)

        super().__init__(AlgorithmModel(**algorithm_configuration))


class CAREamicsClay(L.LightningDataModule):
    def __init__(
        self,
        data_path: Union[str, Path],
        data_extension: str,
        patch_size: List[int],
        axes: str,
        batch_size: int,
        transforms: Optional[Union[List, Compose]] = None,
        target_path: Optional[Union[str, Path]] = None,
        target_extension: Optional[str] = None,
        read_source_func: Optional[Callable] = None,
        num_workers: Optional[int] = 0,
        pin_memory: Optional[bool] = False,
        **kwargs
    ) -> None:
        data_config = {
            "data_extension": data_extension,
            "patch_size": patch_size,
            "axes": axes,
            "transforms": transforms,
            "target_path": target_path,
            "target_extension": target_extension,
            "read_source_func": read_source_func,
            "mean": mean,
        }
        super().__init__()

        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        



        # get file sizes
        # get psutil memory size available
        # if it is within 10 percent, then iterable, otherwise, in memory


    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = None

    def train_dataloader(self) -> Any:
        DataLoader()

    def val_dataloader(self) -> Any:
        DataLoader()

    def predict_dataloader(self) -> Any:
        DataLoader()
