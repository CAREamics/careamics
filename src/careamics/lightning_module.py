from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import numpy as np
import pytorch_lightning as L
from albumentations import Compose
from pytorch_lightning.loops.fetchers import _DataLoaderIterDataFetcher
from pytorch_lightning.loops.utilities import _no_grad_context
from pytorch_lightning.utilities.types import _PREDICT_OUTPUT
from torch import nn, optim
from torch.utils.data import DataLoader

from careamics.config import AlgorithmModel, DataModel
from careamics.config.support import (
    SupportedAlgorithm,
    SupportedArchitecture,
    SupportedLoss,
    SupportedOptimizer,
    SupportedScheduler,
)
from careamics.dataset.dataset_utils import (
    data_type_validator,
    list_files,
    validate_files,
)
from careamics.dataset.in_memory_dataset import (
    InMemoryDataset,
    InMemoryPredictionDataset,
)
from careamics.dataset.iterable_dataset import (
    IterableDataset,
    IterablePredictionDataset,
)
from careamics.losses import create_loss_function
from careamics.models.model_factory import model_registry
from careamics.prediction import stitch_prediction
from careamics.utils import get_ram_size


class CAREamicsFiring(L.loops._PredictionLoop):
    """Predict loop for tiles-based prediction."""

    # def _predict_step(self, batch, batch_idx, dataloader_idx, dataloader_iter):
    #     self.model.predict_step(batch, batch_idx)

    @_no_grad_context
    def run(self) -> Optional[_PREDICT_OUTPUT]:
        self.setup_data()
        if self.skip:
            return None
        self.reset()
        self.on_run_start()
        data_fetcher = self._data_fetcher
        assert data_fetcher is not None
        while True:
            try:
                if isinstance(data_fetcher, _DataLoaderIterDataFetcher):
                    dataloader_iter = next(data_fetcher)
                    # hook's batch_idx and dataloader_idx arguments correctness cannot be guaranteed in this setting
                    batch = data_fetcher._batch
                    batch_idx = data_fetcher._batch_idx
                    dataloader_idx = data_fetcher._dataloader_idx
                else:
                    dataloader_iter = None
                    batch, batch_idx, dataloader_idx = next(data_fetcher)
                self.batch_progress.is_last_batch = data_fetcher.done
                # run step hooks
                self._predict_step(batch, batch_idx, dataloader_idx, dataloader_iter)
            except StopIteration:
                # this needs to wrap the `*_step` call too (not just `next`) for `dataloader_iter` support
                break
            finally:
                self._restarting = False
        return self.on_run_end()

def predict_tiled_simple(
    predictions: list,
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Predict using tiling.

    Parameters
    ----------
    pred_loader : DataLoader
        Prediction dataloader.
    progress_bar : ProgressBar
        Progress bar.
    tta : bool, optional
        Whether to use test time augmentation, by default True.

    Returns
    -------
    Union[np.ndarray, List[np.ndarray]]
        Predicted image, or list of predictions if the images have different sizes.

    Warns
    -----
    UserWarning
        If the samples have different shapes, the prediction then returns a list.
    """
    prediction = []
    tiles = []
    stitching_data = []

    for _i, (_tile, *auxillary) in enumerate(predictions):
        # Unpack auxillary data into last tile indicator and data, required to
        # stitch tiles together
        if auxillary:
            last_tile, *stitching_data = auxillary

        if last_tile:
            # Stitch tiles together if sample is finished
            predicted_sample = stitch_prediction(tiles, stitching_data)
            prediction.append(predicted_sample)
            tiles.clear()
            stitching_data.clear()

        try:
            return np.stack(prediction)
        except ValueError:
            return prediction


class CAREamicsKiln(L.LightningModule):
    def __init__(self, algorithm_config: AlgorithmModel) -> None:
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
            "monitor": "val_loss",  # otherwise one gets a MisconfigurationException
        }


class CAREamicsModule(CAREamicsKiln):
    def __init__(
        self,
        algorithm: Union[SupportedAlgorithm, str],
        loss: Union[SupportedLoss, str],
        architecture: Union[SupportedArchitecture, str],
        model_parameters: dict = {},
        optimizer: Union[SupportedOptimizer, str] = "Adam",
        optimizer_parameters: dict = {},
        lr_scheduler: Union[SupportedScheduler, str] = "ReduceLROnPlateau",
        lr_scheduler_parameters: dict = {},
    ) -> None:

        algorithm_configuration = {
            "algorithm": algorithm,
            "loss": loss,
            "model": {"architecture": architecture},
            "optimizer": {
                "name": optimizer,
                "parameters": optimizer_parameters,
            },
            "lr_scheduler": {
                "name": lr_scheduler,
                "parameters": lr_scheduler_parameters,
            }
        }

        # add model parameters
        algorithm_configuration["model"].update(model_parameters)

        super().__init__(AlgorithmModel(**algorithm_configuration))


class CAREamicsWood(L.LightningDataModule):
    def __init__(
        self,
        data_config: DataModel,
        train_path: Union[Path, str],
        val_path: Union[Path, str],
        train_target_path: Optional[Union[Path, str]] = None,
        val_target_path: Optional[Union[Path, str]] = None,
        read_source_func: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        self.data_config = data_config
        self.train_path = train_path
        self.val_path = val_path
        self.data_type = data_config.data_type
        self.train_target_path = train_target_path
        self.val_target_path = val_target_path
        self.read_source_func = read_source_func
        self.batch_size = data_config.batch_size
        self.num_workers = data_config.num_workers
        self.pin_memory = data_config.pin_memory

    def prepare_data(self) -> None:
        data_type_validator(self.data_type, self.read_source_func)
        self.train_files, self.train_data_size = list_files(
            self.train_path, self.data_type
        )
        self.val_files, _ = list_files(self.val_path, self.data_type)

        if self.train_target_path is not None:
            self.train_target_files, _ = list_files(
                self.train_target_path, self.data_type
            )
            validate_files(self.data_files, self.target_files)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.data_type == "zarr":
            pass
        elif self.data_type == "array":
            self.train_dataset = InMemoryDataset(
                files=self.train_files,
                config=self.data_config,
                target_files=self.train_target_files
                if self.train_target_path
                else None,
                read_source_func=self.read_source_func,
            )
            self.val_dataset = InMemoryDataset(
                files=self.val_files,
                config=self.data_config,
                target_files=self.val_target_files if self.val_target_path else None,
                read_source_func=self.read_source_func,
            )
        else:
            if self.train_data_size > get_ram_size() * 0.8:
                self.train_dataset = IterableDataset(
                    files=self.train_files,
                    config=self.data_config,
                    target_files=self.train_target_files
                    if self.train_target_path
                    else None,
                    read_source_func=self.read_source_func,
                )
                self.val_dataset = IterableDataset(
                    files=self.val_files,
                    config=self.data_config,
                    target_files=self.val_target_files
                    if self.val_target_path
                    else None,
                    read_source_func=self.read_source_func,
                )

            else:
                self.train_dataset = InMemoryDataset(
                    files=self.train_files,
                    config=self.data_config,
                    target_files=self.train_target_files
                    if self.train_target_path
                    else None,
                    read_source_func=self.read_source_func,
                )
                self.val_dataset = InMemoryDataset(
                    files=self.val_files,
                    config=self.data_config,
                    target_files=self.val_target_files
                    if self.val_target_path
                    else None,
                    read_source_func=self.read_source_func,
                )

    def train_dataloader(self) -> Any:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Any:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class CAREamicsClay(L.LightningDataModule):
    def __init__(
        self,
        data_config: DataModel,
        pred_path: Union[Path, str],
        read_source_func: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        self.data_config = data_config
        self.pred_path = pred_path
        self.data_type = data_config.data_type
        self.read_source_func = read_source_func
        self.batch_size = data_config.batch_size
        self.num_workers = data_config.num_workers
        self.pin_memory = data_config.pin_memory

    def prepare_data(self) -> None:
        data_type_validator(self.data_type, self.read_source_func)
        self.pred_files, _ = list_files(self.pred_path, self.data_type)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.data_type == "Zarr":
            pass
        elif self.data_type == "Array":
            self.predict_dataset = InMemoryPredictionDataset(
                files=self.pred_files,
                config=self.data_config,
                read_source_func=self.read_source_func,
            )
        else:
            self.predict_dataset = IterablePredictionDataset(
                files=self.pred_files,
                config=self.data_config,
                read_source_func=self.read_source_func,
            )

    def predict_dataloader(self) -> Any:
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class CAREamicsTrainDataModule(CAREamicsWood):
    def __init__(
        self,
        train_path: Union[str, Path],
        val_path: Union[str, Path],
        data_type: str,
        patch_size: List[int],
        axes: str,
        batch_size: int,
        transforms: Optional[Union[List, Compose]] = None,
        train_target_path: Optional[Union[str, Path]] = None,
        val_target_path: Optional[Union[str, Path]] = None,
        read_source_func: Optional[Callable] = None,
        data_loader_params: Optional[dict] = None,
        **kwargs,
    ) -> None:
        data_loader_params = data_loader_params if data_loader_params else {}
        data_config = {
            "data_type": data_type.lower(),
            "patch_size": patch_size,
            "axes": axes,
            "transforms": transforms,
            "batch_size": batch_size,
            **data_loader_params,
        }
        super().__init__(
            data_config=DataModel(**data_config),
            train_path=train_path,
            val_path=val_path,
            train_target_path=train_target_path,
            val_target_path=val_target_path,
            read_source_func=read_source_func,
        )


class CAREamicsPredictDataModule(CAREamicsClay):
    def __init__(
        self,
        pred_path: Union[str, Path],
        data_type: str,
        tile_size: List[int],
        axes: str,
        batch_size: int,
        transforms: Optional[Union[List, Compose]] = None,
        read_source_func: Optional[Callable] = None,
        data_loader_params: Optional[dict] = None,
        **kwargs,
    ) -> None:
        data_loader_params = data_loader_params if data_loader_params else {}

        data_config = {
            "data_type": data_type,
            "patch_size": tile_size,
            "axes": axes,
            "transforms": transforms,
            "batch_size": batch_size,
            **data_loader_params,
        }
        super().__init__(
            data_config=DataModel(**data_config),
            pred_path=pred_path,
            read_source_func=read_source_func,
        )
