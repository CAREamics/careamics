from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import pytorch_lightning as L
from albumentations import Compose
from torch.utils.data import DataLoader

from careamics.config import DataModel
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
from careamics.utils import get_ram_size

# TODO must be compatible with no validation, path to folder, path to image, and arrays!
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
