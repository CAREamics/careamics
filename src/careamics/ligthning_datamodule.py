from pathlib import Path
from typing import Any, Callable, List, Optional, Union, Tuple

import pytorch_lightning as L
from albumentations import Compose
from torch.utils.data import DataLoader
import numpy as np

from careamics.config import DataModel
from careamics.config.support import SupportedData
from careamics.dataset.dataset_utils import (
    list_files,
    get_files_size,
    validate_source_target_files
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

# TODO must be compatible with no validation being present
class CAREamicsWood(L.LightningDataModule):
    def __init__(
        self,
        data_config: DataModel,
        train_data: Union[Path, str, np.ndarray],
        val_data: Union[Path, str, np.ndarray],
        train_data_target: Optional[Union[Path, str, np.ndarray]] = None,
        val_data_target: Optional[Union[Path, str, np.ndarray]] = None,
        read_source_func: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        # check input types coherence (no mixed types)
        inputs = [
            train_data, val_data, train_data_target, val_data_target
        ]
        types_set = set([type(i) for i in inputs])
        if len(types_set) > 2: # None + expected type
            raise ValueError(
                f"Inputs for `train_data`, `val_data`, `train_data_target` and "
                f"`val_data_target` must be of the same type or None. Got "
                f"{types_set}."
            )
        
        # check that a read source function is provided for custom types
        if data_config.data_type == SupportedData.CUSTOM and read_source_func is None:
            raise ValueError(
                f"Data type {SupportedData.CUSTOM} is not allowed without "
                f"specifying a `read_source_func`."
            )
        
        # and that arrays are passed, if array type specified
        elif data_config.data_type == SupportedData.ARRAY and not isinstance(train_data, np.ndarray):
            raise ValueError(
                f"Expected array input, but got {type(train_data)} instead."
            )

        # configuration
        self.data_config = data_config
        self.data_type = data_config.data_type
        self.batch_size = data_config.batch_size
        self.num_workers = data_config.num_workers
        self.pin_memory = data_config.pin_memory

        # data
        self.train_data = train_data
        self.val_data = val_data
        
        self.train_data_target = train_data_target
        self.val_data_target = val_data_target
        
        # read source function
        self.read_source_func = read_source_func
        
    def prepare_data(self) -> None:
        """Hook used to prepare the data before calling `setup` and creating
        the dataloader.

        Here, we only need to examine the data if it was provided as a str or a Path.
        """
        # if the data is a Path or a str
        if not isinstance(self.train_data, np.ndarray):
            # list training files
            self.train_files = list_files(self.train_data, self.data_type)
            self.train_files_size = get_files_size(self.train_files, self.data_type)
            
            # list validation files
            if self.val_data is not None:
                self.val_files = list_files(self.val_data, self.data_type)

            # same for target data
            if self.train_data_target is not None:
                self.train_target_files = list_files(
                    self.train_data_target, self.data_type
                )

                # verify that they match the training data
                validate_source_target_files(self.train_files, self.train_target_files)
            
            if self.val_data_target is not None:
                self.val_target_files = list_files(self.val_data_target, self.data_type)

                # verify that they match the validation data
                validate_source_target_files(self.val_files, self.val_target_files)


    def setup(self, stage: Optional[str] = None) -> None:
        """Hook called at the beginning of fit (train + validate), validate, test, or 
        predict."""
        # if numpy array
        if self.data_type == SupportedData.ARRAY:
            # train dataset
            self.train_dataset = InMemoryDataset(
                data_config=self.data_config,
                data=self.train_data,
                data_target=self.train_data_target,
            )

            # TODO: how to extract the validation data from the training data?

            # validation dataset
            self.val_dataset = InMemoryDataset(
                data_config=self.data_config,
                data=self.val_data,
                data_target=self.val_data_target,
            )
        # else we read files
        else:
            # heuristics, if the file size is bigger than 80% of the RAM, we iterate
            # through the files
            if self.train_files_size > get_ram_size() * 0.8:
                # TODO here if we don't have validation, we could reserve some files

                # create training dataset
                self.train_dataset = IterableDataset(
                    data_config=self.data_config,
                    src_files=self.train_files,
                    target_files=self.train_target_files
                    if self.train_data_target
                    else None,
                    read_source_func=self.read_source_func,
                )

                # create validation dataset
                self.val_dataset = IterableDataset(
                    data_config=self.data_config,
                    src_files=self.val_files,
                    target_files=self.val_target_files
                    if self.val_data_target
                    else None,
                    read_source_func=self.read_source_func,
                )
            # else, load everything in memory
            else:
                # train dataset
                self.train_dataset = InMemoryDataset(
                    data_config=self.data_config,
                    data=self.train_files,
                    data_target=self.train_target_files
                    if self.train_data_target
                    else None,
                    read_source_func=self.read_source_func,
                )

                # validation dataset
                self.val_dataset = InMemoryDataset(
                    data_config=self.data_config,
                    data=self.val_files,
                    data_target=self.val_target_files
                    if self.val_data_target
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
        pred_data: Union[Path, str, np.ndarray],
        tile_size: Union[List[int], Tuple[int]],
        tile_overlap: Union[List[int], Tuple[int]],
        read_source_func: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        # check that a read source function is provided for custom types
        if data_config.data_type == SupportedData.CUSTOM and read_source_func is None:
            raise ValueError(
                f"Data type {SupportedData.CUSTOM} is not allowed without "
                f"specifying a `read_source_func`."
            )
        
        # and that arrays are passed, if array type specified
        elif data_config.data_type == SupportedData.ARRAY and not isinstance(pred_data, np.ndarray):
            raise ValueError(
                f"Expected array input, but got {type(pred_data)} instead."
            )

        # configuration data
        self.data_config = data_config
        self.data_type = data_config.data_type
        self.batch_size = data_config.batch_size
        self.num_workers = data_config.num_workers
        self.pin_memory = data_config.pin_memory

        self.pred_data = pred_data
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.read_source_func = read_source_func

    def prepare_data(self) -> None:
        # if the data is a Path or a str
        if not isinstance(self.pred_data, np.ndarray):
            self.pred_files = list_files(self.pred_data, self.data_type)

    def setup(self, stage: Optional[str] = None) -> None:
        # if numpy array
        if self.data_type == SupportedData.ARRAY:
            # prediction dataset
            self.predict_dataset = InMemoryPredictionDataset(
                data_config=self.data_config,
                data=self.pred_data,
                tile_size=self.tile_size,
                tile_overlap=self.tile_overlap,
            )
        else:
            self.predict_dataset = IterablePredictionDataset(
                files=self.pred_files,
                data_config=self.data_config,
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
            train_data=train_path,
            val_data=val_path,
            train_data_target=train_target_path,
            val_data_target=val_target_path,
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
            pred_data=pred_path,
            read_source_func=read_source_func,
        )
