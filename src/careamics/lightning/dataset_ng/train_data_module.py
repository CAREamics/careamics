from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Union

import pytorch_lightning as L
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from careamics.config.data import DataConfig
from careamics.config.support import SupportedData
from careamics.config.transformations import TransformModel
from careamics.dataset_ng.dataset import CareamicsDataset, Mode
from careamics.utils import get_logger

logger = get_logger(__name__)


class TrainDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_config: DataConfig,
        train_data: Union[Path, str, NDArray],
        val_data: Optional[Union[Path, str, NDArray]] = None,
        train_data_target: Optional[Union[Path, str, NDArray]] = None,
        val_data_target: Optional[Union[Path, str, NDArray]] = None,
        read_source_func: Optional[Callable] = None,
        read_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        # TODO: checks from current train_data_module

        self.data_config = data_config
        self.data_type: str = data_config.data_type
        self.batch_size: int = data_config.batch_size

        # TODO: figure out the hooks to use
        self.train_dataset = CareamicsDataset(
            data_config=data_config,
            mode=Mode.TRAINING,
            inputs=train_data,
            targets=train_data_target,
            read_func=read_source_func,
            read_kwargs=read_kwargs,
        )
        self.val_dataset = CareamicsDataset(
            data_config=data_config,
            mode=Mode.VALIDATING,
            inputs=val_data,
            targets=val_data_target,
            read_func=read_source_func,
            read_kwargs=read_kwargs,
        )

    def train_dataloader(self) -> Any:
        """
        Create a dataloader for training.

        Returns
        -------
        Any
            Training dataloader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=default_collate,
            **self.data_config.train_dataloader_params,
        )

    def val_dataloader(self) -> Any:
        """
        Create a dataloader for validation.

        Returns
        -------
        Any
            Validation dataloader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=default_collate,
            **self.data_config.val_dataloader_params,
        )


def create_train_datamodule(
    train_data: Union[str, Path, NDArray],
    data_type: Union[Literal["array", "tiff", "custom"], SupportedData],
    patch_size: list[int],
    axes: str,
    batch_size: int,
    val_data: Optional[Union[str, Path, NDArray]] = None,
    train_target_data: Optional[Union[str, Path, NDArray]] = None,
    val_target_data: Optional[Union[str, Path, NDArray]] = None,
    transforms: Optional[list[TransformModel]] = None,
    read_source_func: Optional[Callable] = None,
    dataloader_params: Optional[dict] = None,
) -> TrainDataModule:
    if dataloader_params is None:
        dataloader_params = {}

    data_dict: dict[str, Any] = {
        "mode": "train",
        "data_type": data_type,
        "patch_size": patch_size,
        "axes": axes,
        "batch_size": batch_size,
        "dataloader_params": dataloader_params,
    }
    if transforms is not None:
        data_dict["transforms"] = transforms

    data_config = DataConfig(**data_dict)

    return TrainDataModule(
        data_config=data_config,
        train_data=train_data,
        val_data=val_data,
        train_data_target=train_target_data,
        val_data_target=val_target_data,
        read_source_func=read_source_func,
    )
