from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import pytorch_lightning as L
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from careamics.config.inference_model import InferenceConfig
from careamics.dataset_ng.dataset import CareamicsDataset, Mode
from careamics.utils import get_logger

logger = get_logger(__name__)


class PredictDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_config: InferenceConfig,
        pred_data: Union[Path, str, NDArray],
        pred_target: Optional[Union[Path, str, NDArray]] = None,
        read_source_func: Optional[Callable] = None,
        read_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        self.data_config = data_config
        self.data_type: str = data_config.data_type
        self.batch_size: int = data_config.batch_size

        # TODO: figure out the hooks to use
        self.predict_dataset = CareamicsDataset(
            data_config=data_config,
            mode=Mode.PREDICTING,
            inputs=pred_data,
            targets=pred_target,
            read_func=read_source_func,
            read_kwargs=read_kwargs,
        )

    def predict_dataloader(self) -> Any:
        """
        Create a dataloader for training.

        Returns
        -------
        Any
            Training dataloader.
        """
        return DataLoader(
            self.predict_dataset, batch_size=self.batch_size, collate_fn=default_collate
        )
