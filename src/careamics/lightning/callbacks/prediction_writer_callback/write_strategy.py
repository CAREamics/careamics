from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

from numpy.typing import NDArray
from pytorch_lightning import Trainer, LightningModule

from careamics.config.tile_information import TileInformation

# TODO: where do filenames come from ??

class WriteStrategy(ABC):

    @abstractmethod
    def write_batch(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: Any, # TODO: change to expected type
        batch_indices: Optional[Sequence[int]],
        batch: Any, # TODO: change to expected type
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        ...

class CacheTiles(WriteStrategy):
    
    def __init__(self) -> None:
        super().__init__()
        self.tile_cache: list[NDArray]
        self.tile_info_cache: list[TileInformation]

class WriteTiles(WriteStrategy):
    ...

class WriteImage(WriteStrategy):
    ...