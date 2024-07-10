# from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence, Protocol

import numpy as np
from numpy.typing import NDArray
from pytorch_lightning import Trainer, LightningModule

from careamics.config.tile_information import TileInformation
from careamics.file_io import WriteFunc

# TODO: where do filenames come from ??

class WriteStrategy(Protocol):

    def write_batch(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: Any,  # TODO: change to expected type
        batch_indices: Optional[Sequence[int]],
        batch: Any,  # TODO: change to expected type
        batch_idx: int,
        dataloader_idx: int,
    ) -> None: ...


class CacheTiles(WriteStrategy):

    def __init__(
        self,
        write_func: WriteFunc,
        write_extension: str,
        write_func_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()

        self.write_func: WriteFunc = write_func
        self.write_extension: str = write_extension
        self.write_func_kwargs: Optional[dict[str, Any]] = write_func_kwargs

        # where tiles will be cached until a whole image has been predicted
        # TODO: how to initialise tile_cache shape -> just make it a list?
        self.tile_cache: NDArray
        self.tile_info_cache: list[TileInformation] = []

    def write_batch(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: Any,  # TODO: change to expected type
        batch_indices: Optional[Sequence[int]],
        batch: Any,  # TODO: change to expected type
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self._have_last_tile():
            # get image tiles

            self._clear_cache()

            # stitch prediction

            # write prediction
        else:
            self.tile_cache = np.concatenate(self.tile_cache, prediction[0])
            self.tile_info_cache.extend(prediction[1])

    def _have_last_tile(self) -> bool: ...

    def _clear_cache(self) -> None: ...


class WriteTilesZarr(WriteStrategy):
    def write_batch(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: Any,
        batch_indices: Sequence[int] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        raise NotImplementedError


class WriteImage(WriteStrategy):
    def __init__(
        self,
        write_func: WriteFunc,
        write_extension: str,
        write_func_kwargs: dict[str, Any]
    ) -> None:
        super().__init__()

        self.write_func: WriteFunc = write_func
        self.write_extension: str = write_extension
        self.write_func_kwargs: Optional[dict[str, Any]] = write_func_kwargs
