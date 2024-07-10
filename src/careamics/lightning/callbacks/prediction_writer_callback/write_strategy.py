from pathlib import Path
from typing import Any, Optional, Sequence, Protocol

import numpy as np
from numpy.typing import NDArray
from pytorch_lightning import Trainer, LightningModule
from torch.utils.data import DataLoader

from careamics.prediction_utils import stitch_prediction_single
from careamics.config.tile_information import TileInformation
from careamics.dataset import IterablePredDataset, IterableTiledPredDataset
from careamics.file_io import WriteFunc


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
        dirpath: Path,
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

    @property
    def last_tiles(self):
        return [tile_info.last_tile for tile_info in self.tile_info_cache]

    def write_batch(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: Any,  # TODO: change to expected type
        batch_indices: Optional[Sequence[int]],
        batch: Any,  # TODO: change to expected type
        batch_idx: int,
        dataloader_idx: int,
        dirpath: Path,
    ) -> None:
        dl: DataLoader = trainer.predict_dataloaders[dataloader_idx]
        ds: IterableTiledPredDataset = dl.dataset
        if not isinstance(ds, IterableTiledPredDataset):
            raise TypeError("Prediction dataset is not `IterableTiledPredDataset`.")
        if self._have_last_tile():

            tiles, tile_infos = self._get_image_tiles()
            self._clear_cache()

            # stitch prediction
            prediction_image = stitch_prediction_single(
                tiles=tiles, tile_infos=tile_infos
            )

            # write prediction
            sample_id = tile_infos[0].sample_id  # need this to select correct file name
            file_name_input = ds.data_files[sample_id]
            file_name = Path(file_name_input.stem()).with_suffix(self.write_extension)
            file_path = dirpath / file_name
            self.write_func(
                file_path=file_path, img=prediction_image[0], **self.write_func_kwargs
            )

        else:
            self.tile_cache = np.concatenate(self.tile_cache, prediction[0])
            self.tile_info_cache.extend(prediction[1])

    def _have_last_tile(self) -> bool:
        return any(self.last_tiles)

    def _clear_cache(self) -> None:
        index = self._last_tile_index()
        self.tile_cache = self.tile_cache[index + 1 :]
        self.tile_info_cache = self.tile_info_cache[index + 1 :]

    def _last_tile_index(self) -> int:
        last_tiles = self.last_tiles
        if not any(last_tiles):
            raise ValueError("No last tile.")
        index = np.where(last_tiles)[0][0]
        return index

    def _get_image_tiles(self) -> tuple[NDArray, list[TileInformation]]:
        index = self._last_tile_index()
        tiles = self.tile_cache[: index + 1]
        tile_infos = self.tile_info_cache[: index + 1]
        return tiles, tile_infos


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
        dirpath: Path,
    ) -> None:
        raise NotImplementedError


class WriteImage(WriteStrategy):
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

    def write_batch(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: Any,
        batch_indices: Sequence[int] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
        dirpath: Path,
    ) -> None:
        
        dl: DataLoader = trainer.predict_dataloaders[dataloader_idx]
        ds: IterablePredDataset = dl.dataset
        if not isinstance(ds, IterablePredDataset):
            raise TypeError("Prediction dataset is not `IterablePredDataset`.")

        # TODO: have to check sample_idx is correct
        for i, sample_idx in enumerate(batch_indices):
            prediction_image = batch[i]
            file_name_input = ds.data_files[sample_idx]
            file_name = Path(file_name_input.stem()).with_suffix(self.write_extension)
            file_path = dirpath / file_name
            self.write_func(
                file_path=file_path, img=prediction_image, **self.write_func_kwargs
            )
