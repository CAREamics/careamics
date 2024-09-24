"""Module containing the "cache tiles" write strategy."""

from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader

from careamics.config.tile_information import TileInformation
from careamics.dataset import IterableTiledPredDataset
from careamics.file_io import WriteFunc
from careamics.prediction_utils import stitch_prediction_single

from .utils import TileCache

class WriteTiles:
    """
    A write strategy that will cache tiles.

    Tiles are cached until a whole image is predicted on. Then the stitched
    prediction is saved.

    Parameters
    ----------
    write_func : WriteFunc
        Function used to save predictions.
    write_filenames : list of str, optional
        A list of filenames in the order that predictions will be written in.
    write_extension : str
        Extension added to prediction file paths.
    write_func_kwargs : dict of {str: Any}
        Extra kwargs to pass to `write_func`.

    Attributes
    ----------
    write_func : WriteFunc
        Function used to save predictions.
    write_filenames : list of str, optional
        A list of filenames in the order that predictions will be written in.
    write_extension : str
        Extension added to prediction file paths.
    write_func_kwargs : dict of {str: Any}
        Extra kwargs to pass to `write_func`.
    tile_cache : list of numpy.ndarray
        Tiles cached for stitching prediction.
    tile_info_cache : list of TileInformation
        Cached tile information for stitching prediction.
    current_file_index : int
        Index of current file, increments every time a file is written.
    """

    def __init__(
        self,
        write_func: WriteFunc,
        write_filenames: Optional[list[str]],
        write_extension: str,
        write_func_kwargs: dict[str, Any],
    ) -> None:
        """
        A write strategy that will cache tiles.

        Tiles are cached until a whole image is predicted on. Then the stitched
        prediction is saved.

        Parameters
        ----------
        write_func : WriteFunc
            Function used to save predictions.
        write_filenames : list of str, optional
            A list of filenames in the order that predictions will be written in.
        write_extension : str
            Extension added to prediction file paths.
        write_func_kwargs : dict of {str: Any}
            Extra kwargs to pass to `write_func`.
        """
        super().__init__()

        self.write_func: WriteFunc = write_func
        self.write_filenames: Optional[list[str]] = write_filenames
        self.write_extension: str = write_extension
        self.write_func_kwargs: dict[str, Any] = write_func_kwargs

        # where tiles will be cached until a whole image has been predicted
        self.tile_cache = TileCache()

        self.current_file_index = 0

    @property
    def last_tiles(self) -> list[bool]:
        """
        List of bool to determine whether each tile in the cache is the last tile.

        Returns
        -------
        list of bool
            Whether each tile in the tile cache is the last tile.
        """
        return [tile_info.last_tile for tile_info in self.tile_info_cache]

    def write_batch(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: tuple[NDArray, list[TileInformation]],
        batch_indices: Optional[Sequence[int]],
        batch: tuple[NDArray, list[TileInformation]],
        batch_idx: int,
        dataloader_idx: int,
        dirpath: Path,
    ) -> None:
        """
        Cache tiles until the last tile is predicted; save the stitched prediction.

        Parameters
        ----------
        trainer : Trainer
            PyTorch Lightning Trainer.
        pl_module : LightningModule
            PyTorch Lightning LightningModule.
        prediction : Any
            Predictions on `batch`.
        batch_indices : sequence of int
            Indices identifying the samples in the batch.
        batch : Any
            Input batch.
        batch_idx : int
            Batch index.
        dataloader_idx : int
            Dataloader index.
        dirpath : Path
            Path to directory to save predictions to.

        Raises
        ------
        ValueError
            If `write_filenames` attribute is `None`.
        """
        if self.write_filenames is None:
            raise ValueError("`write_filenames` attribute has not been set.")
        
        # TODO: move dataset type check somewhere else
        dataloaders: Union[DataLoader, list[DataLoader]] = trainer.predict_dataloaders
        dataloader: DataLoader = (
            dataloaders[dataloader_idx]
            if isinstance(dataloaders, list)
            else dataloaders
        )
        dataset: IterableTiledPredDataset = dataloader.dataset
        if not isinstance(dataset, IterableTiledPredDataset):
            raise TypeError("Prediction dataset is not `IterableTiledPredDataset`.")

        self.tile_cache.add(prediction)

        # save stitched prediction
        if self.tile_cache.has_last_tile():

            # get image tiles and remove them from the cache
            tiles, tile_infos = self.tile_cache.pop_image_tiles()

            # stitch prediction
            prediction_image = stitch_prediction_single(
                tiles=tiles, tile_infos=tile_infos
            )

            # write prediction
            file_name = self.write_filenames[self.current_file_index]
            file_path = (dirpath / file_name).with_suffix(self.write_extension)
            self.write_func(
                file_path=file_path, img=prediction_image[0], **self.write_func_kwargs
            )
            self.current_file_index += 1

    def reset(self) -> None:
        """
        Reset the internal attributes.

        Attributes reset are: `write_filenames`, `tile_cache`, and `current_file_index`.
        """
        self.write_filenames = None
        self.current_file_index = 0
        self.tile_cache.reset()

