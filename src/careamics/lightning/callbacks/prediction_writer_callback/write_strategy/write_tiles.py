"""Module containing the "cache tiles" write strategy."""

from pathlib import Path
from typing import Any, Iterator, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader

from careamics.config.tile_information import TileInformation
from careamics.dataset import IterableTiledPredDataset
from careamics.file_io import WriteFunc
from careamics.prediction_utils import stitch_prediction_single

from .utils import SampleCache, TileCache


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
        write_extension: str,
        write_func_kwargs: dict[str, Any],
        write_filenames: Optional[list[str]],
        n_samples_per_file: Optional[list[int]],
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
        self.write_extension: str = write_extension
        self.write_func_kwargs: dict[str, Any] = write_func_kwargs

        # where tiles will be cached until a whole image has been predicted
        self.tile_cache = TileCache()
        # where samples are stored until a whole file has been predicted
        self.sample_cache: Optional[SampleCache]

        self._write_filenames: Optional[list[str]] = write_filenames
        self.filename_iter: Optional[Iterator[str]] = (
            iter(write_filenames) if write_filenames is not None else None
        )
        if write_filenames is not None and n_samples_per_file is not None:
            self.set_file_data(write_filenames, n_samples_per_file)
        else:
            self.sample_cache = None

    def set_file_data(self, write_filenames: list[str], n_samples_per_file: list[int]):
        if len(write_filenames) != len(n_samples_per_file):
            raise ValueError(
                "List of filename and list of number of samples per file are not of "
                "equal length."
            )
        self._write_filenames = write_filenames
        self.filename_iter = iter(write_filenames)
        self.sample_cache = SampleCache(n_samples_per_file)

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
        if self.sample_cache is None:
            raise ValueError(
                "`SampleCache` has not been created. Call `set_file_data` before "
                "calling `write_batch`." 
            )
        # assert for mypy
        assert self.filename_iter is not None, (
            "`filename_iter` is `None` should be set by `set_file_data`."
        )

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

        # early return
        if not self.tile_cache.has_last_tile():
            return

        # if has last tile
        tiles, tile_infos = self.tile_cache.pop_image_tiles()

        # stitch prediction
        prediction_image = stitch_prediction_single(tiles=tiles, tile_infos=tile_infos)

        self.sample_cache.add(prediction_image)

        # early return
        if not self.sample_cache.has_all_file_samples():
            return

        # if has all samples in file
        samples = self.sample_cache.pop_file_samples()

        # combine
        data = np.concatenate(samples)

        # write prediction
        file_name = next(self.filename_iter)
        file_path = (dirpath / file_name).with_suffix(self.write_extension)
        self.write_func(file_path=file_path, img=data, **self.write_func_kwargs)

    def reset(self) -> None:
        """
        Reset the internal attributes.

        Attributes reset are: `write_filenames`, `tile_cache`, and `current_file_index`.
        """
        self._write_filenames = None
        self.filename_iter = None
        if self.tile_cache is not None:
            self.tile_cache.reset()
        if self.sample_cache is not None:
            self.sample_cache.reset()
