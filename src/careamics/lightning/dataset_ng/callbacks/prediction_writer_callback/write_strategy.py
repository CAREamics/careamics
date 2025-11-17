"""Module containing different strategies for writing predictions."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from pytorch_lightning import LightningModule, Trainer

from careamics.dataset_ng.dataset import ImageRegionData
from careamics.file_io import WriteFunc
from careamics.lightning.dataset_ng.prediction import (
    combine_samples,
    stitch_single_prediction,
)

from .file_path_utils import create_write_file_path


class WriteStrategy(Protocol):
    """Protocol for write strategy classes."""

    def write_batch(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: list[ImageRegionData],
        batch_indices: Sequence[int] | None,
        batch: ImageRegionData,
        batch_idx: int,
        dataloader_idx: int,
        dirpath: Path,
    ) -> None:
        """
        WriteStrategy subclasses must contain this function to write a batch.

        Parameters
        ----------
        trainer : Trainer
            PyTorch Lightning Trainer.
        pl_module : LightningModule
            PyTorch Lightning LightningModule.
        prediction : list[ImageRegionData]
            Decollated Predictions on `batch`.
        batch_indices : sequence of int
            Indices identifying the samples in the batch.
        batch : ImageRegionData
            Collated input batch.
        batch_idx : int
            Batch index.
        dataloader_idx : int
            Dataloader index.
        dirpath : Path
            Path to directory to save predictions to.
        """


class CacheTiles(WriteStrategy):
    """
    A write strategy that will cache tiles.

    Tiles are cached until a whole image is predicted on. Then the stitched
    prediction is saved.

    Parameters
    ----------
    write_func : WriteFunc
        Function used to save predictions.
    write_extension : str
        Extension added to prediction file paths.
    write_func_kwargs : dict of {str: Any}
        Extra kwargs to pass to `write_func`.

    Attributes
    ----------
    write_func : WriteFunc
        Function used to save predictions.
    write_extension : str
        Extension added to prediction file paths.
    write_func_kwargs : dict of {str: Any}
        Extra kwargs to pass to `write_func`.
    tile_cache : list of numpy.ndarray
        Tiles cached for stitching prediction.
    tile_info_cache : list of TileInformation
        Cached tile information for stitching prediction.
    """

    def __init__(
        self,
        write_func: WriteFunc,
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
        self.tile_cache: list[ImageRegionData] = []
        self.data_indices: list[int] = []
        self.last_tile: list[bool] = []

    def write_batch(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: list[ImageRegionData],
        batch_indices: Sequence[int] | None,
        batch: ImageRegionData,
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
        prediction : list[ImageRegionData]
            Decollated Predictions on `batch`.
        batch_indices : sequence of int
            Indices identifying the samples in the batch.
        batch : ImageRegionData
            Input batch.
        batch_idx : int
            Batch index.
        dataloader_idx : int
            Dataloader index.
        dirpath : Path
            Path to directory to save predictions to.
        """
        assert prediction is not None

        # cache tiles
        self.tile_cache.extend(prediction)
        self.data_indices.extend(
            [image_region.region_spec["data_idx"] for image_region in prediction]
        )
        self.last_tile.extend(
            [image_region.region_spec["last_tile"] for image_region in prediction]
        )

        # save stitched prediction
        if self._has_last_tile():

            # get ImageRegionData tiles and remove them from the cache
            tiles = self._extract_image_tiles()

            # stitch prediction
            prediction_image = stitch_single_prediction(tiles)

            # write prediction
            source: Path = Path(tiles[0].source)
            file_path = create_write_file_path(
                dirpath=dirpath,
                file_path=source,
                write_extension=self.write_extension,
            )
            self.write_func(
                file_path=file_path, img=prediction_image, **self.write_func_kwargs
            )

    def _has_last_tile(self) -> bool:
        """
        Whether a last tile is contained in the cached tiles.

        Returns
        -------
        bool
            Whether a last tile is contained in the cached tiles.
        """
        return any(self.last_tile)

    def _find_last_tile_index(self) -> int | None:
        """
        Find the index of the last tile in the cache.

        Returns
        -------
        int | None
            Index of the last tile in the cache, or None if not found.
        """
        return next(
            (idx for idx, is_last in enumerate(self.last_tile) if is_last), None
        )

    def _find_image_tile_indices(self, data_idx: int) -> list[int]:
        """
        Find all tile indices in the cache with the given data index.

        Parameters
        ----------
        data_idx : int
            Data index to search for.

        Returns
        -------
        list of int
            List of tile indices in the cache with the given data index.
        """
        return [i for i, d_idx in enumerate(self.data_indices) if d_idx == data_idx]

    def _pop_cache(self, indices: list[int]) -> list[ImageRegionData]:
        """
        Pop tiles from the cache at the given indices.

        Parameters
        ----------
        indices : list of int
            Indices of tiles to pop from the cache.

        Returns
        -------
        list of ImageRegionData
            Popped tiles.
        """
        tiles = [self.tile_cache.pop(i) for i in sorted(indices, reverse=True)]
        _ = [self.data_indices.pop(i) for i in sorted(indices, reverse=True)]
        _ = [self.last_tile.pop(i) for i in sorted(indices, reverse=True)]

        # TODO necessary? it makes testing easier
        # reverse to original order
        tiles.reverse()

        return tiles

    def _extract_image_tiles(self) -> list[ImageRegionData]:
        """
        Get the tiles corresponding to a single image.

        Returns
        -------
        list of ImageRegionData
            Tiles corresponding to a single image.
        """
        # find last tile index
        idx_last_tile = self._find_last_tile_index()

        if idx_last_tile is None:
            raise ValueError("No last tile found in the cache.")

        # find all tiles with same data_idx as the last tile
        data_idx = self.data_indices[idx_last_tile]
        same_data_indices = self._find_image_tile_indices(data_idx)

        # pop tiles and indices from cache
        tiles = self._pop_cache(same_data_indices)

        return tiles


class WriteImage(WriteStrategy):
    """
    A strategy for writing image predictions (i.e. un-tiled predictions).

    Parameters
    ----------
    write_func : WriteFunc
        Function used to save predictions.
    write_extension : str
        Extension added to prediction file paths.
    write_func_kwargs : dict of {str: Any}
        Extra kwargs to pass to `write_func`.

    Attributes
    ----------
    write_func : WriteFunc
        Function used to save predictions.
    write_extension : str
        Extension added to prediction file paths.
    write_func_kwargs : dict of {str: Any}
        Extra kwargs to pass to `write_func`.
    """

    def __init__(
        self,
        write_func: WriteFunc,
        write_extension: str,
        write_func_kwargs: dict[str, Any],
    ) -> None:
        """
        A strategy for writing image predictions (i.e. un-tiled predictions).

        Parameters
        ----------
        write_func : WriteFunc
            Function used to save predictions.
        write_extension : str
            Extension added to prediction file paths.
        write_func_kwargs : dict of {str: Any}
            Extra kwargs to pass to `write_func`.
        """
        super().__init__()

        self.write_func: WriteFunc = write_func
        self.write_extension: str = write_extension
        self.write_func_kwargs: dict[str, Any] = write_func_kwargs

    def write_batch(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: list[ImageRegionData],
        batch_indices: Sequence[int] | None,
        batch: ImageRegionData,
        batch_idx: int,
        dataloader_idx: int,
        dirpath: Path,
    ) -> None:
        """
        Save full images.

        Parameters
        ----------
        trainer : Trainer
            PyTorch Lightning Trainer.
        pl_module : LightningModule
            PyTorch Lightning LightningModule.
        prediction : list[ImageRegionData]
            Decollated predictions on `batch`.
        batch_indices : sequence of int
            Indices identifying the samples in the batch.
        batch : list[ImageRegionData]
            Input batch.
        batch_idx : int
            Batch index.
        dataloader_idx : int
            Dataloader index.
        dirpath : Path
            Path to directory to save predictions to.
        """
        assert prediction is not None
        predictions, sources = combine_samples(prediction)

        for i, image in enumerate(predictions):
            file_path = create_write_file_path(
                dirpath=dirpath,
                file_path=Path(sources[i]),
                write_extension=self.write_extension,
            )
            self.write_func(file_path=file_path, img=image, **self.write_func_kwargs)
