"""A writing strategy that caches tiles until a whole image is predicted."""

from collections import defaultdict
from pathlib import Path
from typing import Any

from careamics.dataset_ng.dataset import ImageRegionData
from careamics.file_io import WriteFunc
from careamics.lightning.dataset_ng.prediction import (
    stitch_single_prediction,
)

from .file_path_utils import create_write_file_path
from .write_strategy import WriteStrategy


class CachedTiles(WriteStrategy):
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
        self.tile_cache: dict[int, list[ImageRegionData]] = defaultdict(list)

    def write_batch(
        self,
        dirpath: Path,
        predictions: list[ImageRegionData],
    ) -> None:
        """
        Cache tiles until the last tile is predicted, then save the stitched image.

        Parameters
        ----------
        dirpath : Path
            Path to directory to save predictions to.
        predictions : list[ImageRegionData]
            Decollated predictions.
        """
        assert predictions is not None

        # cache tiles
        for tile in predictions:
            data_idx = tile.region_spec["data_idx"]
            self.tile_cache[data_idx].append(tile)

        self._write_images(dirpath)

    def _get_full_images(self) -> list[int]:
        """
        Get data indices of full images contained in the cache.

        Returns
        -------
        list of int
            Data indices of full images contained in the cache.
        """
        full_images = []
        for data_idx in self.tile_cache.keys():
            exp_n_tiles = self.tile_cache[data_idx][0].region_spec["total_tiles"]

            if len(self.tile_cache[data_idx]) == exp_n_tiles:
                full_images.append(data_idx)
            elif len(self.tile_cache[data_idx]) > exp_n_tiles:
                raise ValueError(
                    f"More tiles cached for data_idx {data_idx} than expected. "
                    f"Expected {exp_n_tiles}, found "
                    f"{len(self.tile_cache[data_idx])}."
                )

        return full_images

    def _stitch_and_write_single(
        self, dirpath: Path, tiles: list[ImageRegionData]
    ) -> None:
        """
        Stitch and write a single image from tiles.

        Parameters
        ----------
        dirpath : Path
            Path to directory to save predictions to.
        tiles : list[ImageRegionData]
            Tiles to stitch and write.
        """
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

    def _write_images(self, dirpath: Path) -> None:
        """
        Write full images from cached tiles.

        Parameters
        ----------
        dirpath : Path
            Path to directory to save predictions to.
        """
        full_images = self._get_full_images()
        for data_idx in full_images:
            tiles = self.tile_cache.pop(data_idx)
            self._stitch_and_write_single(dirpath, tiles)
