"""A strategy writing whole images directly."""

from collections import defaultdict
from pathlib import Path
from typing import Any

from careamics.dataset.image_region_data import ImageRegionData
from careamics.image_io import WriteFunc
from careamics.lightning.prediction import (
    combine_samples,
)

from .file_path_utils import create_write_file_path
from .image_write_utils import get_complete_images
from .write_strategy import WriteStrategy


class ImageWriteStrategy(WriteStrategy):
    """
    A strategy for writing whole image predictions (i.e. un-tiled predictions).

    Predictions are cached until all samples for a given data_idx are collected,
    then combined and written. This prevents overwrites when S_dim > batch_size.

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
    image_cache : dict of {int: list of ImageRegionData}
        Cache for predictions across batches, keyed by data_idx.
    source_base : pathlib.Path or None
        Common parent of the sources, used to preserve their directory structure.
        Set via `set_source_base`.
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

        self.image_cache: dict[int, list[ImageRegionData]] = defaultdict(list)

        # common parent of the sources, used to preserve their directory structure;
        # set via `set_source_base` by the prediction writer callback
        self.source_base: Path | None = None

    def set_source_base(self, source_base: Path | None) -> None:
        """
        Set the common parent directory of the sources.

        Called by the prediction writer callback so that the directory structure of
        the sources can be preserved in the output.

        Parameters
        ----------
        source_base : pathlib.Path or None
            Common parent of all prediction sources. If None, outputs are written
            directly under the output directory without preserving structure.
        """
        self.source_base = source_base

    def write_batch(
        self,
        dirpath: Path,
        predictions: list[ImageRegionData],
    ) -> None:
        """
        Cache predictions and save full images.

        Predictions are cached by data_idx until all samples (S dimension)
        are collected, then combined and written.

        Parameters
        ----------
        dirpath : Path
            Path to directory to save predictions to.
        predictions : list[ImageRegionData]
            Decollated predictions.
        """
        assert predictions is not None

        for pred in predictions:
            data_idx = pred.region_spec["data_idx"]
            self.image_cache[data_idx].append(pred)

        self._write_complete_images(dirpath)

    def _write_complete_images(self, dirpath: Path) -> None:
        """
        Write complete images from cache and clear them.

        Parameters
        ----------
        dirpath : Path
            Path to directory to save predictions to.
        """
        complete_images = get_complete_images(self.image_cache)

        for data_idx in complete_images:
            cached_preds = self.image_cache.pop(data_idx)

            image_lst, sources = combine_samples(cached_preds, restore_shape=True)

            for i, image in enumerate(image_lst):
                source_path = Path(sources[i])

                postfix = ""
                if source_path.stem == "array":
                    postfix = f"_{data_idx}"

                file_path = create_write_file_path(
                    dirpath=dirpath,
                    file_path=source_path,
                    write_extension=self.write_extension,
                    postfix=postfix,
                    source_base=self.source_base,
                )
                file_path.parent.mkdir(parents=True, exist_ok=True)
                self.write_func(
                    file_path=file_path, img=image, **self.write_func_kwargs
                )
