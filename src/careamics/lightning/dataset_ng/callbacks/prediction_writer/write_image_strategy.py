"""A strategy writing whole images directly."""

from collections import defaultdict
from pathlib import Path
from typing import Any

from careamics.dataset_ng.dataset import ImageRegionData
from careamics.file_io import WriteFunc
from careamics.lightning.dataset_ng.prediction import (
    combine_samples,
)

from .file_path_utils import create_write_file_path
from .write_strategy import WriteStrategy


class WriteImage(WriteStrategy):
    """
    A strategy for writing image predictions (i.e. un-tiled predictions).

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

    def _get_total_samples(self, prediction: ImageRegionData) -> int:
        """
        Get the expected total number of samples from data_shape and axes.

        Parameters
        ----------
        prediction : ImageRegionData
            A prediction containing metadata about the original data.

        Returns
        -------
        int
            Total number of samples in the S dimension, or 1 if no S dimension.
        """
        if "S" in prediction.axes:
            s_idx = prediction.axes.index("S")
            return prediction.data_shape[s_idx]
        return 1

    def _get_complete_images(self) -> list[int]:
        """
        Get data indices where all samples have been collected.

        Returns
        -------
        list of int
            Data indices of complete images in the cache.
        """
        complete_images = []
        for data_idx in self.image_cache.keys():
            total_samples = self._get_total_samples(self.image_cache[data_idx][0])

            if len(self.image_cache[data_idx]) == total_samples:
                complete_images.append(data_idx)
            elif len(self.image_cache[data_idx]) > total_samples:
                raise ValueError(
                    f"More samples cached for data_idx {data_idx} than expected. "
                    f"Expected {total_samples}, found "
                    f"{len(self.image_cache[data_idx])}."
                )

        return complete_images

    def _write_complete_images(self, dirpath: Path) -> None:
        """
        Write complete images from cache and clear them.

        Parameters
        ----------
        dirpath : Path
            Path to directory to save predictions to.
        """
        complete_images = self._get_complete_images()

        for data_idx in complete_images:
            cached_preds = self.image_cache.pop(data_idx)

            image_lst, sources = combine_samples(cached_preds)

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
                )
                self.write_func(
                    file_path=file_path, img=image, **self.write_func_kwargs
                )
