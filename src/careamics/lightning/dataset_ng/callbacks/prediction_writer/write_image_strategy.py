"""A strategy writing whole images directly."""

from pathlib import Path
from typing import Any

from careamics.dataset_ng.dataset import ImageRegionData
from careamics.file_io import WriteFunc
from careamics.lightning.dataset_ng.prediction import (
    combine_samples,
)

from .file_path_utils import create_write_file_path
from .write_strategy import WriteStrategy


# TODO bug: batch is over samples for whole images, if one batch does not cover
# all samples, it will write an incomplete image, then overwrite it whith the next
# batch
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
        dirpath: Path,
        predictions: list[ImageRegionData],
    ) -> None:
        """
        Save full images.

        Parameters
        ----------
        dirpath : Path
            Path to directory to save predictions to.
        predictions : list[ImageRegionData]
            Decollated predictions.
        """
        assert predictions is not None

        image_lst, sources = combine_samples(predictions)

        for i, image in enumerate(image_lst):
            file_path = create_write_file_path(
                dirpath=dirpath,
                file_path=Path(sources[i]),
                write_extension=self.write_extension,
            )
            self.write_func(file_path=file_path, img=image, **self.write_func_kwargs)
