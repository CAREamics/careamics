"""Module containing write strategy for when batches contain full images."""

from pathlib import Path
from typing import Any, Iterator, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader

from careamics.dataset import IterablePredDataset
from careamics.file_io import WriteFunc

from .utils import SampleCache


class WriteImage:
    """
    A strategy for writing image predictions (i.e. un-tiled predictions).

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
        A strategy for writing image predictions (i.e. un-tiled predictions).

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

        # where samples are stored until a whole file has been predicted
        self.sample_cache = SampleCache(n_samples_per_file)

        self._write_filenames: Optional[list[str]] = write_filenames
        self.filename_iter: Optional[Iterator[str]] = (
            iter(write_filenames) if write_filenames is not None else None
        )
        if write_filenames is not None and n_samples_per_file is not None:
            self.set_file_data(write_filenames, n_samples_per_file)

    def set_file_data(self, write_filenames: list[str], n_samples_per_file: list[int]):
        if len(write_filenames) != len(n_samples_per_file):
            raise ValueError(
                "List of filename and list of number of samples per file are not of "
                "equal length."
            )
        self._write_filenames = write_filenames
        self.filename_iter = iter(write_filenames)
        self.sample_cache.n_samples_per_file = n_samples_per_file

    def write_batch(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: NDArray,
        batch_indices: Optional[Sequence[int]],
        batch: NDArray,
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
        TypeError
            If trainer prediction dataset is not `IterablePredDataset`.
        ValueError
            If `write_filenames` attribute is `None`.
        """
        if self._write_filenames is None:
            raise ValueError("`write_filenames` attribute has not been set.")

        dls: Union[DataLoader, list[DataLoader]] = trainer.predict_dataloaders
        dl: DataLoader = dls[dataloader_idx] if isinstance(dls, list) else dls
        ds: IterablePredDataset = dl.dataset
        if not isinstance(ds, IterablePredDataset):
            # TODO: change to warning
            raise TypeError("Prediction dataset is not `IterablePredDataset`.")

        self.sample_cache.add(prediction)
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
        Reset internal attributes.

        Resets the `write_filenames` and `current_file_index` attributes.
        """
        self._write_filenames = None
        self.filename_iter = None
        self.current_file_index = 0
