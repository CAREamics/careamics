"""Module containing write strategy for when batches contain full images."""

from pathlib import Path
from typing import Any, Optional, Sequence, Union

from numpy.typing import NDArray
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader

from careamics.dataset import IterablePredDataset
from careamics.file_io import WriteFunc

from .protocol import WriteStrategy


class WriteImage(WriteStrategy):
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
        write_filenames: Optional[list[str]],
        write_extension: str,
        write_func_kwargs: dict[str, Any],
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
        self.write_filenames: Optional[list[str]] = write_filenames
        self.write_extension: str = write_extension
        self.write_func_kwargs: dict[str, Any] = write_func_kwargs

        self.current_file_index: int = 0

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
        if self.write_filenames is None:
            raise ValueError("`write_filenames` attribute has not been set.")

        dls: Union[DataLoader, list[DataLoader]] = trainer.predict_dataloaders
        dl: DataLoader = dls[dataloader_idx] if isinstance(dls, list) else dls
        ds: IterablePredDataset = dl.dataset
        if not isinstance(ds, IterablePredDataset):
            # TODO: change to warning
            raise TypeError("Prediction dataset is not `IterablePredDataset`.")

        # for i in range(prediction.shape[0]):
        #   prediction_image = prediction[0]
        #   sample_id = batch_idx * dl.batch_size + i

        file_name = self.write_filenames[self.current_file_index]
        file_path = (dirpath / file_name).with_suffix(self.write_extension)
        self.write_func(file_path=file_path, img=prediction, **self.write_func_kwargs)
        self.current_file_index += 1

    def reset(self) -> None:
        """
        Reset internal attributes.

        Resets the `write_filenames` and `current_file_index` attributes.
        """
        self.write_filenames = None
        self.current_file_index = 0
