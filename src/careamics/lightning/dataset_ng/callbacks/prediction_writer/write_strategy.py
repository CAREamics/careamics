"""Module containing different strategies for writing predictions."""

from pathlib import Path
from typing import Protocol

from careamics.dataset_ng.dataset import ImageRegionData


class WriteStrategy(Protocol):
    """Protocol for write strategy classes."""

    def write_batch(
        self,
        dirpath: Path,
        predictions: list[ImageRegionData],
    ) -> None:
        """
        WriteStrategy subclasses must contain this function to write a batch.

        Parameters
        ----------
        dirpath : Path
            Path to directory to save predictions to.
        predictions : list[ImageRegionData]
            Decollated predictions.
        """
        ...
