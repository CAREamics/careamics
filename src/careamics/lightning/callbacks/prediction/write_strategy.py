"""Module containing different strategies for writing predictions."""

from pathlib import Path
from typing import Protocol

from careamics.dataset.image_region_data import ImageRegionData


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

    def set_source_base(self, source_base: Path | None) -> None:
        """
        Set the common parent directory of the prediction sources.

        Called by the prediction writer callback. Strategies that write individual
        files may use it to preserve the sources' directory structure in the output;
        strategies that do not (e.g. writing to a single store) may ignore it.

        Parameters
        ----------
        source_base : pathlib.Path or None
            Common parent of all prediction sources, or None if it cannot be
            determined.
        """
        ...
