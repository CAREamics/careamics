from collections.abc import Sequence
from pathlib import Path

from numpy.typing import NDArray


class ZarrImageStack:
    """
    A class for extracting patches from an image stack that is stored as a zarr array.
    """

    def __init__(
        self,
        source: Path,
        # other args
    ):
        # Note: will probably need to store axes from metadata
        #   - transformation will have to happen in `extract_patch`
        raise NotImplementedError("Not implemented yet.")

    def extract_patch(
        self, sample_idx: int, coords: Sequence[int], patch_size: Sequence[int]
    ) -> NDArray:
        raise NotImplementedError("Not implemented yet.")
