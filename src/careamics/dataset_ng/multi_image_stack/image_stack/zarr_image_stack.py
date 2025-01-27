from collections.abc import Sequence
from pathlib import Path

from numpy.typing import NDArray


class ZarrImageStack:

    def __init__(
        self,
        source: Path,
        # other args
    ):
        raise NotImplementedError("Not implemented yet.")

    def extract_patch(
        self, sample_idx: int, coords: Sequence[int], patch_size: Sequence[int]
    ) -> NDArray:
        raise NotImplementedError("Not implemented yet.")
