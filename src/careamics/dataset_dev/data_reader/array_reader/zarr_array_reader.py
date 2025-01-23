from pathlib import Path

from numpy.typing import NDArray


class ZarrArrayReader:

    def __init__(
        self,
        source: Path,
        # other args
    ):
        raise NotImplementedError("Not implemented yet.")

    def extract_patch(
        self, sample_idx: int, coords: tuple[int, ...], extent: tuple[int, ...]
    ) -> NDArray:
        raise NotImplementedError("Not implemented yet.")
