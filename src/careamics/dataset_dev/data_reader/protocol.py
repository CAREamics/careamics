from pathlib import Path
from typing import Literal, Protocol, Union

from numpy.typing import NDArray


class DataReader(Protocol):

    # TODO: not sure how compatible using Path will be for a zarr array
    #   (for a zarr array need to specify file path and internal zarr path)
    source: Union[Path, Literal["array"]]
    data_shape: tuple[int, ...]

    def extract_patch(
        self, sample_idx: int, coords: tuple[int, ...], extent: tuple[int, ...]
    ) -> NDArray: ...
