from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol, Union

from numpy.typing import NDArray


class ArrayReader(Protocol):

    # TODO: not sure how compatible using Path will be for a zarr array
    #   (for a zarr array need to specify file path and internal zarr path)
    source: Union[Path, Literal["array"]]
    data_shape: Sequence[int]

    def extract_patch(
        self, sample_idx: int, coords: Sequence[int], extent: Sequence[int]
    ) -> NDArray: ...
