from pathlib import Path
from typing import Protocol, Union

from numpy.typing import NDArray

from careamics.config.support import SupportedData
from .tiff import write_tiff

WRITE_FUNCS = {
    SupportedData.TIFF.value: write_tiff,
}


# This is very strict, arguments have to be called fp & img
class WriteFunc(Protocol):
    def __call__(self, fp: Path, img: NDArray, *args, **kwargs) -> None: ...


# Alternative? - doesn't capture *args & **kwargs
# SavePredictFunc = Callable[[Path, NDArray], None]


def get_write_func(data_type: Union[SupportedData, str]) -> WriteFunc:
    if data_type in WRITE_FUNCS:
        return WRITE_FUNCS[data_type]
    else:
        raise NotImplementedError(f"Data type {data_type} is not supported.")