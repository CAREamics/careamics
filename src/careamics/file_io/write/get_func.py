from pathlib import Path
from typing import Protocol, Union

from numpy.typing import NDArray

from careamics.config.support import SupportedData

from .tiff import write_tiff


# This is very strict, arguments have to be called file_path & img
# Alternative? - doesn't capture *args & **kwargs
# WriteFunc = Callable[[Path, NDArray], None]
class WriteFunc(Protocol):
    def __call__(self, file_path: Path, img: NDArray, *args, **kwargs) -> None: ...


WRITE_FUNCS: dict[SupportedData, WriteFunc] = {
    SupportedData.TIFF: write_tiff,
}


def get_write_func(data_type: Union[SupportedData, str]) -> WriteFunc:
    if data_type in WRITE_FUNCS:
        return WRITE_FUNCS[data_type]
    else:
        raise NotImplementedError(f"Data type {data_type} is not supported.")
