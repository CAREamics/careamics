from pathlib import Path
from typing import Protocol, Union

from numpy.typing import NDArray

from ..config.support import SupportedData
from .save_tiff import save_tiff

SAVE_FUNCS = {
    SupportedData.TIFF.value: save_tiff,
}


# This is very strict, arguments have to be called fp & img
class SavePredictFunc(Protocol):
    def __call__(self, fp: Path, img: NDArray, *args, **kwargs) -> None: ...


# Alternative? - doesn't capture *args & **kwargs
# SavePredictFunc = Callable[[Path, NDArray], None]


def get_save_func(data_type: Union[SupportedData, str]) -> SavePredictFunc:
    if data_type in SAVE_FUNCS:
        return SAVE_FUNCS[data_type]
    else:
        raise NotImplementedError(f"Data type {data_type} is not supported.")
