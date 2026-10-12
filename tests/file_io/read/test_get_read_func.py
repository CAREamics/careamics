import pytest

from careamics.config.support import SupportedData
from careamics.image_io import get_read_func
from careamics.image_io.read import read_tiff


def test_get_read_tiff():
    assert get_read_func(SupportedData.TIFF) is read_tiff


def test_get_read_any_error():
    with pytest.raises(NotImplementedError):
        get_read_func("some random")
