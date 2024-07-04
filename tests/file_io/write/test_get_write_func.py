import pytest

from careamics.file_io import get_write_func
from careamics.file_io.write import write_tiff
from careamics.config.support import SupportedData

def test_get_read_tiff():
    assert get_write_func(SupportedData.TIFF) is write_tiff

def test_get_read_any_error():
    with pytest.raises(NotImplementedError):
        get_write_func("some random")