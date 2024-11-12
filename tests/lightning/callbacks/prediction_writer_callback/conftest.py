from unittest.mock import Mock

import pytest

from careamics.file_io import WriteFunc


@pytest.fixture
def write_func():
    """Mock `WriteFunc`."""
    return Mock(spec=WriteFunc)
