import importlib

import pytest

from careamics.dataset_ng.patch_extractor.image_stack import CziImageStack


def test_import_czi_image_stack_failure():
    """Test that in the absence of pylibCZIrw, the CziImageStack import fails
    with an import error."""
    if importlib.util.find_spec("pylibCZIrw") is None:
        with pytest.raises(ImportError):
            CziImageStack(data_path="")
