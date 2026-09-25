"""Tests for the Zarr access factory."""

import pytest

from careamics.dataset.image_stack.zarr_access import (
    TensorstoreAccess,
    ZarrPythonAccess,
    create_zarr_access,
)


@pytest.mark.parametrize(
    ("backend", "access_type", "use_zarrs"),
    [
        pytest.param("zarr", ZarrPythonAccess, False, id="zarr"),
        pytest.param("zarrs", ZarrPythonAccess, True, id="zarrs"),
        pytest.param("tensorstore", TensorstoreAccess, None, id="tensorstore"),
    ],
)
def test_create_zarr_access(backend, access_type, use_zarrs):
    """Test that backend names create the corresponding access implementation."""
    access = create_zarr_access(backend)

    assert isinstance(access, access_type)
    if use_zarrs is not None:
        assert access._use_zarrs is use_zarrs


def test_create_zarr_access_invalid_backend():
    """Test that an unsupported backend raises an error."""
    with pytest.raises(ValueError, match="Unsupported Zarr backend"):
        create_zarr_access("invalid")
