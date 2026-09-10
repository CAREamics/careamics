"""Factory for Zarr access implementations."""

from typing import Literal, TypeAlias

from .tensorstore_access import TensorstoreAccess
from .zarr_access_protocol import ZarrAccessProtocol
from .zarr_python_access import ZarrPythonAccess

ZarrBackend: TypeAlias = Literal["zarr", "zarrs", "tensorstore"]


def create_zarr_access(backend: ZarrBackend) -> ZarrAccessProtocol:
    """Create a Zarr access implementation.

    Parameters
    ----------
    backend : {"zarr", "zarrs", "tensorstore"}
        Zarr I/O backend.

    Returns
    -------
    ZarrAccessProtocol
        Configured Zarr access implementation.
    """
    if backend == "zarr":
        return ZarrPythonAccess(use_zarrs=False)
    if backend == "zarrs":
        return ZarrPythonAccess(use_zarrs=True)
    if backend == "tensorstore":
        return TensorstoreAccess()
    raise ValueError(f"Unsupported Zarr backend '{backend}'.")
