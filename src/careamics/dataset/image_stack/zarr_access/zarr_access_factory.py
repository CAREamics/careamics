"""Factory for Zarr access implementations."""

from typing import Literal, TypeAlias

from .default_zarr_access import DefaultZarrAccess
from .tensorstore_zarr_access import TensorstoreZarrAccess
from .zarr_access import ZarrAccess

ZarrBackend: TypeAlias = Literal["zarr", "zarrs", "tensorstore"]


def create_zarr_access(backend: ZarrBackend) -> ZarrAccess:
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
        return DefaultZarrAccess(use_zarrs=False)
    if backend == "zarrs":
        return DefaultZarrAccess(use_zarrs=True)
    if backend == "tensorstore":
        return TensorstoreZarrAccess()
    raise ValueError(f"Unsupported Zarr backend '{backend}'.")
