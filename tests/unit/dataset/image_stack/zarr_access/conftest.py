from pathlib import Path

import pytest

from tests.conftest import (
    create_image_collection_ome_zarr,
    create_single_image_ome_zarr,
)


@pytest.fixture(scope="session")
def single_image_ome_zarr_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("ome_zarr_single")
    return create_single_image_ome_zarr(root / "single_image.zarr")


@pytest.fixture(scope="session")
def image_collection_ome_zarr_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("ome_zarr_collection")
    return create_image_collection_ome_zarr(root / "collection.zarr")
