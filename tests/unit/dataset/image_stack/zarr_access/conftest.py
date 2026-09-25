from pathlib import Path

import numpy as np
import pytest
from yaozarrs import v05, write


def _ome_image_metadata() -> v05.Image:
    return v05.Image(
        multiscales=[
            v05.Multiscale(
                name="image",
                axes=[
                    v05.SpaceAxis(name="y", unit="micrometer"),
                    v05.SpaceAxis(name="x", unit="micrometer"),
                ],
                datasets=[
                    v05.Dataset(
                        path="0",
                        coordinateTransformations=[
                            v05.ScaleTransformation(scale=[1.0, 1.0])
                        ],
                    ),
                    v05.Dataset(
                        path="1",
                        coordinateTransformations=[
                            v05.ScaleTransformation(scale=[2.0, 2.0])
                        ],
                    ),
                ],
            )
        ]
    )


def _validate_shape_parameters(
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    shards: tuple[int, ...] | None,
) -> None:
    if len(shape) != 2 or len(chunks) != 2:
        raise ValueError("Expected 2D YX shape and chunks.")
    if any(size % 2 != 0 for size in shape):
        raise ValueError("Expected even YX shape for two-level OME-Zarr fixtures.")
    if shards is not None and len(shards) != len(chunks):
        raise ValueError("Shards have the wrong length.")


def _create_multiscale_datasets(shape: tuple[int, int]) -> list[np.ndarray]:
    data_0 = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    shape_1 = tuple(size // 2 for size in shape)
    data_1 = np.arange(np.prod(shape_1), dtype=np.float32).reshape(shape_1)
    return [data_0, data_1]


def create_single_image_ome_zarr(
    store_path: Path,
    *,
    shape: tuple[int, ...] = (16, 16),
    chunks: tuple[int, ...] = (8, 8),
    shards: tuple[int, ...] | None = None,
) -> Path:
    _validate_shape_parameters(shape, chunks, shards)
    metadata = _ome_image_metadata()
    datasets = _create_multiscale_datasets(shape)

    return write.v05.write_image(
        store_path,
        metadata,
        datasets=datasets,
        chunks=chunks,
        shards=shards,
    )


def create_image_collection_ome_zarr(
    store_path: Path,
    *,
    shape: tuple[int, ...] = (16, 16),
    chunks: tuple[int, ...] = (8, 8),
    shards: tuple[int, ...] | None = None,
) -> Path:
    _validate_shape_parameters(shape, chunks, shards)
    metadata = _ome_image_metadata()
    datasets = _create_multiscale_datasets(shape)

    return write.v05.write_bioformats2raw(
        store_path,
        {
            "img_0": (metadata, datasets),
            "img_1": (metadata, datasets),
        },
        chunks=chunks,
        shards=shards,
    )


@pytest.fixture(scope="session")
def single_image_ome_zarr_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("ome_zarr_single")
    return create_single_image_ome_zarr(root / "single_image.zarr")


@pytest.fixture(scope="session")
def image_collection_ome_zarr_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("ome_zarr_collection")
    return create_image_collection_ome_zarr(root / "collection.zarr")
