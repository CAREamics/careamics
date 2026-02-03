"""Benchmark script comparing zarr and TensorStore writing performance."""

import time
from pathlib import Path
from typing import Any

import numpy as np

from careamics.dataset_ng.dataset import ImageRegionData
from careamics.dataset_ng.patching_strategies import TileSpecs, TilingStrategy
from careamics.lightning.dataset_ng.callbacks.prediction_writer.write_tiles_zarr_strategy import (
    WriteTilesZarr,
)
from careamics.lightning.dataset_ng.callbacks.prediction_writer.write_tiles_zarr_ts_strategy import (
    WriteTilesZarrTS,
)


def create_mock_tile(
    tile_spec: TileSpecs,
    data_shape: tuple[int, ...],
    source: str,
    axes: str,
    chunks: tuple[int, ...] | None = None,
    shards: tuple[int, ...] | None = None,
) -> ImageRegionData:
    """Create a mock ImageRegionData for benchmarking.

    Parameters
    ----------
    tile_spec : TileSpecs
        Tile specification.
    data_shape : tuple[int, ...]
        Shape of the full data in SC(Z)YX format.
    source : str
        Source path/URI.
    axes : str
        Axes string.
    chunks : tuple[int, ...] | None
        Chunk sizes.
    shards : tuple[int, ...] | None
        Shard sizes.

    Returns
    -------
    ImageRegionData
        Mock image region data.
    """
    patch_size = tile_spec["patch_size"]

    # Create random tile data in C(Z)YX format
    if len(patch_size) == 2:
        tile_data = np.random.rand(1, *patch_size).astype(np.float32)
    else:
        tile_data = np.random.rand(1, *patch_size).astype(np.float32)

    additional_metadata = {}
    if chunks is not None:
        additional_metadata["chunks"] = chunks
    if shards is not None:
        additional_metadata["shards"] = shards

    return ImageRegionData(
        data=tile_data,
        source=source,
        data_shape=data_shape,
        dtype="float32",
        axes=axes,
        region_spec=tile_spec,
        additional_metadata=additional_metadata,
    )


def benchmark_zarr_writing(
    output_dir: Path,
    data_shape: tuple[int, ...],
    patch_size: tuple[int, ...],
    overlaps: tuple[int, ...],
    axes: str,
    chunks: tuple[int, ...] | None = None,
    shards: tuple[int, ...] | None = None,
    n_iterations: int = 100,
) -> dict[str, Any]:
    """Benchmark writing tiles using zarr library."""
    tiling_strategy = TilingStrategy(
        data_shapes=[data_shape],
        patch_size=patch_size,
        overlaps=overlaps,
    )

    n_tiles = min(n_iterations, tiling_strategy.n_patches)

    # Create mock source URI
    source = "file://test_source.zarr/data"

    # Create writer
    writer = WriteTilesZarr()

    # Create tiles
    tiles = []
    for i in range(n_tiles):
        tile_spec = tiling_strategy.get_patch_spec(i)
        tile = create_mock_tile(tile_spec, data_shape, source, axes, chunks, shards)
        tiles.append(tile)

    # Benchmark batch writing
    start = time.perf_counter()
    writer.write_batch(output_dir, tiles)
    end = time.perf_counter()

    total_time = end - start
    mean_time_per_tile = total_time / n_tiles

    return {
        "n_tiles": n_tiles,
        "total_tiles": tiling_strategy.n_patches,
        "mean_time": mean_time_per_tile,
        "total_time": total_time,
        "tiles_per_second": n_tiles / total_time,
    }


def benchmark_tensorstore_writing(
    output_dir: Path,
    data_shape: tuple[int, ...],
    patch_size: tuple[int, ...],
    overlaps: tuple[int, ...],
    axes: str,
    chunks: tuple[int, ...] | None = None,
    shards: tuple[int, ...] | None = None,
    n_iterations: int = 100,
) -> dict[str, Any]:
    """Benchmark writing tiles using TensorStore."""
    tiling_strategy = TilingStrategy(
        data_shapes=[data_shape],
        patch_size=patch_size,
        overlaps=overlaps,
    )

    n_tiles = min(n_iterations, tiling_strategy.n_patches)

    # Create mock source URI
    source = "file://test_source.zarr/data"

    # Create writer
    writer = WriteTilesZarrTS()

    # Create tiles
    tiles = []
    for i in range(n_tiles):
        tile_spec = tiling_strategy.get_patch_spec(i)
        tile = create_mock_tile(tile_spec, data_shape, source, axes, chunks, shards)
        tiles.append(tile)

    # Benchmark batch writing
    start = time.perf_counter()
    writer.write_batch(output_dir, tiles)
    end = time.perf_counter()

    total_time = end - start
    mean_time_per_tile = total_time / n_tiles

    return {
        "n_tiles": n_tiles,
        "total_tiles": tiling_strategy.n_patches,
        "mean_time": mean_time_per_tile,
        "total_time": total_time,
        "tiles_per_second": n_tiles / total_time,
    }


def main():
    """Run comparison benchmark."""
    import tempfile

    configurations = [
        {
            "name": "Small chunks (64x64)",
            "data_shape": (1, 1, 1024, 1024),
            "axes": "YX",
            "chunks": (64, 64),
            "patch_size": (128, 128),
            "overlaps": (32, 32),
        },
        {
            "name": "Larger chunks (128x128)",
            "data_shape": (1, 1, 1024, 1024),
            "axes": "YX",
            "chunks": (128, 128),
            "patch_size": (256, 256),
            "overlaps": (32, 32),
        },
        {
            "name": "3D data",
            "data_shape": (1, 1, 64, 512, 512),
            "axes": "ZYX",
            "chunks": (16, 64, 64),
            "patch_size": (16, 64, 64),
            "overlaps": (8, 16, 16),
        },
    ]

    print("=" * 80)
    print("Zarr vs TensorStore Writing Benchmark")
    print("=" * 80)

    for config in configurations:
        print(f"\n{config['name']}")
        print("-" * 40)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            print(f"Data shape: {config['data_shape']}")
            print(f"Chunks: {config.get('chunks', 'auto')}")
            print(f"Shards: {config.get('shards', 'None')}")
            print(f"Patch size: {config['patch_size']}")

            # Benchmark zarr
            print("\n  Using zarr library:")
            zarr_output = output_dir / "zarr_test"
            zarr_output.mkdir()
            zarr_results = benchmark_zarr_writing(
                zarr_output,
                data_shape=config["data_shape"],
                patch_size=config["patch_size"],
                overlaps=config["overlaps"],
                axes=config["axes"],
                chunks=config.get("chunks"),
                shards=config.get("shards"),
                n_iterations=100,
            )
            print(f"    Mean time per tile: {zarr_results['mean_time']*1000:.2f} ms")
            print(f"    Tiles per second: {zarr_results['tiles_per_second']:.2f}")
            print(f"    Total time: {zarr_results['total_time']:.2f} s")

            # Benchmark TensorStore
            print("\n  Using TensorStore:")
            ts_output = output_dir / "ts_test"
            ts_output.mkdir()
            ts_results = benchmark_tensorstore_writing(
                ts_output,
                data_shape=config["data_shape"],
                patch_size=config["patch_size"],
                overlaps=config["overlaps"],
                axes=config["axes"],
                chunks=config.get("chunks"),
                shards=config.get("shards"),
                n_iterations=100,
            )
            print(f"    Mean time per tile: {ts_results['mean_time']*1000:.2f} ms")
            print(f"    Tiles per second: {ts_results['tiles_per_second']:.2f}")
            print(f"    Total time: {ts_results['total_time']:.2f} s")

            # Calculate speedup
            speedup = zarr_results["mean_time"] / ts_results["mean_time"]
            print(f"\n  Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
