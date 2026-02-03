"""Benchmark script for writing tiles to zarr arrays."""

import time
from pathlib import Path
from typing import Any

import numpy as np

from careamics.dataset_ng.dataset import ImageRegionData
from careamics.dataset_ng.patching_strategies import TileSpecs, TilingStrategy
from careamics.lightning.dataset_ng.callbacks.prediction_writer.write_tiles_zarr_strategy import (
    WriteTilesZarr,
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


def benchmark_tile_writing(
    output_dir: Path,
    data_shape: tuple[int, ...],
    patch_size: tuple[int, ...],
    overlaps: tuple[int, ...],
    axes: str,
    chunks: tuple[int, ...] | None = None,
    shards: tuple[int, ...] | None = None,
    n_iterations: int = 100,
) -> dict[str, Any]:
    """Benchmark writing tiles to a zarr array.

    Parameters
    ----------
    output_dir : Path
        Directory to write zarr output.
    data_shape : tuple[int, ...]
        Shape of the full data in SC(Z)YX format.
    patch_size : tuple[int, ...]
        Size of patches in spatial dimensions.
    overlaps : tuple[int, ...]
        Overlap between adjacent tiles.
    axes : str
        Axes string.
    chunks : tuple[int, ...] | None
        Chunk sizes for output zarr.
    shards : tuple[int, ...] | None
        Shard sizes for output zarr.
    n_iterations : int
        Number of tiles to write for benchmarking.

    Returns
    -------
    dict[str, Any]
        Benchmark results.
    """
    # Create tiling strategy
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

    # Warmup
    warmup_tiles = []
    for i in range(min(5, n_tiles)):
        tile_spec = tiling_strategy.get_patch_spec(i)
        tile = create_mock_tile(tile_spec, data_shape, source, axes, chunks, shards)
        warmup_tiles.append(tile)

    writer.write_batch(output_dir, warmup_tiles)

    # Reset writer for actual benchmark
    writer = WriteTilesZarr()

    # Benchmark
    times = []
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
    """Run benchmark with different configurations."""
    import tempfile

    configurations = [
        {
            "name": "Small chunks (128x128)",
            "data_shape": (1, 2, 1024, 1024),
            "chunks": (1, 1, 128, 128),
            "patch_size": (256, 256),
            "overlaps": (32, 32),
            "axes": "CZYX",
        },
        {
            "name": "Large chunks (512x512)",
            "data_shape": (1, 2, 1024, 1024),
            "chunks": (1, 1, 512, 512),
            "patch_size": (256, 256),
            "overlaps": (32, 32),
            "axes": "CZYX",
        },
        {
            "name": "3D data",
            "data_shape": (1, 1, 64, 512, 512),
            "chunks": (1, 1, 32, 128, 128),
            "patch_size": (32, 128, 128),
            "overlaps": (8, 16, 16),
            "axes": "CZYX",
        },
    ]

    print("=" * 80)
    print("Zarr Tile Writing Benchmark")
    print("=" * 80)

    for config in configurations:
        print(f"\n{config['name']}")
        print("-" * 40)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Run benchmark
            results = benchmark_tile_writing(
                output_dir,
                data_shape=config["data_shape"],
                patch_size=config["patch_size"],
                overlaps=config["overlaps"],
                axes=config["axes"],
                chunks=config.get("chunks"),
                shards=config.get("shards"),
                n_iterations=100,
            )

            print(f"Data shape: {config['data_shape']}")
            print(f"Chunks: {config.get('chunks', 'auto')}")
            print(f"Patch size: {config['patch_size']}")
            print(f"Overlaps: {config['overlaps']}")
            print(f"Total tiles: {results['total_tiles']}")
            print(f"Benchmarked tiles: {results['n_tiles']}")
            print(f"Mean time per tile: {results['mean_time']*1000:.2f} ms")
            print(f"Tiles per second: {results['tiles_per_second']:.2f}")
            print(f"Total time: {results['total_time']:.2f} s")


if __name__ == "__main__":
    main()
