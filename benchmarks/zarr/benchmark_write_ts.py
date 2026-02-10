"""Benchmark script comparing zarr and TensorStore writing performance."""

import csv
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
    repeat: int = 1,
) -> dict[str, Any]:
    """Benchmark writing tiles using zarr library."""
    tiling_strategy = TilingStrategy(
        data_shapes=[data_shape],
        patch_size=patch_size,
        overlaps=overlaps,
    )

    n_tiles = tiling_strategy.n_patches

    # Create mock source URI
    source = f"file://test_source.zarr/data_{repeat}"

    # Create writer
    writer = WriteTilesZarr()

    # Create tiles
    tiles = []
    for i in range(n_tiles):
        tile_spec = tiling_strategy.get_patch_spec(i)
        tile = create_mock_tile(tile_spec, data_shape, source, axes, chunks, shards)
        tiles.append(tile)

    # Benchmark batch writing
    tot_tiles = 0
    start = time.perf_counter()
    while tot_tiles < n_iterations:
        writer.write_batch(output_dir, tiles)
        tot_tiles += n_tiles
    end = time.perf_counter()

    total_time = (end - start) * 1000  # Convert to milliseconds
    mean_time_per_tile = total_time / tot_tiles

    return {
        "n_tiles": tot_tiles,
        "mean_time": mean_time_per_tile,
        "total_time": total_time,
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
    repeat: int = 1,
) -> dict[str, Any]:
    """Benchmark writing tiles using TensorStore."""
    tiling_strategy = TilingStrategy(
        data_shapes=[data_shape],
        patch_size=patch_size,
        overlaps=overlaps,
    )

    n_tiles = tiling_strategy.n_patches

    # Create mock source URI
    source = f"file://test_source.zarr/data_{repeat}"

    # Create writer
    writer = WriteTilesZarrTS()

    # Create tiles
    tiles = []
    for i in range(n_tiles):
        tile_spec = tiling_strategy.get_patch_spec(i)
        tile = create_mock_tile(tile_spec, data_shape, source, axes, chunks, shards)
        tiles.append(tile)

    # Benchmark batch writing
    tot_tiles = 0
    start = time.perf_counter()
    while tot_tiles < n_iterations:
        writer.write_batch(output_dir, tiles)
        tot_tiles += n_tiles
    end = time.perf_counter()

    total_time = (end - start) * 1000  # Convert to milliseconds
    mean_time_per_tile = total_time / tot_tiles

    return {
        "n_tiles": tot_tiles,
        "mean_time": mean_time_per_tile,
        "total_time": total_time,
    }


class Result:
    shape: int
    patch_size: int
    chunks: int
    shards: int | None

    n_tiles: int
    zarr_mean_time: float
    zarr_std_time: float
    zarr_adj_mean_time: float
    zarr_adj_std_time: float
    ts_mean_time: float
    ts_adj_mean_time: float
    ts_adj_std_time: float
    ts_std_time: float
    speedup: float


def main():
    """Run comparison benchmark."""
    import tempfile

    shape = (1, 1, 2048, 2048)
    results = []
    for p_exp in range(6, 9):
        for c_exp in range(5, 10):
            for s_exp in [0] + [i for i in range(c_exp + 1, 10)]:
                print(f"p {2**p_exp}, c {2**c_exp}, s {2**s_exp}")

                patch_size = (2**p_exp, 2**p_exp)
                overlaps = (0, 0)
                chunks = (1, 1, 2**c_exp, 2**c_exp)
                shards = (1, 1, 2**s_exp, 2**s_exp) if s_exp > 0 else None

                zarr_res = []
                ts_res = []
                for r in range(3):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        output_dir = Path(tmpdir)

                        zarr_results = benchmark_zarr_writing(
                            output_dir,
                            data_shape=shape,
                            patch_size=patch_size,
                            overlaps=overlaps,
                            axes="SCYX",
                            chunks=chunks,
                            shards=shards,
                            n_iterations=100,
                            repeat=r,
                        )

                        ts_results = benchmark_tensorstore_writing(
                            output_dir,
                            data_shape=shape,
                            patch_size=patch_size,
                            overlaps=overlaps,
                            axes="SCYX",
                            chunks=chunks,
                            shards=shards,
                            n_iterations=100,
                            repeat=r,
                        )

                        zarr_res.append(zarr_results["mean_time"])
                        ts_res.append(ts_results["mean_time"])

                    result = Result()
                    result.shape = shape[-1]
                    result.patch_size = patch_size[-1]
                    result.chunks = chunks[-1]
                    result.shards = shards[-1] if shards is not None else 1
                    result.n_tiles = zarr_results["n_tiles"]
                    result.zarr_mean_time = np.mean(zarr_res)
                    result.zarr_std_time = np.std(zarr_res)
                    result.zarr_adj_mean_time = (
                        1000 * np.mean(zarr_res) / np.prod(patch_size)
                    )
                    result.zarr_adj_std_time = (
                        1000 * np.std(zarr_res) / np.prod(patch_size)
                    )
                    result.ts_mean_time = np.mean(ts_res)
                    result.ts_std_time = np.std(ts_res)
                    result.ts_adj_mean_time = (
                        1000 * np.mean(ts_res) / np.prod(patch_size)
                    )
                    result.ts_adj_std_time = 1000 * np.std(ts_res) / np.prod(patch_size)
                    result.speedup = result.ts_mean_time / result.zarr_mean_time

                    results.append(result)

    # TODO write line by line to not loose any
    # save csv with the results
    with open(
        "benchmarks/zarr/benchmark_write_results.csv", "w", newline=""
    ) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Shape",
                "Patch size",
                "Chunks",
                "Shards",
                "N tiles",
                "Zarr mean time (ms)",
                "Zarr std time (ms)",
                "Zarr adj mean time (us)",
                "Zarr adj std time (us)",
                "TensorStore mean time (ms)",
                "TensorStore std time (ms)",
                "TensorStore adj mean time (us)",
                "TensorStore adj std time (us)",
                "Speedup",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.shape,
                    result.patch_size,
                    result.chunks,
                    result.shards,
                    result.n_tiles,
                    result.zarr_mean_time,
                    result.zarr_std_time,
                    result.zarr_adj_mean_time,
                    result.zarr_adj_std_time,
                    result.ts_mean_time,
                    result.ts_std_time,
                    result.ts_adj_mean_time,
                    result.ts_adj_std_time,
                    result.speedup,
                ]
            )


if __name__ == "__main__":
    main()
