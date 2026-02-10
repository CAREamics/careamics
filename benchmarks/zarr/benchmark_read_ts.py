"""Benchmark script comparing zarr and TensorStore reading performance."""

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np

import zarr
from careamics.dataset_ng.image_stack import ZarrImageStack
from careamics.dataset_ng.image_stack.zarr_ts_stack import ZarrTSImageStack
from careamics.dataset_ng.patching_strategies import RandomPatchingStrategy


def create_test_zarr(
    store_path: Path,
    data_shape: tuple[int, ...] = (1, 2, 512, 512),
    chunks: tuple[int, ...] = (1, 1, 128, 128),
    shards: tuple[int, ...] | None = None,
) -> tuple[Path, str]:
    """Create a test zarr array with random data in zarr v3 format."""
    # Use zarr v3 format explicitly
    group = zarr.open_group(store_path, mode="w", zarr_format=3)
    array_name = "data"

    create_kwargs: dict[str, Any] = {
        "name": array_name,
        "shape": data_shape,
        "chunks": chunks,
        "dtype": np.float32,
    }
    if shards is not None:
        create_kwargs["shards"] = shards

    arr = group.create_array(**create_kwargs)
    arr[:] = np.random.rand(*data_shape).astype(np.float32)

    return store_path, array_name


def benchmark_zarr_reading(
    zarr_group: zarr.Group,
    array_name: str,
    axes: str,
    patch_size: tuple[int, ...],
    n_iterations: int = 100,
) -> dict[str, Any]:
    """Benchmark reading tiles using zarr library."""
    image_stack = ZarrImageStack(zarr_group, array_name, axes)

    tiling_strategy = RandomPatchingStrategy(
        data_shapes=[image_stack.data_shape],
        patch_size=patch_size,
    )

    # Warmup
    for i in range(5):
        tile_spec = tiling_strategy.get_patch_spec(i)
        _ = image_stack.extract_patch(
            sample_idx=tile_spec["sample_idx"],
            channels=None,
            coords=tile_spec["coords"],
            patch_size=tile_spec["patch_size"],
        )

    # Benchmark
    times = []
    for _ in range(n_iterations):
        tile_spec = tiling_strategy.get_patch_spec(0)

        start = time.perf_counter()
        patch = image_stack.extract_patch(
            sample_idx=tile_spec["sample_idx"],
            channels=None,
            coords=tile_spec["coords"],
            patch_size=tile_spec["patch_size"],
        )
        end = time.perf_counter()

        times.append(end - start)

    times_array = np.array(times)

    return {
        "n_tiles": n_iterations,
        "mean_time": times_array.mean() * 1000,  # Convert to milliseconds
        "total_time": times_array.sum(),
    }


def benchmark_tensorstore_reading(
    store_path: Path,
    array_name: str,
    axes: str,
    patch_size: tuple[int, ...],
    n_iterations: int = 100,
) -> dict[str, Any]:
    """Benchmark reading tiles using TensorStore."""
    image_stack = ZarrTSImageStack(store_path, array_name, axes)

    tiling_strategy = RandomPatchingStrategy(
        data_shapes=[image_stack.data_shape],
        patch_size=patch_size,
    )

    # Warmup
    for i in range(5):
        tile_spec = tiling_strategy.get_patch_spec(i)
        _ = image_stack.extract_channel_patch(
            sample_idx=tile_spec["sample_idx"],
            channels=None,
            coords=tile_spec["coords"],
            patch_size=tile_spec["patch_size"],
        )

    # Benchmark
    times = []
    for _ in range(n_iterations):
        tile_spec = tiling_strategy.get_patch_spec(0)

        start = time.perf_counter()
        patch = image_stack.extract_channel_patch(
            sample_idx=tile_spec["sample_idx"],
            channels=None,
            coords=tile_spec["coords"],
            patch_size=tile_spec["patch_size"],
        )
        end = time.perf_counter()

        times.append(end - start)

    times_array = np.array(times)

    return {
        "n_tiles": n_iterations,
        "mean_time": times_array.mean() * 1000,  # Convert to milliseconds
        "total_time": times_array.sum(),
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
    ts_std_time: float
    ts_adj_mean_time: float
    ts_adj_std_time: float
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
                chunks = (1, 1, 2**c_exp, 2**c_exp)
                shards = (1, 1, 2**s_exp, 2**s_exp) if s_exp > 0 else None

                with tempfile.TemporaryDirectory() as tmpdir:
                    store_path = Path(tmpdir) / "test.zarr"

                    # Create test data
                    store_path, array_name = create_test_zarr(
                        store_path,
                        data_shape=shape,
                        chunks=chunks,
                        shards=shards,
                    )

                    zarr_group = zarr.open_group(store_path, mode="r", zarr_format=3)

                    zarr_res = []
                    ts_res = []
                    for r in range(3):
                        # Benchmark zarr
                        zarr_results = benchmark_zarr_reading(
                            zarr_group,
                            array_name,
                            "SCYX",
                            patch_size,
                        )

                        # Benchmark TensorStore
                        ts_results = benchmark_tensorstore_reading(
                            store_path,
                            array_name,
                            "SCYX",
                            patch_size,
                            n_iterations=100,
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

    # TODO write line by line to not loose any data
    # save csv with the results
    with open("benchmarks/zarr/benchmark_read_results.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Shape",
                "Patch size",
                "Chunks",
                "Shards",
                "Repeat",
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
