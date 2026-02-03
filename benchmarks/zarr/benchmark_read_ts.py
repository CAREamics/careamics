"""Benchmark script comparing zarr and TensorStore reading performance."""

import time
from pathlib import Path
from typing import Any

import numpy as np

import zarr
from careamics.dataset_ng.image_stack import ZarrImageStack
from careamics.dataset_ng.image_stack.zarr_ts_stack import ZarrTSImageStack
from careamics.dataset_ng.patching_strategies import TilingStrategy


def create_test_zarr(
    store_path: Path,
    data_shape: tuple[int, ...] = (1, 2, 512, 512),
    chunks: tuple[int, ...] = (1, 1, 128, 128),
) -> tuple[Path, str]:
    """Create a test zarr array with random data in zarr v3 format."""
    # Use zarr v3 format explicitly
    group = zarr.open_group(store_path, mode="w", zarr_format=3)
    array_name = "data"

    arr = group.create_array(
        name=array_name,
        shape=data_shape,
        chunks=chunks,
        dtype=np.float32,
    )
    arr[:] = np.random.rand(*data_shape).astype(np.float32)

    return store_path, array_name


def benchmark_zarr_reading(
    zarr_group: zarr.Group,
    array_name: str,
    axes: str,
    patch_size: tuple[int, ...],
    overlaps: tuple[int, ...],
    n_iterations: int = 100,
) -> dict[str, Any]:
    """Benchmark reading tiles using zarr library."""
    image_stack = ZarrImageStack(zarr_group, array_name, axes)

    tiling_strategy = TilingStrategy(
        data_shapes=[image_stack.data_shape],
        patch_size=patch_size,
        overlaps=overlaps,
    )

    n_tiles = min(n_iterations, tiling_strategy.n_patches)

    # Warmup
    for i in range(min(5, n_tiles)):
        tile_spec = tiling_strategy.get_patch_spec(i)
        _ = image_stack.extract_channel_patch(
            sample_idx=tile_spec["sample_idx"],
            channels=None,
            coords=tile_spec["coords"],
            patch_size=tile_spec["patch_size"],
        )

    # Benchmark
    times = []
    for i in range(n_tiles):
        tile_spec = tiling_strategy.get_patch_spec(i)

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
        "n_tiles": n_tiles,
        "mean_time": times_array.mean(),
        "std_time": times_array.std(),
        "median_time": np.median(times_array),
        "total_time": times_array.sum(),
        "tiles_per_second": n_tiles / times_array.sum(),
    }


def benchmark_tensorstore_reading(
    store_path: Path,
    array_name: str,
    axes: str,
    patch_size: tuple[int, ...],
    overlaps: tuple[int, ...],
    n_iterations: int = 100,
) -> dict[str, Any]:
    """Benchmark reading tiles using TensorStore."""
    image_stack = ZarrTSImageStack(store_path, array_name, axes)

    tiling_strategy = TilingStrategy(
        data_shapes=[image_stack.data_shape],
        patch_size=patch_size,
        overlaps=overlaps,
    )

    n_tiles = min(n_iterations, tiling_strategy.n_patches)

    # Warmup
    for i in range(min(5, n_tiles)):
        tile_spec = tiling_strategy.get_patch_spec(i)
        _ = image_stack.extract_channel_patch(
            sample_idx=tile_spec["sample_idx"],
            channels=None,
            coords=tile_spec["coords"],
            patch_size=tile_spec["patch_size"],
        )

    # Benchmark
    times = []
    for i in range(n_tiles):
        tile_spec = tiling_strategy.get_patch_spec(i)

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
        "n_tiles": n_tiles,
        "mean_time": times_array.mean(),
        "std_time": times_array.std(),
        "median_time": np.median(times_array),
        "total_time": times_array.sum(),
        "tiles_per_second": n_tiles / times_array.sum(),
    }


def main():
    """Run comparison benchmark."""
    import tempfile

    configurations = [
        {
            "name": "Small chunks (64x64)",
            "data_shape": (2, 1024, 1024),
            "axes": "CYX",
            "chunks": (1, 64, 64),
            "patch_size": (128, 128),
            "overlaps": (32, 32),
        },
        {
            "name": "Larger chunks (128x128)",
            "data_shape": (2, 1024, 1024),
            "axes": "CYX",
            "chunks": (1, 128, 128),
            "patch_size": (256, 256),
            "overlaps": (32, 32),
        },
        {
            "name": "3D data",
            "data_shape": (64, 512, 512),
            "axes": "ZYX",
            "chunks": (16, 64, 64),
            "patch_size": (16, 64, 64),
            "overlaps": (8, 16, 16),
        },
    ]

    print("=" * 80)
    print("Zarr vs TensorStore Reading Benchmark")
    print("=" * 80)

    for config in configurations:
        print(f"\n{config['name']}")
        print("-" * 40)

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.zarr"

            # Create test data
            store_path, array_name = create_test_zarr(
                store_path,
                data_shape=config["data_shape"],
                chunks=config["chunks"],
            )

            zarr_group = zarr.open_group(store_path, mode="r", zarr_format=3)

            print(f"Data shape: {config['data_shape']}")
            print(f"Chunks: {config['chunks']}")
            print(f"Patch size: {config['patch_size']}")

            # Benchmark zarr
            print("\n  Using zarr library:")
            zarr_results = benchmark_zarr_reading(
                zarr_group,
                array_name,
                config["axes"],
                config["patch_size"],
                config["overlaps"],
                n_iterations=100,
            )
            print(f"    Mean time per tile: {zarr_results['mean_time']*1000:.2f} ms")
            print(f"    Tiles per second: {zarr_results['tiles_per_second']:.2f}")
            print(f"    Total time: {zarr_results['total_time']:.2f} s")

            # Benchmark TensorStore
            print("\n  Using TensorStore:")
            ts_results = benchmark_tensorstore_reading(
                store_path,
                array_name,
                config["axes"],
                config["patch_size"],
                config["overlaps"],
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
