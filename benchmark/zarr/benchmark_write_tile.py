"""Benchmark tiled prediction writes to Zarr arrays."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from time import perf_counter
from typing import cast

import numpy as np
from _benchmark_utils import (
    BACKENDS,
    SEED,
    layout_is_valid,
    non_negative_int,
    parse_optional_shape,
    parse_shape,
    positive_int,
    shape_label,
    write_results,
)

from careamics.dataset.image_region_data import ImageRegionData
from careamics.dataset.image_stack import InMemoryImageStack
from careamics.dataset.patch_constructor import BasicPatchConstr
from careamics.dataset.patch_extractor import PatchExtractor
from careamics.dataset.patching import TiledPatching, TileSpecs
from careamics.lightning.callbacks.prediction import ZarrTileWriteStrategy

DEFAULT_ARRAY_SHAPE = (2048, 2048)
DEFAULT_CHUNKS = ((64, 64), (128, 128), (256, 256))
DEFAULT_SHARDS = (None, (256, 256), (512, 512))
DEFAULT_TILE_SIZES = ((128, 128), (256, 256))
TILE_OVERLAP = (48, 48)
DEFAULT_OUTPUT = Path(__file__).parent / "results" / "write_results.csv"
RESULT_FIELDS = (
    "backend",
    "array_shape",
    "chunks",
    "shards",
    "tile_size",
    "tile_overlap",
    "n_tiles",
    "repeat",
    "seconds",
    "tiles_per_second",
    "mib_per_second",
)


def prepare_tiles(
    data: np.ndarray,
    axes: str,
    tile_size: tuple[int, ...],
    tile_overlap: tuple[int, ...],
    chunks: tuple[int, ...],
    shards: tuple[int, ...] | None,
) -> list[ImageRegionData[TileSpecs]]:
    """Prepare tiled regions outside the timed write.

    Parameters
    ----------
    data : numpy.ndarray
        Source image.
    axes : str
        Source image axes.
    tile_size : tuple[int, ...]
        Spatial tile size.
    tile_overlap : tuple[int, ...]
        Spatial tile overlap.
    chunks : tuple[int, ...]
        Output chunk shape.
    shards : tuple[int, ...] or None
        Output shard shape.

    Returns
    -------
    list[ImageRegionData[TileSpecs]]
        Materialized tiles with reconstruction metadata.
    """
    stack = InMemoryImageStack.from_array(data=data, axes=axes)
    extractor = PatchExtractor([stack])
    patching = TiledPatching(
        data_shapes=extractor.shapes,
        patch_size=tile_size,
        overlaps=tile_overlap,
    )
    constructor = BasicPatchConstr(
        patching_strategy=patching,
        input_extractor=extractor,
    )

    regions: list[ImageRegionData[TileSpecs]] = []
    for index in range(constructor.n_patches):
        tile, _, patch_spec = constructor.construct_patch(index)
        tile_spec = cast(TileSpecs, patch_spec)
        metadata = constructor.get_input_image_metadata(tile_spec)
        regions.append(
            ImageRegionData(
                data=tile,
                source=metadata["source"],
                data_shape=metadata["data_shape"],
                dtype=metadata["dtype"],
                axes=axes,
                target_axes=axes,
                original_data_shape=metadata["original_data_shape"],
                region_spec=tile_spec,
                additional_metadata={"chunks": chunks, "shards": shards},
            )
        )
    return regions


def write_tiles(
    writer: ZarrTileWriteStrategy,
    output_dir: Path,
    regions: list[ImageRegionData[TileSpecs]],
) -> None:
    """Write prepared tiles to a Zarr destination.

    Parameters
    ----------
    writer : ZarrTileWriteStrategy
        Configured tile writer.
    output_dir : pathlib.Path
        Fresh output directory.
    regions : list[ImageRegionData[TileSpecs]]
        Prepared tile regions.

    Returns
    -------
    None
        Tiles are written in place.
    """
    writer.write_batch(output_dir, regions)


def tile_layout_is_valid(
    spatial_shape: tuple[int, ...],
    tile_size: tuple[int, ...],
    tile_overlap: tuple[int, ...],
) -> bool:
    """Return whether a tile layout is valid.

    Parameters
    ----------
    spatial_shape : tuple[int, ...]
        Spatial image shape.
    tile_size : tuple[int, ...]
        Tile shape.
    tile_overlap : tuple[int, ...]
        Tile overlap.

    Returns
    -------
    bool
        Whether the tile layout can be benchmarked.
    """
    if not len(spatial_shape) == len(tile_size) == len(tile_overlap):
        return False
    return all(
        tile <= size and 0 <= overlap < tile
        for size, tile, overlap in zip(
            spatial_shape, tile_size, tile_overlap, strict=True
        )
    )


def run_benchmark(args: argparse.Namespace) -> list[dict[str, object]]:
    """Run the write benchmark matrix.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    list[dict[str, object]]
        Benchmark result rows.
    """
    if len(args.axes) != len(args.array_shape):
        raise ValueError("Array shape rank must match the number of axes.")

    rng = np.random.default_rng(SEED)
    data = rng.integers(0, 256, size=args.array_shape, dtype=np.uint16)
    spatial_shape = tuple(
        size
        for axis, size in zip(args.axes, args.array_shape, strict=True)
        if axis in "ZYX"
    )
    rows: list[dict[str, object]] = []
    valid_layouts = sum(
        layout_is_valid(args.array_shape, chunks, shards)
        and tile_layout_is_valid(spatial_shape, tile_size, TILE_OVERLAP)
        for chunks in args.chunks
        for shards in args.shards
        for tile_size in args.tile_sizes
    )
    total_runs = valid_layouts * len(args.backends)
    current_run = 0

    with tempfile.TemporaryDirectory(prefix="careamics-zarr-write-") as temp_dir:
        root = Path(temp_dir)
        run_index = 0
        for chunks in args.chunks:
            for shards in args.shards:
                if not layout_is_valid(args.array_shape, chunks, shards):
                    print(
                        "Skipping incompatible layout "
                        f"chunks={chunks}, shards={shards}.",
                        file=sys.stderr,
                    )
                    continue

                for tile_size in args.tile_sizes:
                    if not tile_layout_is_valid(spatial_shape, tile_size, TILE_OVERLAP):
                        print(
                            "Skipping incompatible tiles "
                            f"size={tile_size}, overlap={TILE_OVERLAP}.",
                            file=sys.stderr,
                        )
                        continue

                    regions = prepare_tiles(
                        data=data,
                        axes=args.axes,
                        tile_size=tile_size,
                        tile_overlap=TILE_OVERLAP,
                        chunks=chunks,
                        shards=shards,
                    )
                    total_bytes = sum(region.data.nbytes for region in regions)
                    for backend in args.backends:
                        current_run += 1
                        print(
                            f"[{current_run}/{total_runs}] Writing with {backend}: "
                            f"chunks={chunks}, shards={shards}, "
                            f"tile_size={tile_size}, overlap={TILE_OVERLAP}.",
                            file=sys.stderr,
                        )
                        for _ in range(args.warmups):
                            output_dir = root / f"warmup_{run_index}"
                            run_index += 1
                            output_dir.mkdir()
                            write_tiles(
                                ZarrTileWriteStrategy(access=BACKENDS[backend]()),
                                output_dir,
                                regions,
                            )
                            shutil.rmtree(output_dir)

                        for repeat in range(args.repeats):
                            output_dir = root / f"output_{run_index}"
                            run_index += 1
                            output_dir.mkdir()
                            writer = ZarrTileWriteStrategy(access=BACKENDS[backend]())
                            start = perf_counter()
                            write_tiles(writer, output_dir, regions)
                            seconds = perf_counter() - start
                            shutil.rmtree(output_dir)
                            rows.append(
                                {
                                    "backend": backend,
                                    "array_shape": shape_label(args.array_shape),
                                    "chunks": shape_label(chunks),
                                    "shards": shape_label(shards),
                                    "tile_size": shape_label(tile_size),
                                    "tile_overlap": shape_label(TILE_OVERLAP),
                                    "n_tiles": len(regions),
                                    "repeat": repeat,
                                    "seconds": seconds,
                                    "tiles_per_second": len(regions) / seconds,
                                    "mib_per_second": total_bytes / (1024**2 * seconds),
                                }
                            )
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--array-shape", type=parse_shape, default=DEFAULT_ARRAY_SHAPE)
    parser.add_argument("--axes", default="YX")
    parser.add_argument("--chunks", type=parse_shape, nargs="+", default=DEFAULT_CHUNKS)
    parser.add_argument(
        "--shards", type=parse_optional_shape, nargs="+", default=DEFAULT_SHARDS
    )
    parser.add_argument(
        "--tile-sizes", type=parse_shape, nargs="+", default=DEFAULT_TILE_SIZES
    )
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--warmups", type=non_negative_int, default=0)
    parser.add_argument(
        "--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS)
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    """Run the benchmark and write its results.

    Returns
    -------
    None
        Results are written as CSV.
    """
    args = build_parser().parse_args()
    rows = run_benchmark(args)
    write_results(rows, RESULT_FIELDS, args.output)


if __name__ == "__main__":
    main()
