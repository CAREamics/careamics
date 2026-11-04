"""Benchmark whole-image OME-NGFF writes through the CAREamics writer."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import zarr
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
from yaozarrs import validate_zarr_store

from careamics.dataset.image_region_data import ImageRegionData
from careamics.lightning.callbacks.prediction import ZarrImageWriteStrategy

DEFAULT_ARRAY_SHAPE = (2048, 2048)
DEFAULT_CHUNKS = ((64, 64), (128, 128), (256, 256))
DEFAULT_SHARDS = (None, (256, 256), (512, 512))
DEFAULT_OUTPUT = Path(__file__).parent / "results" / "image_write_results.csv"
RESULT_FIELDS = (
    "backend",
    "array_shape",
    "chunks",
    "shards",
    "repeat",
    "seconds",
    "mib_per_second",
    "validated",
    "error_type",
    "error",
)


def prepare_region(
    data: np.ndarray,
    chunks: tuple[int, int],
    shards: tuple[int, int] | None,
) -> ImageRegionData:
    """Prepare one complete SCYX prediction outside the timed region."""
    height, width = data.shape[-2:]
    return ImageRegionData(
        data=data,
        source="array",
        data_shape=(1, 1, height, width),
        dtype="float32",
        axes="SCYX",
        target_axes="SCYX",
        original_data_shape=(1, 1, height, width),
        region_spec={
            "data_idx": 0,
            "sample_idx": 0,
            "coords": (0, 0),
            "patch_size": (height, width),
        },
        additional_metadata={
            "chunks": (1, 1, *chunks),
            "shards": None if shards is None else (1, 1, *shards),
        },
    )


def validate_output(
    output_dir: Path,
    expected: np.ndarray,
    chunks: tuple[int, int],
    shards: tuple[int, int] | None,
) -> None:
    """Validate the OME-NGFF array written by the production strategy."""
    store_path = output_dir / "array_0.ome.zarr"
    output = zarr.open_array(store_path / "0", mode="r")
    np.testing.assert_array_equal(output[:], expected[np.newaxis, ...])
    assert output.chunks == (1, 1, *chunks)
    expected_shards = None if shards is None else (1, 1, *shards)
    assert output.shards == expected_shards
    validate_zarr_store(store_path)


def run_benchmark(args: argparse.Namespace) -> list[dict[str, object]]:
    """Run the whole-image writer benchmark matrix."""
    if len(args.array_shape) != 2:
        raise ValueError("The image benchmark expects a two-dimensional XY shape.")

    rng = np.random.default_rng(SEED)
    data = rng.random((1, *args.array_shape), dtype=np.float32)
    rows: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="careamics-zarr-image-write-") as temp_dir:
        root = Path(temp_dir)
        case_index = 0
        for chunks in args.chunks:
            for shards in args.shards:
                if not layout_is_valid(args.array_shape, chunks, shards):
                    print(
                        f"Skipping incompatible layout chunks={chunks}, "
                        f"shards={shards}.",
                        file=sys.stderr,
                    )
                    continue

                region = prepare_region(data, chunks, shards)
                for backend_name in args.backends:
                    print(
                        f"Benchmarking image write with {backend_name}: "
                        f"chunks={chunks}, shards={shards}.",
                        file=sys.stderr,
                    )
                    for _ in range(args.warmups):
                        warmup_dir = root / f"warmup_{case_index}"
                        warmup_dir.mkdir()
                        ZarrImageWriteStrategy(
                            access=BACKENDS[backend_name]()
                        ).write_batch(warmup_dir, [region])
                        shutil.rmtree(warmup_dir, ignore_errors=True)

                    for repeat in range(args.repeats):
                        output_dir = root / f"output_{case_index}_{repeat}"
                        output_dir.mkdir()
                        writer = ZarrImageWriteStrategy(access=BACKENDS[backend_name]())
                        start = perf_counter()
                        try:
                            writer.write_batch(output_dir, [region])
                            seconds = perf_counter() - start
                            validate_output(output_dir, data, chunks, shards)
                            validated = True
                            error_type = ""
                            error = ""
                        except Exception as exc:
                            validated = False
                            error_type = type(exc).__name__
                            error = str(exc)
                            seconds = perf_counter() - start
                        finally:
                            shutil.rmtree(output_dir, ignore_errors=True)
                        rows.append(
                            {
                                "backend": backend_name,
                                "array_shape": shape_label(args.array_shape),
                                "chunks": shape_label(chunks),
                                "shards": shape_label(shards),
                                "repeat": repeat,
                                "seconds": seconds,
                                "mib_per_second": data.nbytes / (1024**2 * seconds),
                                "validated": validated,
                                "error_type": error_type,
                                "error": error,
                            }
                        )
                    case_index += 1
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--array-shape", type=parse_shape, default=DEFAULT_ARRAY_SHAPE)
    parser.add_argument("--chunks", type=parse_shape, nargs="+", default=DEFAULT_CHUNKS)
    parser.add_argument(
        "--shards", type=parse_optional_shape, nargs="+", default=DEFAULT_SHARDS
    )
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--warmups", type=non_negative_int, default=1)
    parser.add_argument(
        "--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS)
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    """Run the benchmark and write CSV results."""
    args = build_parser().parse_args()
    write_results(run_benchmark(args), RESULT_FIELDS, args.output)


if __name__ == "__main__":
    main()
