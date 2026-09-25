"""Benchmark patch reads from Zarr arrays."""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Sequence
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

from careamics.dataset.image_stack import ZarrImageStack
from careamics.dataset.image_stack.zarr_access import ZarrNode
from careamics.dataset.patch_constructor import BasicPatchConstr
from careamics.dataset.patch_extractor import PatchExtractor
from careamics.dataset.patching import PatchSpecs, RandomPatching

DEFAULT_ARRAY_SHAPE = (2048, 2048)
DEFAULT_CHUNKS = ((64, 64), (128, 128), (256, 256))
DEFAULT_SHARDS = (None, (256, 256), (512, 512))
DEFAULT_PATCH_SIZES = ((64, 64), (128, 128), (256, 256))
DEFAULT_OUTPUT = Path(__file__).parent / "results" / "read_results.csv"
RESULT_FIELDS = (
    "backend",
    "array_shape",
    "chunks",
    "shards",
    "patch_size",
    "n_patches",
    "repeat",
    "seconds",
    "patches_per_second",
    "mib_per_second",
)


class PreparedPatching:
    """Patching strategy backed by precomputed patch specifications.

    Parameters
    ----------
    patch_specs : sequence of PatchSpecs
        Precomputed patch specifications.
    """

    def __init__(self, patch_specs: Sequence[PatchSpecs]) -> None:
        """Initialize the prepared strategy.

        Parameters
        ----------
        patch_specs : sequence of PatchSpecs
            Precomputed patch specifications.
        """
        self._patch_specs = list(patch_specs)

    @property
    def n_patches(self) -> int:
        """Return the number of prepared patches.

        Returns
        -------
        int
            Number of patches.
        """
        return len(self._patch_specs)

    def get_patch_spec(self, index: int) -> PatchSpecs:
        """Return a prepared patch specification.

        Parameters
        ----------
        index : int
            Patch index.

        Returns
        -------
        PatchSpecs
            Prepared specification.
        """
        return self._patch_specs[index]

    def get_patch_indices(self, data_idx: int) -> list[int]:
        """Return prepared indices for an image.

        Parameters
        ----------
        data_idx : int
            Image index.

        Returns
        -------
        list[int]
            Matching patch indices.
        """
        return [
            index
            for index, spec in enumerate(self._patch_specs)
            if spec["data_idx"] == data_idx
        ]


def create_input_store(
    store_path: Path,
    data: np.ndarray,
    chunks: tuple[int, ...],
    shards: tuple[int, ...] | None,
) -> ZarrNode:
    """Create a benchmark input array.

    Parameters
    ----------
    store_path : pathlib.Path
        Destination Zarr store.
    data : numpy.ndarray
        Input data.
    chunks : tuple[int, ...]
        Chunk shape.
    shards : tuple[int, ...] or None
        Shard shape.

    Returns
    -------
    ZarrNode
        Root array node.
    """
    zarr.create_array(
        store=store_path,
        data=data,
        chunks=chunks,
        shards=shards,
        zarr_format=3,
        overwrite=True,
    )
    return ZarrNode(store_uri=store_path.as_uri(), node_type="array")


def prepare_patch_constructor(
    node: ZarrNode,
    axes: str,
    access_name: str,
    patch_size: tuple[int, ...],
    n_patches: int,
) -> BasicPatchConstr:
    """Prepare patch extraction without timing patch generation.

    Parameters
    ----------
    node : ZarrNode
        Input array node.
    axes : str
        Input array axes.
    access_name : str
        Backend name.
    patch_size : tuple[int, ...]
        Spatial patch size.
    n_patches : int
        Number of patch specifications to prepare.

    Returns
    -------
    BasicPatchConstr
        Constructor using precomputed patch specifications.
    """
    stack = ZarrImageStack(node=node, axes=axes, access=BACKENDS[access_name]())
    extractor = PatchExtractor([stack])
    sampling_strategy = RandomPatching(
        data_shapes=extractor.shapes, patch_size=patch_size, seed=SEED
    )
    patch_specs = [
        sampling_strategy.get_patch_spec(index % sampling_strategy.n_patches)
        for index in range(n_patches)
    ]
    return BasicPatchConstr(
        patching_strategy=PreparedPatching(patch_specs),
        input_extractor=extractor,
    )


def read_patches(constructor: BasicPatchConstr, n_patches: int) -> int:
    """Read prepared patches and return their total size.

    Parameters
    ----------
    constructor : BasicPatchConstr
        Prepared patch constructor.
    n_patches : int
        Number of patches to read.

    Returns
    -------
    int
        Total number of bytes read into arrays.
    """
    total_bytes = 0
    for index in range(n_patches):
        patch, _, _ = constructor.construct_patch(index)
        total_bytes += patch.nbytes
    return total_bytes


def run_benchmark(args: argparse.Namespace) -> list[dict[str, object]]:
    """Run the read benchmark matrix.

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
    rows: list[dict[str, object]] = []
    valid_layouts = sum(
        layout_is_valid(args.array_shape, chunks, shards)
        for chunks in args.chunks
        for shards in args.shards
    )
    total_runs = valid_layouts * len(args.patch_sizes) * len(args.backends)
    current_run = 0

    with tempfile.TemporaryDirectory(prefix="careamics-zarr-read-") as temp_dir:
        root = Path(temp_dir)
        layout_index = 0
        for chunks in args.chunks:
            for shards in args.shards:
                if not layout_is_valid(args.array_shape, chunks, shards):
                    print(
                        "Skipping incompatible layout "
                        f"chunks={chunks}, shards={shards}.",
                        file=sys.stderr,
                    )
                    continue

                store_path = root / f"input_{layout_index}.zarr"
                layout_index += 1
                node = create_input_store(store_path, data, chunks, shards)

                for patch_size in args.patch_sizes:
                    for backend in args.backends:
                        current_run += 1
                        print(
                            f"[{current_run}/{total_runs}] Reading with {backend}: "
                            f"chunks={chunks}, shards={shards}, "
                            f"patch_size={patch_size}.",
                            file=sys.stderr,
                        )
                        constructor = prepare_patch_constructor(
                            node=node,
                            axes=args.axes,
                            access_name=backend,
                            patch_size=patch_size,
                            n_patches=args.n_patches,
                        )
                        for _ in range(args.warmups):
                            read_patches(constructor, args.n_patches)

                        for repeat in range(args.repeats):
                            start = perf_counter()
                            total_bytes = read_patches(constructor, args.n_patches)
                            seconds = perf_counter() - start
                            rows.append(
                                {
                                    "backend": backend,
                                    "array_shape": shape_label(args.array_shape),
                                    "chunks": shape_label(chunks),
                                    "shards": shape_label(shards),
                                    "patch_size": shape_label(patch_size),
                                    "n_patches": args.n_patches,
                                    "repeat": repeat,
                                    "seconds": seconds,
                                    "patches_per_second": args.n_patches / seconds,
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
        "--patch-sizes", type=parse_shape, nargs="+", default=DEFAULT_PATCH_SIZES
    )
    parser.add_argument("--n-patches", type=positive_int, default=128)
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--warmups", type=non_negative_int, default=1)
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
