"""Tile Zarr writing strategy."""

import builtins
from collections.abc import Sequence
from pathlib import Path

import zarr
from numpy import float32

from careamics.dataset.image_region_data import ImageRegionData
from careamics.dataset.image_stack_loader.zarr_utils import (
    decipher_zarr_uri,
    is_valid_uri,
)
from careamics.dataset.patching import TileSpecs, is_tile_specs
from careamics.utils.reshape_array import (
    get_original_stitch_slices,
    restore_tile,
)

from .write_strategy import WriteStrategy

OUTPUT_KEY = "_output"


def _auto_chunks(original_axes: str, original_shape: Sequence[int]) -> tuple[int, ...]:
    """Generate automatic chunk sizes based on axes and shape.

    X and Y dimensions will be chunked with a maximum size of 128, other dimensions
    will have chunk size 1.

    Parameters
    ----------
    original_axes : str
        Axes string of the original data.
    original_shape : Sequence[int]
        Shape of the original array.

    Returns
    -------
    tuple[int, ...]
        Chunk sizes for each dimension in SC(Z)YX order, but excluding dimensions that
        are not in the axes string.
    """
    chunk_sizes = []

    for idx, ax in enumerate(original_axes):
        if ax in ("Y", "X"):
            dim_size = original_shape[idx]
            chunk_sizes.append(
                min(128, dim_size)
            )  # TODO arbitrary value, need benchmarking
        else:
            chunk_sizes.append(1)  # chunk size 1 for Z and non spatial dims

    return tuple(chunk_sizes)


def _add_output_key(dirpath: Path, path: str | Path) -> Path:
    """Add `_output` to zarr name.

    Parameters
    ----------
    dirpath : Path
        Directory path to save the output zarr.
    path : str | Path
        Original zarr path.

    Returns
    -------
    Path
        Zarr path with `output` key added.
    """
    p = Path(path)
    new_name = p.stem + OUTPUT_KEY + ".zarr"
    return dirpath / new_name


class ZarrTileWriteStrategy(WriteStrategy):
    """Zarr tile writer strategy.

    This writer creates zarr files, groups and arrays as needed and writes tiles
    into the appropriate locations.
    """

    def __init__(self) -> None:
        """Constructor."""
        self.current_store: zarr.Group | None = None
        self.current_group: zarr.Group | None = None
        self.current_array: zarr.Array | None = None

    def _create_zarr(self, store: str | Path) -> None:
        """Create a new zarr storage.

        Parameters
        ----------
        store : str | Path
            Path to the zarr store.
        """
        if not Path(store).exists():
            self.current_store = zarr.create_group(store)
        else:
            open_store = zarr.open(store)

            if not isinstance(open_store, zarr.Group):
                raise RuntimeError(f"Zarr store at {store} is not a group.")

            self.current_store = open_store

        print(f"Store: {Path(store).absolute()}")

    def _create_group(self, group_path: str) -> None:
        """Create a new group in an existing zarr storage.

        Parameters
        ----------
        group_path : str
            Path to the group within the zarr store.

        Raises
        ------
        RuntimeError
            If the zarr store has not been initialized.
        """
        if self.current_store is None:
            raise RuntimeError("Zarr store not initialized.")

        if group_path not in self.current_store:
            self.current_group = self.current_store.create_group(group_path)
        else:
            current_group = self.current_store[group_path]
            if not isinstance(current_group, zarr.Group):
                raise RuntimeError(f"Zarr group at {group_path} is not a group.")

            self.current_group = current_group

    def _create_array(
        self,
        array_name: str,
        original_axes: str,
        original_shape: Sequence[int],
        shards: tuple[int, ...] | None,
        chunks: tuple[int, ...] | None,
    ) -> None:
        """Create a new array in an existing zarr group.

        Parameters
        ----------
        array_name : str
            Name of the array within the zarr group.
        original_axes : str
            Axes string in SC(Z)YX format with original data order.
        original_shape : Sequence[int]
            Shape of the array.
        shards : tuple[int, ...] or None
            Shard size for the array.
        chunks : tuple[int, ...] or None
            Chunk size for the array.

        Raises
        ------
        RuntimeError
            If the zarr group has not been initialized.
        """
        if self.current_group is None:
            raise RuntimeError("Zarr group not initialized.")

        if array_name not in self.current_group:
            if chunks is not None and len(original_shape) != len(chunks):
                raise ValueError(
                    f"Shape {original_shape} and chunks {chunks} have different "
                    f"lengths."
                )

            if chunks is None:
                chunks = _auto_chunks(original_axes, original_shape)

            # TODO if we auto_chunks, we probably want to auto shards as well
            # there is shards="auto" in zarr, where array.target_shard_size_bytes
            # needs to be used (see zarr-python docs)
            if shards is not None and len(chunks) != len(shards):
                raise ValueError(
                    f"Chunks {chunks} and shards {shards} have different lengths."
                )

            self.current_array = self.current_group.create_array(
                name=array_name,
                shape=original_shape,
                shards=shards,
                chunks=chunks,
                dtype=float32,
            )
        else:
            current_array = self.current_group[array_name]
            if not isinstance(current_array, zarr.Array):
                raise RuntimeError(f"Zarr array at {array_name} is not an array.")
            self.current_array = current_array

    def write_tile(self, dirpath: Path, region: ImageRegionData) -> None:
        """Write cropped tile to zarr array.

        Parameters
        ----------
        dirpath : Path
            Path to directory to save predictions to.
        region : ImageRegionData
            Image region data containing tile information.
        """
        if region.source == "array":
            # data source is an in-memory array:
            # set a new zarr storage output path
            parent_path = ""
            output_store_path = dirpath.joinpath("prediction.zarr")
            # use array data index for array name (in case of having multiple arrays)
            data_idx = region.region_spec["data_idx"]
            array_name = f"{data_idx}"

        elif is_valid_uri(region.source):
            # source is a zarr
            store_path, parent_path, array_name = decipher_zarr_uri(region.source)
            output_store_path = _add_output_key(dirpath, store_path)

        elif ".zarr" not in region.source:
            # data source is a tiff or custom format image:
            # set the zarr storage output path using the source file name
            _source = Path(region.source)
            parent_path = ""
            output_store_path = _source.parent.joinpath(f"{_source.stem}.zarr")
            # use array data index for array name (in case of having multiple tiffs)
            data_idx = region.region_spec["data_idx"]
            array_name = f"{data_idx}"

        else:
            # probably we don't need this
            raise NotImplementedError(
                f"Invalid source: {region.source}. Currently, only predicting from "
                f"array, Zarr, or TIFF files is supported when writing Zarr tiles."
            )

        if (
            self.current_group is None
            or str(self.current_group.store_path)[: len(OUTPUT_KEY)]
            != output_store_path
        ):
            self._create_zarr(output_store_path)

        if self.current_group is None or self.current_group.name != parent_path:
            self._create_group(parent_path)

        original_shape = region.original_data_shape
        original_axes = region.axes

        if self.current_array is None or self.current_array.basename != array_name:
            # If the source is not a Zarr file, then chunks and shards will be `None`.
            chunks: tuple[int, ...] | None = region.additional_metadata.get(
                "chunks", None
            )
            shards: tuple[int, ...] | None = region.additional_metadata.get(
                "shards", None
            )
            self._create_array(array_name, region.axes, original_shape, shards, chunks)

        assert is_tile_specs(region.region_spec)  # for mypy
        tile_spec: TileSpecs = region.region_spec
        crop_coords = tile_spec["crop_coords"]
        crop_size = tile_spec["crop_size"]
        stitch_coords = tile_spec["stitch_coords"]

        # compute sample slice
        sample_idx = tile_spec["sample_idx"]
        stitch_slices = get_original_stitch_slices(
            original_axes, original_shape, sample_idx, stitch_coords, crop_size
        )
        crop_slices: tuple[builtins.ellipsis | slice | int, ...] = (
            ...,
            *[
                slice(start, start + length)
                for start, length in zip(crop_coords, crop_size, strict=True)
            ],
        )

        if self.current_array is not None:
            # region.data has shape C(Z)YX
            crop = region.data[crop_slices]
            reshaped_crop = restore_tile(crop, original_axes, original_shape)

            self.current_array[stitch_slices] = reshaped_crop
        else:
            raise RuntimeError("Zarr array not initialized.")

    def write_batch(
        self,
        dirpath: Path,
        predictions: list[ImageRegionData],
    ) -> None:
        """
        Write all tiles to a Zarr file.

        Parameters
        ----------
        dirpath : Path
            Path to directory to save predictions to.
        predictions : list[ImageRegionData]
            Decollated predictions.
        """
        for region in predictions:
            self.write_tile(dirpath, region)
