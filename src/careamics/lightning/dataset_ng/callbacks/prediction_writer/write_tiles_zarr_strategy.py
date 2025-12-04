"""Tile Zarr writing strategy."""

import builtins
from collections.abc import Sequence
from pathlib import Path

import zarr
from numpy import float32

from careamics.dataset_ng.dataset import ImageRegionData
from careamics.dataset_ng.image_stack_loader.zarr_utils import decipher_zarr_uri
from careamics.dataset_ng.patching_strategies import TileSpecs

OUTPUT_KEY = "_output"


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
    new_name = p.stem + OUTPUT_KEY + p.suffix
    return dirpath / new_name


class WriteTilesZarr:
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
        shape: Sequence[int],
        shards: Sequence[int] | None,
        chunks: Sequence[int],
    ) -> None:
        """Create a new array in an existing zarr group.

        Parameters
        ----------
        array_name : str
            Name of the array within the zarr group.
        shape : Sequence[int]
            Shape of the array.
        shards : Sequence[int] or None
            Shard size for the array.
        chunks : Sequence[int]
            Chunk size for the array.

        Raises
        ------
        RuntimeError
            If the zarr group has not been initialized.
        """
        if self.current_group is None:
            raise RuntimeError("Zarr group not initialized.")

        if array_name not in self.current_group:

            shape = [i for i in shape if i != 1]

            if chunks == (1,):  # guard against the ImageRegionData default
                raise ValueError("Chunks cannot be (1,).")

            if shards == (1,):  # guard against the ImageRegionData default
                shards = None  # no sharding
            elif shards is not None:
                shards = tuple(shards)  # for mypy

            if len(shape) != len(chunks):
                raise ValueError(
                    f"Shape {shape} and chunks {chunks} have different lengths."
                )

            if shards is not None and len(shape) != len(shards):
                raise ValueError(
                    f"Shape {shape} and shards {shards} have different lengths."
                )

            self.current_array = self.current_group.create_array(
                name=array_name,
                shape=shape,
                shards=shards,
                chunks=tuple(chunks),
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
        store_path, parent_path, array_name = decipher_zarr_uri(region.source)
        output_store_path = _add_output_key(dirpath, store_path)

        if (
            self.current_group is None
            or str(self.current_group.store_path)[: len(OUTPUT_KEY)]
            != output_store_path
        ):
            self._create_zarr(output_store_path)

        if self.current_group is None or self.current_group.name != parent_path:
            self._create_group(parent_path)

        if self.current_array is None or self.current_array.basename != array_name:
            shape = region.data_shape
            chunks = region.chunks
            shards = region.shards
            self._create_array(array_name, shape, shards, chunks)

        tile_spec: TileSpecs = region.region_spec  # type: ignore[assignment]
        crop_coords = tile_spec["crop_coords"]
        crop_size = tile_spec["crop_size"]
        stitch_coords = tile_spec["stitch_coords"]

        crop_slices: tuple[builtins.ellipsis | slice, ...] = (
            ...,
            *[
                slice(start, start + length)
                for start, length in zip(crop_coords, crop_size, strict=True)
            ],
        )
        stitch_slices: tuple[builtins.ellipsis | slice, ...] = (
            ...,
            *[
                slice(start, start + length)
                for start, length in zip(stitch_coords, crop_size, strict=True)
            ],
        )

        if self.current_array is not None:
            self.current_array[stitch_slices] = region.data.squeeze()[crop_slices]
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
