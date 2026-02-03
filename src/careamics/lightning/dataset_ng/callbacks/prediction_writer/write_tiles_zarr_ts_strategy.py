"""Tile Zarr writing strategy using TensorStore."""

import builtins
from collections.abc import Sequence
from pathlib import Path

import tensorstore as ts
import zarr

from careamics.dataset_ng.dataset import ImageRegionData
from careamics.dataset_ng.image_stack_loader.zarr_utils import (
    decipher_zarr_uri,
    is_valid_uri,
)
from careamics.dataset_ng.patching_strategies import TileSpecs, is_tile_specs

from .write_tiles_zarr_strategy import (
    _add_output_key,
    _auto_chunks,
    _update_data_shape,
)

# TODO T axis handling
# TODO get metadata from zarr file and pass it to TensorStore array creation


class WriteTilesZarrTS:
    """TensorStore-based Zarr tile writer strategy.

    This writer creates zarr files, groups and arrays as needed and writes tiles
    into the appropriate locations using TensorStore for optimized I/O.
    """

    def __init__(self) -> None:
        """Constructor."""
        self.current_store_path: str | None = None
        self.current_array_path: str | None = None
        self.current_array: ts.TensorStore | None = None

    def _ensure_zarr_groups(self, store_path: str, array_path: str) -> None:
        """Ensure all parent groups exist with proper zarr metadata.

        TensorStore creates arrays but doesn't create parent group metadata,
        so we use zarr-python to ensure the group hierarchy exists.

        Parameters
        ----------
        store_path : str
            Path to the zarr store.
        array_path : str
            Full path to the array (e.g., "group1/group2/data").
        """
        # Open or create the root group
        root = zarr.open_group(store_path, mode="a")

        # Split the path and create intermediate groups
        parts = array_path.split("/")
        if len(parts) > 1:
            # Create all parent groups (excluding the array name itself)
            current_path = ""
            for part in parts[:-1]:
                current_path = f"{current_path}/{part}" if current_path else part
                if current_path not in root:
                    root.create_group(current_path)

    def _create_or_open_array(
        self,
        store_path: str,
        full_array_path: str,
        axes: str,
        data_shape: Sequence[int],
        shards: tuple[int, ...] | None,
        chunks: tuple[int, ...] | None,
    ) -> None:
        """Create or open a zarr array using TensorStore.

        Parameters
        ----------
        store_path : str
            Path to the zarr store.
        full_array_path : str
            Full path to the array (including group path if any).
        axes : str
            Axes string in SC(Z)YX format with original data order.
        data_shape : Sequence[int]
            Shape of the array.
        shards : tuple[int, ...] | None
            Shard size for the array.
        chunks : tuple[int, ...] | None
            Chunk size for the array.
        """
        # Ensure parent groups exist with proper metadata
        self._ensure_zarr_groups(store_path, full_array_path)

        # Get shape without non-existing axes (S or C)
        updated_shape = _update_data_shape(axes, data_shape)

        if chunks is not None and len(updated_shape) != len(chunks):
            raise ValueError(
                f"Shape {updated_shape} and chunks {chunks} have different lengths."
            )

        if chunks is None:
            chunks = _auto_chunks(axes, data_shape)

        if shards is not None and len(chunks) != len(shards):
            raise ValueError(
                f"Chunks {chunks} and shards {shards} have different lengths."
            )

        # TensorStore spec for zarr v3
        spec = {
            "driver": "zarr3",
            "kvstore": {
                "driver": "file",
                "path": store_path,
            },
            "path": full_array_path,
            "metadata": {
                "shape": list(updated_shape),
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {
                        "chunk_shape": (
                            list(shards) if shards is not None else list(chunks)
                        ),
                    },
                },
                "data_type": "float32",
            },
            "create": True,
            "open": True,
            "delete_existing": False,
        }

        # Add sharding codec if shards are provided
        if shards is not None:
            spec["metadata"]["codecs"] = [
                {
                    "name": "sharding_indexed",
                    "configuration": {
                        "chunk_shape": list(chunks),
                    },
                },
            ]

        try:
            self.current_array = ts.open(spec).result()
            self.current_store_path = store_path
            self.current_array_path = full_array_path
        except Exception as e:
            raise RuntimeError(
                f"Failed to create/open zarr array at '{store_path}/{full_array_path}' "
                f"with TensorStore: {e}"
            ) from e

    def write_tile(self, dirpath: Path, region: ImageRegionData) -> None:
        """Write cropped tile to zarr array using TensorStore.

        Parameters
        ----------
        dirpath : Path
            Path to directory to save predictions to.
        region : ImageRegionData
            Image region data containing tile information.
        """
        if is_valid_uri(region.source):
            store_path, parent_path, array_name = decipher_zarr_uri(region.source)
            output_store_path = str(_add_output_key(dirpath, store_path))
        else:
            raise NotImplementedError(
                f"Invalid zarr URI: {region.source}. Currently, only predicting from "
                f"Zarr files is supported when writing Zarr tiles."
            )

        # Construct full array path (group path + array name)
        full_array_path = f"{parent_path}/{array_name}" if parent_path else array_name

        # Check if we need to open/create a new array
        if (
            self.current_array is None
            or self.current_store_path != output_store_path
            or self.current_array_path != full_array_path
        ):
            shape = region.data_shape
            chunks: tuple[int, ...] | None = region.additional_metadata.get(
                "chunks", None
            )
            shards: tuple[int, ...] | None = region.additional_metadata.get(
                "shards", None
            )
            self._create_or_open_array(
                output_store_path, full_array_path, region.axes, shape, shards, chunks
            )

        assert is_tile_specs(region.region_spec)  # for mypy
        tile_spec: TileSpecs = region.region_spec
        crop_coords = tile_spec["crop_coords"]
        crop_size = tile_spec["crop_size"]
        stitch_coords = tile_spec["stitch_coords"]
        sample_idx = tile_spec["sample_idx"]

        # Compute crop slices on the tile data (which is C(Z)YX)
        crop_slices: tuple[builtins.ellipsis | slice | int, ...] = (
            ...,
            *[
                slice(start, start + length)
                for start, length in zip(crop_coords, crop_size, strict=True)
            ],
        )

        # region.data has shape C(Z)YX
        crop = region.data[crop_slices]

        # Build stitch slices for the output array based on its dimensions
        stitch_slices: list[slice | int] = []

        # Add S dimension if present in output array
        if "S" in region.axes:
            stitch_slices.append(sample_idx)

        # Handle C dimension
        if "C" in region.axes:
            # C dimension exists in output array, keep all channels
            stitch_slices.append(slice(None))
        elif crop.shape[0] == 1:
            # C is singleton and not in original axes, remove it from crop
            crop = crop[0]

        # Add spatial dimensions (where the tile will be stitched)
        stitch_slices.extend(
            [
                slice(start, start + length)
                for start, length in zip(stitch_coords, crop_size, strict=True)
            ]
        )

        if self.current_array is not None:
            # Write using TensorStore
            self.current_array[tuple(stitch_slices)].write(crop).result()
        else:
            raise RuntimeError("Zarr array not initialized.")

    def write_batch(
        self,
        dirpath: Path,
        predictions: list[ImageRegionData],
    ) -> None:
        """
        Write all tiles to a Zarr file using TensorStore.

        Parameters
        ----------
        dirpath : Path
            Path to directory to save predictions to.
        predictions : list[ImageRegionData]
            Decollated predictions.
        """
        for region in predictions:
            self.write_tile(dirpath, region)
