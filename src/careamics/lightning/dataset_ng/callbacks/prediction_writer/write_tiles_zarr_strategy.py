"""Tile Zarr writing strategy."""

import builtins
from collections.abc import Sequence
from pathlib import Path

import zarr
from numpy import float32

from careamics.dataset.dataset_utils.dataset_utils import get_axes_order
from careamics.dataset_ng.dataset import ImageRegionData
from careamics.dataset_ng.image_stack_loader.zarr_utils import (
    decipher_zarr_uri,
    is_valid_uri,
)
from careamics.dataset_ng.patching_strategies import TileSpecs, is_tile_specs

OUTPUT_KEY = "_output"


def _update_data_shape(axes: str, data_shape: Sequence[int]) -> tuple[int, ...]:
    """Update data shape to remove non existing dimensions.

    Parameters
    ----------
    axes : str
        Axes string of the original data.
    data_shape : Sequence[int]
        Shape of the array in SC(Z)YX order with potential singleton dimensions.

    Returns
    -------
    tuple[int, ...]
        Updated shape with non-existing axes removed.
    """
    new_shape = []

    if "S" in axes:
        new_shape.append(data_shape[0])

    if "C" in axes:
        new_shape.append(data_shape[1])

    for idx in range(2, len(data_shape)):
        new_shape.append(data_shape[idx])

    return tuple(new_shape)


def _update_T_axis(axes: str) -> str:
    """Update axes string to account for multiplexed S and T dimensions.

    If only `T` is present, then it is relabeled as `S`. If both `S` and `T` are
    present, then `T` is removed.

    Parameters
    ----------
    axes : str
        Axes string of the original data.

    Returns
    -------
    str
        Updated axes string.
    """
    if "T" in axes:
        if "S" in axes:
            # remove T
            axes = axes.replace("T", "")
        else:
            # relabel T as S
            axes = axes.replace("T", "S")
    return axes


def _auto_chunks(axes: str, data_shape: Sequence[int]) -> tuple[int, ...]:
    """Generate automatic chunk sizes based on axes and shape.

    Spatial dimensions will be chunked with a maximum size of 64, other dimensions
    will have chunk size 1.

    Parameters
    ----------
    axes : str
        Axes string of the original data.
    data_shape : Sequence[int]
        Shape of the array in SC(Z)YX order with potential singleton dimensions.

    Returns
    -------
    tuple[int, ...]
        Chunk sizes for each dimension in SC(Z)YX order, but excluding dimensions that
        are not in the axes string.
    """
    chunk_sizes = []

    # axes may contain T, which is now multiplexed with S
    updated_axes = _update_T_axis(axes)

    # axes reshaping indices in the order SC(Z)YX
    indices = get_axes_order(updated_axes, ref_axes="SCZYX")

    sczyx_offset = 0

    if "S" not in updated_axes:
        sczyx_offset = 1  # singleton S dim added to data_shape

    if "C" not in updated_axes:
        sczyx_offset += 1  # singleton C dim added to data_shape

    # loop through the original axes in order SC(Z)YX
    #   - original_index is the index of the axis in the original `axes` string
    #   - idx is the index in SC(Z)YX order of the axes present in `axes`
    #   - since all non spatial are treated the same, we can recover the spatial dims
    # index in SC(Z)YX order by using sczyx_offset
    for idx, original_index in enumerate(indices):
        axis = updated_axes[original_index]

        # TODO we should probably not chunk along Z (#658)
        if axis in ("Z", "Y", "X"):
            dim_size = data_shape[idx + sczyx_offset]
            chunk_sizes.append(
                min(128, dim_size)
            )  # TODO arbitrary value, about 1MB for float64
        else:
            chunk_sizes.append(1)

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
        axes: str,
        data_shape: Sequence[int],
        shards: tuple[int, ...] | None,
        chunks: tuple[int, ...] | None,
    ) -> None:
        """Create a new array in an existing zarr group.

        Parameters
        ----------
        array_name : str
            Name of the array within the zarr group.
        axes : str
            Axes string in SC(Z)YX format with original data order.
        data_shape : Sequence[int]
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
            # get shape without non-existing axes (S or C)
            updated_shape = _update_data_shape(axes, data_shape)

            if chunks is not None and len(updated_shape) != len(chunks):
                raise ValueError(
                    f"Shape {updated_shape} and chunks {chunks} have different lengths."
                )

            if chunks is None:
                chunks = _auto_chunks(axes, data_shape)

            # TODO if we auto_chunks, we probably want to auto shards as well
            # there is shards="auto" in zarr, where array.target_shard_size_bytes
            # needs to be used (see zarr-python docs)
            if shards is not None and len(chunks) != len(shards):
                raise ValueError(
                    f"Chunks {chunks} and shards {shards} have different lengths."
                )

            self.current_array = self.current_group.create_array(
                name=array_name,
                shape=updated_shape,
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
        if is_valid_uri(region.source):
            store_path, parent_path, array_name = decipher_zarr_uri(region.source)
            output_store_path = _add_output_key(dirpath, store_path)
        else:
            raise NotImplementedError(
                f"Invalid zarr URI: {region.source}. Currently, only predicting from "
                f"Zarr files is supported when writing Zarr tiles."
            )

        if (
            self.current_group is None
            or str(self.current_group.store_path)[: len(OUTPUT_KEY)]
            != output_store_path
        ):
            self._create_zarr(output_store_path)

        if self.current_group is None or self.current_group.name != parent_path:
            self._create_group(parent_path)

        if self.current_array is None or self.current_array.basename != array_name:
            # data_shape, chunks and shards are in SC(Z)YX order since they are reshaped
            # in the zarr image stack loader
            # If the source is not a Zarr file, then chunks and shards will be `None`.
            shape = region.data_shape
            chunks: tuple[int, ...] | None = region.additional_metadata.get(
                "chunks", None
            )
            shards: tuple[int, ...] | None = region.additional_metadata.get(
                "shards", None
            )
            self._create_array(array_name, region.axes, shape, shards, chunks)

        assert is_tile_specs(region.region_spec)  # for mypy
        tile_spec: TileSpecs = region.region_spec
        crop_coords = tile_spec["crop_coords"]
        crop_size = tile_spec["crop_size"]
        stitch_coords = tile_spec["stitch_coords"]

        # compute sample slice
        sample_idx = tile_spec["sample_idx"]

        # TODO there is duplicated code in stitch_prediction
        crop_slices: tuple[builtins.ellipsis | slice | int, ...] = (
            ...,
            *[
                slice(start, start + length)
                for start, length in zip(crop_coords, crop_size, strict=True)
            ],
        )
        stitch_slices: tuple[builtins.ellipsis | slice | int, ...] = (
            ...,
            *[
                slice(start, start + length)
                for start, length in zip(stitch_coords, crop_size, strict=True)
            ],
        )

        if self.current_array is not None:
            # region.data has shape C(Z)YX, broadcast can fail with singleton dims
            crop = region.data[crop_slices]

            if region.data.shape[0] == 1 and "C" not in region.axes:
                # singleton C dim, need to remove it before writing
                # unless it was present in the original axes
                crop = crop[0]

            if "S" in region.axes:
                if "C" in region.axes:
                    stitch_slices = (sample_idx, *stitch_slices[0:])
                else:
                    stitch_slices = (sample_idx, *stitch_slices[1:])

            self.current_array[stitch_slices] = crop
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
