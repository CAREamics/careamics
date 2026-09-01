"""Tile Zarr writing strategy."""

from collections.abc import Sequence
from pathlib import Path
from types import EllipsisType

import zarr
from numpy import float32
from numpy.typing import NDArray

from careamics.dataset.image_region_data import ImageRegionData
from careamics.dataset.image_stack.zarr_access import (
    ZarrAccessProtocol,
    ZarrNode,
    ZarrPythonAccess,
    file_uri_to_path,
    path_to_file_uri,
)
from careamics.dataset.image_stack_loader.zarr_utils import (
    is_valid_uri,
    to_zarr_node,
)
from careamics.dataset.patching import TileSpecs
from careamics.utils.reshape_array import RestoredAxesTransform

from .write_strategy import WriteStrategy

OUTPUT_KEY = "_output"


class ZarrTileHandler:
    """A class handling metadata creation, cropping, restoring and stitching of a tile.

    Parameters
    ----------
    region : ImageRegionData
        The image region data containing the tile information.

    Attributes
    ----------
    original_chunks : Sequence[int] | None
        Original chunks of the array, if available.

    original_shards : Sequence[int] | None
        Original shards of the array, if available.

    crop_size : Sequence[int]
        Size of the tile to crop.

    crop_coords : Sequence[int]
        Coordinates where to crop the tile.

    stitch_coords : Sequence[int]
        Coordinates in the array for stitching the tile.

    sample_idx : int
        Sample index of the tile.

    tile : NDArray
        Tile data.
    """

    def __init__(self, region: ImageRegionData) -> None:
        """Initialize the TileHandler with the given ImageRegionData.

        Parameters
        ----------
        region : ImageRegionData
            The image region data containing the tile information.
        """
        self.tile = region.data
        tile_shape = region.data.shape
        original_shape = region.original_data_shape
        original_axes = region.axes
        target_axes = region.target_axes
        self.original_chunks = region.additional_metadata.get("chunks", None)
        self.original_shards = region.additional_metadata.get("shards", None)

        tile_spec: TileSpecs = region.region_spec
        self.crop_coords = tile_spec["crop_coords"]
        self.crop_size = tile_spec["crop_size"]
        self.stitch_coords = tile_spec["stitch_coords"]
        self.sample_idx = tile_spec["sample_idx"]

        # get adjusted shapes in original orders
        # axes and shapes may differ in C channel
        self.transform = RestoredAxesTransform(
            original_axes=original_axes,
            original_shape=original_shape,
            target_axes=target_axes,
            current_shape=tile_shape,
            current_is_tile=True,
        )

        if not self.transform.canonical_order:
            raise ValueError(
                f"Axes {original_axes} are not in canonical order, which is "
                f"incompatible with writing Zarr tiles. Please ensure axes are in the "
                f"expected order: (S)(T)(C)(Z)YX."
            )

    @property
    def pred_array_shape(self) -> tuple[int, ...]:
        """Shape of the prediction array after restoring axes.

        Returns
        -------
        tuple[int, ...]
            Shape of the prediction array after restoring axes.
        """
        return self.transform.restored_array_shape

    @property
    def pred_array_axes(self) -> str:
        """Prediction array axes after restoring axes.

        Returns
        -------
        str
            Axes of the prediction array after restoring axes.
        """
        return self.transform.target_axes

    @property
    def pred_chunks(self) -> tuple[int, ...]:
        """Chunk sizes of the prediction array after restoring axes.

        Returns
        -------
        tuple[int, ...]
            Chunk sizes of the prediction array after restoring axes.
        """
        return (
            self.transform.adjust_shape(self.original_chunks)
            if self.original_chunks is not None
            else _auto_chunks(self.pred_array_axes, self.pred_array_shape)
        )

    @property
    def pred_shards(self) -> tuple[int, ...] | None:
        """Shard sizes of the prediction array after restoring axes.

        Returns
        -------
        tuple[int, ...] | None
            Shard sizes of the prediction array after restoring axes, or None if not
            available.
        """
        return (
            self.transform.adjust_shape(self.original_shards)
            if self.original_shards is not None
            else None
        )

    @property
    def crop_slices(self) -> tuple[EllipsisType | slice | int, ...]:
        """Tuple of slices for cropping the tile.

        Returns
        -------
        tuple[slice | int, ...]
            Slices for cropping the tile.
        """
        return (
            ...,
            *[
                slice(start, start + length)
                for start, length in zip(self.crop_coords, self.crop_size, strict=True)
            ],
        )

    @property
    def stitch_slices(self) -> tuple[slice | int, ...]:
        """Tuple of slices for stitching the tile into the prediction array.

        Returns
        -------
        tuple[slice | int, ...]
            Slices for stitching the tile into the prediction array.
        """
        return self.transform.stitch_slices(
            self.sample_idx,
            self.stitch_coords,
            self.crop_size,
        )

    @property
    def crop(self) -> NDArray:
        """Cropped tile array.

        Returns
        -------
        NDArray
            Cropped tile array.
        """
        return self.tile[self.crop_slices]

    @property
    def restored_crop(self) -> NDArray:
        """Cropped tile array with restored axes.

        Returns
        -------
        NDArray
            Cropped tile array with restored axes.
        """
        return self.transform.restore(self.crop)


def _auto_chunks(axes: str, shape: Sequence[int]) -> tuple[int, ...]:
    """Generate automatic chunk sizes based on axes and shape.

    X and Y dimensions will be chunked with a maximum size of 128, other dimensions
    will have chunk size 1.

    Parameters
    ----------
    axes : str
        Axes string of the data.
    shape : Sequence[int]
        Shape of the original array.

    Returns
    -------
    tuple[int, ...]
        Chunk sizes for each dimension in SC(Z)YX order, but excluding dimensions that
        are not in the axes string.
    """
    chunk_sizes = []

    for idx, ax in enumerate(axes):
        if ax in ("Y", "X"):
            dim_size = shape[idx]
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


def _get_destination(region: ImageRegionData, dirpath: Path) -> ZarrNode:
    """Generate the destination node for the zarr array based on the source.

    Parameters
    ----------
    region : ImageRegionData
        The region data containing the source information.
    dirpath : Path
        The directory path to save the output zarr.

    Returns
    -------
    ZarrNode
        Output array node.
    """
    if region.source == "array":
        data_idx = region.region_spec["data_idx"]
        return ZarrNode(
            store_uri=path_to_file_uri(dirpath.joinpath("prediction.zarr")),
            path=f"{data_idx}",
            node_type="array",
        )
    elif is_valid_uri(region.source):
        source_node = to_zarr_node(region.source)
        output_store_path = _add_output_key(
            dirpath, file_uri_to_path(source_node.store_uri)
        )
        return ZarrNode(
            store_uri=path_to_file_uri(output_store_path),
            path=source_node.path,
            node_type="array",
        )
    elif ".zarr" not in region.source:
        _source = Path(region.source)
        data_idx = region.region_spec["data_idx"]
        return ZarrNode(
            store_uri=path_to_file_uri(_source.parent.joinpath(f"{_source.stem}.zarr")),
            path=f"{data_idx}",
            node_type="array",
        )
    else:
        raise NotImplementedError(
            f"Invalid source: {region.source}. Currently, only predicting from "
            f"array, Zarr, or TIFF files is supported when writing Zarr tiles."
        )


class ZarrTileWriteStrategy(WriteStrategy):
    """Zarr tile writer strategy.

    This writer creates zarr files, groups and arrays as needed and writes tiles
    into the appropriate locations.

    Parameters
    ----------
    access : ZarrAccessProtocol or None, default=None
        Zarr backend access implementation.
    """

    def __init__(self, access: ZarrAccessProtocol | None = None) -> None:
        """Constructor.

        Parameters
        ----------
        access : ZarrAccessProtocol or None, default=None
            Zarr backend access implementation.
        """
        self.access = ZarrPythonAccess() if access is None else access
        self.current_array: zarr.Array | None = None
        self._current_node_source: str | None = None

    def set_source_base(self, source_base: Path | None) -> None:
        """
        No-op.

        Parameters
        ----------
        source_base : pathlib.Path or None
            Ignored.

        Returns
        -------
        None
            This method does nothing for Zarr outputs.
        """
        pass

    def _create_array(
        self,
        node: ZarrNode,
        shape: Sequence[int],
        shards: tuple[int, ...] | None,
        chunks: tuple[int, ...],
    ) -> None:
        """Create a new array in an existing zarr store or group.

        Parameters
        ----------
        node : ZarrNode
            Output array node.
        shape : Sequence[int]
            Shape of the array.
        shards : tuple[int, ...] or None
            Shard size for the array.
        chunks : tuple[int, ...]
            Chunk size for the array.

        Returns
        -------
        None
            The current output array cache is updated in place.
        """
        if len(shape) != len(chunks):
            raise ValueError(
                f"Shape {shape} and chunks {chunks} have different lengths."
            )

        if shards is not None and len(chunks) != len(shards):
            raise ValueError(
                f"Chunks {chunks} and shards {shards} have different lengths."
            )

        self.current_array = self.access.create_array(
            node=node,
            shape=shape,
            shards=shards,
            chunks=chunks,
            dtype=float32,
        )
        self._current_node_source = node.source

    def write_tile(self, dirpath: Path, region: ImageRegionData) -> None:
        """Write cropped tile to zarr array.

        Parameters
        ----------
        dirpath : Path
            Path to directory to save predictions to.
        region : ImageRegionData
            Image region data containing tile information.

        Returns
        -------
        None
            The tile is written in place to the destination array.
        """
        output_node = _get_destination(region, dirpath)

        # create a TileHandler to manage the array and tile metadata, cropping,
        # restoring and stitching
        handler = ZarrTileHandler(region)

        # create array
        if (
            self.current_array is None
            or self._current_node_source != output_node.source
        ):
            self._create_array(
                output_node,
                handler.pred_array_shape,
                handler.pred_shards,
                handler.pred_chunks,
            )

        if self.current_array is None:
            raise RuntimeError("Zarr array not initialized.")

        self.access.write_array_tile(
            output_node,
            handler.stitch_slices,
            handler.restored_crop,
        )

    def write_batch(
        self,
        dirpath: Path,
        predictions: list[ImageRegionData],
    ) -> None:
        """Write all tiles to a Zarr file.

        Parameters
        ----------
        dirpath : Path
            Path to directory to save predictions to.
        predictions : list[ImageRegionData]
            Decollated predictions.

        Returns
        -------
        None
            All tiles are written in place.
        """
        for region in predictions:
            self.write_tile(dirpath, region)
