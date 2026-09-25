"""Tile Zarr writing strategy."""

from pathlib import Path
from types import EllipsisType

from numpy import float32
from numpy.typing import NDArray

from careamics.dataset.image_region_data import ImageRegionData
from careamics.dataset.image_stack.zarr_access import (
    ZarrAccessProtocol,
    ZarrArraySpec,
    get_ome_dimension_names,
)
from careamics.dataset.patching import TileSpecs
from careamics.utils.reshape_array import RestoredAxesTransform

from .zarr_write_utils import (
    ZarrWriteStrategyBase,
    auto_chunks,
    get_zarr_destination,
)


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
            else auto_chunks(self.pred_array_axes, self.pred_array_shape)
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


class ZarrTileWriteStrategy(ZarrWriteStrategyBase):
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
        super().__init__(access)
        self._current_node_source: str | None = None

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
        output_node = get_zarr_destination(region, dirpath)

        # create a TileHandler to manage the array and tile metadata, cropping,
        # restoring and stitching
        handler = ZarrTileHandler(region)

        # create array
        if self._current_node_source != output_node.source:
            source_ome = region.additional_metadata.get("ome")
            self._create_array(
                region=region,
                node=output_node,
                spec=ZarrArraySpec(
                    shape=handler.pred_array_shape,
                    shards=handler.pred_shards,
                    chunks=handler.pred_chunks,
                    dtype=float32,
                    dimension_names=tuple(
                        get_ome_dimension_names(
                            handler.pred_array_axes,
                            source_ome if isinstance(source_ome, dict) else None,
                        )
                    ),
                ),
                axes=handler.pred_array_axes,
            )
            self._current_node_source = output_node.source

        self.access.write_array_tile(
            output_node,
            handler.stitch_slices,
            handler.restored_crop,
        )

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
