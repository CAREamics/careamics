"""Memory-mapped MRC tile writing strategy."""

from pathlib import Path
from typing import Any

import mrcfile
import numpy as np
from numpy.typing import NDArray

from careamics.dataset.image_region_data import ImageRegionData
from careamics.dataset.patching import TileSpecs
from careamics.utils.reshape_array import RestoredAxesTransform

from .file_path_utils import create_write_file_path, remove_even_odd_suffix
from .write_strategy import WriteStrategy

_OUTPUT_HEADER_FIELDS = {"nx", "ny", "nz", "mode", "mx", "my", "mz"}


def _set_header_field(header: Any, name: str, value: Any) -> None:
    """Set a field from metadata returned by :class:`MrcImageStack`.

    Parameters
    ----------
    header : numpy.ndarray
        Structured MRC header.
    name : str
        Header field name.
    value : Any
        Field value to restore.
    """
    if name not in header.dtype.names:
        return

    current = header[name]
    if isinstance(value, dict) and getattr(current.dtype, "names", None):
        for child_name, child_value in value.items():
            if child_name in current.dtype.names:
                current[child_name] = child_value
        return

    header[name] = value


class MrcTileHandler:
    """Restore and locate one prediction tile in an MRC array.

    Parameters
    ----------
    region : ImageRegionData
        Prediction tile and its source-region metadata.
    """

    def __init__(self, region: ImageRegionData) -> None:
        """Initialize a tile handler.

        Parameters
        ----------
        region : ImageRegionData
            Prediction tile and its source-region metadata.
        """
        self.tile = region.data
        tile_spec: TileSpecs = region.region_spec
        self.crop_coords = tile_spec["crop_coords"]
        self.crop_size = tile_spec["crop_size"]
        self.stitch_coords = tile_spec["stitch_coords"]
        self.sample_idx = tile_spec["sample_idx"]
        self.transform = RestoredAxesTransform(
            original_axes=region.axes,
            original_shape=region.original_data_shape,
            target_axes=region.target_axes,
            current_shape=tuple(region.data.shape),
            current_is_tile=True,
        )

        if self.transform.target_axes not in ("YX", "ZYX"):
            raise ValueError(
                "MRC tile writing requires target axes 'YX' or 'ZYX'; "
                f"found '{self.transform.target_axes}'."
            )

    @property
    def crop(self) -> NDArray:
        """Return the valid, non-padding part of the tile.

        Returns
        -------
        numpy.ndarray
            Cropped prediction tile.
        """
        crop_slices = (
            ...,
            *[
                slice(start, start + length)
                for start, length in zip(self.crop_coords, self.crop_size, strict=True)
            ],
        )
        return self.tile[crop_slices]  # type: ignore[index]

    @property
    def restored_crop(self) -> NDArray:
        """Return the cropped tile in MRC axis order.

        Returns
        -------
        numpy.ndarray
            Cropped prediction in YX or ZYX order.
        """
        return self.transform.restore(self.crop)

    @property
    def stitch_slices(self) -> tuple[slice, ...]:
        """Return the destination slices in the MRC array.

        Returns
        -------
        tuple of slice
            Destination slices for the cropped tile.
        """
        slices = self.transform.stitch_slices(
            self.sample_idx, self.stitch_coords, self.crop_size
        )
        return tuple(slices)  # type: ignore[arg-type]

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Return the complete MRC array shape.

        Returns
        -------
        tuple of int
            Shape of the destination MRC array.
        """
        return self.transform.restored_array_shape


class MrcTileWriteStrategy(WriteStrategy):
    """Write prediction tiles directly to memory-mapped MRC files.

    A destination MRC is initialized once per source, then kept open as a writable
    memory map while tiles are written into their final locations.
    """

    def __init__(self) -> None:
        """Initialize an empty collection of destination memory maps."""
        self._maps: dict[Path, mrcfile.mrcfile.MrcFile] = {}
        self._shapes: dict[Path, tuple[int, ...]] = {}
        self._dtypes: dict[Path, np.dtype] = {}
        self.source_base: Path | None = None

    def set_source_base(self, source_base: Path | None) -> None:
        """Set the common source directory used for output paths.

        Parameters
        ----------
        source_base : pathlib.Path or None
            Common parent directory of source files.
        """
        self.source_base = source_base

    def _destination(self, region: ImageRegionData, dirpath: Path) -> Path:
        """Build the output path for a prediction region.

        Parameters
        ----------
        region : ImageRegionData
            Prediction tile containing the source path.
        dirpath : pathlib.Path
            Root output directory.

        Returns
        -------
        pathlib.Path
            Destination MRC path.
        """
        source = Path(region.source)
        postfix = ""
        if region.source == "array" or source.stem == "array":
            postfix = f"_{region.region_spec['data_idx']}"
            source = Path("array.mrc")
        else:
            # for cryoCARE, we want to remove "odd" or "even"
            source = remove_even_odd_suffix(source)
        return create_write_file_path(
            dirpath=dirpath,
            file_path=source,
            write_extension=".mrc",
            postfix=postfix,
            source_base=self.source_base,
        )

    def _open_map(
        self, destination: Path, region: ImageRegionData, handler: MrcTileHandler
    ) -> mrcfile.mrcfile.MrcFile:
        """Initialize and open a writable MRC memory map.

        Parameters
        ----------
        destination : pathlib.Path
            Output MRC path.
        region : ImageRegionData
            First prediction region for the destination.
        handler : MrcTileHandler
            Handler describing the destination shape.

        Returns
        -------
        mrcfile.mrcfile.MrcFile
            Writable memory-mapped MRC file.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"File already exists at '{destination}'.")

        with mrcfile.new(destination, overwrite=True) as mrc:
            mrc.set_data(np.zeros(handler.output_shape, dtype=region.data.dtype))
            for name, value in region.additional_metadata.get("mrc", {}).items():
                if name not in _OUTPUT_HEADER_FIELDS:
                    _set_header_field(mrc.header, name, value)

        mapped = mrcfile.mmap(destination, mode="r+")
        self._maps[destination] = mapped
        return mapped

    def write_tile(self, dirpath: Path, region: ImageRegionData) -> None:
        """Write one prediction tile to its destination MRC.

        Parameters
        ----------
        dirpath : pathlib.Path
            Directory where output files are written.
        region : ImageRegionData
            Prediction tile and its source-region metadata.
        """
        handler = MrcTileHandler(region)
        destination = self._destination(region, dirpath)
        mapped = self._maps.get(destination)
        if mapped is None:
            mapped = self._open_map(destination, region, handler)
            self._shapes[destination] = handler.output_shape
            self._dtypes[destination] = np.dtype(region.data.dtype)
        elif self._shapes[destination] != handler.output_shape or self._dtypes[
            destination
        ] != np.dtype(region.data.dtype):
            raise ValueError(
                f"Tiles written to '{destination}' have incompatible output shape "
                "or dtype."
            )
        mapped.data[handler.stitch_slices] = handler.restored_crop
        mapped.flush()

    def write_batch(self, dirpath: Path, predictions: list[ImageRegionData]) -> None:
        """Write all prediction tiles in a batch.

        Parameters
        ----------
        dirpath : pathlib.Path
            Directory where output files are written.
        predictions : list of ImageRegionData
            Prediction tiles to write.
        """
        for region in predictions:
            self.write_tile(dirpath, region)

    def close(self) -> None:
        """Flush and close all active MRC memory maps."""
        for mapped in self._maps.values():
            mapped.close()
        self._maps.clear()
        self._shapes.clear()
        self._dtypes.clear()

    def __del__(self) -> None:
        """Close any maps still owned by this writer."""
        self.close()
