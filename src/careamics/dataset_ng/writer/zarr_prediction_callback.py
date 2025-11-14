from collections.abc import Sequence
from pathlib import Path

import zarr
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import BasePredictionWriter

from careamics.dataset_ng.dataset import ImageRegionData
from careamics.dataset_ng.patch_extractor.image_stack.image_utils.zarr_utils import (
    decipher_zarr_path,
)
from careamics.dataset_ng.patching_strategies import TileSpecs


def _add_output_key(path: str | Path) -> Path:
    """Add `output` key to zarr name.

    Parameters
    ----------
    path : str | Path
        Original zarr path.

    Returns
    -------
    str
        Zarr path with `output` key added.
    """
    p = Path(path)
    new_name = p.stem + "_output" + p.suffix
    return p.with_name(new_name)


# TODO output shape may not be input shape, check what the image region info holds!!
# TODO dealing with non tiled
class ZarrTileWriter:
    """Zarr tile writer class.

    This writer creates zarr files, groups and arrays as needed and writes tiles
    into the appropriate locations.
    """

    def __init__(self) -> None:
        """Constructor."""
        self.current_store = None
        self.current_group = None
        self.current_array = None

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
            self.current_store = zarr.open(store)

        print(f"Store: {Path(store).absolute()}")

    def _create_group(self, group_path: str | Path) -> None:
        """Create a new group in an existing zarr storage.

        Parameters
        ----------
        group_path : str | Path
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
            self.current_group = self.current_store[group_path]

    def _create_array(
        self,
        array_name: str,
        shape: Sequence[int],
        chunks: Sequence[int],
        dtype: str,
    ) -> None:
        """Create a new array in an existing zarr group.

        Parameters
        ----------
        array_name : str
            Name of the array within the zarr group.
        shape : Sequence[int]
            Shape of the array.
        chunks : Sequence[int]
            Chunk size for the array.
        dtype : str
            Data type of the array.

        Raises
        ------
        RuntimeError
            If the zarr group has not been initialized.
        """
        if self.current_group is None:
            raise RuntimeError("Zarr group not initialized.")

        if array_name not in self.current_group:

            shape = [i for i in shape if i != 1]

            self.current_array = self.current_group.create_array(
                name=array_name, shape=shape, chunks=chunks, dtype=dtype
            )
        else:
            self.current_array = self.current_group[array_name]

    def write_tile(self, region: ImageRegionData) -> None:
        """Write cropped tile to zarr array.

        Parameters
        ----------
        region : ImageRegionData
            Image region data containing tile information.
        """

        store_path, parent_path, array_name = decipher_zarr_path(region.source)
        output_store_path = _add_output_key(store_path)

        if (
            self.current_group is None
            or
            # TODO remove hard coded 7
            str(self.current_group.store_path)[7:] != output_store_path
        ):
            self._create_zarr(output_store_path)

        if self.current_group is None or self.current_group.name != parent_path:
            self._create_group(parent_path)

        if self.current_array is None or self.current_array.basename != array_name:
            shape = region.data_shape
            chunks = region.chunks
            region_dtype = region.dtype
            self._create_array(array_name, shape, chunks, region_dtype)

        # TODO region_spec is PatchSpecs, TileSpecs inherit from PatchSpecs
        tile_spec: TileSpecs = region.region_spec  # type: ignore[assignment]
        crop_coords = tile_spec["crop_coords"]
        crop_size = tile_spec["crop_size"]
        stitch_coords = tile_spec["stitch_coords"]

        slices = tuple(
            slice(start, start + length)
            for start, length in zip(crop_coords, crop_size, strict=True)
        )
        stitch_slices = tuple(
            slice(start, start + length)
            for start, length in zip(stitch_coords, crop_size, strict=True)
        )

        if self.current_array is not None:
            self.current_array[stitch_slices] = region.data.squeeze()[slices]
        else:
            raise RuntimeError("Zarr array not initialized.")


# TODO generalizes prediction writer as done previously
# TODO switch off if not tiled
class ZarrPredictionWriterCallback(BasePredictionWriter):

    def __init__(
        self,
    ) -> None:
        """
        A PyTorch Lightning callback to save predictions.
        """
        super().__init__(write_interval="batch")

        # Toggle for CAREamist to switch off saving if desired
        self.writing_predictions: bool = True
        self.writer = ZarrTileWriter()

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: list[ImageRegionData],
        batch_indices: Sequence[int] | None,
        batch: list[ImageRegionData],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        assert prediction is not None
        for pred_region in prediction:
            self.writer.write_tile(pred_region)
