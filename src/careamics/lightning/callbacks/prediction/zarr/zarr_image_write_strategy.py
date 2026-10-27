"""Strategy for writing whole-image predictions to Zarr."""

from collections import defaultdict
from pathlib import Path

from careamics.dataset.image_region_data import ImageRegionData
from careamics.dataset.image_stack.zarr_access import (
    ZarrAccess,
    ZarrArraySpec,
    ZarrPythonAccess,
)
from careamics.lightning.callbacks.prediction.image_write_utils import (
    get_complete_images,
)
from careamics.lightning.prediction import combine_samples
from careamics.utils.reshape_array import RestoredAxesTransform

from .ome_write_utils import get_ome_dimension_names
from .zarr_write_utils import (
    auto_chunks,
    create_array,
    get_zarr_destination,
)


class ZarrImageWriteStrategy:
    """Write complete image predictions to Zarr.

    Parameters
    ----------
    access : ZarrAccess or None, default=None
        Zarr backend access implementation.
    """

    def __init__(self, access: ZarrAccess | None = None) -> None:
        """Initialize the writer.

        Parameters
        ----------
        access : ZarrAccess or None, default=None
            Zarr backend access implementation.
        """
        self.access = ZarrPythonAccess() if access is None else access
        self._store_image_groups: dict[str, set[str]] = {}
        self.image_cache: dict[int, list[ImageRegionData]] = defaultdict(list)

    def set_source_base(self, source_base: Path | None) -> None:
        """
        No-op for Zarr writer.

        Parameters
        ----------
        source_base : pathlib.Path or None
            Ignored.
        """

    def write_batch(
        self,
        dirpath: Path,
        predictions: list[ImageRegionData],
    ) -> None:
        """Cache samples and write complete images.

        Parameters
        ----------
        dirpath : Path
            Output directory.
        predictions : list[ImageRegionData]
            Decollated predictions.
        """
        for prediction in predictions:
            data_idx = prediction.region_spec["data_idx"]
            self.image_cache[data_idx].append(prediction)

        for data_idx in get_complete_images(self.image_cache):
            self._write_image(dirpath, self.image_cache.pop(data_idx))

    def _write_image(
        self,
        dirpath: Path,
        regions: list[ImageRegionData],
    ) -> None:
        """Write one complete image.

        Parameters
        ----------
        dirpath : Path
            Output directory.
        regions : list[ImageRegionData]
            Samples belonging to one image.
        """
        region = regions[0]

        # recombine image
        images, _ = combine_samples(regions, restore_shape=True)
        image = images[0]  # we know there is single sample only
        assert image.data.shape is not None

        transform = RestoredAxesTransform(
            original_axes=region.axes,
            original_shape=region.original_data_shape,
            target_axes=region.target_axes,
            current_shape=region.data.shape,
            # before combine sample region is effectively a whole-image tile missing S
            current_is_tile=True,
        )

        if not transform.canonical_order:
            raise ValueError(
                f"Axes {region.axes} are not in canonical order, which is "
                f"incompatible with writing Zarr files. Please ensure axes are in the "
                f"expected order: (S)(T)(C)(Z)YX."
            )

        if image.ndim > 5:
            raise ValueError(
                f"OME-NGFF (v0.5) only supports a maximum of 5 axes (TCZYX), got "
                f"{image.ndim}."
            )

        original_chunks = region.additional_metadata.get("chunks")
        chunks = (
            transform.adjust_shape(original_chunks)
            if original_chunks is not None
            else auto_chunks(region.target_axes, image.shape)
        )
        original_shards = region.additional_metadata.get("shards")
        shards = (
            transform.adjust_shape(original_shards)
            if original_shards is not None
            else None
        )

        source_ome = region.additional_metadata.get("ome")
        node = get_zarr_destination(region, dirpath)
        create_array(
            access=self.access,
            region=region,
            node=node,
            spec=ZarrArraySpec(
                shape=image.shape,
                chunks=chunks,
                shards=shards,
                dtype=image.dtype,
                dimension_names=tuple(
                    get_ome_dimension_names(
                        region.target_axes,
                        source_ome if isinstance(source_ome, dict) else None,
                    )
                ),
            ),
            axes=region.target_axes,
            store_image_groups=self._store_image_groups,
        )
        self.access.write_array_tile(
            node,
            tuple(slice(None) for _ in image.shape),
            image,
        )
