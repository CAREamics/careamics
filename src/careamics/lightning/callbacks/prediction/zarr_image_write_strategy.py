"""Strategy for writing whole-image predictions to Zarr."""

from collections import defaultdict
from pathlib import Path

from careamics.dataset.image_region_data import ImageRegionData
from careamics.dataset.image_stack.zarr_access import (
    ZarrAccessProtocol,
    ZarrArraySpec,
    get_ome_dimension_names,
)
from careamics.lightning.prediction import combine_samples
from careamics.utils.reshape_array import RestoredAxesTransform

from .image_write_utils import get_complete_images
from .zarr_write_utils import (
    ZarrWriteStrategyBase,
    auto_chunks,
    get_zarr_destination,
)


class ZarrImageWriteStrategy(ZarrWriteStrategyBase):
    """Write complete image predictions to Zarr.

    Parameters
    ----------
    access : ZarrAccessProtocol or None, default=None
        Zarr backend access implementation.
    """

    def __init__(self, access: ZarrAccessProtocol | None = None) -> None:
        """Initialize the writer.

        Parameters
        ----------
        access : ZarrAccessProtocol or None, default=None
            Zarr backend access implementation.
        """
        super().__init__(access)
        self.image_cache: dict[int, list[ImageRegionData]] = defaultdict(list)

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

        Returns
        -------
        None
            Complete images are written in place.
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

        Returns
        -------
        None
            The image is written in place.
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
        self._create_array(
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
        )
        self.access.write_array_tile(
            node,
            tuple(slice(None) for _ in image.shape),
            image,
        )
