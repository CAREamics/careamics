"""Strategy for writing whole-image predictions to Zarr."""

from collections import defaultdict
from pathlib import Path

from careamics.dataset.image_region_data import ImageRegionData
from careamics.dataset.image_stack.zarr_access import (
    DefaultZarrAccess,
    ZarrAccess,
    ZarrArraySpec,
    ZarrNode,
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
        self.access = DefaultZarrAccess() if access is None else access
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

        # check if we need to split axes in multiple arrays
        has_both_ST = "S" in region.target_axes and "T" in region.target_axes
        if image.ndim > 5 and not has_both_ST:
            raise ValueError(
                f"OME-NGFF only supports a maximum of 5 axes (TCZYX), got "
                f"{image.ndim} axes."
            )

        # adjust output axes and shape
        output_axes = (
            region.target_axes.replace("S", "") if has_both_ST else region.target_axes
        )
        output_shape = image.shape[1:] if has_both_ST else image.shape

        # compute chunks
        original_chunks = region.additional_metadata.get("chunks")
        if original_chunks is not None:
            transformed_chunks = transform.adjust_shape(original_chunks)
            chunks = transformed_chunks[1:] if has_both_ST else transformed_chunks
        else:
            chunks = auto_chunks(output_axes, output_shape)

        # compute shards
        original_shards = region.additional_metadata.get("shards")
        if original_shards is not None:
            transformed_shards = transform.adjust_shape(original_shards)
            shards = transformed_shards[1:] if has_both_ST else transformed_shards
        else:
            shards = None

        # write each sample (ST dimensions) or image
        source_ome = region.additional_metadata.get("ome")
        base_node = get_zarr_destination(region, dirpath)
        samples = list(image) if has_both_ST else [image]
        for sample_idx, sample in enumerate(samples):
            # choose destination node:
            #  - array in collection if both S and T are present
            #  - base array in OME-NGFF Image otherwise
            if has_both_ST:
                node = ZarrNode(
                    store_uri=base_node.store_uri,
                    path=f"{sample_idx}/0",
                    node_type="array",
                )
            else:
                node = base_node

            create_array(
                access=self.access,
                region=region,
                node=node,
                spec=ZarrArraySpec(
                    shape=sample.shape,
                    chunks=chunks,
                    shards=shards,
                    dtype=sample.dtype,
                    dimension_names=tuple(
                        get_ome_dimension_names(
                            output_axes,
                            source_ome if isinstance(source_ome, dict) else None,
                        )
                    ),
                ),
                axes=output_axes,
                store_image_groups=self._store_image_groups,
            )
            self.access.write_array_tile(
                node,
                tuple(slice(None) for _ in sample.shape),
                sample,
            )
