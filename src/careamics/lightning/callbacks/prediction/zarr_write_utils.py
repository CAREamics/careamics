"""Shared functionality for Zarr prediction writers."""

from collections.abc import Sequence
from pathlib import Path

from careamics.dataset.image_region_data import ImageRegionData
from careamics.dataset.image_stack.zarr_access import (
    ZarrAccessProtocol,
    ZarrArraySpec,
    ZarrNode,
    ZarrPythonAccess,
    ensure_ome_store_structure,
    file_uri_to_path,
    is_valid_uri,
    path_to_file_uri,
    resolve_ome_output_node,
    to_ome_write_target,
    to_zarr_node,
)

OUTPUT_KEY = "_output"


def auto_chunks(axes: str, shape: Sequence[int]) -> tuple[int, ...]:
    """Generate chunk sizes for an array.

    Parameters
    ----------
    axes : str
        Array axes.
    shape : Sequence[int]
        Array shape.

    Returns
    -------
    tuple[int, ...]
        Chunk sizes in array axis order.
    """
    return tuple(
        min(128, shape[index]) if axis in ("Y", "X") else 1
        for index, axis in enumerate(axes)
    )


def _add_output_key(dirpath: Path, path: str | Path) -> Path:
    """Add the output suffix to a Zarr store name.

    Parameters
    ----------
    dirpath : Path
        Output directory.
    path : str or Path
        Source Zarr store path.

    Returns
    -------
    Path
        Output Zarr store path.
    """
    source_path = Path(path)
    return dirpath / f"{source_path.stem}{OUTPUT_KEY}.zarr"


def get_zarr_destination(region: ImageRegionData, dirpath: Path) -> ZarrNode:
    """Resolve the destination array for a prediction.

    If the source is an `"array"`, then it is placed in "prediction.zarr" under the
    path `"{data_idx}/0".
    If the source is a valid Zarr source URI, then the destination is a new zarr file
    with name `"{original_name}_output.zarr"` and the array original internal path is
    conserved.
    If the source is another file type, then it is placed in "prediction.zarr" under
    the path `"{original_file_name_tem}/0"`

    Parameters
    ----------
    region : ImageRegionData
        Prediction metadata.
    dirpath : Path
        Output directory.

    Returns
    -------
    ZarrNode
        Output array node.
    """
    if region.source == "array":
        data_idx = region.region_spec["data_idx"]
        return ZarrNode(
            store_uri=path_to_file_uri(dirpath / "prediction.zarr"),
            path=f"{data_idx}/0",
            node_type="array",
        )

    if is_valid_uri(region.source):
        source_node = to_zarr_node(region.source)
        output_store_path = _add_output_key(
            dirpath, file_uri_to_path(source_node.store_uri)
        )
        source_ome = region.additional_metadata.get("ome")
        return resolve_ome_output_node(
            source_node=source_node,
            output_store_uri=path_to_file_uri(output_store_path),
            source_ome=source_ome if isinstance(source_ome, dict) else None,
        )

    if ".zarr" not in region.source:
        return ZarrNode(
            store_uri=path_to_file_uri(dirpath / "prediction.zarr"),
            path=f"{Path(region.source).stem}/0",
            node_type="array",
        )

    raise NotImplementedError(f"Invalid source: {region.source}.")


class ZarrWriteStrategyBase:
    """Shared state and array creation for Zarr write strategies.

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
        self.access = ZarrPythonAccess() if access is None else access
        self._store_image_groups: dict[str, set[str]] = {}

    def set_source_base(self, source_base: Path | None) -> None:
        """Ignore the common source directory for Zarr output.

        Parameters
        ----------
        source_base : Path or None
            Common source directory.

        Returns
        -------
        None
            This method does nothing.
        """
        pass

    def _create_array(
        self,
        region: ImageRegionData,
        node: ZarrNode,
        spec: ZarrArraySpec,
        axes: str,
    ) -> None:
        """Create an OME-Zarr output array.

        Parameters
        ----------
        region : ImageRegionData
            Prediction metadata.
        node : ZarrNode
            Output array node.
        spec : ZarrArraySpec
            Output array specification.
        axes : str
            Output array axes.

        Returns
        -------
        None
            The array and its OME metadata are created in place.
        """
        if len(spec.shape) != len(spec.chunks):
            raise ValueError(
                f"Shape {spec.shape} and chunks {spec.chunks} have different lengths."
            )
        if spec.shards is not None and len(spec.chunks) != len(spec.shards):
            raise ValueError(
                f"Chunks {spec.chunks} and shards {spec.shards} have different "
                "lengths."
            )

        target = to_ome_write_target(node)
        image_groups = self._store_image_groups.setdefault(target.store_uri, set())
        if target.image_group_path != "":
            image_groups.add(target.image_group_path)

        source_ome = region.additional_metadata.get("ome")
        ensure_ome_store_structure(
            target=target,
            axes=axes,
            source_ome=source_ome if isinstance(source_ome, dict) else None,
            image_group_paths=sorted(image_groups),
        )
        self.access.create_array(node=target.array_node, spec=spec)
