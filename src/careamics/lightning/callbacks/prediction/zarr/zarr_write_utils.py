"""Shared functionality for Zarr prediction writers."""

from collections.abc import Sequence
from pathlib import Path

from careamics.dataset.image_region_data import ImageRegionData
from careamics.dataset.image_stack.zarr_access import (
    ZarrAccess,
    ZarrArraySpec,
    ZarrNode,
    file_uri_to_path,
    is_valid_uri,
    path_to_file_uri,
    to_zarr_node,
)

from .ome_write_utils import ensure_ome_store_structure, resolve_ome_output_node

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
    name = source_path.name
    if name.endswith(".ome.zarr"):
        output_name = f"{name[:-len('.ome.zarr')]}{OUTPUT_KEY}.ome.zarr"
    elif name.endswith(".zarr"):
        output_name = f"{name[:-len('.zarr')]}{OUTPUT_KEY}.zarr"
    else:
        output_name = f"{source_path.stem}{OUTPUT_KEY}.zarr"
    return dirpath / output_name


def get_zarr_destination(region: ImageRegionData, dirpath: Path) -> ZarrNode:
    """Resolve the destination array for a prediction.

    If the source is an `"array"`, then it is placed in "prediction.zarr" under the
    path `"{data_idx}/0".
    If the source is a valid Zarr source URI, then the destination is a new zarr file
    with name `"{original_name}_output.zarr"` and the array original internal path is
    conserved.
    If the source is another file type, then it is written to an individual
    `"{original_file_name_stem}.ome.zarr"`.

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
            store_uri=path_to_file_uri(
                dirpath / f"{Path(region.source).stem}.ome.zarr"
            ),
            path="0",
            node_type="array",
        )

    raise NotImplementedError(f"Invalid source: {region.source}.")


def create_array(
    *,
    access: ZarrAccess,
    region: ImageRegionData,
    node: ZarrNode,
    spec: ZarrArraySpec,
    axes: str,
    store_image_groups: dict[str, set[str]],
) -> None:
    """Create an OME-Zarr output array and its metadata.

    Parameters
    ----------
    access : ZarrAccess
        Backend used to create the array.
    region : ImageRegionData
        Prediction metadata, including source OME metadata.
    node : ZarrNode
        Output array node.
    spec : ZarrArraySpec
        Output array specification.
    axes : str
        Output array axes.
    store_image_groups : dict[str, set[str]]
        Image-group paths already registered for each output store. This mapping is
        updated in place.
    """
    if len(spec.shape) != len(spec.chunks):
        raise ValueError(
            f"Shape {spec.shape} and chunks {spec.chunks} have different lengths."
        )
    if spec.shards is not None and len(spec.chunks) != len(spec.shards):
        raise ValueError(
            f"Chunks {spec.chunks} and shards {spec.shards} have different lengths."
        )

    image_groups = store_image_groups.setdefault(node.store_uri, set())
    if node.parent_path != "":
        image_groups.add(node.parent_path)

    source_ome = region.additional_metadata.get("ome")
    ensure_ome_store_structure(
        array_node=node,
        axes=axes,
        source_ome=source_ome if isinstance(source_ome, dict) else None,
        image_group_paths=sorted(image_groups),
    )
    access.create_array(node=node, spec=spec)
