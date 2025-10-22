import warnings
from collections.abc import Sequence
from pathlib import Path

import zarr

from careamics.dataset_ng.patch_extractor.image_stack import ZarrImageStack

FILE = "file://"

# TODO convenience function to return source path?


def is_file_uri(source: str) -> bool:
    """Check if a source string is a file URI.

    Parameters
    ----------
    source : str
        The source string to check.

    Returns
    -------
    bool
        True if the source string is a file URI, False otherwise.
    """
    return source.startswith(FILE)


def collect_arrays(zarr_group: zarr.Group) -> list[str]:
    """
    Collect all arrays in a Zarr group into a list.

    Parameters
    ----------
    zarr_group : zarr.Group
        The Zarr group to collect arrays from.

    Returns
    -------
    listof str
        A list of Zarr arrays contained in the group as relative path to the group.
    """
    arrays: list[str] = []

    _collect_arrays_recursive(zarr_group, parent_path="", arrays=arrays)

    return arrays


def _collect_arrays_recursive(
    zarr_group: zarr.Group, parent_path: str, arrays: list[str]
) -> None:
    """Recursively collect arrays in a Zarr group.

    Parameters
    ----------
    zarr_group : zarr.Group
        The Zarr group to collect arrays from.
    parent_path : str
        The parent path of the current group.
    arrays : list of str
        The list to append the array paths to.
    """
    for name in zarr_group.keys():
        current_path = f"{parent_path}/{name}" if parent_path else name
        if isinstance(zarr_group[name], zarr.Array):
            arrays.append(current_path)
        elif isinstance(zarr_group[name], zarr.Group):
            _collect_arrays_recursive(zarr_group[name], current_path, arrays)


def decipher_zarr_path(source: str) -> tuple[str, str, str]:
    """Extract the zarr store path, group path and array path from a zarr source string.

    The input string is expected to be in the format:
    \"file://path/to/zarr_store.zarr/group/path/array_name\"

    Note that the root folder of the zarr store must end with ".zarr".

    Parameters
    ----------
    source : str
        The zarr source string.

    Returns
    -------
    str
        The path to the zarr store.
    str
        The group path within the zarr store.
    str
        The array name within the group.

    Raises
    ------
    ValueError
        If the source string does not start with "file://".
    ValueError
        If the source string does not contain a ".zarr" file extension.
    """
    key = "file://"

    if source[: len(key)] != key:
        raise ValueError(f"Remote file not supported: {source}")

    if ".zarr" not in source:
        raise ValueError(f"No .zarr file extension found in source: {source}")

    _source = source[len(key) :]
    groups = _source.split("/")

    # find .zarr entry
    zarr_index = next((i for i, p in enumerate(groups) if p.endswith(".zarr")))

    path_to_zarr = groups[: zarr_index + 1]
    group_path = groups[zarr_index + 1 : -1]
    array_path = groups[-1]

    return "/".join(path_to_zarr), "/".join(group_path), array_path


def add_segmentation_key(path_to_zarr: str) -> str:
    """Add '_seg' before the '.zarr' extension in a zarr store path.

    Parameters
    ----------
    path_to_zarr : str
        The path to the zarr store.

    Returns
    -------
    str
        The modified path with '_seg' added before the '.zarr' extension.
    """
    return path_to_zarr[:-5] + "_seg.zarr"


# TODO Does this hold also for old zarr? Pydantic models from Talley might be better
def _is_ome_zarr(zarr_group: zarr.Group) -> bool:
    """Check if a Zarr group is an OME-Zarr.

    Parameters
    ----------
    zarr_group : zarr.Group
        The Zarr group to check.

    Returns
    -------
    bool
        True if the Zarr group is an OME-Zarr, False otherwise.
    """
    return "multiscales" in zarr_group.attrs


def _extract_metadata_from_ome_zarr(
    zarr_group: zarr.Group,
    multiscale_level: str = "0",
) -> tuple[str, str]:
    """Extract metadata from an OME-Zarr group.

    Parameters
    ----------
    zarr_group : zarr.Group
        The OME-Zarr group to extract metadata from.
    multiscale_level : str, default = "0"
        The multiscale level to extract metadata for.

    Returns
    -------
    str
        Validated multiscale level.
    str
        Axes string.
    """
    # extract metadata
    multiscales_metadata = zarr_group.attrs["multiscales"][0]

    # retrieve all datasets (resolution levels) in multiscale
    levels = [d["path"] for d in multiscales_metadata["datasets"]]

    if multiscale_level not in levels:
        raise ValueError(
            f"Multiscale level '{multiscale_level}' not found in OME-Zarr group. "
            f"Available levels are {levels}."
        )

    # get axes
    axes_list = [axes_data["name"] for axes_data in multiscales_metadata["axes"]]
    axes = "".join(axes_list).upper()

    return multiscale_level, axes


def create_zarr_image_stacks(
    source: Sequence[str | Path],
    axes: str,
    multiscale_level: str = "0",
) -> list[ZarrImageStack]:
    """Create a list of ZarrImageStack from a sequence of zarr file paths or URIs.

    File paths must point to a zarr store (ending with .zarr) and URIs must be in the
    format "file://path/to/zarr_store.zarr/group/path/array_name".

    If the zarr file is an OME-Zarr, the specified multiscale level will be used. Note
    that OME-Zarrs are only supported when providing a path to the zarr store, not when
    using a file URI. One can, however, provide a file URI to the specific resolution
    array within the OME-Zarr.

    Parameters
    ----------
    source : sequence of str or Path
        The source zarr file paths or URIs.
    axes : str
        The original axes of the data, must be a subset of "STCZYX".
    multiscale_level : str, default = "0"
        The multiscale level to use when loading OME-Zarr data.

    Returns
    -------
    list of ZarrImageStack
        A list of ZarrImageStack created from the sources.
    """

    image_stacks: list[ZarrImageStack] = []

    for data_source in source:
        data_str = str(data_source)

        # either a path to a zarr file or a uri "file://path/to/zarr/array_path"
        if data_str.endswith(".zarr"):
            zarr_group = zarr.open(data_str, mode="r")

            # test if ome-zarr (minimum assumption of multiscales)
            if _is_ome_zarr(zarr_group):
                data_path, metadata_axes = _extract_metadata_from_ome_zarr(
                    zarr_group, multiscale_level=multiscale_level
                )

                if metadata_axes != axes:
                    warnings.warn(
                        f"Axes mismatch for OME-Zarr at '{data_str}': "
                        f"expected {axes}, got {metadata_axes}.",
                        UserWarning,
                        stacklevel=2,
                    )

                array_paths = [data_path]
            else:
                # collect all arrays
                array_paths = collect_arrays(zarr_group)

            # instantiate image stacks
            for array_path in array_paths:
                image_stacks.append(
                    ZarrImageStack(group=zarr_group, data_path=array_path, axes=axes)
                )

        elif is_file_uri(data_str):
            # decipher the uri and open the group
            store_path, group_path, array_name = decipher_zarr_path(data_str)

            zarr_group = zarr.open(store_path, mode="r")[group_path]

            # create image stack from a single array
            image_stacks.append(
                ZarrImageStack(
                    group=zarr_group,
                    data_path=array_name,
                    axes=axes,
                )
            )

        else:
            raise ValueError(
                f"Source '{data_source}' is neither a zarr file nor a file URI."
            )

    return image_stacks
