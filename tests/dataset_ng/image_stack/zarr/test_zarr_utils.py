import pytest
import zarr

from careamics.dataset_ng.image_stack_loader import load_zarrs
from careamics.dataset_ng.image_stack_loader.zarr_utils import (
    collect_arrays,
    decipher_zarr_uri,
    is_valid_uri,
)


@pytest.mark.parametrize("zarr_source", ["zarr_linear", "zarr_groups", "zarr_multiple"])
def test_fixtures(request, zarr_source):
    """Test that the fixtures work as intended."""

    zarr_s = request.getfixturevalue(zarr_source)
    assert len(zarr_s) == 3

    for source in zarr_s:
        zarr_archive = zarr.open(source["zarr_file"], mode="r")
        array_path = source["array_path"]

        assert array_path in zarr_archive

        image_data = zarr_archive[array_path]
        assert isinstance(image_data, zarr.Array)


def test_collect_arrays(zarr_linear):
    """Test that collect_arrays returns the correct arrays."""
    zarr_s = zarr_linear[0]["zarr_file"]
    g = zarr.open(zarr_s, mode="r")

    assert len(collect_arrays(g)) == 3


@pytest.mark.parametrize(
    "uri, expected",
    [
        # True
        (
            "file://data/bsd68_group_in_group.zarr/group_1/group_2/bsd68_gaussian25_7",
            True,
        ),
        ("file:///absolute/path/to/zarr/store.zarr/array0", True),
        ("gs://gcs_bucket/mydata.zarr", True),
        ("az://container/store.zarr", True),
        ("https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr", True),
        # False
        ("data/local_store.zarr", False),
        ("/absolute/path/to/store.zarr", False),
        ("./relative/path/to/store.zarr", False),
        ("C:\\data\\store.zarr", False),
        ("C:/data/store.zarr", False),
        (r"\\server\share\store.zarr", False),
        ("", False),  # empty
        (None, False),  # not a string
    ],
)
def test_valid_zarr_uris(uri, expected):
    assert is_valid_uri(uri) == expected


@pytest.mark.parametrize(
    "path, zarr_path, group_path, array_name",
    [
        (
            "file:///path/to/data.zarr/groupA/array0",
            "/path/to/data.zarr",
            "groupA",
            "array0",
        ),
        (
            "file:///path/to/data.zarr/groupA/groupB/zarr_array0",
            "/path/to/data.zarr",
            "groupA/groupB",
            "zarr_array0",
        ),
        (
            "file:///another/path/data.zarr/array1",
            "/another/path/data.zarr",
            "",
            "array1",
        ),
    ],
)
def test_decipher_zarr_path(path, zarr_path, group_path, array_name):
    """Test that decipher_zarr_path correctly deciphers the zarr URI."""

    decoded_zarr_path, decoded_group_path, decoded_array_name = decipher_zarr_uri(path)

    assert decoded_zarr_path == zarr_path
    assert decoded_group_path == group_path
    assert decoded_array_name == array_name


@pytest.mark.parametrize("zarr_source", ["zarr_linear", "zarr_groups", "zarr_multiple"])
def test_create_image_stacks_uris(request, zarr_source):
    """Test that create_image_stacks creates the correct number of ZarrImageStack."""

    zarr_s = request.getfixturevalue(zarr_source)

    # create list of file URIs
    source_uris = []
    for source in zarr_s:
        zarr_file = source["zarr_file"]
        array_path = source["array_path"]

        uri = f"file://{zarr_file}/{array_path}"
        source_uris.append(uri)

    image_stacks = load_zarrs(
        source=source_uris,
        axes="YX",
    )

    assert len(image_stacks) == 3


@pytest.mark.parametrize("zarr_source", ["zarr_linear", "zarr_multiple"])
def test_create_image_stacks_paths(request, zarr_source):
    """Test that create_image_stacks creates the correct number of ZarrImageStack
    for file paths to zarr containing arrays in the root group."""

    zarr_s = request.getfixturevalue(zarr_source)

    # create set of files
    source_files = []
    for source in zarr_s:
        source_files.append(source["zarr_file"])

    source_files = list(set(source_files))

    image_stacks = load_zarrs(
        source=source_files,
        axes="YX",
    )

    assert len(image_stacks) == 3


# def test_create_image_stacks_ome(ome_zarr_url):
#     """Test that create_image_stacks can create a ZarrImageStack from an OME-Zarr
#     URL."""

#     image_stacks = create_zarr_image_stacks(
#         source=[ome_zarr_url],
#         axes="ZYX",
#         multiscale_level="0",
#     )

#     assert len(image_stacks) == 1

#     # warning raised if axes do not match
#     with pytest.warns(UserWarning):
#         _ = create_zarr_image_stacks(
#             source=[ome_zarr_url],
#             axes="CYX",  # wrong axes
#             multiscale_level="0",
#         )

#     # error raised if level does not exist
#     with pytest.raises(ValueError):
#         _ = create_zarr_image_stacks(
#             source=[ome_zarr_url],
#             axes="ZYX",
#             multiscale_level="lvl1",  # non-existing level
#         )
