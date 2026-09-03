import itertools
import os
from contextlib import nullcontext as does_not_raise

import pytest

from careamics.dataset.image_stack.zarr_access import (
    file_uri_to_path,
    is_valid_uri,
    path_to_file_uri,
    to_zarr_node,
)

# --- Test utilities

if os.name == "nt":
    VALID_URIS = [
        "file:///D:/data/store.zarr",
        "file:///D:/data/store.zarr/array_0",
        r"file://C:\data\store.zarr",
        r"file://C:\data\store.zarr/array_0",
        "file://server/share/store.zarr",
        "file://server/share/store.zarr/array_0",
    ]

    VALID_URI_NODES = [
        ("file:///D:/data/store.zarr", "file:///D:/data/store.zarr", ""),
        ("file:///D:/data/store.zarr/array_0", "file:///D:/data/store.zarr", "array_0"),
        (r"file://C:\data\store.zarr", r"file://C:\data\store.zarr", ""),
        (
            r"file://C:\data\store.zarr/array_0",
            r"file://C:\data\store.zarr",
            "array_0",
        ),
        ("file://server/share/store.zarr", "file://server/share/store.zarr", ""),
        (
            "file://server/share/store.zarr/array_0",
            "file://server/share/store.zarr",
            "array_0",
        ),
    ]
else:
    VALID_URIS = [
        "file:///absolute/path/to/store.zarr",
        "file:///absolute/path/to/store.zarr/array_0",
        "file:///tmp/input%20data/store.zarr/group/array_1",
        "file:///tmp/input%20data/store.zarr/my_group/",
        "file://localhost/absolute/path/to/store.zarr",
        "file:///D:/data/store.zarr",
        "file:///D:/data/store.zarr/array_0",
    ]

    VALID_URI_NODES = [
        (
            "file:///absolute/path/to/store.zarr",
            "file:///absolute/path/to/store.zarr",
            "",
        ),
        (
            "file:///absolute/path/to/store.zarr/array_0",
            "file:///absolute/path/to/store.zarr",
            "array_0",
        ),
        (
            "file:///tmp/input%20data/store.zarr/group/array_1",
            "file:///tmp/input%20data/store.zarr",
            "group/array_1",
        ),
        (
            "file:///tmp/input%20data/store.zarr/my_group/",
            "file:///tmp/input%20data/store.zarr",
            "my_group/",
        ),
        (
            "file://localhost/absolute/path/to/store.zarr",
            "file://localhost/absolute/path/to/store.zarr",
            "",
        ),
        ("file:///D:/data/store.zarr", "file:///D:/data/store.zarr", ""),
        ("file:///D:/data/store.zarr/array_0", "file:///D:/data/store.zarr", "array_0"),
    ]

INVALID_URIS = [
    "gs://bucket/store.zarr",
    "az://container/store.zarr",
    "https://example.org/store.zarr",
    "http://example.org/store.zarr",
    "zip://archive.zip::store.zarr",
    "file:relative/path/store.zarr",
    "file:store.zarr",
    "data/local_store.zarr",
    "C:/data/store.zarr",
    r"C:\data\store.zarr",
    "",
]

if os.name == "nt":
    INVALID_URIS.extend(
        [
            "file:///absolute/path/to/store.zarr",
            "file:///absolute/path/to/store.zarr/array_0",
            "file:///tmp/input%20data/store.zarr/group/array_1",
            "file:///tmp/input%20data/store.zarr/my_group/",
            "file://localhost/absolute/path/to/store.zarr",
            "/absolute/path/to/store.zarr",
            "./relative/path/to/store.zarr",
        ]
    )
else:
    INVALID_URIS.extend(
        [
            "file://data/store.zarr",
            "file://relative/path/store.zarr",
            "file://server/share/store.zarr",
            "file://server/share/store.zarr/array_0",
        ]
    )


# --- Unit tests


@pytest.mark.parametrize(
    "uri, expected",
    list(itertools.product(VALID_URIS, [True]))
    + list(itertools.product(INVALID_URIS, [False])),
)
def test_is_valid_uri(uri, expected) -> None:
    """Test validity of URIs."""
    assert is_valid_uri(uri) == expected


@pytest.mark.parametrize(
    "uri, expected",
    list(itertools.product(VALID_URIS, [does_not_raise()]))
    + list(itertools.product(INVALID_URIS, [pytest.raises(ValueError, match="URI")])),
)
def test_file_uri_to_path(uri, expected) -> None:
    """Test URI to path."""
    with expected:
        path = file_uri_to_path(uri)

        assert path.is_absolute()


@pytest.mark.parametrize("uri", VALID_URIS)
def test_file_uri_to_path_roundtrip(uri):
    """Test URI to path roundtrip."""
    path = file_uri_to_path(uri)
    uri_back = path_to_file_uri(path)
    assert is_valid_uri(uri_back)


@pytest.mark.parametrize("uri_nodes", VALID_URI_NODES)
def test_to_zarr_node(uri_nodes):
    """Test Zarr node instantiation from uri."""
    uri, store_uri, node_path = uri_nodes
    zarr_node = to_zarr_node(uri)

    assert zarr_node.store_uri == store_uri
    assert zarr_node.node_type is None
    assert zarr_node.path == node_path
