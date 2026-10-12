import itertools

import pytest

from careamics.utils import get_run_version


def _create_versioned_folders(tmp_path, folder_name):
    """Create versioned folders in the tmp_path for testing get_run_version."""
    for i in range(5):
        if i != 1:
            (tmp_path / f"{folder_name}_{i}").mkdir()


def _create_non_versioned_folders(tmp_path, folder_name):
    """Create non-versioned folders in the tmp_path for testing get_run_version."""
    for i in range(5):
        (tmp_path / f"{folder_name}_non_versioned_{i}").mkdir()


def _create_single_non_versioned_folder(tmp_path, folder_name):
    """Create a single non-versioned folder in the tmp_path for testing
    get_run_version."""
    (tmp_path / f"{folder_name}").mkdir()


@pytest.mark.parametrize(
    "create_folders, exp_version",
    # should return version 0
    list(
        itertools.product(
            [_create_non_versioned_folders, _create_single_non_versioned_folder], [0]
        )
    )
    # should return expected version 5
    + [(_create_versioned_folders, 5)],
)
def test_folder_versioning(tmp_path, create_folders, exp_version):
    """Test that get_run_version returns the correct version number based on the content
    of the root folder."""
    folder_name = "test_folder"
    create_folders(tmp_path, folder_name)

    assert get_run_version(tmp_path, folder_name) == exp_version
