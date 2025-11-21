from pathlib import Path

import numpy as np
import pytest
import zarr

from careamics.dataset_ng.image_stack_loader.zarr_utils import is_valid_uri
from careamics.lightning.dataset_ng.data_module_utils import initialize_data_pair


@pytest.fixture
def tiff_folder(tmp_path) -> Path:
    # Create a temporary folder with some dummy tiff files
    folder_path = tmp_path / "tiff_data"
    folder_path.mkdir(exist_ok=True)
    return folder_path


@pytest.fixture
def target_tiff_folder(tmp_path) -> Path:
    # Create a temporary folder with some dummy tiff files
    folder_path = tmp_path / "target_tiff_data"
    folder_path.mkdir(exist_ok=True)
    return folder_path


@pytest.fixture
def tiff_files(tiff_folder) -> list[Path]:
    # Create some dummy tiff files
    file_paths = []
    for i in range(3):
        file_path = tiff_folder / f"image_{i}.tiff"
        file_path.touch()  # create empty file
        file_paths.append(file_path)
    return file_paths


@pytest.fixture
def target_tiff_files(target_tiff_folder) -> list[Path]:
    # Create some dummy tiff files
    file_paths = []
    for i in range(3):
        file_path = target_tiff_folder / f"image_{i}.tiff"
        file_path.touch()  # create empty file
        file_paths.append(file_path)
    return file_paths


@pytest.fixture
def zarr_folder(tmp_path) -> Path:
    # Create a temporary folder with some dummy zarr files
    folder_path = tmp_path / "zarr_data"
    folder_path.mkdir(exist_ok=True)
    return folder_path


class TestInitializeDataPairArrays:

    def test_array(self):
        input_data = np.random.rand(10, 10)
        target_data = np.random.rand(10, 10)
        inp, tgt = initialize_data_pair(
            data_type="array",
            input_data=input_data,
            target_data=target_data,
        )
        assert isinstance(inp, list) and len(inp) == 1
        assert isinstance(tgt, list) and len(tgt) == 1

    def test_array_list(self):
        input_data = [np.random.rand(10, 10) for _ in range(3)]
        target_data = [np.random.rand(10, 10) for _ in range(3)]
        inp, tgt = initialize_data_pair(
            data_type="array",
            input_data=input_data,
            target_data=target_data,
        )
        assert isinstance(inp, list) and len(inp) == 3
        assert isinstance(tgt, list) and len(tgt) == 3

    def test_array_list_mismatching_lengths(self):
        input_data = [np.random.rand(10, 10) for _ in range(3)]
        target_data = [np.random.rand(10, 10) for _ in range(2)]

        with pytest.raises(ValueError):
            initialize_data_pair(
                data_type="array",
                input_data=input_data,
                target_data=target_data,
            )

    def test_dropping_non_arrays(self):
        input_data = [np.random.rand(10, 10), "not an array", np.random.rand(5, 5)]

        inp, _ = initialize_data_pair(
            data_type="array",
            input_data=input_data,
        )
        assert isinstance(inp, list) and len(inp) == 2


class TestInitializeDataPairPaths:

    def test_single_path(self, tiff_files, target_tiff_files):
        input_file = tiff_files[0]
        target_file = target_tiff_files[0]
        inp, tgt = initialize_data_pair(
            data_type="tiff",
            input_data=input_file,
            target_data=target_file,
        )
        assert isinstance(inp, list) and inp == [input_file]
        assert isinstance(tgt, list) and tgt == [target_file]

    def test_single_str(self, tiff_files, target_tiff_files):
        input_file = str(tiff_files[0])
        target_file = str(target_tiff_files[0])
        inp, tgt = initialize_data_pair(
            data_type="tiff",
            input_data=input_file,
            target_data=target_file,
        )
        assert isinstance(inp, list) and inp == [tiff_files[0]]
        assert isinstance(tgt, list) and tgt == [target_tiff_files[0]]

    def test_list_paths(self, tiff_files, target_tiff_files):
        inp, tgt = initialize_data_pair(
            data_type="tiff",
            input_data=tiff_files,
            target_data=target_tiff_files,
        )
        assert isinstance(inp, list) and inp == tiff_files
        assert isinstance(tgt, list) and tgt == target_tiff_files

    def test_mismatching_lengths_paths(self, tiff_files, target_tiff_files):
        input_files = tiff_files
        target_files = target_tiff_files[:2]  # shorter list

        with pytest.raises(ValueError):
            initialize_data_pair(
                data_type="tiff",
                input_data=input_files,
                target_data=target_files,
            )

    def test_list_strs(self, tiff_files, target_tiff_files):
        input_files = [str(p) for p in tiff_files]
        target_files = [str(p) for p in target_tiff_files]
        inp, tgt = initialize_data_pair(
            data_type="tiff",
            input_data=input_files,
            target_data=target_files,
        )
        assert isinstance(inp, list) and inp == tiff_files
        assert isinstance(tgt, list) and tgt == target_tiff_files

    def test_mismatching_lengths(self, tiff_files, target_tiff_files):
        input_files = [str(p) for p in tiff_files]
        target_files = [str(p) for p in target_tiff_files[:2]]  # shorter list

        with pytest.raises(ValueError):
            initialize_data_pair(
                data_type="tiff",
                input_data=input_files,
                target_data=target_files,
            )

    def test_non_path_elements(self, tiff_files):
        input_data = [tiff_files[0], "not a path", tiff_files[1]]

        inp, _ = initialize_data_pair(
            data_type="tiff",
            input_data=input_data,
        )
        assert isinstance(inp, list) and len(inp) == 2


class TestInitializeDataPairFolder:

    def test_folder_paths(
        self, tiff_folder, target_tiff_folder, tiff_files, target_tiff_files
    ):
        inp, tgt = initialize_data_pair(
            data_type="tiff",
            input_data=tiff_folder,
            target_data=target_tiff_folder,
        )
        assert isinstance(inp, list) and set(inp) == set(tiff_files)
        assert isinstance(tgt, list) and set(tgt) == set(target_tiff_files)


class TestIntializeDataPairWrongType:

    def test_wrong_type(self):
        input_data = np.random.rand(10, 10)
        target_data = np.random.rand(10, 10)

        with pytest.raises(NotImplementedError):
            initialize_data_pair(
                data_type="unsupported_type",
                input_data=input_data,
                target_data=target_data,
            )

        with pytest.raises(ValueError):
            initialize_data_pair(
                data_type="tiff",
                input_data=input_data,
                target_data=target_data,
            )


# TODO: currently paths to .zarr means the files need be named the same
# TODO: how to enforce that the arrays are matching between input and target?
class TestInitializeDataPairZarr:

    def test_single_zarr(self, zarr_with_arrays):
        inp, _ = initialize_data_pair(
            data_type="zarr",
            input_data=zarr_with_arrays,
        )
        assert isinstance(inp, list) and len(inp) == 1

    def test_single_zarr_uri(self, zarr_with_arrays, target_zarr_with_arrays):
        g = zarr.open(zarr_with_arrays)
        g_tar = zarr.open(target_zarr_with_arrays)

        zarr_uri = str(g.store_path)
        assert is_valid_uri(zarr_uri), "Not a valid zarr URI"

        tar_zarr_uri = str(g_tar.store_path)
        assert is_valid_uri(tar_zarr_uri), "Not a valid zarr URI for target"

        inp, targ = initialize_data_pair(
            data_type="zarr",
            input_data=zarr_uri,
            target_data=tar_zarr_uri,
        )
        assert isinstance(inp, list) and len(inp) == 1
        assert isinstance(targ, list) and len(targ) == 1

    def test_list_zarrs(self, zarr_with_arrays, zarr_with_groups):
        inp, _ = initialize_data_pair(
            data_type="zarr",
            input_data=[zarr_with_arrays, zarr_with_groups],
        )
        assert isinstance(inp, list) and len(inp) == 2

    def test_list_zarr_uris(self, zarr_with_arrays, zarr_with_groups):
        assert zarr_with_arrays.exists() and zarr_with_groups.exists()
        g1 = zarr.open(zarr_with_arrays)
        g2 = zarr.open(zarr_with_groups)

        zarr_uri1 = str(g1.store_path)
        zarr_uri2 = str(g2.store_path)

        assert is_valid_uri(zarr_uri1), "Not a valid zarr URI"
        assert is_valid_uri(zarr_uri2), "Not a valid zarr URI"

        inp, _ = initialize_data_pair(
            data_type="zarr",
            input_data=[zarr_uri1, zarr_uri2],
        )
        assert isinstance(inp, list) and len(inp) == 2

    def test_folder_zarrs(self, zarr_with_arrays, zarr_with_groups, zarr_folder):
        assert zarr_with_arrays.exists() and zarr_with_groups.exists()
        assert zarr_with_arrays.parent == zarr_with_groups.parent == zarr_folder

        inp, _ = initialize_data_pair(
            data_type="zarr",
            input_data=zarr_folder,
        )
        assert isinstance(inp, list) and len(inp) == 2

    def test_single_uri_to_array(self, zarr_with_arrays, target_zarr_with_arrays):
        g = zarr.open(zarr_with_arrays)
        g_tar = zarr.open(target_zarr_with_arrays)

        array_name = sorted(g.array_keys())[0]
        array_uri = str(g[array_name].store_path)
        assert is_valid_uri(array_uri), "Not a valid zarr URI"

        tar_array_name = sorted(g_tar.array_keys())[0]
        tar_array_uri = str(g_tar[tar_array_name].store_path)
        assert is_valid_uri(tar_array_uri), "Not a valid zarr URI for target"

        assert array_name == tar_array_name, "Array names do not match"

        inp, tar = initialize_data_pair(
            data_type="zarr",
            input_data=array_uri,
            target_data=tar_array_uri,
        )
        assert isinstance(inp, list) and len(inp) == 1
        assert isinstance(tar, list) and len(tar) == 1

    def test_uris_to_arrays(
        self, zarr_with_arrays, target_zarr_with_arrays, zarr_with_groups
    ):

        array_uris = []

        g_arrays = zarr.open(zarr_with_arrays)
        for array_name in g_arrays.array_keys():
            array_uris.append(str(g_arrays[array_name].store_path))

        g_groups = zarr.open(zarr_with_groups)
        for array_name in g_groups["groupA"].array_keys():
            array_uris.append(str(g_groups["groupA"][array_name].store_path))

        for array_name in g_groups["groupB"].array_keys():
            array_uris.append(str(g_groups["groupB"][array_name].store_path))

        assert len(array_uris) == 6

        # targets
        tar_array_uris = []

        g_arrays_tar = zarr.open(target_zarr_with_arrays)
        for array_name in g_arrays_tar.array_keys():
            tar_array_uris.append(str(g_arrays_tar[array_name].store_path))

        g_groups_tar = zarr.open(zarr_with_groups)
        for array_name in g_groups_tar["target_groupA"].array_keys():
            tar_array_uris.append(
                str(g_groups_tar["target_groupA"][array_name].store_path)
            )
        for array_name in g_groups_tar["target_groupB"].array_keys():
            tar_array_uris.append(
                str(g_groups_tar["target_groupB"][array_name].store_path)
            )
        assert len(tar_array_uris) == 6

        inp, tar = initialize_data_pair(
            data_type="zarr",
            input_data=array_uris,
            target_data=tar_array_uris,
        )
        assert isinstance(inp, list) and len(inp) == len(array_uris)
        assert isinstance(tar, list) and len(tar) == len(tar_array_uris)

    def test_mixed_uris_and_path(self, zarr_with_arrays, zarr_with_groups):
        g_groups = zarr.open(zarr_with_groups)
        group_group_uri = str(g_groups["groupA"].store_path)  # uri to group
        array_uris = [
            str(g_groups["groupB"][array_name].store_path)
            for array_name in g_groups["groupB"].array_keys()
        ]
        g_zarr_path = str(zarr_with_arrays)

        all_uris = array_uris + [g_zarr_path] + [group_group_uri]

        inp, _ = initialize_data_pair(
            data_type="zarr",
            input_data=all_uris,
        )
        assert len(inp) == len(all_uris) - 1  # it removed the path to zarr
