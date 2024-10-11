"""CZI dataset  utilities."""

from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Callable, NamedTuple, Optional, Union

import numpy as np
from pylibCZIrw import czi as pyczi
from torch.utils.data import get_worker_info

from careamics.file_io.read import read_czi_roi


class Rectangle(NamedTuple):
    """Rectangle object."""

    x: int
    y: int
    w: int
    h: int


def iterate_file_names(
    data_files: list[Path],
    target_files: Optional[list[Path]] = None,
) -> Generator[tuple[Path, Optional[Path]], None, None]:
    """Iterate over the filenames.

    This function yields the filenames from the provided list of data files.

    Parameters
    ----------
    data_files : list of pathlib.Path
        List of data files.
    target_files : Optional[list[Path]], optional
        List of target data files, by default None.

    Yields
    ------
    Generator[tuple[str, Optional[str]], None, None]
        The filename as a string

    """
    # iterate over the files
    for i, filename in enumerate(data_files):
        if target_files is not None:
            if filename.name != target_files[i].name:
                raise ValueError(
                    f"File {filename} does not match target file " f"{target_files[i]}."
                )

            yield filename, target_files[i]
        else:
            yield filename, None


def get_czi_shape(
    czi_reader: pyczi.CziReader,
) -> list[int]:
    """Return the dimesion of the CZI object (Z,X,Y).

       We explicitly check CZTXY dimensions
       Rest of the dimensions are considered as samples (planes) to extract the data
       The depth of the the data is decided by Z or T which ever is highest

    Parameters
    ----------
    czi_reader : pyczi.CziReader
       The czi reader object.

    Returns
    -------
    List[int]
        The (Z),X,Y dimension of the czi file.
    """
    total_dim = czi_reader.total_bounding_box
    if "Z" in total_dim and "T" in total_dim:
        Z_shape = total_dim["Z"][1] - total_dim["Z"][0]
        T_shape = total_dim["T"][1] - total_dim["T"][0]

        return [
            Z_shape if Z_shape > T_shape else T_shape,
            total_dim["X"][1] - total_dim["X"][0],
            total_dim["Y"][1] - total_dim["Y"][0],
        ]

    elif "Z" in total_dim and "T" not in total_dim:
        Z_shape = total_dim["Z"][1] - total_dim["Z"][0]
        return [
            Z_shape,
            total_dim["X"][1] - total_dim["X"][0],
            total_dim["Y"][1] - total_dim["Y"][0],
        ]

    elif "Z" not in total_dim and "T" in total_dim:
        T_shape = total_dim["T"][1] - total_dim["T"][0]
        return [
            T_shape,
            total_dim["X"][1] - total_dim["X"][0],
            total_dim["Y"][1] - total_dim["Y"][0],
        ]
    return [
        total_dim["X"][1] - total_dim["X"][0],
        total_dim["Y"][1] - total_dim["Y"][0],
    ]


def check_for_scenes(czi_reader: pyczi.CziReader) -> dict[int, Rectangle]:
    """Check for existence of scenes in a czi file.

    Parameters
    ----------
    czi_reader : pyczi.CziReader
        CZI reader object of the file.

    Returns
    -------
    dict[int,Rectangle]
        The scene information in the czi file.
    """
    total_dim = czi_reader.total_bounding_box
    scene_info = czi_reader.scenes_bounding_rectangle
    if len(scene_info) >= 1:
        return scene_info
    else:
        return {
            0: Rectangle(
                x=total_dim["X"][0],
                y=total_dim["Y"][0],
                w=total_dim["X"][1] - total_dim["X"][0],
                h=total_dim["Y"][1] - total_dim["Y"][0],
            )
        }


def get_number_patches(
    czi_reader: pyczi.CziReader,
    patch_size: Union[list[int], tuple[int, ...]],
    scene_info: dict[int, Rectangle],
    num_workers=int,
) -> int:
    """Calculate the number of patches that can be extracted from a czi file.

    Parameters
    ----------
    czi_reader : pyczi.CziReader
       The czi reader object of the file.
    patch_size : Union[list[int], tuple[int, ...]]
        Patch_size ((Z),X,Y).
    scene_info : dict[int, Rectangle]
        The scene information in the czi file.
    num_workers : int
        Number of workers to split the patches.

    Returns
    -------
    int
        Number of patches that can be extracted.
    """
    n_patches = 0
    for _, scene in scene_info.items():
        data_shape = list(get_czi_shape(czi_reader))
        data_shape[-2:] = scene.w, scene.h
        if len(patch_size) == 2 and len(data_shape) == 3:
            # treat the "Z/Taxis" as 2D samples
            n_patches += (np.prod(data_shape[1:]) / np.prod(patch_size)).astype(int)
        else:
            n_patches += (np.prod(data_shape) / np.prod(patch_size)).astype(int)
    return np.ceil(n_patches / num_workers).astype(int)


def plane_iterator(
    czi_reader: pyczi.CziReader, patch_size: Union[list[int], tuple[int, ...]]
) -> tuple[Iterator[tuple[int, ...]], list[str], Union[list[int], tuple[int, ...]]]:
    """Update patch_size (C,Z,X,Y) and get iterator over the planes of the CZI file.

    Parameters
    ----------
    czi_reader : pyczi.CziReader
        The czi reader object of the file.
    patch_size : Union[List[int], tuple[int, ...]]
        Patch_size (X,Y) or (Z,X,Y).

    Returns
    -------
    tuple[Iterator[tuple[int, ...]], List[str], Union[List[int], tuple[int, ...]]]
        Iterator over the czi dimensions except C,(Z/T) depending of the patch_size.
        The list of dimensions of the czi file.
        The updated patch_size (C,Z,X,Y).
    """
    total_dim = czi_reader.total_bounding_box
    # Remove C dimension from the plane_iterator
    if "C" in total_dim:
        patch_size = (total_dim["C"][1] - total_dim["C"][0], *patch_size)
        total_dim.pop("C")
    else:
        # update the patch_size to C(Z)XY
        patch_size = (1, *patch_size)

    # Remove Z/T dimension from the plane_iterator if 3D model
    if len(patch_size) == 4:
        if "Z" in total_dim and "T" in total_dim:
            Z_shape = total_dim["Z"][1] - total_dim["Z"][0]
            T_shape = total_dim["T"][1] - total_dim["T"][0]
            total_dim.pop("Z") if Z_shape > T_shape else total_dim.pop("T")
        elif "Z" in total_dim and "T" not in total_dim:
            total_dim.pop("Z")
        else:
            total_dim.pop("T")
    else:
        # update the patchsize to CZXY (Z=1 for 2D)
        patch_size = (patch_size[0], 1, *patch_size[1:])

    arrays = [
        np.arange(total_dim[key][1] - total_dim[key][0]) for key in total_dim.keys()
    ]
    # updated key_list after removing C and/or Z dimensions
    key_list = list(total_dim.keys())
    shape = tuple(array.shape[0] for array in arrays)
    plane_iter = iter(np.ndindex(shape[:-2]))
    return plane_iter, key_list, patch_size


def extract_random_czi_patches(
    sample_file_path: Path,
    patch_size: Union[list[int], tuple[int, ...]],
    target_file_path: Optional[Path] = None,
    read_source_func: Callable = read_czi_roi,
) -> Generator[tuple[np.ndarray, Optional[np.ndarray]], None, None]:
    """Generate random patches from CZI files.

    The method calculates how many patches the image can be divided into and then
    extracts an equal number of random patches.

    It returns a generator that yields the following:

    - patch: np.ndarray, dimension C(Z)YX.
    - target_patch: np.ndarray, dimension C(Z)YX, if the target is present, None
        otherwise.

    Parameters
    ----------
    sample_file_path : Path
        The czi filepath.
    patch_size : tuple[int]
        Patch sizes in each dimension.
    target_file_path : Optional[Path], optional
        Target filepath, by default None.
    read_source_func : Callable
        CZI reader function.

    Yields
    ------
    Generator[np.ndarray, None, None]
        Generator of patches.
    """
    worker_info = get_worker_info()
    num_workers = worker_info.num_workers if worker_info is not None else 1
    rng = np.random.default_rng(seed=None)
    with pyczi.open_czi(str(sample_file_path)) as sample_reader:

        scene_info = check_for_scenes(sample_reader)
        data_shape = get_czi_shape(sample_reader)

        # if Z not in the data and patch_size is 3D
        if len(patch_size) == 3 and len(data_shape) < 3:
            raise ValueError(
                "Dimension in the data do not match for a 3D model."
                " Data needs to have Z/T dimensions for 3D models."
            )
        elif (
            len(patch_size) == 3
            and len(data_shape) == 3
            and patch_size[0] > data_shape[0]
        ):
            raise ValueError(
                "Do not have enough data in the Z/T dimension to create 3D patches."
                "Reduce the depth of the patch_size."
            )
        else:
            n_patches = get_number_patches(
                sample_reader, patch_size, scene_info, num_workers
            )

        data_shape = [1, *data_shape] if len(data_shape) == 2 else data_shape

        if target_file_path is not None:
            with pyczi.open_czi(str(target_file_path)) as target_reader:
                plane_indices, key_list, updated_patch_size = plane_iterator(
                    sample_reader, patch_size
                )
                list_plane_indices = list(plane_indices)
                for _ in range(n_patches * len(list_plane_indices)):
                    plane_key = rng.integers(0, len(list_plane_indices))
                    scene_key = rng.integers(0, len(scene_info))
                    scene_value = scene_info[scene_key]
                    plane_indice = list_plane_indices[plane_key]
                    # change the data chape according to the scene
                    data_shape[-2:] = scene_value.w, scene_value.h
                    # (Z,X,Y) coordinates
                    coords: list[int] = [
                        rng.integers(
                            0,
                            data_shape[i] - updated_patch_size[1:][i],
                            endpoint=True,
                        )
                        for i in range(len(updated_patch_size[1:]))
                    ]
                    coords[1] = scene_value.x + coords[1]
                    coords[2] = scene_value.y + coords[2]
                    plane = {
                        key: plane_indice[i] for i, key in enumerate(key_list[:-2])
                    }
                    patch = read_source_func(
                        sample_file_path,
                        sample_reader,
                        updated_patch_size,
                        coords,
                        plane,
                        scene_key,
                    )
                    target_patch = read_source_func(
                        target_file_path,
                        target_reader,
                        updated_patch_size,
                        coords,
                        plane,
                        scene_key,
                    )
                    yield patch, target_patch
        else:
            plane_indices, key_list, updated_patch_size = plane_iterator(
                sample_reader, patch_size
            )
            list_plane_indices = list(plane_indices)
            for _ in range(n_patches * len(list_plane_indices)):
                plane_key = rng.integers(0, len(list_plane_indices))
                scene_key = rng.integers(0, len(scene_info))
                scene_value = scene_info[scene_key]
                plane_indice = list_plane_indices[plane_key]
                # change the data chape according to the scene
                data_shape[-2:] = scene_value.w, scene_value.h
                coords = [
                    rng.integers(
                        0, data_shape[i] - updated_patch_size[1:][i], endpoint=True
                    )
                    for i in range(len(updated_patch_size[1:]))
                ]
                coords[1] = scene_value.x + coords[1]
                coords[2] = scene_value.y + coords[2]
                plane = {key: plane_indice[i] for i, key in enumerate(key_list[:-2])}
                patch = read_source_func(
                    sample_file_path,
                    sample_reader,
                    updated_patch_size,
                    coords,
                    plane,
                    scene_key,
                )
                yield patch, None


def get_tiles(data_shape: list[int]) -> list[list[int]]:
    """Non-overlapping patch locations along with patch shape from a given data_shape.

    Parameters
    ----------
    data_shape : list[int]
        Shape of the data (Z,X,Y).

    Returns
    -------
    List[int]
        A list of coordinates along with the size of the patch.
    """
    tile = []
    for x in range(0, data_shape[1], 2048):
        for y in range(0, data_shape[2], 2048):
            tile.append(
                [x, y, min(2048, data_shape[1] - x), min(2048, data_shape[2] - y)]
            )
    return tile


def extract_sequential_czi_patches(
    read_source_func: Callable,
    czi_file_path: Path,
) -> Generator[tuple[np.ndarray]]:
    """Generate sequntial patches from CZI file covering the entre data.

    Used to calculate data statistics without loading the entire data in memory

    It returns a generator that yields the following:

    - patch: np.ndarray, dimension C(Z)YX.

    Parameters
    ----------
    read_source_func : Callable
        CZI reader function.
    czi_file_path : Path
        CZI file path.

    Yields
    ------
    Generator[np.ndarray, None, None]
        Generator of patches.
    """
    with pyczi.open_czi(str(czi_file_path)) as czi_reader:
        scene_info = check_for_scenes(czi_reader)
        for scene_key, scene in scene_info.items():
            data_shape = get_czi_shape(czi_reader)
            data_shape[-2:] = scene.w, scene.h
            # data shape= (Z)XY shape; for 2D make Z=1
            data_shape = [1, *data_shape] if len(data_shape) == 2 else data_shape
            tiles = get_tiles(data_shape)

            for tile in tiles:
                tile_size = [tile[2], tile[3]]
                plane_indices, key_list, updated_patch_size = plane_iterator(
                    czi_reader, tile_size
                )
                for plane_indice in plane_indices:
                    # get crop coordinates
                    crop_coords = [tile[0] + scene.x, tile[1] + scene.y]
                    plane = {
                        key: plane_indice[i] for i, key in enumerate(key_list[:-2])
                    }
                    patch = read_source_func(
                        czi_file_path,
                        czi_reader,
                        updated_patch_size,
                        (0, *crop_coords),
                        plane,
                        scene_key,
                    )
                    yield patch
