"""Utility functions for file and paths solver."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from numpy import ndarray
from numpy.typing import NDArray

from careamics.config.support import SupportedData
from careamics.dataset.dataset_utils import list_files, validate_source_target_files
from careamics.dataset_ng.image_stack_loader.zarr_utils import is_valid_uri

ItemType = Path | str | NDArray[Any]
"""Type of input items passed to the dataset."""

InputType = ItemType | Sequence[ItemType] | None
"""Type of input data passed to the dataset."""


def list_files_in_directory(
    data_type: Literal["tiff", "zarr", "czi", "custom"],
    input_data,
    target_data=None,
    extension_filter: str = "",
) -> tuple[list[Path], list[Path] | None]:
    """List files from input and target directories.

    Parameters
    ----------
    data_type : Literal["tiff", "zarr", "czi", "custom"]
        The type of data to validate.
    input_data : InputType
        Input data, can be a path to a folder, a list of paths, or a numpy array.
    target_data : Optional[InputType]
        Target data, can be None, a path to a folder, a list of paths, or a numpy
        array.
    extension_filter : str, default=""
        File extension filter to apply when listing files.

    Returns
    -------
    list[Path]
        A list of file paths for input data.
    list[Path] | None
        A list of file paths for target data, or None if target_data is None.
    """
    input_data = Path(input_data)

    # list_files will return a list with a single element if the path is a file with
    # the correct extension
    input_files = list_files(input_data, data_type, extension_filter)
    if target_data is None:
        return input_files, None
    else:
        target_data = Path(target_data)
        target_files = list_files(target_data, data_type, extension_filter)
        validate_source_target_files(input_files, target_files)
        return input_files, target_files


def convert_paths_to_pathlib(
    input_data: Sequence[str | Path],
    target_data: Sequence[str | Path] | None = None,
) -> tuple[list[Path], list[Path] | None]:
    """Create a list of file paths from the input and target data.

    Parameters
    ----------
    input_data : Sequence[str | Path]
        Input data, can be a path to a folder, or a list of paths.
    target_data : Sequence[str | Path] | None
        Target data, can be None, a path to a folder, or a list of paths.

    Returns
    -------
    list[Path]
        A list of file paths for input data.
    list[Path] | None
        A list of file paths for target data, or None if target_data is None.
    """
    input_files = [Path(item) if isinstance(item, str) else item for item in input_data]
    if target_data is None:
        return input_files, None
    else:
        target_files = [
            Path(item) if isinstance(item, str) else item for item in target_data
        ]
        validate_source_target_files(input_files, target_files)
        return input_files, target_files


def validate_input_target_type_consistency(
    input_data: InputType,
    target_data: InputType | None,
) -> None:
    """Validate if the input and target data types are consistent.

    Parameters
    ----------
    input_data : InputType
        Input data, can be a path to a folder, a list of paths, or a numpy array.
    target_data : Optional[InputType]
        Target data, can be None, a path to a folder, a list of paths, or a numpy
        array.

    Raises
    ------
    ValueError
        If the input and target data types are not consistent.
    """
    if input_data is not None and target_data is not None:
        if not isinstance(input_data, type(target_data)):
            raise ValueError(
                f"Inputs for input and target must be of the same type or None. "
                f"Got {type(input_data)} and {type(target_data)}."
            )
    if isinstance(input_data, list) and isinstance(target_data, list):
        if len(input_data) != len(target_data):
            raise ValueError(
                f"Inputs and targets must have the same length. "
                f"Got {len(input_data)} and {len(target_data)}."
            )
        if not isinstance(input_data[0], type(target_data[0])):
            raise ValueError(
                f"Inputs and targets must have the same type. "
                f"Got {type(input_data[0])} and {type(target_data[0])}."
            )


def validate_array_input(
    input_data: NDArray | list[NDArray],
    target_data: NDArray | list[NDArray] | None,
) -> tuple[list[NDArray], list[NDArray] | None]:
    """Validate if the input data is a numpy array.

    Parameters
    ----------
    input_data : InputType
        Input data, can be a path to a folder, a list of paths, or a numpy array.
    target_data : Optional[InputType]
        Target data, can be None, a path to a folder, a list of paths, or a numpy
        array.

    Returns
    -------
    list[numpy.ndarray]
        Validated input data.
    list[numpy.ndarray] | None
        Validated target data, None if the target data is None.

    Raises
    ------
    ValueError
        If the input data is not a numpy array or a list of numpy arrays.
    """
    if isinstance(input_data, ndarray):
        input_list = [input_data]

        if target_data is not None and not isinstance(target_data, ndarray):
            raise ValueError(
                f"Wrong target type. Expected numpy.ndarray, got {type(target_data)}. "
                f"Check the data_type parameter or your inputs."
            )
        target_list = [target_data] if target_data is not None else None
        return input_list, target_list
    elif isinstance(input_data, list):
        # TODO warn if wrong types inside list
        input_list = [array for array in input_data if isinstance(array, ndarray)]

        if target_data is None:
            target_list = None
        else:
            assert isinstance(target_data, list)
            target_list = [array for array in target_data if isinstance(array, ndarray)]
        return input_list, target_list
    else:
        raise ValueError(
            f"Wrong input type. Expected numpy.ndarray or list of numpy.ndarray, got "
            f"{type(input_data)}. Check the data_type parameter or your inputs."
        )


def validate_path_input(
    data_type: Literal["tiff", "zarr", "czi", "custom"],
    input_data: str | Path | list[str | Path],
    target_data: str | Path | list[str | Path] | None,
    extension_filter: str = "",
) -> tuple[list[Path], list[Path] | None]:
    """Validate if the input data is a path or a list of paths.

    Parameters
    ----------
    data_type : Literal["tiff", "zarr", "czi", "custom"]
        The type of data to validate.
    input_data : str | Path | list[str | Path]
        Input data, can be a path to a folder, a list of paths, or a numpy array.
    target_data : str | Path | list[str | Path] | None
        Target data, can be None, a path to a folder, a list of paths, or a numpy
        array.
    extension_filter : str, default=""
        File extension filter to apply when listing files.

    Returns
    -------
    list[Path]
        A list of file paths for input data.
    list[Path] | None
        A list of file paths for target data, or None if target_data is None.

    Raises
    ------
    ValueError
        If the input data is not a path or a list of paths.
    """
    if isinstance(input_data, (str, Path)):
        input_list, target_list = list_files_in_directory(
            data_type, input_data, target_data, extension_filter
        )
        return input_list, target_list
    elif isinstance(input_data, list):
        # TODO warn if wrong types inside list
        input_list = [
            Path(item)
            for item in input_data
            if isinstance(item, (str, Path)) and Path(item).exists()
        ]

        target_list = None
        if target_data is not None:
            assert isinstance(target_data, list)
            target_list = [
                Path(item)
                for item in target_data
                if isinstance(item, (str, Path)) and Path(item).exists()
            ]  # consistency with input is enforced by convert_paths_to_pathlib

        return convert_paths_to_pathlib(input_list, target_list)
    else:
        raise ValueError(
            f"Wrong input type, expected str or Path or list[str | Path], got "
            f"{type(input_data)}. Check the data_type parameter or your inputs."
        )


def validate_zarr_input(
    input_data: str | Path | list[str | Path],
    target_data: str | Path | list[str | Path] | None,
) -> tuple[list[str] | list[Path], list[str] | list[Path] | None]:
    """Validate if the input data corresponds a zarr input.

    Parameters
    ----------
    input_data : str | Path | list[str | Path]
        Input data, can be a path to a folder, to zarr file, a URI pointing to a zarr
        dataset, or a list.
    target_data : str | Path | list[str | Path] | None
        Target data, can be None.

    Returns
    -------
    list[str] or list[Path]
        A list of zarr URIs or path for input data.
    list[str] or list[Path] | None
        A list of zarr URIs or paths for target data, or None if target_data is None.

    Raises
    ------
    ValueError
        If the input and target data types are not consistent.
    ValueError
        If the input data is not a zarr URI or path, or a list of zarr URIs or paths.
    """
    # validate_input_target_type_consistency is called beforehand, ensuring the types
    # of input and target are the same
    if isinstance(input_data, (str, Path)):
        if Path(input_data).exists():
            # either a path to a folder or a zarr file
            # path to a folder will trigger collection of all zarr files in that folder
            assert target_data is None or isinstance(target_data, (str, Path))
            if target_data is not None and not Path(target_data).exists():
                raise ValueError(
                    f"Target provided as path, but does not exist: {target_data}."
                )

            return validate_path_input("zarr", input_data, target_data)
        elif isinstance(input_data, str) and is_valid_uri(input_data):
            input_list = [input_data]

            assert target_data is None or isinstance(target_data, str)
            if target_data is not None and not is_valid_uri(target_data):
                raise ValueError(
                    f"Wrong target type for zarr data. Expected a zarr URI, got "
                    f"{type(target_data)}."
                )
            target_list = [target_data] if target_data is not None else None
            return input_list, target_list
        else:
            raise ValueError(
                f"Wrong input type for zarr data. Expected a file URI or a path to a "
                f" file, got {input_data}. Path may not exist."
            )
    elif isinstance(input_data, list):
        # use first element as determinant of type
        if isinstance(input_data[0], (str, Path)):
            if Path(input_data[0]).exists():
                return validate_path_input("zarr", input_data, target_data)
            else:
                final_input_list = [
                    str(item) for item in input_data if is_valid_uri(item)
                ]
                if target_data is not None:
                    assert isinstance(target_data, list)
                    final_target_list = [
                        str(item) for item in target_data if is_valid_uri(item)
                    ]
                else:
                    final_target_list = None
                return final_input_list, final_target_list
        else:
            raise ValueError(
                f"Wrong input type for zarr data. Expected a list of file URIs or "
                f" paths to files, got {type(input_data[0])}."
            )
    else:
        raise ValueError(
            f"Wrong input type for zarr data. Expected a file URI, a path to a file, "
            f" or a list of those, got {type(input_data)}."
        )


def initialize_data_pair(
    data_type: Literal["array", "tiff", "zarr", "czi", "custom"],
    input_data: InputType,
    target_data: InputType | None = None,
    extension_filter: str = "",
    custom_loader: bool = False,
) -> tuple[InputType | list[InputType], InputType | list[InputType] | None]:
    """
    Initialize a pair of input and target data.

    Parameters
    ----------
    data_type : Literal["array", "tiff", "zarr", "czi", "custom"]
        The type of data to initialize.
    input_data : InputType
        Input data, can be None, a path to a folder, a list of paths, or a numpy
        array.
    target_data : InputType | None
        Target data, can be None, a path to a folder, a list of paths, or a numpy
        array.
    extension_filter : str, default=""
        File extension filter to apply when listing files.
    custom_loader : bool, default=False
        Whether a custom image stack loader is used.

    Returns
    -------
    list[numpy.ndarray] | list[pathlib.Path]
        Initialized input data. For file paths, returns a list of Path objects. For
        numpy arrays, returns the arrays directly.
    list[numpy.ndarray] | list[pathlib.Path] | None
        Initialized target data. For file paths, returns a list of Path objects. For
        numpy arrays, returns the arrays directly. Returns None if target_data is None.
    """
    if input_data is None:
        return None, None

    validate_input_target_type_consistency(input_data, target_data)

    if data_type == SupportedData.ARRAY:
        return validate_array_input(input_data, target_data)
    elif data_type in (SupportedData.TIFF, SupportedData.CZI):
        assert data_type != SupportedData.ARRAY.value  # for mypy

        if isinstance(input_data, (str, Path)):
            assert target_data is None or isinstance(target_data, (str, Path))

            return validate_path_input(data_type, input_data, target_data)
        elif isinstance(input_data, list):
            assert target_data is None or isinstance(target_data, list)

            return validate_path_input(data_type, input_data, target_data)
        else:
            raise ValueError(
                f"Unsupported input type for {data_type}: {type(input_data)}"
            )
    elif data_type == SupportedData.ZARR:
        return validate_zarr_input(input_data, target_data)
    elif data_type == SupportedData.CUSTOM:
        if custom_loader:
            return input_data, target_data
        return validate_path_input(data_type, input_data, target_data, extension_filter)
    else:
        raise NotImplementedError(f"Unsupported data type: {data_type}")
