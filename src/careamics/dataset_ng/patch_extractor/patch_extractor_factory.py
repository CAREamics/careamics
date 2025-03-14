from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Optional, Union, overload

from numpy.typing import NDArray
from typing_extensions import ParamSpec

from careamics.config.support import SupportedData
from careamics.dataset_ng.patch_extractor import PatchExtractor
from careamics.file_io.read import ReadFunc

from .image_stack_loader import (
    ImageStackLoader,
    SupportedDataDev,
    get_image_stack_loader,
)

P = ParamSpec("P")


# Define overloads for each implemented ImageStackLoader case
# Array case
@overload
def create_patch_extractor(
    source: Sequence[NDArray], axes: str, data_type: Literal[SupportedData.ARRAY]
) -> PatchExtractor:
    """
    Create a patch extractor from a sequence of numpy arrays.

    Parameters
    ----------
    source: sequence of numpy.ndarray
        The source arrays of the data.
    data_config: DataConfig
        The data configuration, `data_config.data_type` should have the value "array",
        and `data_config.axes` should describe the axes of every array in the `source`.

    Returns
    -------
    PatchExtractor
    """


# TIFF and ZARR case
@overload
def create_patch_extractor(
    source: Sequence[Path],
    axes: str,
    data_type: Literal[SupportedData.TIFF, SupportedDataDev.ZARR],
) -> PatchExtractor:
    """
    Create a patch extractor from a sequence of files that match our supported types.

    Supported file types include TIFF and ZARR.

    If the files are ZARR files they must follow the OME standard. If you have ZARR
    files that do not follow the OME standard, see documentation on how to create
    a custom `image_stack_loader`. (TODO: Add link).

    Parameters
    ----------
    source: sequence of Path
        The source files for the data.
    data_config: DataConfig
        The data configuration, `data_config.data_type` should have the value "tiff" or
        "zarr", and `data_config.axes` should describe the axes of every image in the
        `source`.

    Returns
    -------
    PatchExtractor
    """


# Custom file type case (loaded into memory)
@overload
def create_patch_extractor(
    source: Any,
    axes: str,
    data_type: Literal[SupportedData.CUSTOM],
    *,
    read_func: ReadFunc,
    read_kwargs: dict[str, Any],
) -> PatchExtractor:
    """
    Create a patch extractor from a sequence of files of a custom type.

    Parameters
    ----------
    source: sequence of Path
        The source files for the data.
    data_config: DataConfig
        The data configuration, `data_config.data_type` should have the value "custom".
    read_func : ReadFunc
        A function to read the custom file type, see the `ReadFunc` protocol.
    read_kwargs : dict of {str: Any}
        Kwargs that will be passed to the custom `read_func`.

    Returns
    -------
    PatchExtractor
    """


# Custom ImageStackLoader case
@overload
def create_patch_extractor(
    source: Any,
    axes: str,
    data_type: Literal[SupportedData.CUSTOM],
    image_stack_loader: ImageStackLoader[P],
    *args: P.args,
    **kwargs: P.kwargs,
) -> PatchExtractor:
    """
    Create a patch extractor using a custom `ImageStackLoader`.

    The custom image stack loader must follow the `ImageStackLoader` protocol, i.e.
    it must have the following function signature:
    ```
    def image_loader_example(
        source: Any, data_config: DataConfig, *args, **kwargs
    ) -> Sequence[ImageStack]:
    ```

    Parameters
    ----------
    source: sequence of Path
        The source files for the data.
    data_config: DataConfig
        The data configuration, `data_config.data_type` should have the value "custom".
    image_stack_loader: ImageStackLoader
        A custom image stack loader callable.
    *args: Any
        Positional arguments that will be passed to the custom image stack loader.
    **kwargs: Any
        Keyword arguments that will be passed to the custom image stack loader.

    Returns
    -------
    PatchExtractor
    """


# final overload to match the implentation function signature
# Need this so it works later in the code
#   (bec there aren't created overloads for create_patch_extractors below)
@overload
def create_patch_extractor(
    source: Any,
    axes: str,
    data_type: Union[SupportedData, SupportedDataDev],
    image_stack_loader: Optional[ImageStackLoader[P]] = None,
    *args: P.args,
    **kwargs: P.kwargs,
) -> PatchExtractor: ...


def create_patch_extractor(
    source: Any,
    axes: str,
    data_type: Union[SupportedData, SupportedDataDev],
    image_stack_loader: Optional[ImageStackLoader[P]] = None,
    *args: P.args,
    **kwargs: P.kwargs,
) -> PatchExtractor:
    # TODO: Do we need to catch data_config.data_type and source mismatches?
    #   e.g. data_config.data_type is "array" but source is not Sequence[NDArray]
    loader: ImageStackLoader[P] = get_image_stack_loader(data_type, image_stack_loader)
    image_stacks = loader(source, axes, *args, **kwargs)
    return PatchExtractor(image_stacks)


# TODO: Remove this and just call `create_patch_extractor` within the Dataset class
# Keeping for consistency for now
def create_patch_extractors(
    source: Any,
    target_source: Optional[Any],
    axes: str,
    data_type: Union[SupportedData, SupportedDataDev],
    image_stack_loader: Optional[ImageStackLoader] = None,
    *args,
    **kwargs,
) -> tuple[PatchExtractor, Optional[PatchExtractor]]:

    # --- data extractor
    patch_extractor: PatchExtractor = create_patch_extractor(
        source,
        axes,
        data_type,
        image_stack_loader,
        *args,
        **kwargs,
    )
    # --- optional target extractor
    if target_source is not None:
        target_patch_extractor = create_patch_extractor(
            target_source,
            axes,
            data_type,
            image_stack_loader,
            *args,
            **kwargs,
        )

        return patch_extractor, target_patch_extractor

    return patch_extractor, None
