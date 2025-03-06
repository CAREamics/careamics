from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, ParamSpec, overload

from numpy.typing import NDArray

from careamics.config import DataConfig
from careamics.dataset_ng.patch_extractor import PatchExtractor
from careamics.file_io.read import ReadFunc

from .image_stack_loader import ImageStackLoader, get_image_stack_loader

P = ParamSpec("P")


@overload
def create_patch_extractor(
    data_config: DataConfig,
    source: Sequence[NDArray],
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


@overload
def create_patch_extractor(
    data_config: DataConfig,
    source: Sequence[Path],
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


@overload
def create_patch_extractor(
    data_config: DataConfig,
    source: Any,
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


@overload
def create_patch_extractor(
    data_config: DataConfig,
    source: Any,
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


@overload
def create_patch_extractor(
    data_config: DataConfig,
    source: Any,
    image_stack_loader: Optional[ImageStackLoader[P]] = None,
    *args: P.args,
    **kwargs: P.kwargs,
) -> PatchExtractor: ...


def create_patch_extractor(
    data_config: DataConfig,
    source: Any,
    image_stack_loader: Optional[ImageStackLoader[P]] = None,
    *args: P.args,
    **kwargs: P.kwargs,
) -> PatchExtractor:
    loader = get_image_stack_loader(data_config.data_type, image_stack_loader)
    image_stacks = loader(source, data_config, *args, **kwargs)
    return PatchExtractor(image_stacks)


# TODO: Remove this and just call `create_patch_extractor` within the Dataset class
# Keeping for consistency for now
def create_patch_extractors(
    data_config: DataConfig,
    source: Any,
    target_source: Optional[Any] = None,
    image_stack_loader: Optional[ImageStackLoader] = None,
    *args,
    **kwargs,
) -> tuple[PatchExtractor, Optional[PatchExtractor]]:

    # --- data extractor
    patch_extractor: PatchExtractor = create_patch_extractor(
        data_config,
        source,
        image_stack_loader,
        *args,
        **kwargs,
    )
    # --- optional target extractor
    if target_source is not None:
        target_patch_extractor = create_patch_extractor(
            data_config,
            target_source,
            image_stack_loader,
            *args,
            **kwargs,
        )

        return patch_extractor, target_patch_extractor

    return patch_extractor, None
