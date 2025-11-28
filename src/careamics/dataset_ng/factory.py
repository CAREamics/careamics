from functools import partial
from typing import Any

from typing_extensions import ParamSpec

from careamics.config.data.ng_data_config import NGDataConfig
from careamics.config.support import SupportedData
from careamics.file_io.read import ReadFunc

from .dataset import CareamicsDataset
from .image_stack import (
    GenericImageStack,
    ImageStack,
)
from .image_stack_loader import (
    ImageStackLoader,
    load_arrays,
    load_custom_file,
    load_czis,
    load_iter_tiff,
    load_tiffs,
    load_zarrs,
)
from .patch_extractor import LimitFilesPatchExtractor, PatchExtractor

P = ParamSpec("P")


# convenience function but should use `create_dataloader` function instead
# For lazy loading custom batch sampler also needs to be set.
def create_dataset(
    config: NGDataConfig,
    inputs: Any,
    targets: Any,
    masks: Any = None,
    read_func: ReadFunc | None = None,
    read_kwargs: dict[str, Any] | None = None,
    image_stack_loader: ImageStackLoader | None = None,
    image_stack_loader_kwargs: dict[str, Any] | None = None,
) -> CareamicsDataset[ImageStack]:
    """
    Convenience function to create the CAREamicsDataset.

    Parameters
    ----------
    config : DataConfig or InferenceConfig
        The data configuration.
    inputs : Any
        The input sources to the dataset.
    targets : Any, optional
        The target sources to the dataset.
    masks : Any, optional
        The mask sources used to filter patches.
    read_func : ReadFunc, optional
        A function that can that can be used to load custom data. This argument is
        ignored unless the `data_type` in the `config` is "custom".
    read_kwargs : dict of {str, Any}, optional
        Additional key-word arguments to pass to the `read_func`.
    image_stack_loader : ImageStackLoader, optional
        A function for custom image stack loading. This argument is ignored unless the
        `data_type` in the `config` is "custom".
    image_stack_loader_kwargs : {str, Any}, optional
        Additional key-word arguments to pass to the `image_stack_loader`.
    """
    image_stack_loader = select_image_stack_loader(
        data_type=SupportedData(config.data_type),
        in_memory=config.in_memory,
        read_func=read_func,
        read_kwargs=read_kwargs,
        image_stack_loader=image_stack_loader,
        image_stack_loader_kwargs=image_stack_loader_kwargs,
    )
    patch_extractor_type = select_patch_extractor_type(
        data_type=SupportedData(config.data_type), in_memory=config.in_memory
    )
    input_extractor = init_patch_extractor(
        patch_extractor_type, image_stack_loader, inputs, config.axes
    )
    if targets is not None:
        target_extractor = init_patch_extractor(
            patch_extractor_type, image_stack_loader, targets, config.axes
        )
    else:
        target_extractor = None
    if masks is not None:
        mask_extractor = init_patch_extractor(
            patch_extractor_type, image_stack_loader, masks, config.axes
        )
    else:
        mask_extractor = None
    return CareamicsDataset(
        data_config=config,
        input_extractor=input_extractor,
        target_extractor=target_extractor,
        mask_extractor=mask_extractor,
    )


def init_patch_extractor(
    patch_extractor: type[PatchExtractor],
    image_stack_loader: ImageStackLoader[..., GenericImageStack],
    source: Any,
    axes: str,
) -> PatchExtractor[GenericImageStack]:
    image_stacks = image_stack_loader(source, axes)
    return patch_extractor(image_stacks)


def select_patch_extractor_type(
    data_type: SupportedData,
    in_memory: bool,
) -> type[PatchExtractor]:
    """Select the appropriate PatchExtractor type based on data type and memory mode.

    If `in_memory` is True, or `data_type` is ZARR or CZI, the standard
    `PatchExtractor` is selected, otherwise the `LimitFilesPatchExtractor` will be used.

    Parameters
    ----------
    data_type : SupportedData
        The type of data being handled.
    in_memory : bool
        Indicates whether data is to be loaded into memory.

    Returns
    -------
    type[PatchExtractor]
        The selected PatchExtractor type.
    """
    if not in_memory and data_type in (SupportedData.TIFF, SupportedData.CUSTOM):
        return LimitFilesPatchExtractor
    else:
        return PatchExtractor


def select_image_stack_loader(
    data_type: SupportedData,
    in_memory: bool,
    read_func: ReadFunc | None = None,
    read_kwargs: dict[str, Any] | None = None,
    image_stack_loader: ImageStackLoader | None = None,
    image_stack_loader_kwargs: dict[str, Any] | None = None,
) -> ImageStackLoader:
    match data_type:
        case SupportedData.ARRAY:
            return load_arrays
        case SupportedData.TIFF:
            if in_memory:
                return load_tiffs
            else:
                return load_iter_tiff
        case SupportedData.CUSTOM:
            if (read_func is not None) and (image_stack_loader is None):
                read_kwargs = {} if read_kwargs is None else read_kwargs
                return partial(
                    load_custom_file, read_func=read_func, read_kwargs=read_kwargs
                )
            elif (read_func is None) and (image_stack_loader is not None):
                image_stack_loader_kwargs = (
                    {}
                    if image_stack_loader_kwargs is None
                    else image_stack_loader_kwargs
                )
                return partial(image_stack_loader, **image_stack_loader_kwargs)
            else:
                raise ValueError(
                    "Found `data_type='custom'` **one** of `read_func` or "
                    "`image_stack_loader` must be provided."
                )
        case SupportedData.ZARR:
            # TODO: in_memory or not
            return load_zarrs
        case SupportedData.CZI:
            # TODO: in_memory or not
            return load_czis
        case _:
            raise NotImplementedError(
                f"Selecting an image stack for data type '{data_type}' has not been "
                "implemented yet."
            )
