"""Factory functions and data types for building CAREamics datasets and loaders."""

from dataclasses import dataclass
from functools import partial
from typing import Any, Generic, TypeVar

import numpy as np
from typing_extensions import ParamSpec

from careamics.config.data.data_config import DataConfig
from careamics.config.support import SupportedData
from careamics.file_io.read import ReadFunc
from careamics.lightning.dataset_ng.lightning_modules.constraints import (
    ModelConstraints,
)

from .dataset import CareamicsDataset
from .filter_bg import filter_background, filter_background_with_mask
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
from .patching_strategies import StratifiedPatchingStrategy, create_patching_strategy
from .val_split import create_val_split

P = ParamSpec("P")
T = TypeVar("T")


@dataclass
class ReadFuncLoading:
    """Loading specification using a custom read function."""

    read_source_func: ReadFunc
    read_kwargs: dict[str, Any] | None = None
    extension_filter: str = ""


@dataclass
class ImageStackLoading:
    """Loading spec. for a custom image stack loader (chunked / memory-mapped)."""

    image_stack_loader: ImageStackLoader[..., ImageStack]
    image_stack_loader_kwargs: dict[str, Any] | None = None


Loading = ReadFuncLoading | ImageStackLoading | None
"""
The type of loading used for custom data. `ReadFuncLoading` is the use of
a simple function that will load full images into memory.
`ImageStackLoading` is for custom chunked or memory-mapped next-generation
file formats enabling  single patches to be read from disk at a time.
If the data type is not custom `loading` should be `None`.
"""


@dataclass
class TrainValData(Generic[T]):
    """Data for training with validation data provided."""

    train_data: T
    val_data: T
    train_data_target: T | None = None
    val_data_target: T | None = None
    train_data_mask: T | None = None


@dataclass
class TrainValSplitData(Generic[T]):
    """Data for training with automatic validation splitting."""

    train_data: T
    n_val_patches: int
    train_data_target: T | None = None
    train_data_mask: T | None = None


@dataclass
class PredData(Generic[T]):
    """Data for prediction."""

    pred_data: T
    pred_data_target: T | None = None


# convenience function but should use `create_dataloader` function instead
# For lazy loading custom batch sampler also needs to be set.
# TODO: remove this function? Or is it good for tests
def create_dataset(
    config: DataConfig,
    inputs: Any,
    targets: Any,
    loading: ReadFuncLoading | ImageStackLoading | None = None,
    model_constraints: ModelConstraints | None = None,
) -> CareamicsDataset[ImageStack]:
    """Create a CAREamicsDataset.

    Parameters
    ----------
    config : DataConfig
        The data configuration (data type, axes, patching, etc.).
    inputs : Any
        The input data sources (paths, arrays, or custom).
    targets : Any, optional
        The target data sources, or None.
    loading : ReadFuncLoading or ImageStackLoading or None, optional
        Custom loading specification. Required when `data_type` is "custom":
        use ReadFuncLoading for a read function, or ImageStackLoading for a
        custom image stack loader. Otherwise None.
    model_constraints : ModelConstraints, optional
        If provided, the data module will validate that the input data shape is
        compatible with the model constraints. Only used for prediction datasets.

    Returns
    -------
    CareamicsDataset[ImageStack]
        The configured dataset instance.
    """
    image_stack_loader = select_image_stack_loader(
        data_type=SupportedData(config.data_type),
        in_memory=config.in_memory,
        loading=loading,
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

    patching_strategy = create_patching_strategy(
        input_extractor.shapes, config.patching
    )

    return CareamicsDataset(
        data_config=config,
        patching_strategy=patching_strategy,
        input_extractor=input_extractor,
        target_extractor=target_extractor,
        model_constraints=model_constraints,
    )


def init_patch_extractor(
    patch_extractor: type[PatchExtractor],
    image_stack_loader: ImageStackLoader[..., GenericImageStack],
    source: Any,
    axes: str,
) -> PatchExtractor[GenericImageStack]:
    """Build a patch extractor by loading image stacks from the source.

    Parameters
    ----------
    patch_extractor : type[PatchExtractor]
        The PatchExtractor class to instantiate (e.g. PatchExtractor).
    image_stack_loader : ImageStackLoader
        Callable that takes (source, axes) and returns a list of image stacks.
    source : Any
        Data source (paths, arrays, etc.) passed to the loader.
    axes : str
        Axis order string passed to the loader.

    Returns
    -------
    PatchExtractor[GenericImageStack]
        The constructed patch extractor instance.
    """
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
    loading: ReadFuncLoading | ImageStackLoading | None = None,
) -> ImageStackLoader[..., ImageStack]:
    """Select image stack loader function for the given data type and loading options.

    Parameters
    ----------
    data_type : SupportedData
        The type of data (array, tiff, zarr, czi, custom).
    in_memory : bool
        Whether to load full data into memory (True) or use lazy loading.
    loading : ReadFuncLoading or ImageStackLoading or None, optional
        Custom loading spec, required when data_type is custom.

    Returns
    -------
    ImageStackLoader
        A callable that takes (source, axes) and returns a list of image stacks.
    """
    match data_type:
        case SupportedData.ARRAY:
            return load_arrays
        case SupportedData.TIFF:
            if in_memory:
                return load_tiffs
            else:
                return load_iter_tiff
        case SupportedData.CUSTOM:
            match loading:
                case ReadFuncLoading(read_func, read_kwargs):
                    read_kwargs = {} if read_kwargs is None else read_kwargs
                    return partial(
                        load_custom_file, read_func=read_func, read_kwargs=read_kwargs
                    )
                case ImageStackLoading(image_stack_loader, image_stack_loader_kwargs):
                    if image_stack_loader_kwargs is None:
                        image_stack_loader_kwargs = {}
                    return partial(image_stack_loader, **image_stack_loader_kwargs)
                case None:
                    raise ValueError(
                        "Found `data_type='custom'`, a custom read function or a "
                        "custom image stack loader must be provided."
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


def create_train_dataset(
    config: DataConfig,
    data: TrainValData[Any] | TrainValSplitData,
    loading: Loading,
    model_constraints: ModelConstraints | None = None,
) -> CareamicsDataset[ImageStack]:
    """Create a dataset for training.

    Parameters
    ----------
    config : DataConfig
        Data configuration (must have mode='training').
    data : TrainValData | TrainValSplitData
        Train and validation data sources (and optional targets/masks).
    loading : ReadFuncLoading or ImageStackLoading or None
        Custom loading specification when using custom data type.
    model_constraints : ModelConstraints, optional
        If provided, the dataset will validate that the input data shape is
        compatible with the model constraints.

    Returns
    -------
    CareamicsDataset
        The training dataset.
    """
    if config.mode != "training":
        raise ValueError(
            f"CAREamicsDataModule configured for {config.mode} cannot be "
            f"used for training. Please create a new CareamicsDataModule with "
            f"a configuration with mode='training'."
        )
    inputs = data.train_data
    targets = data.train_data_target
    masks = data.train_data_mask

    # init dataset components
    image_stack_loader = select_image_stack_loader(
        data_type=SupportedData(config.data_type),
        in_memory=config.in_memory,
        loading=loading,
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

    patching_strategy = create_patching_strategy(
        input_extractor.shapes, config.patching
    )

    # patch filtering
    if config.patch_filter is not None:
        if not isinstance(patching_strategy, StratifiedPatchingStrategy):
            raise TypeError(
                "Background patch filtering is only compatible with stratified "
                f"patching. Found {config.patching.name} patching in the configuration."
            )
        filter_background(
            config.patch_filter,
            patching_strategy,
            input_extractor,
        )
    # if a mask is provided, apply mask filtering
    if mask_extractor is not None:
        if config.mask_filter is None:
            raise ValueError("No `MaskFilterConfig` found for mask filtering.")
        if not isinstance(patching_strategy, StratifiedPatchingStrategy):
            raise TypeError(
                "Mask filtering is only compatible with stratified "
                f"patching. Found {config.patching.name} patching in the configuration."
            )
        filter_background_with_mask(
            config.mask_filter,
            patching_strategy,
            mask_extractor,
        )

    return CareamicsDataset(
        data_config=config,
        patching_strategy=patching_strategy,
        input_extractor=input_extractor,
        target_extractor=target_extractor,
        model_constraints=model_constraints,
    )


def create_train_val_datasets(
    config: DataConfig,
    data: TrainValData[Any],
    loading: Loading,
    model_constraints: ModelConstraints | None = None,
) -> tuple[CareamicsDataset[ImageStack], CareamicsDataset[ImageStack]]:
    """Create train and validation datasets when validation data is provided explicitly.

    Parameters
    ----------
    config : DataConfig
        Data configuration (must have mode='training').
    data : TrainValData
        Train and validation data sources (and optional targets/masks).
    loading : ReadFuncLoading or ImageStackLoading or None
        Custom loading specification when using custom data type.
    model_constraints : ModelConstraints, optional
        If provided, the dataset will validate that the input data shape is
        compatible with the model constraints.

    Returns
    -------
    tuple of (CareamicsDataset, CareamicsDataset)
        (train_dataset, val_dataset).
    """
    if config.mode != "training":
        raise ValueError(
            f"CAREamicsDataModule configured for {config.mode} cannot be "
            f"used for training. Please create a new CareamicsDataModule with "
            f"a configuration with mode='training'."
        )

    train_dataset = create_train_dataset(
        config=config,
        data=data,
        loading=loading,
        model_constraints=model_constraints,
    )

    validation_config = config.convert_mode("validating")

    val_dataset = create_dataset(
        config=validation_config,
        inputs=data.val_data,
        targets=data.val_data_target,
        loading=loading,
        model_constraints=model_constraints,
    )

    return train_dataset, val_dataset


def create_val_split_datasets(
    config: DataConfig,
    data: TrainValSplitData[Any],
    loading: Loading,
    rng: np.random.Generator,
    model_constraints: ModelConstraints | None = None,
) -> tuple[CareamicsDataset[ImageStack], CareamicsDataset[ImageStack]]:
    """Create train and validation datasets by splitting from training data.

    Requires stratified patching in config.

    Parameters
    ----------
    config : DataConfig
        Data configuration (must have mode='training', patching.name='stratified').
    data : TrainValSplitData
        Training data sources and number of validation patches.
    loading : ReadFuncLoading or ImageStackLoading or None
        Custom loading specification when using custom data type.
    rng : numpy.random.Generator
        Random generator for reproducible validation split.
    model_constraints : ModelConstraints, optional
        If provided, the dataset will validate that the input data shape is
        compatible with the model constraints.

    Returns
    -------
    tuple of (CareamicsDataset, CareamicsDataset)
        (train_dataset, val_dataset).
    """
    if config.mode != "training":
        raise ValueError(
            f"CAREamicsDataModule configured for {config.mode} cannot be "
            f"used for training. Please create a new CareamicsDataModule with "
            f"a configuration with mode='training'."
        )
    if config.patching.name != "stratified":
        # TODO: we could optionally split by samples instead.
        raise ValueError(
            "Validation split is only compatible with stratified patching."
        )

    train_dataset = create_train_dataset(config, data, loading, model_constraints)
    train_patching = train_dataset.patching_strategy
    # ensured by guard on config at the start of function
    assert isinstance(train_patching, StratifiedPatchingStrategy)

    # TODO: select val patches according to filtered background distribution
    # val split applied to patching strat
    train_patching, val_patching = create_val_split(
        train_patching, data.n_val_patches, rng=rng
    )

    val_dataset = CareamicsDataset(
        data_config=config.convert_mode("validating"),
        input_extractor=train_dataset.input_extractor,
        target_extractor=train_dataset.target_extractor,
        patching_strategy=val_patching,
        model_constraints=model_constraints,
    )
    return train_dataset, val_dataset


def create_pred_dataset(
    config: DataConfig,
    data: PredData[Any],
    loading: Loading,
    model_constraints: ModelConstraints | None = None,
) -> CareamicsDataset[ImageStack]:
    """Create the dataset for prediction.

    Parameters
    ----------
    config : DataConfig
        Data configuration.
    data : PredData
        Prediction data sources.
    loading : ReadFuncLoading or ImageStackLoading or None
        Custom loading specification when using custom data type.
    model_constraints : ModelConstraints, optional
        If provided, the dataset will validate that the prediction data shape is
        compatible with the model constraints.

    Returns
    -------
    CareamicsDataset[ImageStack]
        Dataset configured for prediction.
    """
    if config.mode == "validating":
        raise ValueError(
            "CAREamicsDataModule configured for validating cannot be used for "
            "prediction. Please create a new CareamicsDataModule with a "
            "configuration with mode='predicting'."
        )
    return create_dataset(
        config=(
            config.convert_mode("predicting") if config.mode == "training" else config
        ),
        inputs=data.pred_data,
        targets=data.pred_data_target,
        loading=loading,
        model_constraints=model_constraints,
    )
