"""Factory functions and data types for MicroSplit datasets."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar, overload

import numpy as np
from numpy.typing import NDArray

from careamics.config.data.microsplit_data_config import MicroSplitDataConfig
from careamics.config.support import SupportedData
from careamics.dataset.dataset import CareamicsDataset
from careamics.dataset.image_stack import ImageStack
from careamics.dataset.patch_constructor import PatchConstr
from careamics.dataset.patch_constructor.microsplit_patch_constructors import (
    IndependentTargetsMsPatchConstr,
    MultiChannelTargetMsPatchConstr,
    PairedInputTargetMsPatchConstr,
    PredMsPatchConstr,
)
from careamics.dataset.patching import create_patching
from careamics.models.constraints import ModelConstraints
from careamics.utils import get_logger

from .factory import (
    ImageStackLoading,
    Loading,
    ReadFuncLoading,
    init_patch_extractor,
    select_image_stack_loader,
    select_patch_extractor_type,
)

T = TypeVar("T")
logger = get_logger("MicroSplitFactory")
AlphaRanges = Sequence[tuple[float, float]]


def _warn_unused_config_fields(
    config: MicroSplitDataConfig,
    field_names: Sequence[str],
    mode_name: str,
) -> None:
    """Warn when explicitly configured fields are not used by a factory path.

    Parameters
    ----------
    config : MicroSplitDataConfig
        Configuration to inspect for explicitly set fields.
    field_names : Sequence[str]
        Field names that are unused by the selected factory path.
    mode_name : str
        Human-readable MicroSplit mode name used in the warning message.
    """
    config_values = config.model_dump()
    unused_fields = sorted(
        field_name
        for field_name in set(field_names).intersection(config.model_fields_set)
        if config_values[field_name] is not None
    )
    if unused_fields:
        logger.warning(
            "MicroSplit %s does not use config field(s): %s.",
            mode_name,
            ", ".join(unused_fields),
        )


def _warn_unused_training_config_fields(
    config: MicroSplitDataConfig,
    mode_name: str,
    unused_fields: Sequence[str] = (),
) -> None:
    """Warn for MicroSplit training config fields unused by a constructor mode.

    Parameters
    ----------
    config : MicroSplitDataConfig
        Configuration to inspect for explicitly set fields.
    mode_name : str
        Human-readable MicroSplit mode name used in the warning message.
    unused_fields : Sequence[str], default=()
        Additional fields unused by the selected training constructor.
    """
    _warn_unused_config_fields(
        config,
        unused_fields,
        mode_name,
    )


@dataclass
class MultiChannelTarget(Generic[T]):
    """MicroSplit data with target channels acquired together.

    The input will be a synthetically generated superposition of the target channels.
    """

    target_data: T


@dataclass
class IndependentTargets(Generic[T]):
    """MicroSplit data with independent target structures.

    The data for different target structures may have a different shape.

    The input will be a synthetically generated superposition of the target structures.
    """

    target_channel_data: Sequence[T]


@dataclass
class PairedInputTarget(Generic[T]):
    """MicroSplit data with paired inputs and multi-channel targets."""

    input_data: T
    target_data: T


MicroSplitTrainingData = (
    MultiChannelTarget[Any] | IndependentTargets[Any] | PairedInputTarget[Any]
)


def create_microsplit_dataset(
    config: MicroSplitDataConfig,
    data: MultiChannelTarget[Any] | IndependentTargets[Any] | PairedInputTarget[Any],
    loading: Loading = None,
    model_constraints: ModelConstraints | None = None,
    rng: np.random.Generator | None = None,
) -> CareamicsDataset[ImageStack]:
    """Create a MicroSplit training or validation dataset.

    The `data` type determines which MicroSplit training mode to use. There are three
    options:

        - `MicroSplitMultiplexedTargetData`: When only multiplexed target channels are
        available the inputs can be synthesized by summing together the target channels.
        - `MicroSplitSeparateTargetData`: Multiplexed target channels are not available,
        instead, each channel are acquired separately. This should only be used for
        structures which are not spatially correlated.
        - `MicroSplitPairedData`: When both the multiplexed target channels and the
        real input are available.

    Parameters
    ----------
    config : MicroSplitDataConfig
        MicroSplit data configuration.
    data : MicroSplitTrainingData
        Data sources used to construct MicroSplit training patches. Either
        `MicroSplitMultiplexedTargetData`, `MicroSplitSeparateTargetData` or
        `MicroSplitPairedData`.
    loading : Loading, default=None
        Loading specification for custom data.
    model_constraints : ModelConstraints, optional
        Optional model constraints for dataset validation.
    rng : numpy.random.Generator, optional
        Random number generator passed to stochastic MicroSplit constructors.

    Returns
    -------
    CareamicsDataset
        The configured MicroSplit dataset.
    """
    if config.mode == "predicting":
        raise ValueError(
            "Use `create_microsplit_pred_dataset` to create prediction datasets."
        )

    image_stack_loader = select_image_stack_loader(
        data_type=SupportedData(config.data_type),
        in_memory=config.in_memory,
        loading=loading,
    )
    patch_extractor_type = select_patch_extractor_type(
        data_type=SupportedData(config.data_type), in_memory=config.in_memory
    )
    rng = rng if rng is not None else np.random.default_rng(config.seed)

    patch_constructor: PatchConstr
    match data:
        case MultiChannelTarget(target_data):
            _warn_unused_training_config_fields(config, "multiplexed-target mode")
            target_extractor = init_patch_extractor(
                patch_extractor_type, image_stack_loader, target_data, config.axes
            )
            patching_strategy = create_patching(
                target_extractor.shapes, config.patching
            )
            patch_constructor = MultiChannelTargetMsPatchConstr(
                patching_strategy=patching_strategy,
                target_extractor=target_extractor,
                multiscale_count=config.multiscale_count,
                padding_mode=config.padding_mode,
                alpha_ranges=config.alpha_ranges,
                uncorrelated_channel_prob=config.uncorrelated_channel_prob,
                channels=config.channels,
                rng=rng,
            )
        case IndependentTargets(target_channel_data):
            _warn_unused_training_config_fields(
                config,
                "separate-target mode",
                unused_fields=("channels", "uncorrelated_channel_prob"),
            )
            if len(target_channel_data) < 2:
                raise ValueError(
                    "At least two target channel sources must be provided."
                )
            target_extractors = [
                init_patch_extractor(
                    patch_extractor_type, image_stack_loader, source, config.axes
                )
                for source in target_channel_data
            ]
            patching_strategies = [
                create_patching(extractor.shapes, config.patching)
                for extractor in target_extractors
            ]
            patch_constructor = IndependentTargetsMsPatchConstr(
                patching_strategies=patching_strategies,
                target_extractors=target_extractors,
                multiscale_count=config.multiscale_count,
                padding_mode=config.padding_mode,
                alpha_ranges=config.alpha_ranges,
                rng=rng,
            )
        case PairedInputTarget(input_data, target_data):
            # TODO: currently input data has to have singleton C dimension
            _warn_unused_training_config_fields(
                config,
                "paired-input-target mode",
                unused_fields=(
                    "alpha_ranges",
                    "channels",
                    "uncorrelated_channel_prob",
                ),
            )
            input_extractor = init_patch_extractor(
                patch_extractor_type, image_stack_loader, input_data, config.axes
            )
            target_extractor = init_patch_extractor(
                patch_extractor_type, image_stack_loader, target_data, config.axes
            )
            patching_strategy = create_patching(input_extractor.shapes, config.patching)
            patch_constructor = PairedInputTargetMsPatchConstr(
                patching_strategy=patching_strategy,
                input_extractor=input_extractor,
                target_extractor=target_extractor,
                multiscale_count=config.multiscale_count,
                padding_mode=config.padding_mode,
            )
        case _:
            raise TypeError(
                "`data` must be one of MicroSplitJointTargetData, "
                "MicroSplitSeparateTargetData or MicroSplitPairedData."
            )

    return CareamicsDataset(
        data_config=config,
        patch_constructor=patch_constructor,
        model_constraints=model_constraints,
    )


@overload
def create_microsplit_pred_dataset(  # numpydoc ignore=GL08
    config: MicroSplitDataConfig,
    input_data: Sequence[NDArray[Any]] | Sequence[Path],
    loading: ReadFuncLoading | None = None,
    model_constraints: ModelConstraints | None = None,
) -> CareamicsDataset[ImageStack]: ...


@overload
def create_microsplit_pred_dataset(  # numpydoc ignore=GL08
    config: MicroSplitDataConfig,
    input_data: Any,
    loading: ImageStackLoading,
    model_constraints: ModelConstraints | None = None,
) -> CareamicsDataset[ImageStack]: ...


def create_microsplit_pred_dataset(
    config: MicroSplitDataConfig,
    input_data: Any,
    loading: Loading = None,
    model_constraints: ModelConstraints | None = None,
) -> CareamicsDataset[ImageStack]:
    """Create a MicroSplit prediction dataset.

    Parameters
    ----------
    config : MicroSplitDataConfig
        MicroSplit prediction data configuration.
    input_data : Sequence[NDArray], Sequence[Path] or Any
        Prediction data sources. For default loading, this is a list of numpy arrays
        or a list of file paths. If using a custom image stack loader the input can be
        any type that is supported by the loader.
    loading : Loading, default=None
        Loading specification. `None` or `ReadFuncLoading` is used for standard array
        and path inputs, while `ImageStackLoading` is used for custom input data.
    model_constraints : ModelConstraints, optional
        Optional model constraints for dataset validation.

    Returns
    -------
    CareamicsDataset
        The configured MicroSplit prediction dataset.
    """
    if config.mode != "predicting":
        raise ValueError(
            "`create_microsplit_pred_dataset` requires a config with mode='predicting'."
        )
    is_standard_loading = loading is None or isinstance(loading, ReadFuncLoading)
    if is_standard_loading and not isinstance(input_data, Sequence):
        raise TypeError(
            "Prediction input must be a sequence of numpy arrays or paths for standard "
            "loading."
        )

    image_stack_loader = select_image_stack_loader(
        data_type=SupportedData(config.data_type),
        in_memory=config.in_memory,
        loading=loading,
    )
    patch_extractor_type = select_patch_extractor_type(
        data_type=SupportedData(config.data_type), in_memory=config.in_memory
    )
    input_extractor = init_patch_extractor(
        patch_extractor_type, image_stack_loader, input_data, config.axes
    )
    patching_strategy = create_patching(input_extractor.shapes, config.patching)

    patch_constructor = PredMsPatchConstr(
        patching_strategy=patching_strategy,
        input_extractor=input_extractor,
        multiscale_count=config.multiscale_count,
        padding_mode=config.padding_mode,
    )
    return CareamicsDataset(
        data_config=config,
        patch_constructor=patch_constructor,
        model_constraints=model_constraints,
    )
