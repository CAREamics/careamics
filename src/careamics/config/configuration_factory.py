"""Convenience functions to create configurations for training and inference."""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from albumentations import Compose

from .algorithm_model import AlgorithmModel
from .architectures import UNetModel
from .configuration_model import Configuration
from .data_model import DataModel
from .inference_model import InferenceModel
from .support import (
    SupportedAlgorithm,
    SupportedArchitecture,
    SupportedLoss,
    SupportedPixelManipulation,
    SupportedTransform,
)
from .training_model import TrainingModel


def create_n2n_configuration(
    experiment_name: str,
    data_type: Literal["array", "tiff", "custom"],
    axes: str,
    patch_size: List[int],
    batch_size: int,
    num_epochs: int,
    use_augmentations: bool = True,
    use_n2v2: bool = False,
    n_channels: int = 1,
    logger: Literal["wandb", "tensorboard", "none"] = "none",
    model_kwargs: Optional[dict] = None,
) -> Configuration:
    """
    Create a configuration for training N2V.

    If "Z" is present in `axes`, then `path_size` must be a list of length 3, otherwise
    2.

    By setting `use_augmentations` to False, the only transformation applied will be
    normalization and N2V manipulation.

    The parameter `use_n2v2` overrides the corresponding `n2v2` that can be passed
    in `model_kwargs`.

    If you pass "horizontal" or "vertical" to `struct_n2v_axis`, then structN2V mask
    will be applied to each manipulated pixel.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    data_type : Literal["array", "tiff", "custom"]
        Type of the data.
    axes : str
        Axes of the data (e.g. SYX).
    patch_size : List[int]
        Size of the patches along the spatial dimensions (e.g. [64, 64]).
    batch_size : int
        Batch size.
    num_epochs : int
        Number of epochs.
    use_augmentations : bool, optional
        Whether to use augmentations, by default True.
    use_n2v2 : bool, optional
        Whether to use N2V2, by default False.
    n_channels : int, optional
        Number of channels (in and out), by default 1.
    roi_size : int, optional
        N2V pixel manipulation area, by default 11.
    masked_pixel_percentage : float, optional
        Percentage of pixels masked in each patch, by default 0.2.
    struct_n2v_axis : Literal["horizontal", "vertical", "none"], optional
        Axis along which to apply structN2V mask, by default "none".
    struct_n2v_span : int, optional
        Span of the structN2V mask, by default 5.
    logger : Literal["wandb", "tensorboard", "none"], optional
        Logger to use, by default "none".
    model_kwargs : dict, optional
        UNetModel parameters, by default {}.

    Returns
    -------
    Configuration
        Configuration for training N2V.
    """
    # model
    if model_kwargs is None:
        model_kwargs = {}
    model_kwargs["n2v2"] = use_n2v2
    model_kwargs["conv_dims"] = 3 if "Z" in axes else 2
    model_kwargs["in_channels"] = n_channels
    model_kwargs["num_classes"] = n_channels

    unet_model = UNetModel(
        architecture=SupportedArchitecture.UNET.value,
        **model_kwargs,
    )

    # algorithm model
    algorithm = AlgorithmModel(
        algorithm=SupportedAlgorithm.N2N.value,
        loss=SupportedLoss.MAE.value,
        model=unet_model,
    )

    # augmentations
    if use_augmentations:
        transforms: List[Dict[str, Any]] = [
            {
                "name": SupportedTransform.NORMALIZE.value,
            },
            {
                "name": SupportedTransform.NDFLIP.value,
            },
            {
                "name": SupportedTransform.XY_RANDOM_ROTATE90.value,
            },
        ]
    else:
        transforms = [
            {
                "name": SupportedTransform.NORMALIZE.value,
            },
        ]

    # data model
    data = DataModel(
        data_type=data_type,
        axes=axes,
        patch_size=patch_size,
        batch_size=batch_size,
        transforms=transforms,
    )

    # training model
    training = TrainingModel(
        num_epochs=num_epochs,
        batch_size=batch_size,
        logger=None if logger == "none" else logger,
    )

    # create configuration
    configuration = Configuration(
        experiment_name=experiment_name,
        algorithm_config=algorithm,
        data_config=data,
        training_config=training,
    )

    return configuration


def create_n2v_configuration(
    experiment_name: str,
    data_type: Literal["array", "tiff", "custom"],
    axes: str,
    patch_size: List[int],
    batch_size: int,
    num_epochs: int,
    use_augmentations: bool = True,
    use_n2v2: bool = False,
    n_channels: int = -1,
    roi_size: int = 11,
    masked_pixel_percentage: float = 0.2,
    struct_n2v_axis: Literal["horizontal", "vertical", "none"] = "none",
    struct_n2v_span: int = 5,
    logger: Literal["wandb", "tensorboard", "none"] = "none",
    model_kwargs: Optional[dict] = None,
) -> Configuration:
    """
    Create a configuration for training N2V.

    N2V uses a UNet model to denoise images in a self-supervised manner. To use its
    variants structN2V and N2V2, set the `struct_n2v_axis` and `struct_n2v_span`
    (structN2V) parameters, or set `use_n2v2` to True (N2V2).

    N2V2 modifies the UNet architecture by adding blur pool layers and removes the skip
    connections, thus removing checkboard artefacts. StructN2V is used when vertical
    or horizontal correlations are present in the noise; it applies an additional mask
    to the manipulated pixel neighbors.

    If "C" is present in `axes`, then you need to set `n_channels` to the number of
    channels.

    If "Z" is present in `axes`, then `path_size` must be a list of length 3, otherwise
    2.

    By setting `use_augmentations` to False, the only transformations applied will be
    normalization and N2V manipulation.

    The `roi_size` parameter specifies the size of the area around each pixel that will
    be manipulated by N2V. The `masked_pixel_percentage` parameter specifies how many
    pixels per patch will be manipulated.

    The parameters of the UNet can be specified in the `model_kwargs` (passed as a
    parameter-value dictionary). Note that `use_n2v2` and 'n_channels' override the
    corresponding parameters passed in `model_kwargs`.

    If you pass "horizontal" or "vertical" to `struct_n2v_axis`, then structN2V mask
    will be applied to each manipulated pixel.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    data_type : Literal["array", "tiff", "custom"]
        Type of the data.
    axes : str
        Axes of the data (e.g. SYX).
    patch_size : List[int]
        Size of the patches along the spatial dimensions (e.g. [64, 64]).
    batch_size : int
        Batch size.
    num_epochs : int
        Number of epochs.
    use_augmentations : bool, optional
        Whether to use augmentations, by default True.
    use_n2v2 : bool, optional
        Whether to use N2V2, by default False.
    n_channels : int, optional
        Number of channels (in and out), by default -1.
    roi_size : int, optional
        N2V pixel manipulation area, by default 11.
    masked_pixel_percentage : float, optional
        Percentage of pixels masked in each patch, by default 0.2.
    struct_n2v_axis : Literal["horizontal", "vertical", "none"], optional
        Axis along which to apply structN2V mask, by default "none".
    struct_n2v_span : int, optional
        Span of the structN2V mask, by default 5.
    logger : Literal["wandb", "tensorboard", "none"], optional
        Logger to use, by default "none".
    model_kwargs : dict, optional
        UNetModel parameters, by default {}.

    Returns
    -------
    Configuration
        Configuration for training N2V.

    Examples
    --------
    Minimum example:
    >>> config = create_n2v_configuration(
    ...     experiment_name="n2v_experiment",
    ...     data_type="array",
    ...     axes="YX",
    ...     patch_size=[64, 64],
    ...     batch_size=32,
    ...     num_epochs=100
    ... )

    To use N2V2, simply pass the `use_n2v2` parameter:
    >>> config = create_n2v_configuration(
    ...     experiment_name="n2v2_experiment",
    ...     data_type="tiff",
    ...     axes="YX",
    ...     patch_size=[64, 64],
    ...     batch_size=32,
    ...     num_epochs=100,
    ...     use_n2v2=True
    ... )

    For structN2V, there are two parameters to set, `struct_n2v_axis` and
    `struct_n2v_span`:
    >>> config = create_n2v_configuration(
    ...     experiment_name="structn2v_experiment",
    ...     data_type="tiff",
    ...     axes="YX",
    ...     patch_size=[64, 64],
    ...     batch_size=32,
    ...     num_epochs=100,
    ...     struct_n2v_axis="horizontal",
    ...     struct_n2v_span=7
    ... )

    If you are training multiple channels together, then you need to specify the number
    of channels:
    >>> config = create_n2v_configuration(
    ...     experiment_name="n2v_experiment",
    ...     data_type="array",
    ...     axes="YXC",
    ...     patch_size=[64, 64],
    ...     batch_size=32,
    ...     num_epochs=100,
    ...     n_channels=3
    ... )

    To turn off the augmentations, except normalization and N2V manipulation, use the
    relevant keyword argument:
    >>> config = create_n2v_configuration(
    ...     experiment_name="n2v_experiment",
    ...     data_type="array",
    ...     axes="YX",
    ...     patch_size=[64, 64],
    ...     batch_size=32,
    ...     num_epochs=100,
    ...     use_augmentations=False
    ... )
    """
    # if there are channels, we need to specify their number
    if "C" in axes and n_channels == -1:
        raise ValueError(
            f"Number of channels must be specified when using channels "
            f"(got {n_channels} channel)."
        )
    elif "C" not in axes and n_channels != -1:
        raise ValueError(
            f"C is not present in the axes, but number of channels is specified "
            f"(got {n_channels} channel)."
        )
    elif n_channels == -1:
        n_channels = 1

    # model
    if model_kwargs is None:
        model_kwargs = {}
    model_kwargs["n2v2"] = use_n2v2
    model_kwargs["conv_dims"] = 3 if "Z" in axes else 2
    model_kwargs["in_channels"] = n_channels
    model_kwargs["num_classes"] = n_channels

    unet_model = UNetModel(
        architecture=SupportedArchitecture.UNET.value,
        **model_kwargs,
    )

    # algorithm model
    algorithm = AlgorithmModel(
        algorithm=SupportedAlgorithm.N2V.value,
        loss=SupportedLoss.N2V.value,
        model=unet_model,
    )

    # augmentations
    if use_augmentations:
        transforms: List[Dict[str, Any]] = [
            {
                "name": SupportedTransform.NORMALIZE.value,
            },
            {
                "name": SupportedTransform.NDFLIP.value,
            },
            {
                "name": SupportedTransform.XY_RANDOM_ROTATE90.value,
            },
        ]
    else:
        transforms = [
            {
                "name": SupportedTransform.NORMALIZE.value,
            },
        ]

    # n2v2 and structn2v
    nv2_transform = {
        "name": SupportedTransform.N2V_MANIPULATE.value,
        "strategy": SupportedPixelManipulation.MEDIAN.value
        if use_n2v2
        else SupportedPixelManipulation.UNIFORM.value,
        "roi_size": roi_size,
        "masked_pixel_percentage": masked_pixel_percentage,
        "struct_mask_axis": struct_n2v_axis,
        "struct_mask_span": struct_n2v_span,
    }
    transforms.append(nv2_transform)

    # data model
    data = DataModel(
        data_type=data_type,
        axes=axes,
        patch_size=patch_size,
        batch_size=batch_size,
        transforms=transforms,
    )

    # training model
    training = TrainingModel(
        num_epochs=num_epochs,
        batch_size=batch_size,
        logger=None if logger == "none" else logger,
    )

    # create configuration
    configuration = Configuration(
        experiment_name=experiment_name,
        algorithm_config=algorithm,
        data_config=data,
        training_config=training,
    )

    return configuration


# TODO add tests
def create_inference_configuration(
    training_configuration: Configuration,
    tile_size: Optional[Tuple[int, ...]] = None,
    tile_overlap: Optional[Tuple[int, ...]] = None,
    data_type: Optional[Literal["array", "tiff", "custom"]] = None,
    axes: Optional[str] = None,
    transforms: Optional[Union[List[Dict[str, Any]], Compose]] = None,
    tta_transforms: bool = True,
    batch_size: Optional[int] = 1,
) -> InferenceModel:
    """
    Create a configuration for inference with N2V.

    If not provided, `data_type` and `axes` are taken from the training
    configuration. If `transforms` are not provided, only normalization is applied.

    Parameters
    ----------
    training_configuration : Configuration
        Configuration used for training.
    tile_size : Tuple[int, ...], optional
        Size of the tiles.
    tile_overlap : Tuple[int, ...], optional
        Overlap of the tiles.
    data_type : str, optional
        Type of the data, by default "tiff".
    axes : str, optional
        Axes of the data, by default "YX".
    transforms : List[Dict[str, Any]] or Compose, optional
        Transformations to apply to the data, by default None.
    tta_transforms : bool, optional
        Whether to apply test-time augmentations, by default True.
    batch_size : int, optional
        Batch size, by default 1.

    Returns
    -------
    InferenceConfiguration
        Configuration for inference with N2V.
    """
    if (
        training_configuration.data_config.mean is None
        or training_configuration.data_config.std is None
    ):
        raise ValueError("Mean and std must be provided in the training configuration.")

    if transforms is None:
        transforms = [
            {
                "name": SupportedTransform.NORMALIZE.value,
            },
        ]

    return InferenceModel(
        data_type=data_type or training_configuration.data_config.data_type,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        axes=axes or training_configuration.data_config.axes,
        mean=training_configuration.data_config.mean,
        std=training_configuration.data_config.std,
        transforms=transforms,
        tta_transforms=tta_transforms,
        batch_size=batch_size,
    )
