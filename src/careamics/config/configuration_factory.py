from typing import List, Literal, Optional

from .algorithm_model import AlgorithmModel
from .architectures import UNetModel
from .configuration_model import Configuration
from .data_model import DataModel
from .prediction_model import InferenceModel
from .support import (
    SupportedAlgorithm,
    SupportedArchitecture,
    SupportedLoss,
    SupportedPixelManipulation,
    SupportedTransform,
)
from .training_model import TrainingModel


def create_n2n_training_configuration(
    experiment_name: str,
    data_type: Literal["array", "tiff", "custom"],
    axes: str,
    patch_size: List[int],
    batch_size: int,
    num_epochs: int,
    use_augmentations: bool = True,
    use_n2v2: bool = False,
    n_channels: int = 1,
    roi_size: int = 11,
    masked_pixel_percentage: float = 0.2,
    struct_n2v_axis: Literal["horizontal", "vertical", "none"] = "none",
    struct_n2v_span: int= 5,
    model_kwargs: dict = {},
) -> Configuration:
    """Create a configuration for training N2V.

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
    data_type : Literal[&quot;array&quot;, &quot;tiff&quot;, &quot;custom&quot;]
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
        Whether to use N2V2, by default False
    n_channels : int, optional
        Number of channels (in and out), by default 1.
    roi_size : int, optional
        N2V pixel manipulation area, by default 11.
    masked_pixel_percentage : float, optional
        Percentage of pixels masked in each patch, by default 0.2.
    struct_n2v_axis : Literal[&quot;horizontal&quot;, &quot;vertical&quot;, &quot;none&quot;], optional
        Axis along which to apply structN2V mask, by default "none"
    struct_n2v_span : int, optional
        Span of the structN2V mask, by default 5
    model_kwargs : dict, optional
        UNetModel parameters, by default {}

    Returns
    -------
    Configuration
        Configuration for training N2V.
    """
    # model
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
        transforms = [
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
        "parameters": {
            "strategy": SupportedPixelManipulation.MEDIAN.value if use_n2v2 \
                        else SupportedPixelManipulation.UNIFORM.value,
            "roi_size": roi_size,
            "masked_pixel_percentage": masked_pixel_percentage,
            "struct_mask_axis": struct_n2v_axis,
            "struct_mask_span": struct_n2v_span,
        }
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
    )

    # create configuration
    configuration = Configuration(
        experiment_name=experiment_name,
        algorithm=algorithm,
        data=data,
        training=training,
    )

    return configuration


def create_n2v_training_configuration(
    experiment_name: str,
    data_type: Literal["array", "tiff", "custom"],
    axes: str,
    patch_size: List[int],
    batch_size: int,
    num_epochs: int,
    use_augmentations: bool = True,
    use_n2v2: bool = False,
    n_channels: int = 1,
    roi_size: int = 11,
    masked_pixel_percentage: float = 0.2,
    struct_n2v_axis: Literal["horizontal", "vertical", "none"] = "none",
    struct_n2v_span: int= 5,
    model_kwargs: dict = {},
) -> Configuration:
    """Create a configuration for training N2V.

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
    data_type : Literal[&quot;array&quot;, &quot;tiff&quot;, &quot;custom&quot;]
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
        Whether to use N2V2, by default False
    n_channels : int, optional
        Number of channels (in and out), by default 1.
    roi_size : int, optional
        N2V pixel manipulation area, by default 11.
    masked_pixel_percentage : float, optional
        Percentage of pixels masked in each patch, by default 0.2.
    struct_n2v_axis : Literal[&quot;horizontal&quot;, &quot;vertical&quot;, &quot;none&quot;], optional
        Axis along which to apply structN2V mask, by default "none"
    struct_n2v_span : int, optional
        Span of the structN2V mask, by default 5
    model_kwargs : dict, optional
        UNetModel parameters, by default {}

    Returns
    -------
    Configuration
        Configuration for training N2V.
    """
    # model
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
        transforms = [
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
        "parameters": {
            "strategy": SupportedPixelManipulation.MEDIAN.value if use_n2v2 \
                        else SupportedPixelManipulation.UNIFORM.value,
            "roi_size": roi_size,
            "masked_pixel_percentage": masked_pixel_percentage,
            "struct_mask_axis": struct_n2v_axis,
            "struct_mask_span": struct_n2v_span,
        }
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
    )

    # create configuration
    configuration = Configuration(
        experiment_name=experiment_name,
        algorithm=algorithm,
        data=data,
        training=training,
    )

    return configuration


def create_inference_configuration(
    training_configuration: Configuration,
    tile_size: List[int],
    tile_overlap: Optional[List[int]] = None,
    data_type: Optional[Literal["array", "tiff", "custom"]] = None,
    axes: Optional[str] = None,
    transforms: Optional[List] = None,
    tta_transforms: Optional[bool] = True,
    batch_size: Optional[int] = 1,
    extension_filter: Optional[str] = "",
) -> InferenceModel:
    """Create a configuration for inference with N2V.

    By default all parameters, except `tile_size`, `tile_overlap`, and
    `extension_filter`, are taken from the training configuration.

    Parameters
    ----------
    training_configuration : Configuration
        Configuration used for training.
    data_type : str, optional
        Type of the data, by default "tiff".
    axes : str, optional
        Axes of the data, by default "YX".
    tile_size : List[int], optional
        Size of the tiles.
    tile_overlap : List[int], optional
        Overlap of the tiles.
    batch_size : int, optional
        Batch size, by default 1.
    extension_filter : str, optional
        Filter for the extensions, by default "".

    Returns
    -------
    InferenceConfiguration
        Configuration for inference with N2V.
    """
    if data_type is None:
        try:
            data_type = training_configuration.data.data_type
        except AttributeError as e:
            raise ValueError("data_type must be provided.") from e

    if axes is None:
        try:
            axes = training_configuration.data.axes
        except AttributeError as e:
            raise ValueError("axes must be provided.") from e


    return InferenceModel(
        data_type=data_type,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        axes=axes,
        transforms=transforms,
        tta_transforms=tta_transforms,
        batch_size=batch_size,
        extension_filter=extension_filter,
    )



