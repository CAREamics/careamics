"""Convenience functions to create configurations for training and inference."""

from typing import Any, Literal, Optional, Union

from .architectures import UNetModel
from .configuration_model import Configuration
from .data_model import DataConfig
from .fcn_algorithm_model import FCNAlgorithmConfig
from .support import (
    SupportedArchitecture,
    SupportedPixelManipulation,
    SupportedTransform,
)
from .training_model import TrainingConfig
from .transformations import (
    N2VManipulateModel,
    XYFlipModel,
    XYRandomRotate90Model,
)


def _list_augmentations(
    augmentations: Optional[list[Union[XYFlipModel, XYRandomRotate90Model]]],
) -> list[Union[XYFlipModel, XYRandomRotate90Model]]:
    """
    List the augmentations to apply.

    Parameters
    ----------
    augmentations : list of transforms, optional
        List of transforms to apply, either both or one of XYFlipModel and
        XYRandomRotate90Model.

    Returns
    -------
    list of transforms
        List of transforms to apply.

    Raises
    ------
    ValueError
        If the transforms are not XYFlipModel or XYRandomRotate90Model.
    ValueError
        If there are duplicate transforms.
    """
    if augmentations is None:
        transform_list: list[Union[XYFlipModel, XYRandomRotate90Model]] = [
            XYFlipModel(),
            XYRandomRotate90Model(),
        ]
    else:
        # throw error if not all transforms are pydantic models
        if not all(
            isinstance(t, XYFlipModel) or isinstance(t, XYRandomRotate90Model)
            for t in augmentations
        ):
            raise ValueError(
                "Accepted transforms are either XYFlipModel or "
                "XYRandomRotate90Model."
            )

        # check that there is no duplication
        aug_types = [t.__class__ for t in augmentations]
        if len(set(aug_types)) != len(aug_types):
            raise ValueError("Duplicate transforms are not allowed.")

        transform_list = augmentations

    return transform_list


def _create_unet_configuration(
    axes: str,
    n_channels_in: int,
    n_channels_out: int,
    independent_channels: bool,
    use_n2v2: bool,
    model_params: Optional[dict[str, Any]] = None,
) -> UNetModel:
    """
    Create a dictionary with the parameters of the UNet model.

    Parameters
    ----------
    axes : str
        Axes of the data.
    n_channels_in : int
        Number of input channels.
    n_channels_out : int
        Number of output channels.
    independent_channels : bool
        Whether to train all channels independently.
    use_n2v2 : bool
        Whether to use N2V2.
    model_params : dict
        UNetModel parameters.

    Returns
    -------
    UNetModel
        UNet model with the specified parameters.
    """
    if model_params is None:
        model_params = {}

    model_params["n2v2"] = use_n2v2
    model_params["conv_dims"] = 3 if "Z" in axes else 2
    model_params["in_channels"] = n_channels_in
    model_params["num_classes"] = n_channels_out
    model_params["independent_channels"] = independent_channels

    return UNetModel(
        architecture=SupportedArchitecture.UNET.value,
        **model_params,
    )


def _create_configuration(
    algorithm: Literal["n2v", "care", "n2n"],
    experiment_name: str,
    data_type: Literal["array", "tiff", "custom"],
    axes: str,
    patch_size: list[int],
    batch_size: int,
    num_epochs: int,
    augmentations: list[Union[XYFlipModel, XYRandomRotate90Model]],
    independent_channels: bool,
    loss: Literal["n2v", "mae", "mse"],
    n_channels_in: int,
    n_channels_out: int,
    logger: Literal["wandb", "tensorboard", "none"],
    use_n2v2: bool = False,
    model_params: Optional[dict] = None,
    dataloader_params: Optional[dict] = None,
) -> Configuration:
    """
    Create a configuration for training N2V, CARE or Noise2Noise.

    Parameters
    ----------
    algorithm : {"n2v", "care", "n2n"}
        Algorithm to use.
    experiment_name : str
        Name of the experiment.
    data_type : {"array", "tiff", "custom"}
        Type of the data.
    axes : str
        Axes of the data (e.g. SYX).
    patch_size : list of int
        Size of the patches along the spatial dimensions (e.g. [64, 64]).
    batch_size : int
        Batch size.
    num_epochs : int
        Number of epochs.
    augmentations : list of transforms
        List of transforms to apply, either both or one of XYFlipModel and
        XYRandomRotate90Model.
    independent_channels : bool
        Whether to train all channels independently.
    loss : {"n2v", "mae", "mse"}
        Loss function to use.
    n_channels_in : int
        Number of channels in.
    n_channels_out : int
        Number of channels out.
    logger : {"wandb", "tensorboard", "none"}
        Logger to use.
    use_n2v2 : bool, optional
        Whether to use N2V2, by default False.
    model_params : dict
        UNetModel parameters.
    dataloader_params : dict
        Parameters for the dataloader, see PyTorch notes, by default None.

    Returns
    -------
    Configuration
        Configuration for training N2V, CARE or Noise2Noise.
    """
    # model
    unet_model = _create_unet_configuration(
        axes=axes,
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
        independent_channels=independent_channels,
        use_n2v2=use_n2v2,
        model_params=model_params,
    )

    # algorithm model
    algorithm_config = FCNAlgorithmConfig(
        algorithm=algorithm,
        loss=loss,
        model=unet_model,
    )

    # data model
    data = DataConfig(
        data_type=data_type,
        axes=axes,
        patch_size=patch_size,
        batch_size=batch_size,
        transforms=augmentations,
        dataloader_params=dataloader_params,
    )

    # training model
    training = TrainingConfig(
        num_epochs=num_epochs,
        batch_size=batch_size,
        logger=None if logger == "none" else logger,
    )

    # create configuration
    configuration = Configuration(
        experiment_name=experiment_name,
        algorithm_config=algorithm_config,
        data_config=data,
        training_config=training,
    )

    return configuration


# TODO reconsider naming once we officially support LVAE approaches
def _create_supervised_configuration(
    algorithm: Literal["care", "n2n"],
    experiment_name: str,
    data_type: Literal["array", "tiff", "custom"],
    axes: str,
    patch_size: list[int],
    batch_size: int,
    num_epochs: int,
    augmentations: Optional[list[Union[XYFlipModel, XYRandomRotate90Model]]] = None,
    independent_channels: bool = True,
    loss: Literal["mae", "mse"] = "mae",
    n_channels_in: int = 1,
    n_channels_out: int = 1,
    logger: Literal["wandb", "tensorboard", "none"] = "none",
    model_params: Optional[dict] = None,
    dataloader_params: Optional[dict] = None,
) -> Configuration:
    """
    Create a configuration for training CARE or Noise2Noise.

    Parameters
    ----------
    algorithm : Literal["care", "n2n"]
        Algorithm to use.
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
    augmentations : list of transforms, default=None
        List of transforms to apply, either both or one of XYFlipModel and
        XYRandomRotate90Model. By default, it applies both XYFlip (on X and Y)
        and XYRandomRotate90 (in XY) to the images.
    independent_channels : bool, optional
        Whether to train all channels independently, by default False.
    loss : Literal["mae", "mse"], optional
        Loss function to use, by default "mae".
    n_channels_in : int, optional
        Number of channels in, by default 1.
    n_channels_out : int, optional
        Number of channels out, by default 1.
    logger : Literal["wandb", "tensorboard", "none"], optional
        Logger to use, by default "none".
    model_params : dict, optional
        UNetModel parameters, by default {}.
    dataloader_params : dict, optional
        Parameters for the dataloader, see PyTorch notes, by default None.

    Returns
    -------
    Configuration
        Configuration for training CARE or Noise2Noise.
    """
    # if there are channels, we need to specify their number
    if "C" in axes and n_channels_in == 1:
        raise ValueError(
            f"Number of channels in must be specified when using channels "
            f"(got {n_channels_in} channel)."
        )
    elif "C" not in axes and n_channels_in > 1:
        raise ValueError(
            f"C is not present in the axes, but number of channels is specified "
            f"(got {n_channels_in} channels)."
        )

    # augmentations
    transform_list = _list_augmentations(augmentations)

    return _create_configuration(
        algorithm=algorithm,
        experiment_name=experiment_name,
        data_type=data_type,
        axes=axes,
        patch_size=patch_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        augmentations=transform_list,
        independent_channels=independent_channels,
        loss=loss,
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
        logger=logger,
        model_params=model_params,
        dataloader_params=dataloader_params,
    )


def create_care_configuration(
    experiment_name: str,
    data_type: Literal["array", "tiff", "custom"],
    axes: str,
    patch_size: list[int],
    batch_size: int,
    num_epochs: int,
    augmentations: Optional[list[Union[XYFlipModel, XYRandomRotate90Model]]] = None,
    independent_channels: bool = True,
    loss: Literal["mae", "mse"] = "mae",
    n_channels_in: int = 1,
    n_channels_out: int = -1,
    logger: Literal["wandb", "tensorboard", "none"] = "none",
    model_params: Optional[dict] = None,
    dataloader_params: Optional[dict] = None,
) -> Configuration:
    """
    Create a configuration for training CARE.

    If "Z" is present in `axes`, then `path_size` must be a list of length 3, otherwise
    2.

    If "C" is present in `axes`, then you need to set `n_channels_in` to the number of
    channels. Likewise, if you set the number of channels, then "C" must be present in
    `axes`.

    To set the number of output channels, use the `n_channels_out` parameter. If it is
    not specified, it will be assumed to be equal to `n_channels_in`.

    By default, all channels are trained together. To train all channels independently,
    set `independent_channels` to True.

    By setting `augmentations` to `None`, the default transformations (flip in X and Y,
    rotations by 90 degrees in the XY plane) are applied. Rather than the default
    transforms, a list of transforms can be passed to the `augmentations` parameter. To
    disable the transforms, simply pass an empty list.

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
    augmentations : list of transforms, default=None
        List of transforms to apply, either both or one of XYFlipModel and
        XYRandomRotate90Model. By default, it applies both XYFlip (on X and Y)
        and XYRandomRotate90 (in XY) to the images.
    independent_channels : bool, optional
        Whether to train all channels independently, by default False.
    loss : Literal["mae", "mse"], optional
        Loss function to use, by default "mae".
    n_channels_in : int, optional
        Number of channels in, by default 1.
    n_channels_out : int, optional
        Number of channels out, by default -1.
    logger : Literal["wandb", "tensorboard", "none"], optional
        Logger to use, by default "none".
    model_params : dict, optional
        UNetModel parameters, by default None.
    dataloader_params : dict, optional
        Parameters for the dataloader, see PyTorch notes, by default None.

    Returns
    -------
    Configuration
        Configuration for training CARE.

    Examples
    --------
    Minimum example:
    >>> config = create_care_configuration(
    ...     experiment_name="care_experiment",
    ...     data_type="array",
    ...     axes="YX",
    ...     patch_size=[64, 64],
    ...     batch_size=32,
    ...     num_epochs=100
    ... )

    To disable transforms, simply set `augmentations` to an empty list:
    >>> config = create_care_configuration(
    ...     experiment_name="care_experiment",
    ...     data_type="array",
    ...     axes="YX",
    ...     patch_size=[64, 64],
    ...     batch_size=32,
    ...     num_epochs=100,
    ...     augmentations=[]
    ... )

    A list of transforms can be passed to the `augmentations` parameter to replace the
    default augmentations:
    >>> from careamics.config.transformations import XYFlipModel
    >>> config = create_care_configuration(
    ...     experiment_name="care_experiment",
    ...     data_type="array",
    ...     axes="YX",
    ...     patch_size=[64, 64],
    ...     batch_size=32,
    ...     num_epochs=100,
    ...     augmentations=[
    ...         # No rotation and only Y flipping
    ...         XYFlipModel(flip_x = False, flip_y = True)
    ...     ]
    ... )

    If you are training multiple channels they will be trained independently by default,
    you simply need to specify the number of channels input (and optionally, the number
    of channels output):
    >>> config = create_care_configuration(
    ...     experiment_name="care_experiment",
    ...     data_type="array",
    ...     axes="YXC", # channels must be in the axes
    ...     patch_size=[64, 64],
    ...     batch_size=32,
    ...     num_epochs=100,
    ...     n_channels_in=3, # number of input channels
    ...     n_channels_out=1 # if applicable
    ... )

    If instead you want to train multiple channels together, you need to turn off the
    `independent_channels` parameter:
    >>> config = create_care_configuration(
    ...     experiment_name="care_experiment",
    ...     data_type="array",
    ...     axes="YXC", # channels must be in the axes
    ...     patch_size=[64, 64],
    ...     batch_size=32,
    ...     num_epochs=100,
    ...     independent_channels=False,
    ...     n_channels_in=3,
    ...     n_channels_out=1 # if applicable
    ... )
    """
    if n_channels_out == -1:
        n_channels_out = n_channels_in

    return _create_supervised_configuration(
        algorithm="care",
        experiment_name=experiment_name,
        data_type=data_type,
        axes=axes,
        patch_size=patch_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        augmentations=augmentations,
        independent_channels=independent_channels,
        loss=loss,
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
        logger=logger,
        model_params=model_params,
        dataloader_params=dataloader_params,
    )


def create_n2n_configuration(
    experiment_name: str,
    data_type: Literal["array", "tiff", "custom"],
    axes: str,
    patch_size: list[int],
    batch_size: int,
    num_epochs: int,
    augmentations: Optional[list[Union[XYFlipModel, XYRandomRotate90Model]]] = None,
    independent_channels: bool = True,
    loss: Literal["mae", "mse"] = "mae",
    n_channels_in: int = 1,
    n_channels_out: int = -1,
    logger: Literal["wandb", "tensorboard", "none"] = "none",
    model_params: Optional[dict] = None,
    dataloader_params: Optional[dict] = None,
) -> Configuration:
    """
    Create a configuration for training Noise2Noise.

    If "Z" is present in `axes`, then `path_size` must be a list of length 3, otherwise
    2.

    If "C" is present in `axes`, then you need to set `n_channels_in` to the number of
    channels. Likewise, if you set the number of channels, then "C" must be present in
    `axes`.

    To set the number of output channels, use the `n_channels_out` parameter. If it is
    not specified, it will be assumed to be equal to `n_channels_in`.

    By default, all channels are trained together. To train all channels independently,
    set `independent_channels` to True.

    By setting `augmentations` to `None`, the default transformations (flip in X and Y,
    rotations by 90 degrees in the XY plane) are applied. Rather than the default
    transforms, a list of transforms can be passed to the `augmentations` parameter. To
    disable the transforms, simply pass an empty list.

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
    augmentations : list of transforms, default=None
        List of transforms to apply, either both or one of XYFlipModel and
        XYRandomRotate90Model. By default, it applies both XYFlip (on X and Y)
        and XYRandomRotate90 (in XY) to the images.
    independent_channels : bool, optional
        Whether to train all channels independently, by default False.
    loss : Literal["mae", "mse"], optional
        Loss function to use, by default "mae".
    n_channels_in : int, optional
        Number of channels in, by default 1.
    n_channels_out : int, optional
        Number of channels out, by default -1.
    logger : Literal["wandb", "tensorboard", "none"], optional
        Logger to use, by default "none".
    model_params : dict, optional
        UNetModel parameters, by default {}.
    dataloader_params : dict, optional
        Parameters for the dataloader, see PyTorch notes, by default None.

    Returns
    -------
    Configuration
        Configuration for training Noise2Noise.

    Examples
    --------
    Minimum example:
    >>> config = create_n2n_configuration(
    ...     experiment_name="n2n_experiment",
    ...     data_type="array",
    ...     axes="YX",
    ...     patch_size=[64, 64],
    ...     batch_size=32,
    ...     num_epochs=100
    ... )

    To disable transforms, simply set `augmentations` to an empty list:
    >>> config = create_n2n_configuration(
    ...     experiment_name="n2n_experiment",
    ...     data_type="array",
    ...     axes="YX",
    ...     patch_size=[64, 64],
    ...     batch_size=32,
    ...     num_epochs=100,
    ...     augmentations=[]
    ... )

    A list of transforms can be passed to the `augmentations` parameter to replace the
    default augmentations:
    >>> from careamics.config.transformations import XYFlipModel
    >>> config = create_n2n_configuration(
    ...     experiment_name="n2n_experiment",
    ...     data_type="array",
    ...     axes="YX",
    ...     patch_size=[64, 64],
    ...     batch_size=32,
    ...     num_epochs=100,
    ...     augmentations=[
    ...         # No rotation and only Y flipping
    ...         XYFlipModel(flip_x = False, flip_y = True)
    ...     ]
    ... )

    If you are training multiple channels they will be trained independently by default,
    you simply need to specify the number of channels input (and optionally, the number
    of channels output):
    >>> config = create_n2n_configuration(
    ...     experiment_name="n2n_experiment",
    ...     data_type="array",
    ...     axes="YXC", # channels must be in the axes
    ...     patch_size=[64, 64],
    ...     batch_size=32,
    ...     num_epochs=100,
    ...     n_channels_in=3, # number of input channels
    ...     n_channels_out=1 # if applicable
    ... )

    If instead you want to train multiple channels together, you need to turn off the
    `independent_channels` parameter:
    >>> config = create_n2n_configuration(
    ...     experiment_name="n2n_experiment",
    ...     data_type="array",
    ...     axes="YXC", # channels must be in the axes
    ...     patch_size=[64, 64],
    ...     batch_size=32,
    ...     num_epochs=100,
    ...     independent_channels=False,
    ...     n_channels_in=3,
    ...     n_channels_out=1 # if applicable
    ... )
    """
    if n_channels_out == -1:
        n_channels_out = n_channels_in

    return _create_supervised_configuration(
        algorithm="n2n",
        experiment_name=experiment_name,
        data_type=data_type,
        axes=axes,
        patch_size=patch_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        augmentations=augmentations,
        independent_channels=independent_channels,
        loss=loss,
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
        logger=logger,
        model_params=model_params,
        dataloader_params=dataloader_params,
    )


def create_n2v_configuration(
    experiment_name: str,
    data_type: Literal["array", "tiff", "custom"],
    axes: str,
    patch_size: list[int],
    batch_size: int,
    num_epochs: int,
    augmentations: Optional[list[Union[XYFlipModel, XYRandomRotate90Model]]] = None,
    independent_channels: bool = True,
    use_n2v2: bool = False,
    n_channels: int = 1,
    roi_size: int = 11,
    masked_pixel_percentage: float = 0.2,
    struct_n2v_axis: Literal["horizontal", "vertical", "none"] = "none",
    struct_n2v_span: int = 5,
    logger: Literal["wandb", "tensorboard", "none"] = "none",
    model_params: Optional[dict] = None,
    dataloader_params: Optional[dict] = None,
) -> Configuration:
    """
    Create a configuration for training Noise2Void.

    N2V uses a UNet model to denoise images in a self-supervised manner. To use its
    variants structN2V and N2V2, set the `struct_n2v_axis` and `struct_n2v_span`
    (structN2V) parameters, or set `use_n2v2` to True (N2V2).

    N2V2 modifies the UNet architecture by adding blur pool layers and removes the skip
    connections, thus removing checkboard artefacts. StructN2V is used when vertical
    or horizontal correlations are present in the noise; it applies an additional mask
    to the manipulated pixel neighbors.

    If "Z" is present in `axes`, then `path_size` must be a list of length 3, otherwise
    2.

    If "C" is present in `axes`, then you need to set `n_channels` to the number of
    channels.

    By default, all channels are trained independently. To train all channels together,
    set `independent_channels` to False.

    By default, the transformations applied are a random flip along X or Y, and a random
    90 degrees rotation in the XY plane. Normalization is always applied, as well as the
    N2V manipulation.

    By setting `augmentations` to `None`, the default transformations (flip in X and Y,
    rotations by 90 degrees in the XY plane) are applied. Rather than the default
    transforms, a list of transforms can be passed to the `augmentations` parameter. To
    disable the transforms, simply pass an empty list.

    The `roi_size` parameter specifies the size of the area around each pixel that will
    be manipulated by N2V. The `masked_pixel_percentage` parameter specifies how many
    pixels per patch will be manipulated.

    The parameters of the UNet can be specified in the `model_params` (passed as a
    parameter-value dictionary). Note that `use_n2v2` and 'n_channels' override the
    corresponding parameters passed in `model_params`.

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
    augmentations : list of transforms, default=None
        List of transforms to apply, either both or one of XYFlipModel and
        XYRandomRotate90Model. By default, it applies both XYFlip (on X and Y)
        and XYRandomRotate90 (in XY) to the images.
    independent_channels : bool, optional
        Whether to train all channels together, by default True.
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
    model_params : dict, optional
        UNetModel parameters, by default None.
    dataloader_params : dict, optional
        Parameters for the dataloader, see PyTorch notes, by default None.

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

    To disable transforms, simply set `augmentations` to an empty list:
    >>> config = create_n2v_configuration(
    ...     experiment_name="n2v_experiment",
    ...     data_type="array",
    ...     axes="YX",
    ...     patch_size=[64, 64],
    ...     batch_size=32,
    ...     num_epochs=100,
    ...     augmentations=[]
    ... )

    A list of transforms can be passed to the `augmentations` parameter:
    >>> from careamics.config.transformations import XYFlipModel
    >>> config = create_n2v_configuration(
    ...     experiment_name="n2v_experiment",
    ...     data_type="array",
    ...     axes="YX",
    ...     patch_size=[64, 64],
    ...     batch_size=32,
    ...     num_epochs=100,
    ...     augmentations=[
    ...         # No rotation and only Y flipping
    ...         XYFlipModel(flip_x = False, flip_y = True)
    ...     ]
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

    If you are training multiple channels they will be trained independently by default,
    you simply need to specify the number of channels:
    >>> config = create_n2v_configuration(
    ...     experiment_name="n2v_experiment",
    ...     data_type="array",
    ...     axes="YXC",
    ...     patch_size=[64, 64],
    ...     batch_size=32,
    ...     num_epochs=100,
    ...     n_channels=3
    ... )

    If instead you want to train multiple channels together, you need to turn off the
    `independent_channels` parameter:
    >>> config = create_n2v_configuration(
    ...     experiment_name="n2v_experiment",
    ...     data_type="array",
    ...     axes="YXC",
    ...     patch_size=[64, 64],
    ...     batch_size=32,
    ...     num_epochs=100,
    ...     independent_channels=False,
    ...     n_channels=3
    ... )
    """
    # if there are channels, we need to specify their number
    if "C" in axes and n_channels == 1:
        raise ValueError(
            f"Number of channels must be specified when using channels "
            f"(got {n_channels} channel)."
        )
    elif "C" not in axes and n_channels > 1:
        raise ValueError(
            f"C is not present in the axes, but number of channels is specified "
            f"(got {n_channels} channel)."
        )

    # augmentations
    transform_list = _list_augmentations(augmentations)

    # create the N2VManipulate transform using the supplied parameters
    n2v_transform = N2VManipulateModel(
        name=SupportedTransform.N2V_MANIPULATE.value,
        strategy=(
            SupportedPixelManipulation.MEDIAN.value
            if use_n2v2
            else SupportedPixelManipulation.UNIFORM.value
        ),
        roi_size=roi_size,
        masked_pixel_percentage=masked_pixel_percentage,
        struct_mask_axis=struct_n2v_axis,
        struct_mask_span=struct_n2v_span,
    )
    transform_list.append(n2v_transform)

    return _create_configuration(
        algorithm="n2v",
        experiment_name=experiment_name,
        data_type=data_type,
        axes=axes,
        patch_size=patch_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        augmentations=transform_list,
        independent_channels=independent_channels,
        loss="n2v",
        use_n2v2=use_n2v2,
        n_channels_in=n_channels,
        n_channels_out=n_channels,
        logger=logger,
        model_params=model_params,
        dataloader_params=dataloader_params,
    )
