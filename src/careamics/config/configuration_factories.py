"""Convenience functions to create configurations for training and inference."""

from collections.abc import Sequence
from typing import Annotated, Any, Literal, Union

from pydantic import Field, TypeAdapter

from careamics.config.algorithms import (
    CAREAlgorithm,
    MicroSplitAlgorithm,
    N2NAlgorithm,
    N2VAlgorithm,
)
from careamics.config.architectures import LVAEModel, UNetModel
from careamics.config.data import DataConfig, NGDataConfig
from careamics.config.likelihood_model import (
    GaussianLikelihoodConfig,
    NMLikelihoodConfig,
)
from careamics.config.loss_model import LVAELossConfig
from careamics.config.nm_model import GaussianMixtureNMConfig, MultiChannelNMConfig
from careamics.config.support import (
    SupportedArchitecture,
    SupportedPixelManipulation,
    SupportedTransform,
)
from careamics.config.training_model import TrainingConfig
from careamics.config.transformations import (
    SPATIAL_TRANSFORMS_UNION,
    N2VManipulateModel,
    XYFlipModel,
    XYRandomRotate90Model,
)
from careamics.lvae_training.dataset.config import MicroSplitDataConfig

from .configuration import Configuration


def algorithm_factory(
    algorithm: dict[str, Any],
) -> Union[N2VAlgorithm, N2NAlgorithm, CAREAlgorithm]:
    """
    Create an algorithm model for training CAREamics.

    Parameters
    ----------
    algorithm : dict
        Algorithm dictionary.

    Returns
    -------
    N2VAlgorithm or N2NAlgorithm or CAREAlgorithm
        Algorithm model for training CAREamics.
    """
    adapter: TypeAdapter = TypeAdapter(
        Annotated[
            Union[N2VAlgorithm, N2NAlgorithm, CAREAlgorithm],
            Field(discriminator="algorithm"),
        ]
    )
    return adapter.validate_python(algorithm)


def _list_spatial_augmentations(
    augmentations: list[SPATIAL_TRANSFORMS_UNION] | None = None,
) -> list[SPATIAL_TRANSFORMS_UNION]:
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
        transform_list: list[SPATIAL_TRANSFORMS_UNION] = [
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
    model_params: dict[str, Any] | None = None,
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


def _create_algorithm_configuration(
    axes: str,
    algorithm: Literal["n2v", "care", "n2n"],
    loss: Literal["n2v", "mae", "mse"],
    independent_channels: bool,
    n_channels_in: int,
    n_channels_out: int,
    use_n2v2: bool = False,
    model_params: dict | None = None,
    optimizer: Literal["Adam", "Adamax", "SGD"] = "Adam",
    optimizer_params: dict[str, Any] | None = None,
    lr_scheduler: Literal["ReduceLROnPlateau", "StepLR"] = "ReduceLROnPlateau",
    lr_scheduler_params: dict[str, Any] | None = None,
) -> dict:
    """
    Create a dictionary with the parameters of the algorithm model.

    Parameters
    ----------
    axes : str
        Axes of the data.
    algorithm : {"n2v", "care", "n2n"}
        Algorithm to use.
    loss : {"n2v", "mae", "mse"}
        Loss function to use.
    independent_channels : bool
        Whether to train all channels independently.
    n_channels_in : int
        Number of input channels.
    n_channels_out : int
        Number of output channels.
    use_n2v2 : bool, default=false
        Whether to use N2V2.
    model_params : dict, default=None
        UNetModel parameters.
    optimizer : {"Adam", "Adamax", "SGD"}, default="Adam"
        Optimizer to use.
    optimizer_params : dict, default=None
        Parameters for the optimizer, see PyTorch documentation for more details.
    lr_scheduler : {"ReduceLROnPlateau", "StepLR"}, default="ReduceLROnPlateau"
        Learning rate scheduler to use.
    lr_scheduler_params : dict, default=None
        Parameters for the learning rate scheduler, see PyTorch documentation for more
        details.


    Returns
    -------
    dict
        Algorithm model as dictionnary with the specified parameters.
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

    return {
        "algorithm": algorithm,
        "loss": loss,
        "model": unet_model,
        "optimizer": {
            "name": optimizer,
            "parameters": {} if optimizer_params is None else optimizer_params,
        },
        "lr_scheduler": {
            "name": lr_scheduler,
            "parameters": {} if lr_scheduler_params is None else lr_scheduler_params,
        },
    }


def _create_data_configuration(
    data_type: Literal["array", "tiff", "czi", "custom"],
    axes: str,
    patch_size: Sequence[int],
    batch_size: int,
    augmentations: Union[list[SPATIAL_TRANSFORMS_UNION]],
    train_dataloader_params: dict[str, Any] | None = None,
    val_dataloader_params: dict[str, Any] | None = None,
) -> DataConfig:
    """
    Create a dictionary with the parameters of the data model.

    Parameters
    ----------
    data_type : {"array", "tiff", "czi", "custom"}
        Type of the data.
    axes : str
        Axes of the data.
    patch_size : list of int
        Size of the patches along the spatial dimensions.
    batch_size : int
        Batch size.
    augmentations : list of transforms
        List of transforms to apply.
    train_dataloader_params : dict
        Parameters for the training dataloader, see PyTorch notes, by default None.
    val_dataloader_params : dict
        Parameters for the validation dataloader, see PyTorch notes, by default None.

    Returns
    -------
    DataConfig
        Data model with the specified parameters.
    """
    # data model
    data = {
        "data_type": data_type,
        "axes": axes,
        "patch_size": patch_size,
        "batch_size": batch_size,
        "transforms": augmentations,
    }
    # Don't override defaults set in DataConfig class
    if train_dataloader_params is not None:
        # DataConfig enforces the presence of `shuffle` key in the dataloader parameters
        if "shuffle" not in train_dataloader_params:
            train_dataloader_params["shuffle"] = True

        data["train_dataloader_params"] = train_dataloader_params

    if val_dataloader_params is not None:
        data["val_dataloader_params"] = val_dataloader_params

    return DataConfig(**data)


def _create_microsplit_data_configuration(
    data_type: Literal["array", "tiff", "custom"],
    axes: str,
    patch_size: Sequence[int],
    grid_size: int,
    multiscale_count: int,
    batch_size: int,
    augmentations: Union[list[SPATIAL_TRANSFORMS_UNION]],
    train_dataloader_params: dict[str, Any] | None = None,
    val_dataloader_params: dict[str, Any] | None = None,
) -> DataConfig:
    """
    Create a dictionary with the parameters of the data model.

    Parameters
    ----------
    data_type : {"array", "tiff", "czi", "custom"}
        Type of the data.
    axes : str
        Axes of the data.
    patch_size : list of int
        Size of the patches along the spatial dimensions.
    grid_size : int
        Grid size for patch extraction.
    multiscale_count : int
        Number of LC scales.
    batch_size : int
        Batch size.
    augmentations : list of transforms
        List of transforms to apply.
    train_dataloader_params : dict
        Parameters for the training dataloader, see PyTorch notes, by default None.
    val_dataloader_params : dict
        Parameters for the validation dataloader, see PyTorch notes, by default None.

    Returns
    -------
    DataConfig
        Data model with the specified parameters.
    """
    # data model
    data = {
        "data_type": data_type,
        "axes": axes,
        "image_size": patch_size,
        "grid_size": grid_size,
        "multiscale_lowres_count": multiscale_count,
        "batch_size": batch_size,
        "transforms": augmentations,
    }
    # Don't override defaults set in DataConfig class
    if train_dataloader_params is not None:
        # DataConfig enforces the presence of `shuffle` key in the dataloader parameters
        if "shuffle" not in train_dataloader_params:
            train_dataloader_params["shuffle"] = True

        data["train_dataloader_params"] = train_dataloader_params

    if val_dataloader_params is not None:
        data["val_dataloader_params"] = val_dataloader_params

    return MicroSplitDataConfig(**data)


def _create_ng_data_configuration(
    data_type: Literal["array", "tiff", "custom"],
    axes: str,
    patch_size: Sequence[int],
    batch_size: int,
    augmentations: list[SPATIAL_TRANSFORMS_UNION],
    patch_overlaps: Sequence[int] | None = None,
    train_dataloader_params: dict[str, Any] | None = None,
    val_dataloader_params: dict[str, Any] | None = None,
    test_dataloader_params: dict[str, Any] | None = None,
    seed: int | None = None,
) -> NGDataConfig:
    """
    Create a dictionary with the parameters of the data model.

    Parameters
    ----------
    data_type : {"array", "tiff", "custom"}
        Type of the data.
    axes : str
        Axes of the data.
    patch_size : list of int
        Size of the patches along the spatial dimensions.
    batch_size : int
        Batch size.
    augmentations : list of transforms
        List of transforms to apply.
    patch_overlaps : Sequence of int, default=None
        Overlaps between patches in each spatial dimension, only used with "sequential"
        patching. If `None`, no overlap is applied. The overlap must be smaller than
        the patch size in each spatial dimension, and the number of dimensions be either
        2 or 3.
    train_dataloader_params : dict
        Parameters for the training dataloader, see PyTorch notes, by default None.
    val_dataloader_params : dict
        Parameters for the validation dataloader, see PyTorch notes, by default None.
    test_dataloader_params : dict
        Parameters for the test dataloader, see PyTorch notes, by default None.
    seed : int, default=None
        Random seed for reproducibility. If `None`, no seed is set.

    Returns
    -------
    NGDataConfig
        Next-Generation Data model with the specified parameters.
    """
    # data model
    data = {
        "data_type": data_type,
        "axes": axes,
        "batch_size": batch_size,
        "transforms": augmentations,
        "seed": seed,
    }
    # don't override defaults set in DataConfig class
    if train_dataloader_params is not None:
        # the presence of `shuffle` key in the dataloader parameters is enforced
        # by the NGDataConfig class
        if "shuffle" not in train_dataloader_params:
            train_dataloader_params["shuffle"] = True

        data["train_dataloader_params"] = train_dataloader_params

    if val_dataloader_params is not None:
        data["val_dataloader_params"] = val_dataloader_params

    if test_dataloader_params is not None:
        data["test_dataloader_params"] = test_dataloader_params

    # add training patching
    data["patching"] = {
        "name": "random",
        "patch_size": patch_size,
        "overlaps": patch_overlaps,
    }

    return NGDataConfig(**data)


def _create_training_configuration(
    trainer_params: dict,
    logger: Literal["wandb", "tensorboard", "none"],
    checkpoint_params: dict[str, Any] | None = None,
) -> TrainingConfig:
    """
    Create a dictionary with the parameters of the training model.

    Parameters
    ----------
    trainer_params : dict
        Parameters for Lightning Trainer class, see PyTorch Lightning documentation.
    logger : {"wandb", "tensorboard", "none"}
        Logger to use.
    checkpoint_params : dict, default=None
        Parameters for the checkpoint callback, see PyTorch Lightning documentation
        (`ModelCheckpoint`) for the list of available parameters.

    Returns
    -------
    TrainingConfig
        Training model with the specified parameters.
    """
    return TrainingConfig(
        lightning_trainer_config=trainer_params,
        logger=None if logger == "none" else logger,
        checkpoint_callback={} if checkpoint_params is None else checkpoint_params,
    )


# TODO reconsider naming once we officially support LVAE approaches
def _create_supervised_config_dict(
    algorithm: Literal["care", "n2n"],
    experiment_name: str,
    data_type: Literal["array", "tiff", "czi", "custom"],
    axes: str,
    patch_size: Sequence[int],
    batch_size: int,
    trainer_params: dict | None = None,
    augmentations: list[SPATIAL_TRANSFORMS_UNION] | None = None,
    independent_channels: bool = True,
    loss: Literal["mae", "mse"] = "mae",
    n_channels_in: int | None = None,
    n_channels_out: int | None = None,
    logger: Literal["wandb", "tensorboard", "none"] = "none",
    model_params: dict | None = None,
    optimizer: Literal["Adam", "Adamax", "SGD"] = "Adam",
    optimizer_params: dict[str, Any] | None = None,
    lr_scheduler: Literal["ReduceLROnPlateau", "StepLR"] = "ReduceLROnPlateau",
    lr_scheduler_params: dict[str, Any] | None = None,
    train_dataloader_params: dict[str, Any] | None = None,
    val_dataloader_params: dict[str, Any] | None = None,
    checkpoint_params: dict[str, Any] | None = None,
    num_epochs: int | None = None,
    num_steps: int | None = None,
) -> dict:
    """
    Create a configuration for training CARE or Noise2Noise.

    Parameters
    ----------
    algorithm : Literal["care", "n2n"]
        Algorithm to use.
    experiment_name : str
        Name of the experiment.
    data_type : Literal["array", "tiff", "czi", "custom"]
        Type of the data.
    axes : str
        Axes of the data (e.g. SYX).
    patch_size : List[int]
        Size of the patches along the spatial dimensions (e.g. [64, 64]).
    batch_size : int
        Batch size.
    trainer_params : dict
        Parameters for the training configuration.
    augmentations : list of transforms, default=None
        List of transforms to apply, either both or one of XYFlipModel and
        XYRandomRotate90Model. By default, it applies both XYFlip (on X and Y)
        and XYRandomRotate90 (in XY) to the images.
    independent_channels : bool, optional
        Whether to train all channels independently, by default False.
    loss : Literal["mae", "mse"], optional
        Loss function to use, by default "mae".
    n_channels_in : int or None, default=None
        Number of channels in.
    n_channels_out : int or None, default=None
        Number of channels out.
    logger : Literal["wandb", "tensorboard", "none"], optional
        Logger to use, by default "none".
    model_params : dict, default=None
        UNetModel parameters.
    optimizer : {"Adam", "Adamax", "SGD"}, default="Adam"
        Optimizer to use.
    optimizer_params : dict, default=None
        Parameters for the optimizer, see PyTorch documentation for more details.
    lr_scheduler : {"ReduceLROnPlateau", "StepLR"}, default="ReduceLROnPlateau"
        Learning rate scheduler to use.
    lr_scheduler_params : dict, default=None
        Parameters for the learning rate scheduler, see PyTorch documentation for more
        details.
    train_dataloader_params : dict
        Parameters for the training dataloader, see PyTorch notes, by default None.
    val_dataloader_params : dict
        Parameters for the validation dataloader, see PyTorch notes, by default None.
    checkpoint_params : dict, default=None
        Parameters for the checkpoint callback, see PyTorch Lightning documentation
        (`ModelCheckpoint`) for the list of available parameters.
    num_epochs : int or None, default=None
        Number of epochs to train for. If provided, this will be added to
        trainer_params.
    num_steps : int or None, default=None
        Number of batches in 1 epoch. If provided, this will be added to trainer_params.
        Translates to `limit_train_batches` in PyTorch Lightning Trainer. See relevant
        documentation for more details.

    Returns
    -------
    Configuration
        Configuration for training CARE or Noise2Noise.

    Raises
    ------
    ValueError
        If the number of channels is not specified when using channels.
    ValueError
        If the number of channels is specified but "C" is not in the axes.
    """
    # if there are channels, we need to specify their number
    if "C" in axes and n_channels_in is None:
        raise ValueError("Number of channels in must be specified when using channels ")
    elif "C" not in axes and (n_channels_in is not None and n_channels_in > 1):
        raise ValueError(
            f"C is not present in the axes, but number of channels is specified "
            f"(got {n_channels_in} channels)."
        )

    if n_channels_in is None:
        n_channels_in = 1

    if n_channels_out is None:
        n_channels_out = n_channels_in

    # augmentations
    spatial_transform_list = _list_spatial_augmentations(augmentations)

    # algorithm
    algorithm_params = _create_algorithm_configuration(
        axes=axes,
        algorithm=algorithm,
        loss=loss,
        independent_channels=independent_channels,
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
        model_params=model_params,
        optimizer=optimizer,
        optimizer_params=optimizer_params,
        lr_scheduler=lr_scheduler,
        lr_scheduler_params=lr_scheduler_params,
    )

    # data
    data_params = _create_data_configuration(
        data_type=data_type,
        axes=axes,
        patch_size=patch_size,
        batch_size=batch_size,
        augmentations=spatial_transform_list,
        train_dataloader_params=train_dataloader_params,
        val_dataloader_params=val_dataloader_params,
    )

    # Handle trainer parameters with num_epochs and num_steps
    final_trainer_params = {} if trainer_params is None else trainer_params.copy()

    # Add num_epochs and num_steps if provided
    if num_epochs is not None:
        final_trainer_params["max_epochs"] = num_epochs
    if num_steps is not None:
        final_trainer_params["limit_train_batches"] = num_steps

    # training
    training_params = _create_training_configuration(
        trainer_params=final_trainer_params,
        logger=logger,
        checkpoint_params=checkpoint_params,
    )

    return {
        "experiment_name": experiment_name,
        "algorithm_config": algorithm_params,
        "data_config": data_params,
        "training_config": training_params,
    }


def create_care_configuration(
    experiment_name: str,
    data_type: Literal["array", "tiff", "czi", "custom"],
    axes: str,
    patch_size: Sequence[int],
    batch_size: int,
    num_epochs: int = 100,
    num_steps: int | None = None,
    augmentations: list[Union[XYFlipModel, XYRandomRotate90Model]] | None = None,
    independent_channels: bool = True,
    loss: Literal["mae", "mse"] = "mae",
    n_channels_in: int | None = None,
    n_channels_out: int | None = None,
    logger: Literal["wandb", "tensorboard", "none"] = "none",
    trainer_params: dict | None = None,
    model_params: dict | None = None,
    optimizer: Literal["Adam", "Adamax", "SGD"] = "Adam",
    optimizer_params: dict[str, Any] | None = None,
    lr_scheduler: Literal["ReduceLROnPlateau", "StepLR"] = "ReduceLROnPlateau",
    lr_scheduler_params: dict[str, Any] | None = None,
    train_dataloader_params: dict[str, Any] | None = None,
    val_dataloader_params: dict[str, Any] | None = None,
    checkpoint_params: dict[str, Any] | None = None,
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
    data_type : Literal["array", "tiff", "czi", "custom"]
        Type of the data.
    axes : str
        Axes of the data (e.g. SYX).
    patch_size : List[int]
        Size of the patches along the spatial dimensions (e.g. [64, 64]).
    batch_size : int
        Batch size.
    num_epochs : int, default=100
        Number of epochs to train for. If provided, this will be added to
        trainer_params.
    num_steps : int, optional
        Number of batches in 1 epoch. If provided, this will be added to trainer_params.
        Translates to `limit_train_batches` in PyTorch Lightning Trainer. See relevant
        documentation for more details.
    augmentations : list of transforms, default=None
        List of transforms to apply, either both or one of XYFlipModel and
        XYRandomRotate90Model. By default, it applies both XYFlip (on X and Y)
        and XYRandomRotate90 (in XY) to the images.
    independent_channels : bool, optional
        Whether to train all channels independently, by default False.
    loss : Literal["mae", "mse"], default="mae"
        Loss function to use.
    n_channels_in : int or None, default=None
        Number of channels in.
    n_channels_out : int or None, default=None
        Number of channels out.
    logger : Literal["wandb", "tensorboard", "none"], default="none"
        Logger to use.
    trainer_params : dict, optional
        Parameters for the trainer class, see PyTorch Lightning documentation.
    model_params : dict, default=None
        UNetModel parameters.
    optimizer : Literal["Adam", "Adamax", "SGD"], default="Adam"
        Optimizer to use.
    optimizer_params : dict, default=None
        Parameters for the optimizer, see PyTorch documentation for more details.
    lr_scheduler : Literal["ReduceLROnPlateau", "StepLR"], default="ReduceLROnPlateau"
        Learning rate scheduler to use.
    lr_scheduler_params : dict, default=None
        Parameters for the learning rate scheduler, see PyTorch documentation for more
        details.
    train_dataloader_params : dict, optional
        Parameters for the training dataloader, see the PyTorch docs for `DataLoader`.
        If left as `None`, the dict `{"shuffle": True}` will be used, this is set in
        the `GeneralDataConfig`.
    val_dataloader_params : dict, optional
        Parameters for the validation dataloader, see PyTorch the docs for `DataLoader`.
        If left as `None`, the empty dict `{}` will be used, this is set in the
        `GeneralDataConfig`.
    checkpoint_params : dict, default=None
        Parameters for the checkpoint callback, see PyTorch Lightning documentation
        (`ModelCheckpoint`) for the list of available parameters.

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

    You can also limit the number of batches per epoch:
    >>> config = create_care_configuration(
    ...     experiment_name="care_experiment",
    ...     data_type="array",
    ...     axes="YX",
    ...     patch_size=[64, 64],
    ...     batch_size=32,
    ...     num_steps=100  # limit to 100 batches per epoch
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

    If you would like to train on CZI files, use `"czi"` as `data_type` and `"SCYX"` as
    `axes` for 2-D or `"SCZYX"` for 3-D denoising. Note that `"SCYX"` can also be used
    for 3-D data but spatial context along the Z dimension will then not be taken into
    account.
    >>> config_2d = create_care_configuration(
    ...     experiment_name="care_experiment",
    ...     data_type="czi",
    ...     axes="SCYX",
    ...     patch_size=[64, 64],
    ...     batch_size=32,
    ...     num_epochs=100,
    ...     n_channels_in=1,
    ... )
    >>> config_3d = create_care_configuration(
    ...     experiment_name="care_experiment",
    ...     data_type="czi",
    ...     axes="SCZYX",
    ...     patch_size=[16, 64, 64],
    ...     batch_size=16,
    ...     num_epochs=100,
    ...     n_channels_in=1,
    ... )
    """
    return Configuration(
        **_create_supervised_config_dict(
            algorithm="care",
            experiment_name=experiment_name,
            data_type=data_type,
            axes=axes,
            patch_size=patch_size,
            batch_size=batch_size,
            augmentations=augmentations,
            independent_channels=independent_channels,
            loss=loss,
            n_channels_in=n_channels_in,
            n_channels_out=n_channels_out,
            logger=logger,
            trainer_params=trainer_params,
            model_params=model_params,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            train_dataloader_params=train_dataloader_params,
            val_dataloader_params=val_dataloader_params,
            checkpoint_params=checkpoint_params,
            num_epochs=num_epochs,
            num_steps=num_steps,
        )
    )


def create_n2n_configuration(
    experiment_name: str,
    data_type: Literal["array", "tiff", "czi", "custom"],
    axes: str,
    patch_size: Sequence[int],
    batch_size: int,
    num_epochs: int = 100,
    num_steps: int | None = None,
    augmentations: list[Union[XYFlipModel, XYRandomRotate90Model]] | None = None,
    independent_channels: bool = True,
    loss: Literal["mae", "mse"] = "mae",
    n_channels_in: int | None = None,
    n_channels_out: int | None = None,
    logger: Literal["wandb", "tensorboard", "none"] = "none",
    trainer_params: dict | None = None,
    model_params: dict | None = None,
    optimizer: Literal["Adam", "Adamax", "SGD"] = "Adam",
    optimizer_params: dict[str, Any] | None = None,
    lr_scheduler: Literal["ReduceLROnPlateau", "StepLR"] = "ReduceLROnPlateau",
    lr_scheduler_params: dict[str, Any] | None = None,
    train_dataloader_params: dict[str, Any] | None = None,
    val_dataloader_params: dict[str, Any] | None = None,
    checkpoint_params: dict[str, Any] | None = None,
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
    data_type : Literal["array", "tiff", "czi", "custom"]
        Type of the data.
    axes : str
        Axes of the data (e.g. SYX).
    patch_size : List[int]
        Size of the patches along the spatial dimensions (e.g. [64, 64]).
    batch_size : int
        Batch size.
    num_epochs : int, default=100
        Number of epochs to train for. If provided, this will be added to
        trainer_params.
    num_steps : int, optional
        Number of batches in 1 epoch. If provided, this will be added to trainer_params.
        Translates to `limit_train_batches` in PyTorch Lightning Trainer. See relevant
        documentation for more details.
    augmentations : list of transforms, default=None
        List of transforms to apply, either both or one of XYFlipModel and
        XYRandomRotate90Model. By default, it applies both XYFlip (on X and Y)
        and XYRandomRotate90 (in XY) to the images.
    independent_channels : bool, optional
        Whether to train all channels independently, by default False.
    loss : Literal["mae", "mse"], optional
        Loss function to use, by default "mae".
    n_channels_in : int or None, default=None
        Number of channels in.
    n_channels_out : int or None, default=None
        Number of channels out.
    logger : Literal["wandb", "tensorboard", "none"], optional
        Logger to use, by default "none".
    trainer_params : dict, optional
        Parameters for the trainer class, see PyTorch Lightning documentation.
    model_params : dict, default=None
        UNetModel parameters.
    optimizer : Literal["Adam", "Adamax", "SGD"], default="Adam"
        Optimizer to use.
    optimizer_params : dict, default=None
        Parameters for the optimizer, see PyTorch documentation for more details.
    lr_scheduler : Literal["ReduceLROnPlateau", "StepLR"], default="ReduceLROnPlateau"
        Learning rate scheduler to use.
    lr_scheduler_params : dict, default=None
        Parameters for the learning rate scheduler, see PyTorch documentation for more
        details.
    train_dataloader_params : dict, optional
        Parameters for the training dataloader, see the PyTorch docs for `DataLoader`.
        If left as `None`, the dict `{"shuffle": True}` will be used, this is set in
        the `GeneralDataConfig`.
    val_dataloader_params : dict, optional
        Parameters for the validation dataloader, see PyTorch the docs for `DataLoader`.
        If left as `None`, the empty dict `{}` will be used, this is set in the
        `GeneralDataConfig`.
    checkpoint_params : dict, default=None
        Parameters for the checkpoint callback, see PyTorch Lightning documentation
        (`ModelCheckpoint`) for the list of available parameters.

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

    You can also limit the number of batches per epoch:
    >>> config = create_n2n_configuration(
    ...     experiment_name="n2n_experiment",
    ...     data_type="array",
    ...     axes="YX",
    ...     patch_size=[64, 64],
    ...     batch_size=32,
    ...     num_steps=100  # limit to 100 batches per epoch
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

    A list of transforms can be passed to the `augmentations` parameter:
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

    If you would like to train on CZI files, use `"czi"` as `data_type` and `"SCYX"` as
    `axes` for 2-D or `"SCZYX"` for 3-D denoising. Note that `"SCYX"` can also be used
    for 3-D data but spatial context along the Z dimension will then not be taken into
    account.
    >>> config_2d = create_n2n_configuration(
    ...     experiment_name="n2n_experiment",
    ...     data_type="czi",
    ...     axes="SCYX",
    ...     patch_size=[64, 64],
    ...     batch_size=32,
    ...     num_epochs=100,
    ...     n_channels_in=1,
    ... )
    >>> config_3d = create_n2n_configuration(
    ...     experiment_name="n2n_experiment",
    ...     data_type="czi",
    ...     axes="SCZYX",
    ...     patch_size=[16, 64, 64],
    ...     batch_size=16,
    ...     num_epochs=100,
    ...     n_channels_in=1,
    ... )
    """
    return Configuration(
        **_create_supervised_config_dict(
            algorithm="n2n",
            experiment_name=experiment_name,
            data_type=data_type,
            axes=axes,
            patch_size=patch_size,
            batch_size=batch_size,
            trainer_params=trainer_params,
            augmentations=augmentations,
            independent_channels=independent_channels,
            loss=loss,
            n_channels_in=n_channels_in,
            n_channels_out=n_channels_out,
            logger=logger,
            model_params=model_params,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params,
            train_dataloader_params=train_dataloader_params,
            val_dataloader_params=val_dataloader_params,
            checkpoint_params=checkpoint_params,
            num_epochs=num_epochs,
            num_steps=num_steps,
        )
    )


def create_n2v_configuration(
    experiment_name: str,
    data_type: Literal["array", "tiff", "czi", "custom"],
    axes: str,
    patch_size: Sequence[int],
    batch_size: int,
    num_epochs: int = 100,
    num_steps: int | None = None,
    augmentations: list[Union[XYFlipModel, XYRandomRotate90Model]] | None = None,
    independent_channels: bool = True,
    use_n2v2: bool = False,
    n_channels: int | None = None,
    roi_size: int = 11,
    masked_pixel_percentage: float = 0.2,
    struct_n2v_axis: Literal["horizontal", "vertical", "none"] = "none",
    struct_n2v_span: int = 5,
    trainer_params: dict | None = None,
    logger: Literal["wandb", "tensorboard", "none"] = "none",
    model_params: dict | None = None,
    optimizer: Literal["Adam", "Adamax", "SGD"] = "Adam",
    optimizer_params: dict[str, Any] | None = None,
    lr_scheduler: Literal["ReduceLROnPlateau", "StepLR"] = "ReduceLROnPlateau",
    lr_scheduler_params: dict[str, Any] | None = None,
    train_dataloader_params: dict[str, Any] | None = None,
    val_dataloader_params: dict[str, Any] | None = None,
    checkpoint_params: dict[str, Any] | None = None,
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
    data_type : Literal["array", "tiff", "czi", "custom"]
        Type of the data.
    axes : str
        Axes of the data (e.g. SYX).
    patch_size : List[int]
        Size of the patches along the spatial dimensions (e.g. [64, 64]).
    batch_size : int
        Batch size.
    num_epochs : int, default=100
        Number of epochs to train for. If provided, this will be added to
        trainer_params.
    num_steps : int, optional
        Number of batches in 1 epoch. If provided, this will be added to trainer_params.
        Translates to `limit_train_batches` in PyTorch Lightning Trainer. See relevant
        documentation for more details.
    augmentations : list of transforms, default=None
        List of transforms to apply, either both or one of XYFlipModel and
        XYRandomRotate90Model. By default, it applies both XYFlip (on X and Y)
        and XYRandomRotate90 (in XY) to the images.
    independent_channels : bool, optional
        Whether to train all channels together, by default True.
    use_n2v2 : bool, optional
        Whether to use N2V2, by default False.
    n_channels : int or None, default=None
        Number of channels (in and out).
    roi_size : int, optional
        N2V pixel manipulation area, by default 11.
    masked_pixel_percentage : float, optional
        Percentage of pixels masked in each patch, by default 0.2.
    struct_n2v_axis : Literal["horizontal", "vertical", "none"], optional
        Axis along which to apply structN2V mask, by default "none".
    struct_n2v_span : int, optional
        Span of the structN2V mask, by default 5.
    trainer_params : dict, optional
        Parameters for the trainer, see the relevant documentation.
    logger : Literal["wandb", "tensorboard", "none"], optional
        Logger to use, by default "none".
    model_params : dict, default=None
        UNetModel parameters.
    optimizer : Literal["Adam", "Adamax", "SGD"], default="Adam"
        Optimizer to use.
    optimizer_params : dict, default=None
        Parameters for the optimizer, see PyTorch documentation for more details.
    lr_scheduler : Literal["ReduceLROnPlateau", "StepLR"], default="ReduceLROnPlateau"
        Learning rate scheduler to use.
    lr_scheduler_params : dict, default=None
        Parameters for the learning rate scheduler, see PyTorch documentation for more
        details.
    train_dataloader_params : dict, optional
        Parameters for the training dataloader, see the PyTorch docs for `DataLoader`.
        If left as `None`, the dict `{"shuffle": True}` will be used, this is set in
        the `GeneralDataConfig`.
    val_dataloader_params : dict, optional
        Parameters for the validation dataloader, see PyTorch the docs for `DataLoader`.
        If left as `None`, the empty dict `{}` will be used, this is set in the
        `GeneralDataConfig`.
    checkpoint_params : dict, default=None
        Parameters for the checkpoint callback, see PyTorch Lightning documentation
        (`ModelCheckpoint`) for the list of available parameters.

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

    You can also limit the number of batches per epoch:
    >>> config = create_n2v_configuration(
    ...     experiment_name="n2v_experiment",
    ...     data_type="array",
    ...     axes="YX",
    ...     patch_size=[64, 64],
    ...     batch_size=32,
    ...     num_steps=100  # limit to 100 batches per epoch
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

    If you would like to train on CZI files, use `"czi"` as `data_type` and `"SCYX"` as
    `axes` for 2-D or `"SCZYX"` for 3-D denoising. Note that `"SCYX"` can also be used
    for 3-D data but spatial context along the Z dimension will then not be taken into
    account.
    >>> config_2d = create_n2v_configuration(
    ...     experiment_name="n2v_experiment",
    ...     data_type="czi",
    ...     axes="SCYX",
    ...     patch_size=[64, 64],
    ...     batch_size=32,
    ...     num_epochs=100,
    ...     n_channels=1,
    ... )
    >>> config_3d = create_n2v_configuration(
    ...     experiment_name="n2v_experiment",
    ...     data_type="czi",
    ...     axes="SCZYX",
    ...     patch_size=[16, 64, 64],
    ...     batch_size=16,
    ...     num_epochs=100,
    ...     n_channels=1,
    ... )
    """
    # if there are channels, we need to specify their number
    if "C" in axes and n_channels is None:
        raise ValueError("Number of channels must be specified when using channels.")
    elif "C" not in axes and (n_channels is not None and n_channels > 1):
        raise ValueError(
            f"C is not present in the axes, but number of channels is specified "
            f"(got {n_channels} channel)."
        )

    if n_channels is None:
        n_channels = 1

    # augmentations
    spatial_transforms = _list_spatial_augmentations(augmentations)

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

    # algorithm
    algorithm_params = _create_algorithm_configuration(
        axes=axes,
        algorithm="n2v",
        loss="n2v",
        independent_channels=independent_channels,
        n_channels_in=n_channels,
        n_channels_out=n_channels,
        use_n2v2=use_n2v2,
        model_params=model_params,
        optimizer=optimizer,
        optimizer_params=optimizer_params,
        lr_scheduler=lr_scheduler,
        lr_scheduler_params=lr_scheduler_params,
    )
    algorithm_params["n2v_config"] = n2v_transform

    # data
    data_params = _create_data_configuration(
        data_type=data_type,
        axes=axes,
        patch_size=patch_size,
        batch_size=batch_size,
        augmentations=spatial_transforms,
        train_dataloader_params=train_dataloader_params,
        val_dataloader_params=val_dataloader_params,
    )

    # training
    # Handle trainer parameters with num_epochs and nun_steps
    final_trainer_params = {} if trainer_params is None else trainer_params.copy()

    # Add num_epochs and nun_steps if provided
    if num_epochs is not None:
        final_trainer_params["max_epochs"] = num_epochs
    if num_steps is not None:
        final_trainer_params["limit_train_batches"] = num_steps

    training_params = _create_training_configuration(
        trainer_params=final_trainer_params,
        logger=logger,
        checkpoint_params=checkpoint_params,
    )

    return Configuration(
        experiment_name=experiment_name,
        algorithm_config=algorithm_params,
        data_config=data_params,
        training_config=training_params,
    )


def _create_vae_configuration(
    input_shape: Sequence[int],
    encoder_conv_strides: tuple[int, ...],
    decoder_conv_strides: tuple[int, ...],
    multiscale_count: int,
    z_dims: tuple[int, ...],
    output_channels: int,
    encoder_n_filters: int,
    decoder_n_filters: int,
    encoder_dropout: float,
    decoder_dropout: float,
    nonlinearity: Literal[
        "None", "Sigmoid", "Softmax", "Tanh", "ReLU", "LeakyReLU", "ELU"
    ],
    predict_logvar: Literal[None, "pixelwise"],
    analytical_kl: bool,
) -> LVAEModel:
    """Create a dictionary with the parameters of the vae based algorithm model.

    Parameters
    ----------
    input_shape : tuple[int, ...]
        Shape of the input patch (Z, Y, X) or (Y, X) if the data is 2D.
    encoder_conv_strides : tuple[int, ...]
        Strides of the encoder convolutional layers, length also defines 2D or 3D.
    decoder_conv_strides : tuple[int, ...]
        Strides of the decoder convolutional layers, length also defines 2D or 3D.
    multiscale_count : int
        Number of lateral context layers, specific to MicroSplit.
    z_dims : tuple[int, ...]
        Number of hierarchies in the LVAE model.
    output_channels : int
        Number of output channels.
    encoder_n_filters : int
        Number of filters in the convolutional layers of the encoder.
    decoder_n_filters : int
        Number of filters in the convolutional layers of the decoder.
    encoder_dropout : float
        Dropout rate for the encoder.
    decoder_dropout : float
        Dropout rate for the decoder.
    nonlinearity : Literal
        Type of nonlinearity function to use.
    predict_logvar : Literal # TODO needs review
        _description_.
    analytical_kl : bool # TODO needs clarification
        _description_.

    Returns
    -------
    LVAEModel
        LVAE model with the specified parameters.
    """
    return LVAEModel(
        architecture=SupportedArchitecture.LVAE.value,
        input_shape=input_shape,
        encoder_conv_strides=encoder_conv_strides,
        decoder_conv_strides=decoder_conv_strides,
        multiscale_count=multiscale_count,
        z_dims=z_dims,
        output_channels=output_channels,
        encoder_n_filters=encoder_n_filters,
        decoder_n_filters=decoder_n_filters,
        encoder_dropout=encoder_dropout,
        decoder_dropout=decoder_dropout,
        nonlinearity=nonlinearity,
        predict_logvar=predict_logvar,
        analytical_kl=analytical_kl,
    )


def _create_vae_based_algorithm(
    algorithm: Literal["hdn", "microsplit"],
    loss: LVAELossConfig,
    input_shape: Sequence[int],
    encoder_conv_strides: tuple[int, ...],
    decoder_conv_strides: tuple[int, ...],
    multiscale_count: int,
    z_dims: tuple[int, ...],
    output_channels: int,
    encoder_n_filters: int,
    decoder_n_filters: int,
    encoder_dropout: float,
    decoder_dropout: float,
    nonlinearity: Literal[
        "None", "Sigmoid", "Softmax", "Tanh", "ReLU", "LeakyReLU", "ELU"
    ],
    predict_logvar: Literal[None, "pixelwise"],
    analytical_kl: bool,
    gaussian_likelihood: GaussianLikelihoodConfig | None = None,
    nm_likelihood: NMLikelihoodConfig | None = None,
) -> dict:
    """
    Create a dictionary with the parameters of the VAE-based algorithm model.

    Parameters
    ----------
    algorithm : Literal["hdn"]
        The algorithm type.
    loss : Literal["hdn"]
        The loss function type.
    input_shape : tuple[int, ...]
        The shape of the input data.
    encoder_conv_strides : list[int]
        The strides of the encoder convolutional layers.
    decoder_conv_strides : list[int]
        The strides of the decoder convolutional layers.
    multiscale_count : int
        The number of multiscale layers.
    z_dims : list[int]
        The dimensions of the latent space.
    output_channels : int
        The number of output channels.
    encoder_n_filters : int
        The number of filters in the encoder.
    decoder_n_filters : int
        The number of filters in the decoder.
    encoder_dropout : float
        The dropout rate for the encoder.
    decoder_dropout : float
        The dropout rate for the decoder.
    nonlinearity : Literal
        The nonlinearity function to use.
    predict_logvar : Literal[None, "pixelwise"]
        The type of log variance prediction.
    analytical_kl : bool
        Whether to use analytical KL divergence.
    gaussian_likelihood : Optional[GaussianLikelihoodConfig], optional
        The Gaussian likelihood model, by default None.
    nm_likelihood : Optional[NMLikelihoodConfig], optional
        The noise model likelihood model, by default None.

    Returns
    -------
    dict
        A dictionary with the parameters of the VAE-based algorithm model.
    """
    network_model = _create_vae_configuration(
        input_shape=input_shape,
        encoder_conv_strides=encoder_conv_strides,
        decoder_conv_strides=decoder_conv_strides,
        multiscale_count=multiscale_count,
        z_dims=z_dims,
        output_channels=output_channels,
        encoder_n_filters=encoder_n_filters,
        decoder_n_filters=decoder_n_filters,
        encoder_dropout=encoder_dropout,
        decoder_dropout=decoder_dropout,
        nonlinearity=nonlinearity,
        predict_logvar=predict_logvar,
        analytical_kl=analytical_kl,
    )
    assert gaussian_likelihood or nm_likelihood, "Likelihood model must be specified"
    return {
        "algorithm": algorithm,
        "loss": loss,
        "model": network_model,
        "gaussian_likelihood": gaussian_likelihood,
        "noise_model_likelihood": nm_likelihood,
    }


def get_likelihood_config(
    loss_type: Literal["musplit", "denoisplit", "denoisplit_musplit"],
    # TODO remove different microsplit loss types, refac
    predict_logvar: Literal["pixelwise"] | None = None,
    logvar_lowerbound: float | None = -5.0,
    nm_paths: list[str] | None = None,
    data_stats: tuple[float, float] | None = None,
) -> tuple[
    GaussianLikelihoodConfig | None,
    MultiChannelNMConfig | None,
    NMLikelihoodConfig | None,
]:
    """Get the likelihood configuration for split models.

    Returns a tuple containing the following optional entries:
        - GaussianLikelihoodConfig: Gaussian likelihood configuration for musplit losses
        - MultiChannelNMConfig: Multi-channel noise model configuration for denoisplit
        losses
        - NMLikelihoodConfig: Noise model likelihood configuration for denoisplit losses

    Parameters
    ----------
    loss_type : Literal["musplit", "denoisplit", "denoisplit_musplit"]
        The type of loss function to use.
    predict_logvar : Literal["pixelwise"] | None, optional
        Type of log variance prediction, by default None.
        Required when loss_type is "musplit" or "denoisplit_musplit".
    logvar_lowerbound : float | None, optional
        Lower bound for the log variance, by default -5.0.
        Used when loss_type is "musplit" or "denoisplit_musplit".
    nm_paths : list[str] | None, optional
        Paths to the noise model files, by default None.
        Required when loss_type is "denoisplit" or "denoisplit_musplit".
    data_stats : tuple[float, float] | None, optional
        Data statistics (mean, std), by default None.
        Required when loss_type is "denoisplit" or "denoisplit_musplit".

    Returns
    -------
    GaussianLikelihoodConfig or None
        Configuration for the Gaussian likelihood model.
    MultiChannelNMConfig or None
        Configuration for the multi-channel noise model.
    NMLikelihoodConfig or None
        Configuration for the noise model likelihood.

    Raises
    ------
    ValueError
        If required parameters are missing for the specified loss_type.
    """
    # gaussian likelihood
    if loss_type in ["musplit", "denoisplit_musplit"]:
        # if predict_logvar is None:
        #    raise ValueError(f"predict_logvar is required for loss_type '{loss_type}'")
        # TODO validators should be in pydantic models
        gaussian_lik_config = GaussianLikelihoodConfig(
            predict_logvar=predict_logvar,
            logvar_lowerbound=logvar_lowerbound,
        )
    else:
        gaussian_lik_config = None

    # noise model likelihood
    if loss_type in ["denoisplit", "denoisplit_musplit"]:
        # if nm_paths is None:
        #     raise ValueError(f"nm_paths is required for loss_type '{loss_type}'")
        # if data_stats is None:
        #     raise ValueError(f"data_stats is required for loss_type '{loss_type}'")
        # TODO validators should be in pydantic models
        gmm_list = []
        if nm_paths is not None:
            for NM_path in nm_paths:
                gmm_list.append(
                    GaussianMixtureNMConfig(
                        model_type="GaussianMixtureNoiseModel",
                        path=NM_path,
                    )
                )
        noise_model_config = MultiChannelNMConfig(noise_models=gmm_list)
        nm_lik_config = NMLikelihoodConfig()  # TODO this config isn't needed probably
    else:
        noise_model_config = None
        nm_lik_config = None

    return gaussian_lik_config, noise_model_config, nm_lik_config


# TODO wrap parameters into model, loss etc
# TODO refac likelihood configs to make it 1. Can it be done ?
def create_hdn_configuration(
    experiment_name: str,
    data_type: Literal["array", "tiff", "custom"],
    axes: str,
    patch_size: Sequence[int],
    batch_size: int,
    num_epochs: int = 100,
    num_steps: int | None = None,
    encoder_conv_strides: tuple[int, ...] = (2, 2),
    decoder_conv_strides: tuple[int, ...] = (2, 2),
    multiscale_count: int = 1,
    z_dims: tuple[int, ...] = (128, 128),
    output_channels: int = 1,
    encoder_n_filters: int = 32,
    decoder_n_filters: int = 32,
    encoder_dropout: float = 0.0,
    decoder_dropout: float = 0.0,
    nonlinearity: Literal[
        "None", "Sigmoid", "Softmax", "Tanh", "ReLU", "LeakyReLU", "ELU"
    ] = "ReLU",
    analytical_kl: bool = False,
    predict_logvar: Literal["pixelwise"] | None = None,
    logvar_lowerbound: Union[float, None] = None,
    logger: Literal["wandb", "tensorboard", "none"] = "none",
    trainer_params: dict | None = None,
    augmentations: list[Union[XYFlipModel, XYRandomRotate90Model]] | None = None,
    train_dataloader_params: dict[str, Any] | None = None,
    val_dataloader_params: dict[str, Any] | None = None,
) -> Configuration:
    """
    Create a configuration for training HDN.

    If "Z" is present in `axes`, then `path_size` must be a list of length 3, otherwise
    2.

    If "C" is present in `axes`, then you need to set `n_channels_in` to the number of
    channels. Likewise, if you set the number of channels, then "C" must be present in
    `axes`.

    To set the number of output channels, use the `n_channels_out` parameter. If it is
    not specified, it will be assumed to be equal to `n_channels_in`.

    By default, all channels are trained independently. To train all channels together,
    set `independent_channels` to False.

    By setting `augmentations` to `None`, the default transformations (flip in X and Y,
    rotations by 90 degrees in the XY plane) are applied. Rather than the default
    transforms, a list of transforms can be passed to the `augmentations` parameter. To
    disable the transforms, simply pass an empty list.

    # TODO revisit the necessity of model_params

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
    num_epochs : int, default=100
        Number of epochs to train for. If provided, this will be added to
        trainer_params.
    num_steps : int, optional
        Number of batches in 1 epoch. If provided, this will be added to trainer_params.
        Translates to `limit_train_batches` in PyTorch Lightning Trainer. See relevant
        documentation for more details.
    encoder_conv_strides : tuple[int, ...], optional
        Strides for the encoder convolutional layers, by default (2, 2).
    decoder_conv_strides : tuple[int, ...], optional
        Strides for the decoder convolutional layers, by default (2, 2).
    multiscale_count : int, optional
        Number of scales in the multiscale architecture, by default 1.
    z_dims : tuple[int, ...], optional
        Dimensions of the latent space, by default (128, 128).
    output_channels : int, optional
        Number of output channels, by default 1.
    encoder_n_filters : int, optional
        Number of filters in the encoder, by default 32.
    decoder_n_filters : int, optional
        Number of filters in the decoder, by default 32.
    encoder_dropout : float, optional
        Dropout rate for the encoder, by default 0.0.
    decoder_dropout : float, optional
        Dropout rate for the decoder, by default 0.0.
    nonlinearity : Literal, optional
        Nonlinearity function to use, by default "ReLU".
    analytical_kl : bool, optional
        Whether to use analytical KL divergence, by default False.
    predict_logvar : Literal[None, "pixelwise"], optional
        Type of log variance prediction, by default None.
    logvar_lowerbound : Union[float, None], optional
        Lower bound for the log variance, by default None.
    logger : Literal["wandb", "tensorboard", "none"], optional
        Logger to use for training, by default "none".
    trainer_params : dict, optional
        Parameters for the trainer class, see PyTorch Lightning documentation.
    augmentations : Optional[list[Union[XYFlipModel, XYRandomRotate90Model]]], optional
        List of augmentations to apply, by default None.
    train_dataloader_params : Optional[dict[str, Any]], optional
        Parameters for the training dataloader, by default None.
    val_dataloader_params : Optional[dict[str, Any]], optional
        Parameters for the validation dataloader, by default None.

    Returns
    -------
    Configuration
        The configuration object for training HDN.

    Examples
    --------
    Minimum example:
    >>> config = create_hdn_configuration(
    ...     experiment_name="hdn_experiment",
    ...     data_type="array",
    ...     axes="YX",
    ...     patch_size=[64, 64],
    ...     batch_size=32,
    ...     num_epochs=100
    ... )

    You can also limit the number of batches per epoch:
    >>> config = create_hdn_configuration(
    ...     experiment_name="hdn_experiment",
    ...     data_type="array",
    ...     axes="YX",
    ...     patch_size=[64, 64],
    ...     batch_size=32,
    ...     num_steps=100  # limit to 100 batches per epoch
    ... )
    """
    transform_list = _list_spatial_augmentations(augmentations)

    loss_config = LVAELossConfig(
        loss_type="hdn", denoisplit_weight=1, musplit_weight=0
    )  # TODO what are the correct defaults for HDN?

    gaussian_likelihood = GaussianLikelihoodConfig(
        predict_logvar=predict_logvar, logvar_lowerbound=logvar_lowerbound
    )

    # algorithm & model
    algorithm_params = _create_vae_based_algorithm(
        algorithm="hdn",
        loss=loss_config,
        input_shape=patch_size,
        encoder_conv_strides=encoder_conv_strides,
        decoder_conv_strides=decoder_conv_strides,
        multiscale_count=multiscale_count,
        z_dims=z_dims,
        output_channels=output_channels,
        encoder_n_filters=encoder_n_filters,
        decoder_n_filters=decoder_n_filters,
        encoder_dropout=encoder_dropout,
        decoder_dropout=decoder_dropout,
        nonlinearity=nonlinearity,
        predict_logvar=predict_logvar,
        analytical_kl=analytical_kl,
        gaussian_likelihood=gaussian_likelihood,
        nm_likelihood=None,
    )

    # data
    data_params = _create_data_configuration(
        data_type=data_type,
        axes=axes,
        patch_size=patch_size,
        batch_size=batch_size,
        augmentations=transform_list,
        train_dataloader_params=train_dataloader_params,
        val_dataloader_params=val_dataloader_params,
    )

    # Handle trainer parameters with num_epochs and num_steps
    final_trainer_params = {} if trainer_params is None else trainer_params.copy()

    # Add num_epochs and num_steps if provided
    if num_epochs is not None:
        final_trainer_params["max_epochs"] = num_epochs
    if num_steps is not None:
        final_trainer_params["limit_train_batches"] = num_steps

    # training
    training_params = _create_training_configuration(
        trainer_params=final_trainer_params,
        logger=logger,
    )

    return Configuration(
        experiment_name=experiment_name,
        algorithm_config=algorithm_params,
        data_config=data_params,
        training_config=training_params,
    )


def create_microsplit_configuration(
    experiment_name: str,
    data_type: Literal["array", "tiff", "custom"],
    axes: str,
    patch_size: Sequence[int],
    batch_size: int,
    num_epochs: int = 100,
    num_steps: int | None = None,
    encoder_conv_strides: tuple[int, ...] = (2, 2),
    decoder_conv_strides: tuple[int, ...] = (2, 2),
    multiscale_count: int = 3,
    grid_size: int = 32,  # TODO most likely can be derived from patch size
    z_dims: tuple[int, ...] = (128, 128),
    output_channels: int = 1,
    encoder_n_filters: int = 32,
    decoder_n_filters: int = 32,
    encoder_dropout: float = 0.0,
    decoder_dropout: float = 0.0,
    nonlinearity: Literal[
        "None", "Sigmoid", "Softmax", "Tanh", "ReLU", "LeakyReLU", "ELU"
    ] = "ReLU",  # TODO do we need all these?
    analytical_kl: bool = False,
    predict_logvar: Literal["pixelwise"] = "pixelwise",
    logvar_lowerbound: Union[float, None] = None,
    logger: Literal["wandb", "tensorboard", "none"] = "none",
    trainer_params: dict | None = None,
    augmentations: list[Union[XYFlipModel, XYRandomRotate90Model]] | None = None,
    nm_paths: list[str] | None = None,
    data_stats: tuple[float, float] | None = None,
    train_dataloader_params: dict[str, Any] | None = None,
    val_dataloader_params: dict[str, Any] | None = None,
) -> Configuration:
    """
    Create a configuration for training MicroSplit.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    data_type : Literal["array", "tiff", "custom"]
        Type of the data.
    axes : str
        Axes of the data (e.g. SYX).
    patch_size : Sequence[int]
        Size of the patches along the spatial dimensions (e.g. [64, 64]).
    batch_size : int
        Batch size.
    num_epochs : int, default=100
        Number of epochs to train for. If provided, this will be added to
        trainer_params.
    num_steps : int, optional
        Number of batches in 1 epoch. If provided, this will be added to trainer_params.
        Translates to `limit_train_batches` in PyTorch Lightning Trainer. See relevant
        documentation for more details.
    encoder_conv_strides : tuple[int, ...], optional
        Strides for the encoder convolutional layers, by default (2, 2).
    decoder_conv_strides : tuple[int, ...], optional
        Strides for the decoder convolutional layers, by default (2, 2).
    multiscale_count : int, optional
        Number of multiscale levels, by default 1.
    grid_size : int, optional
        Size of the grid for the lateral context, by default 32.
    z_dims : tuple[int, ...], optional
        List of latent dimensions for each hierarchy level in the LVAE, by default
        (128, 128).
    output_channels : int, optional
        Number of output channels for the model, by default 1.
    encoder_n_filters : int, optional
        Number of filters in the encoder, by default 32.
    decoder_n_filters : int, optional
        Number of filters in the decoder, by default 32.
    encoder_dropout : float, optional
        Dropout rate for the encoder, by default 0.0.
    decoder_dropout : float, optional
        Dropout rate for the decoder, by default 0.0.
    nonlinearity : Literal, optional
        Nonlinearity to use in the model, by default "ReLU".
    analytical_kl : bool, optional
        Whether to use analytical KL divergence, by default False.
    predict_logvar : Literal["pixelwise"] | None, optional
        Type of log-variance prediction, by default None.
    logvar_lowerbound : Union[float, None], optional
        Lower bound for the log variance, by default None.
    logger : Literal["wandb", "tensorboard", "none"], optional
        Logger to use for training, by default "none".
    trainer_params : dict, optional
        Parameters for the trainer class, see PyTorch Lightning documentation.
    augmentations : list[Union[XYFlipModel, XYRandomRotate90Model]] | None, optional
        List of augmentations to apply, by default None.
    nm_paths : list[str] | None, optional
        Paths to the noise model files, by default None.
    data_stats : tuple[float, float] | None, optional
        Data statistics (mean, std), by default None.
    train_dataloader_params : dict[str, Any] | None, optional
        Parameters for the training dataloader, by default None.
    val_dataloader_params : dict[str, Any] | None, optional
        Parameters for the validation dataloader, by default None.

    Returns
    -------
    Configuration
        A configuration object for the microsplit algorithm.

    Examples
    --------
    Minimum example:
    # >>> config = create_microsplit_configuration(
    # ...     experiment_name="microsplit_experiment",
    # ...     data_type="array",
    # ...     axes="YX",
    # ...     patch_size=[64, 64],
    # ...     batch_size=32,
    # ...     num_epochs=100

    # ... )

    # You can also limit the number of batches per epoch:
    # >>> config = create_microsplit_configuration(
    # ...     experiment_name="microsplit_experiment",
    # ...     data_type="array",
    # ...     axes="YX",
    # ...     patch_size=[64, 64],
    # ...     batch_size=32,
    # ...     num_steps=100  # limit to 100 batches per epoch
    # ... )
    """
    transform_list = _list_spatial_augmentations(augmentations)

    loss_config = LVAELossConfig(
        loss_type="denoisplit_musplit", denoisplit_weight=0.9, musplit_weight=0.1
    )  # TODO losses need to be refactored! just for example. Add validator if sum to 1

    # Create likelihood configurations
    gaussian_likelihood_config, noise_model_config, nm_likelihood_config = (
        get_likelihood_config(
            loss_type="denoisplit_musplit",
            predict_logvar=predict_logvar,
            logvar_lowerbound=logvar_lowerbound,
            nm_paths=nm_paths,
            data_stats=data_stats,
        )
    )

    # Create the LVAE model
    network_model = _create_vae_configuration(
        input_shape=patch_size,
        encoder_conv_strides=encoder_conv_strides,
        decoder_conv_strides=decoder_conv_strides,
        multiscale_count=multiscale_count,
        z_dims=z_dims,
        output_channels=output_channels,
        encoder_n_filters=encoder_n_filters,
        decoder_n_filters=decoder_n_filters,
        encoder_dropout=encoder_dropout,
        decoder_dropout=decoder_dropout,
        nonlinearity=nonlinearity,
        predict_logvar=predict_logvar,
        analytical_kl=analytical_kl,
    )

    # Create the MicroSplit algorithm configuration
    algorithm_params = {
        "algorithm": "microsplit",
        "loss": loss_config,
        "model": network_model,
        "gaussian_likelihood": gaussian_likelihood_config,
        "noise_model": noise_model_config,
        "noise_model_likelihood": nm_likelihood_config,
    }

    # Convert to MicroSplitAlgorithm instance
    algorithm_config = MicroSplitAlgorithm(**algorithm_params)

    # data
    data_params = _create_microsplit_data_configuration(
        data_type=data_type,
        axes=axes,
        patch_size=patch_size,
        grid_size=grid_size,
        multiscale_count=multiscale_count,
        batch_size=batch_size,
        augmentations=transform_list,
        train_dataloader_params=train_dataloader_params,
        val_dataloader_params=val_dataloader_params,
    )

    # Handle trainer parameters with num_epochs and num_steps
    final_trainer_params = {} if trainer_params is None else trainer_params.copy()

    # Add num_epochs and num_steps if provided
    if num_epochs is not None:
        final_trainer_params["max_epochs"] = num_epochs
    if num_steps is not None:
        final_trainer_params["limit_train_batches"] = num_steps

    # training
    training_params = _create_training_configuration(
        trainer_params=final_trainer_params,
        logger=logger,
    )

    return Configuration(
        experiment_name=experiment_name,
        algorithm_config=algorithm_config,
        data_config=data_params,
        training_config=training_params,
    )
