"""Convenience functions to create CryoCARE configurations."""

from collections.abc import Sequence
from typing import Any, Literal

from careamics.config.algorithms import CryoCAREAlgorithm
from careamics.config.cryocare_configuration import CryoCAREConfiguration
from careamics.config.data import CryoCAREDataConfig
from careamics.config.lightning.optimizer_configs import (
    LrSchedulerConfig,
    OptimizerConfig,
)

from .training_factory import create_training_configuration, update_trainer_params


def create_cryocare_config(
    *,
    experiment_name: str,
    data_type: Literal["array", "tiff", "zarr", "czi", "mrc", "custom"],
    axes: str,
    patch_size: Sequence[int],
    batch_size: int,
    # optional parameters
    num_epochs: int = 100,
    num_steps: int | None = None,
    tilt_axis: Literal["X", "Y", "Z"] = "Y",
    validation_fraction: float = 0.1,
    n_val_patches: int = 50,
) -> CryoCAREConfiguration:
    """Create a standard configuration for training CryoCARE.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    data_type : {"array", "tiff", "zarr", "czi", "mrc", "custom"}
        Type of the data.
    axes : str
        Axes of the input data. Only permutations of `ZYX` are allowed.
    patch_size : sequence of int
        Three-dimensional patch size in `ZYX` order.
    batch_size : int
        Batch size.
    num_epochs : int, default=100
        Number of epochs to train for.
    num_steps : int or None, default=None
        Number of batches in one epoch.
    tilt_axis : {"X", "Y", "Z"}, default="Y"
        Physical tilt axis used for the train/validation split and augmentation.
    validation_fraction : float, default=0.1
        Fraction of the reconstructed stack reserved for validation along the tilt axis.
    n_val_patches : int, default=50
        Number of validation patches to draw from the validation region. Must be
        positive.

    Returns
    -------
    CryoCAREConfiguration
        Configuration for training CryoCARE.
    """
    return create_advanced_cryocare_config(**locals())


def create_advanced_cryocare_config(
    *,
    experiment_name: str,
    data_type: Literal["array", "tiff", "zarr", "czi", "mrc", "custom"],
    axes: str,
    patch_size: Sequence[int],
    batch_size: int,
    # optional parameters
    num_epochs: int = 100,
    num_steps: int | None = None,
    tilt_axis: Literal["X", "Y", "Z"] = "Y",
    validation_fraction: float = 0.1,
    n_val_patches: int = 50,
    # advanced parameters
    project_name: str | None = None,
    normalization: Literal["mean_std", "min_max", "quantile", "none"] = "mean_std",
    normalization_params: dict[str, Any] | None = None,
    # lightning parameters
    num_workers: int = -1,
    trainer_params: dict[str, Any] | None = None,
    optimizer: Literal["Adam", "Adamax", "SGD"] = "Adam",
    optimizer_params: dict[str, Any] | None = None,
    lr_scheduler: Literal["ReduceLROnPlateau", "StepLR"] = "ReduceLROnPlateau",
    lr_scheduler_params: dict[str, Any] | None = None,
    train_dataloader_params: dict[str, Any] | None = None,
    val_dataloader_params: dict[str, Any] | None = None,
    checkpoint_params: dict[str, Any] | None = None,
    logger: Literal["wandb", "tensorboard", "none"] = "none",
    # reproducibility
    seed: int | None = None,
) -> CryoCAREConfiguration:
    """Create an advanced configuration for training CryoCARE.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    data_type : {"array", "tiff", "zarr", "czi", "mrc", "custom"}
        Type of the data.
    axes : str
        Axes of the input data. Must contain `Z` and must not contain `C`.
    patch_size : sequence of int
        Three-dimensional patch size in `ZYX` order.
    batch_size : int
        Batch size.
    num_epochs : int, default=100
        Number of epochs to train for.
    num_steps : int or None, default=None
        Number of batches in one epoch.
    tilt_axis : {"X", "Y", "Z"}, default="Y"
        Physical tilt axis used for the train/validation split and augmentation.
    validation_fraction : float, default=0.1
        Fraction of the reconstructed stack reserved for validation along the tilt axis.
    n_val_patches : int, default=50
        Number of validation patches to draw from the validation region. Must be
        positive.
    project_name : str or None, default=None
        Name of the project, used to group related experiments.
    normalization : {"mean_std", "min_max", "quantile", "none"}
        Normalization strategy. Default is `mean_std`.
    normalization_params : dict, optional
        Parameters for the selected normalization strategy.
    num_workers : int, default=-1
        Number of dataloader workers. `-1` uses the platform default.
    trainer_params : dict, optional
        Additional Lightning Trainer parameters.
    optimizer : {"Adam", "Adamax", "SGD"}, default="Adam"
        Optimizer to use.
    optimizer_params : dict, optional
        Optimizer parameters. The CryoCARE default learning rate is used when omitted.
    lr_scheduler : {"ReduceLROnPlateau", "StepLR"}, default="ReduceLROnPlateau"
        Learning-rate scheduler to use.
    lr_scheduler_params : dict, optional
        Scheduler parameters.
    train_dataloader_params : dict, optional
        Training dataloader parameters.
    val_dataloader_params : dict, optional
        Validation dataloader parameters.
    checkpoint_params : dict, optional
        Model checkpoint parameters.
    logger : {"wandb", "tensorboard", "none"}, default="none"
        Logger to use.
    seed : int or None, default=None
        Seed for reproducibility.

    Returns
    -------
    CryoCAREConfiguration
        Configuration for training CryoCARE.
    """
    norm_config: dict[str, Any] = {"name": normalization}
    if normalization_params is not None:
        norm_config.update(normalization_params)

    data_kwargs: dict[str, Any] = {}
    if num_workers != -1:
        data_kwargs["num_workers"] = num_workers
    if train_dataloader_params is not None:
        data_kwargs["train_dataloader_params"] = train_dataloader_params
    if val_dataloader_params is not None:
        data_kwargs["val_dataloader_params"] = val_dataloader_params
    # if in_memory is not None:
    #     data_kwargs["in_memory"] = in_memory
    if seed is not None:
        data_kwargs["seed"] = seed

    data_config = CryoCAREDataConfig(
        mode="training",
        data_type=data_type,
        axes=axes,
        patching={"name": "stratified", "patch_size": patch_size},
        normalization=norm_config,
        batch_size=batch_size,
        n_val_patches=n_val_patches,
        channels=None,
        tilt_axis=tilt_axis,
        validation_fraction=validation_fraction,
        **data_kwargs,
    )

    algorithm_config = CryoCAREAlgorithm(
        optimizer=OptimizerConfig(
            name=optimizer,
            parameters=optimizer_params or {"lr": 4e-4},
        ),
        lr_scheduler=LrSchedulerConfig(
            name=lr_scheduler,
            parameters=lr_scheduler_params or {},
        ),
    )

    training_config = create_training_configuration(
        algorithm="cryocare",
        trainer_params=update_trainer_params(
            trainer_params=trainer_params,
            num_epochs=num_epochs,
            num_steps=num_steps,
        ),
        logger=logger,
        checkpoint_params=checkpoint_params,
    )

    return CryoCAREConfiguration(
        experiment_name=experiment_name,
        project_name=project_name,
        algorithm_config=algorithm_config,
        data_config=data_config,
        training_config=training_config,
    )
