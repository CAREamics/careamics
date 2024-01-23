"""Algorithm configuration."""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationInfo,
    model_validator,
    field_validator
)
from torch import optim

from .config_filter import remove_default_optionals
from .models import Model
from .torch_optim import TorchLRScheduler, TorchOptimizer, get_parameters

#from .noise_models import NoiseModel

# python 3.11: https://docs.python.org/3/library/enum.html
class AlgorithmType(str, Enum):
    """
    Available types of algorithms.

    Currently supported algorithms:
        - CARE: CARE. https://www.nature.com/articles/s41592-018-0216-7
        - n2v: Noise2Void. https://arxiv.org/abs/1811.10980
        - n2n: Noise2Noise. https://arxiv.org/abs/1803.04189
        - pn2v: Probabilistic Noise2Void. https://arxiv.org/abs/1906.00651
        - hdn: Hierarchical DivNoising. https://arxiv.org/abs/2104.01374
    """

    # CARE = "care"
    N2V = "n2v"
    # N2N = "n2n"
    # PN2V = "pn2v"
    # HDN = "hdn"
    # CUSTOM = "custom"
    # SEGM = "segmentation"


class Loss(str, Enum):
    """
    Available loss functions.

    Currently supported losses:

        - n2v: Noise2Void loss.
        - n2n: Noise2Noise loss.
        - pn2v: Probabilistic Noise2Void loss.
        - hdn: Hierarchical DivNoising loss.
    """

    # MSE = "mse"
    # MAE = "mae"
    N2V = "n2v"
    # PN2V = "pn2v"
    # HDN = "hdn"
    # CE = "ce"
    # DICE = "dice"
    # CUSTOM = "custom"


class Optimizer(BaseModel):
    """
    Torch optimizer.

    Only parameters supported by the corresponding torch optimizer will be taken
    into account. For more details, check:
    https://pytorch.org/docs/stable/optim.html#algorithms

    Note that mandatory parameters (see the specific Optimizer signature in the
    link above) must be provided. For example, SGD requires `lr`.

    Attributes
    ----------
    name : TorchOptimizer
        Name of the optimizer.
    parameters : dict
        Parameters of the optimizer (see torch documentation).
    """

    # Pydantic class configuration
    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
    )

    # Mandatory field
    name: TorchOptimizer

    # Optional parameters
    parameters: dict = {}

    @field_validator("parameters")
    def filter_parameters(cls, user_params: dict, values: ValidationInfo) -> Dict:
        """
        Validate optimizer parameters.

        This method filters out unknown parameters, given the optimizer name.

        Parameters
        ----------
        user_params : dict
            Parameters passed on to the torch optimizer.
        values : ValidationInfo
            Pydantic field validation info, used to get the optimizer name.

        Returns
        -------
        Dict
            Filtered optimizer parameters.

        Raises
        ------
        ValueError
            If the optimizer name is not specified.
        """
        if "name" in values.data:
            optimizer_name = values.data["name"]

            # retrieve the corresponding optimizer class
            optimizer_class = getattr(optim, optimizer_name)

            # filter the user parameters according to the optimizer's signature
            return get_parameters(optimizer_class, user_params)
        else:
            raise ValueError(
                "Cannot validate optimizer parameters without `name`, check that it "
                "has correctly been specified."
            )

    @model_validator(mode="after")
    def sgd_lr_parameter(cls, optimizer: Optimizer) -> Optimizer:
        """
        Check that SGD optimizer has the mandatory `lr` parameter specified.

        Parameters
        ----------
        optimizer : Optimizer
            Optimizer to validate.

        Returns
        -------
        Optimizer
            Validated optimizer.

        Raises
        ------
        ValueError
            If the optimizer is SGD and the lr parameter is not specified.
        """
        if optimizer.name == TorchOptimizer.SGD and "lr" not in optimizer.parameters:
            raise ValueError(
                "SGD optimizer requires `lr` parameter, check that it has correctly "
                "been specified in `parameters`."
            )

        return optimizer

    def model_dump(
        self, exclude_optionals: bool = True, *args: List, **kwargs: Dict
    ) -> Dict:
        """
        Override model_dump method.

        The purpose of this method is to ensure smooth export to yaml. It
        includes:
            - removing entries with None value.
            - removing optional values if they have the default value.

        Parameters
        ----------
        exclude_optionals : bool, optional
            Whether to exclude optional arguments if they are default, by default True.
        *args : List
            Positional arguments, unused.
        **kwargs : Dict
            Keyword arguments, unused.

        Returns
        -------
        dict
            Dictionary containing the model parameters.
        """
        dictionary = super().model_dump(exclude_none=True)

        if exclude_optionals:
            # remove optional arguments if they are default
            default_optionals: dict = {"parameters": {}}

            remove_default_optionals(dictionary, default_optionals)

        return dictionary


class LrScheduler(BaseModel):
    """
    Torch learning rate scheduler.

    Only parameters supported by the corresponding torch lr scheduler will be taken
    into account. For more details, check:
    https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate

    Note that mandatory parameters (see the specific LrScheduler signature in the
    link above) must be provided. For example, StepLR requires `step_size`.

    Attributes
    ----------
    name : TorchLRScheduler
        Name of the learning rate scheduler.
    parameters : dict
        Parameters of the learning rate scheduler (see torch documentation).
    """

    # Pydantic class configuration
    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
    )

    # Mandatory field
    name: TorchLRScheduler

    # Optional parameters
    parameters: dict = {}

    @field_validator("parameters")
    def filter_parameters(cls, user_params: dict, values: ValidationInfo) -> Dict:
        """
        Validate lr scheduler parameters.

        This method filters out unknown parameters, given the lr scheduler name.

        Parameters
        ----------
        user_params : dict
            Parameters passed on to the torch lr scheduler.
        values : ValidationInfo
            Pydantic field validation info, used to get the lr scheduler name.

        Returns
        -------
        Dict
            Filtered lr scheduler parameters.

        Raises
        ------
        ValueError
            If the lr scheduler name is not specified.
        """
        if "name" in values.data:
            lr_scheduler_name = values.data["name"]

            # retrieve the corresponding lr scheduler class
            lr_scheduler_class = getattr(optim.lr_scheduler, lr_scheduler_name)

            # filter the user parameters according to the lr scheduler's signature
            return get_parameters(lr_scheduler_class, user_params)
        else:
            raise ValueError(
                "Cannot validate lr scheduler parameters without `name`, check that it "
                "has correctly been specified."
            )

    @model_validator(mode="after")
    def step_lr_step_size_parameter(cls, lr_scheduler: LrScheduler) -> LrScheduler:
        """
        Check that StepLR lr scheduler has `step_size` parameter specified.

        Parameters
        ----------
        lr_scheduler : LrScheduler
            Lr scheduler to validate.

        Returns
        -------
        LrScheduler
            Validated lr scheduler.

        Raises
        ------
        ValueError
            If the lr scheduler is StepLR and the step_size parameter is not specified.
        """
        if (
            lr_scheduler.name == TorchLRScheduler.StepLR
            and "step_size" not in lr_scheduler.parameters
        ):
            raise ValueError(
                "StepLR lr scheduler requires `step_size` parameter, check that it has "
                "correctly been specified in `parameters`."
            )

        return lr_scheduler

    def model_dump(
        self, exclude_optionals: bool = True, *args: List, **kwargs: Dict
    ) -> Dict:
        """
        Override model_dump method.

        The purpose of this method is to ensure smooth export to yaml. It includes:
            - removing entries with None value.
            - removing optional values if they have the default value.

        Parameters
        ----------
        exclude_optionals : bool, optional
            Whether to exclude optional arguments if they are default, by default True.
        *args : List
            Positional arguments, unused.
        **kwargs : Dict
            Keyword arguments, unused.

        Returns
        -------
        dict
            Dictionary containing the model parameters.
        """
        dictionary = super().model_dump(exclude_none=True)

        if exclude_optionals:
            # remove optional arguments if they are default
            default_optionals: dict = {"parameters": {}}
            remove_default_optionals(dictionary, default_optionals)

        return dictionary


# TODO: validate model parameters for those that are fully CAREamics?
class Algorithm(BaseModel):
    """
    Algorithm configuration.

    The minimum algorithm configuration is composed of the following fields:
        - loss:
            Loss to use, currently only supports n2v.
        - model:
            Model to use, currently only supports UNet.
        - is_3D:
            Whether to use a 3D model or not, this should be coherent with the
            data configuration (axes).

    Other optional fields are:
        - masking_strategy:
            Masking strategy to use, currently only supports default masking.
        - masked_pixel_percentage:
            Percentage of pixels to be masked in each patch.
        - roi_size:
            Size of the region of interest to use in the masking algorithm.
        - model_parameters:
            Model parameters, see ModelParameters for more details.

    Attributes
    ----------
    loss : List[Losses]
        List of losses to use, currently only supports n2v.
    model : Models
        Model to use, currently only supports UNet.
    is_3D : bool
        Whether to use a 3D model or not.
    masking_strategy : MaskingStrategies
        Masking strategy to use, currently only supports default masking.
    masked_pixel_percentage : float
        Percentage of pixels to be masked in each patch.
    roi_size : int
        Size of the region of interest used in the masking scheme.
    model_parameters : ModelParameters
        Model parameters, see ModelParameters for more details.
    """

    # Pydantic class configuration
    model_config = ConfigDict(
        use_enum_values=True,
        protected_namespaces=(),  # allows to use model_* as a field name
        validate_assignment=True,
    )

    # Mandatory fields
    algorithm_type: AlgorithmType
    loss: Loss
    model: Model

    optimizer: Optimizer
    lr_scheduler: LrScheduler

    # Optional fields, define a default value
    #noise_model: Optional[NoiseModel] = None

    def get_conv_dim(self) -> int:
        """
        Get the convolution layers dimension (2D or 3D).

        Returns
        -------
        int
            Dimension (2 or 3).
        """
        return 3 if self.model.is_3D else 2

    # def get_noise_model(self, noise_model: Dict, info: ValidationInfo) -> Dict:
    #     """
    #     Validate noise model.

    #     Returns
    #     -------
    #     Dict
    #         Validated noise model.
    #     """
    #     # TODO validate noise model
    #     if "noise_model_type" not in info.data:
    #         raise ValueError("Noise model is missing.")

    #     noise_model_type = info.data["noise_model_type"]

    #     # TODO this does not exist
    #     if noise_model is not None:
    #         _ = NoiseModel.get_noise_model(noise_model_type, noise_model)

    #     return noise_model

    # TODO think in terms of validation of Algorithm and entry point in Lightning
    # TODO we might need to do the model validation in the overall configuration
    # @model_validator(mode="after")
    # def algorithm_cross_validation(cls, data: Algorithm) -> Algorithm:
    #     """Validate loss.

    #     Returns
    #     -------
    #     Loss
    #         Validated loss.

    #     Raises
    #     ------
    #     ValueError
    #         If the loss is not supported or inconsistent with the noise model.
    #     """
    #     if data.algorithm_type in [
    #         AlgorithmType.CARE,
    #         AlgorithmType.N2N,
    #     ] and data.loss not in [
    #         Loss.MSE,
    #         Loss.MAE,
    #         Loss.CUSTOM,
    #     ]:
    #         raise ValueError(
    #             f"Algorithm {data.algorithm_type} does not support"
    #             f" {data.loss.upper()} loss. Please refer to the documentation"
    #             # TODO add link to documentation
    #         )

    #     if (
    #         data.algorithm_type in [AlgorithmType.CARE, AlgorithmType.N2N]
    #         and data.noise_model is not None
    #     ):
    #         raise ValueError(
    #             f"Algorithm {data.algorithm_type} isn't compatible with a noise model."
    #         )

    #     if data.algorithm_type in [AlgorithmType.N2V, AlgorithmType.PN2V]:
    #         if data.transforms is None:
    #             raise ValueError(
    #                 f"Algorithm {data.algorithm_type} requires a masking strategy."
    #                 "Please add ManipulateN2V to transforms."
    #             )
    #         else:
    #             if "ManipulateN2V" not in data.transforms:
    #                 raise ValueError(
    #                     f"Algorithm {data.algorithm_type} requires a masking strategy."
    #                     "Please add ManipulateN2V to transforms."
    #                 )
    #     elif "ManipulateN2V" in data.transforms:
    #         raise ValueError(
    #             f"Algorithm {data.algorithm_type} doesn't require a masking strategy."
    #             "Please remove ManipulateN2V from the image or patch_transform."
    #         )
    #     if (
    #         data.loss == Loss.PN2V or data.loss == Loss.HDN
    #     ) and data.noise_model is None:
    #         raise ValueError(f"Loss {data.loss.upper()} requires a noise model.")

    #     if data.loss in [Loss.N2V, Loss.MAE, Loss.MSE] and data.noise_model is not None:
    #         raise ValueError(
    #             f"Loss {data.loss.upper()} does not support a noise model."
    #         )
    #     if data.loss == Loss.N2V and data.algorithm_type != AlgorithmType.N2V:
    #         raise ValueError(
    #             f"Loss {data.loss.upper()} is only supported by "
    #             f"{AlgorithmType.N2V}."
    #         )

    #     return data

    def model_dump(
        self, exclude_optionals: bool = True, *args: List, **kwargs: Dict
    ) -> Dict:
        """
        Override model_dump method.

        The purpose is to ensure export smooth import to yaml. It includes:
            - remove entries with None value.
            - remove optional values if they have the default value.

        Parameters
        ----------
        exclude_optionals : bool, optional
            Whether to exclude optional arguments if they are default, by default True.
        *args : List
            Positional arguments, unused.
        **kwargs : Dict
            Keyword arguments, unused.

        Returns
        -------
        Dict
            Dictionary representation of the model.
        """
        dictionary = super().model_dump(exclude_none=True)

        if exclude_optionals is True:
            # remove optional arguments if they are default
            defaults = {
                "model": {
                    # "architecture": "UNet",
                    "parameters": {"depth": 2, "num_channels_init": 32},
                },
                "masking_strategy": {
                    # "strategy_type": "default",
                    "parameters": {"masked_pixel_percentage": 0.2, "roi_size": 11},
                },
            }

            remove_default_optionals(dictionary, defaults)

        return dictionary
