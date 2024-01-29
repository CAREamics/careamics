"""Algorithm configuration."""
from typing import Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from .optimizers import OptimizerModel, LrSchedulerModel
from .architectures import UNetModel, VAEModel
from .filters import remove_default_optionals
from .torch_optim import LrSchedulerModel, OptimizerModel

#from .noise_models import NoiseModel

class AlgorithmModel(BaseModel):
    """
    Algorithm configuration.

    The minimum algorithm configuration is composed of the following fields:
        - loss:
            Loss to use, currently only supports n2v.
        - model:
            Model to use, currently only supports UNet.

    Attributes
    ----------
    loss : str
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
        protected_namespaces=(),  # allows to use model_* as a field name
        validate_assignment=True,
    )

    # Mandatory fields
    algorithm: Literal["n2v", "n2v2"]
    loss: Literal["n2v", "mae", "mse"]
    model: Union[VAEModel, UNetModel] = Field(discriminator="architecture")

    optimizer: OptimizerModel
    lr_scheduler: LrSchedulerModel

    # Optional fields, define a default value
    #noise_model: Optional[NoiseModel] = None


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
