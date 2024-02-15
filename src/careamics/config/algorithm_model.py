from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .architectures import UNetModel, VAEModel, CustomModel
from .optimizer_models import LrSchedulerModel, OptimizerModel


class AlgorithmModel(BaseModel):
    """Pydantic model describing CAREamics' algorithm.

    # TODO
    """

    # Pydantic class configuration
    model_config = ConfigDict(
        protected_namespaces=(),  # allows to use model_* as a field name
        validate_assignment=True,
    )

    # Mandatory fields
    algorithm: Literal["n2v", "n2v2", "structn2v", "custom"]
    loss: Literal["n2v", "mae", "mse"]
    model: Union[UNetModel, VAEModel, CustomModel] = Field(discriminator="architecture")

    # Optional fields
    optimizer: OptimizerModel = OptimizerModel()
    lr_scheduler: LrSchedulerModel = LrSchedulerModel()

    @model_validator(mode="after")
    def algorithm_cross_validation(self) -> AlgorithmModel:
        """Validate the algorithm model based on `algorithm`.

        N2V:
        - loss must be n2v
        - model must be a `UNetModel`
        - model n2v2 parameters must be `False`

        N2V2:
        - loss must be `n2v`
        - model must be a `UnetModel`
        - model n2v2 parameters must be `True`

        StructN2V:
        - loss must be `n2v`
        - model must be a `UNetModel`
        """
        # N2V
        if self.algorithm == "n2v":
            if self.loss != "n2v":
                raise ValueError(
                    f"Algorithm {self.algorithm} only supports loss `n2v`."
                )

            if not isinstance(self.model, UNetModel):
                raise ValueError(
                    f"Model for algorithm {self.algorithm} must be a `UNetModel`."
                )

            if self.model.n2v2:
                raise ValueError(
                    f"Model for algorithm {self.algorithm} must have `n2v2` parameters "
                    f"set to `False`."
                )

        # N2V2
        elif self.algorithm == "n2v2":
            if self.loss != "n2v":
                raise ValueError(
                    f"Algorithm {self.algorithm} only supports loss `n2v`."
                )

            if not isinstance(self.model, UNetModel):
                raise ValueError(
                    f"Model for algorithm {self.algorithm} must be a `UNetModel`."
                )

            if not self.model.n2v2:
                raise ValueError(
                    f"Model for algorithm {self.algorithm} must have `n2v2` parameters "
                    f"set to `True`."
                )

        # StructN2V
        elif self.algorithm == "structn2v":
            if self.loss != "n2v":
                raise ValueError(
                    f"Algorithm {self.algorithm} only supports loss `n2v`."
                )

            if not isinstance(self.model, UNetModel):
                raise ValueError(
                    f"Model for algorithm {self.algorithm} must be a `UNetModel`."
                )

        if isinstance(self.model, VAEModel):
            raise ValueError("VAE are currently not implemented.")

        return self
