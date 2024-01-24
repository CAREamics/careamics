from enum import Enum

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationInfo,
    field_validator,
)


# python 3.11: https://docs.python.org/3/library/enum.html
class LossType(str, Enum):
    """
    Available loss functions.

    Currently supported losses:

        - n2v: Noise2Void loss.
        - n2n: Noise2Noise loss.
        - pn2v: Probabilistic Noise2Void loss.
        - hdn: Hierarchical DivNoising loss.
    """

    CARE = "care"
    N2V = "n2v"
    N2N = "n2n"
    PN2V = "pn2v"
    HDN = "hdn"


class Loss(BaseModel):
    """
    Loss function.

    Attributes
    ----------
    loss : LossType
        Loss function.
    """

    model_config = ConfigDict(
        use_enum_values=True,
        protected_namespaces=(),  # allows to use model_* as a field name
        validate_assignment=True,
    )

    loss_type: LossType

    @field_validator("loss_type")
    def validate_loss(cls, loss: LossType, info: ValidationInfo) -> LossType:
        """Validate loss function.

        Parameters
        ----------
        loss : LossType
            Loss function.

        Raises
        ------
        ValueError
            If the loss is unknown.
        """
        if (loss == Loss.PN2V or loss == Loss.HDN) and info.data["noise_model"] is None:
            raise ValueError(f"Loss {loss} requires a noise model.")
        elif loss in [Loss.N2V, Loss.N2N] and info.data["noise_model"] is not None:
            raise ValueError(f"Loss {loss} does not support a noise model.")

        return loss
