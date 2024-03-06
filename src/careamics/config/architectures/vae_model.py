from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
)


class VAEModel(BaseModel):
    model_config = ConfigDict(
        use_enum_values=True, protected_namespaces=(), validate_assignment=True
    )

    architecture: Literal["VAE"]

    def set_3D(self, is_3D: bool) -> None:
        """
        Set 3D model by setting the `conv_dims` parameters.

        Parameters
        ----------
        is_3D : bool
            Whether the algorithm is 3D or not.
        """
        raise NotImplementedError("VAE is not implemented yet.")

    def is_3D(self) -> bool:
        """
        Return whether the model is 3D or not.

        Returns
        -------
        bool
            Whether the model is 3D or not.
        """
        raise NotImplementedError("VAE is not implemented yet.")
