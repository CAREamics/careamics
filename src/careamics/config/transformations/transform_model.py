"""Parent model for the transforms."""

from typing import Any

from pydantic import BaseModel, ConfigDict


class TransformModel(BaseModel):
    """
    Pydantic model used to represent a transformation.

    The `model_dump` method is overwritten to exclude the name field.

    Attributes
    ----------
    name : str
        Name of the transformation.
    """

    model_config = ConfigDict(
        extra="forbid",  # throw errors if the parameters are not properly passed
    )

    name: str

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """
        Return the model as a dictionary.

        Parameters
        ----------
        **kwargs
            Pydantic BaseMode model_dump method keyword arguments.

        Returns
        -------
        {str: Any}
            Dictionary representation of the model.
        """
        model_dict = super().model_dump(**kwargs)

        # remove the name field
        model_dict.pop("name")

        return model_dict
