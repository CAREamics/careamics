"""CAREamics configuration compatible with the NG Dataset."""

from __future__ import annotations

import re
from pprint import pformat
from typing import Any, Literal, Self, Union

from bioimageio.spec.generic.v0_3 import CiteEntry
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from careamics.config.algorithms import (
    CAREAlgorithm,
    N2NAlgorithm,
    N2VAlgorithm,
)
from careamics.config.data import NGDataConfig
from careamics.config.lightning.training_config import TrainingConfig

ALGORITHMS = Union[
    CAREAlgorithm,
    N2NAlgorithm,
    N2VAlgorithm,
]


class NGConfiguration(BaseModel):
    """
    CAREamics configuration.

    The configuration defines all parameters used to build and train a CAREamics model.
    These parameters are validated to ensure that they are compatible with each other.

    It contains three sub-configurations:

    - AlgorithmModel: configuration for the algorithm training, which includes the
        architecture, loss function, optimizer, and other hyperparameters.
    - DataModel: configuration for the dataloader, which includes the type of data,
        transformations, mean/std and other parameters.
    - TrainingModel: configuration for the training, which includes the number of
        epochs or the callbacks.

    Attributes
    ----------
    experiment_name : str
        Name of the experiment, used when saving logs and checkpoints.
    algorithm : AlgorithmModel
        Algorithm configuration.
    data : DataModel
        Data configuration.
    training : TrainingModel
        Training configuration.

    Methods
    -------
    set_3D(is_3D: bool, axes: str, patch_size: List[int]) -> None
        Switch configuration between 2D and 3D.
    model_dump(
        exclude_defaults: bool = False, exclude_none: bool = True, **kwargs: Dict
        ) -> Dict
        Export configuration to a dictionary.

    Raises
    ------
    ValueError
        Configuration parameter type validation errors.
    ValueError
        If the experiment name contains invalid characters or is empty.
    ValueError
        If the algorithm is 3D but there is not "Z" in the data axes, or 2D algorithm
        with "Z" in data axes.
    ValueError
        Algorithm, data or training validation errors.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    # version
    version: Literal["0.1.0"] = "0.1.0"
    """CAREamics configuration version."""

    # required parameters
    experiment_name: str
    """Name of the experiment, used to name logs and checkpoints."""

    # Sub-configurations
    algorithm_config: ALGORITHMS = Field(discriminator="algorithm")
    """Algorithm configuration, holding all parameters required to configure the
    model."""

    data_config: NGDataConfig
    """Data configuration, holding all parameters required to configure the training
    data loader."""

    training_config: TrainingConfig
    """Training configuration, holding all parameters required to configure the
    training process."""

    @field_validator("experiment_name")
    @classmethod
    def no_symbol(cls, name: str) -> str:
        """
        Validate experiment name.

        A valid experiment name is a non-empty string with only contains letters,
        numbers, underscores, dashes and spaces.

        Parameters
        ----------
        name : str
            Name to validate.

        Returns
        -------
        str
            Validated name.

        Raises
        ------
        ValueError
            If the name is empty or contains invalid characters.
        """
        if len(name) == 0 or name.isspace():
            raise ValueError("Experiment name is empty.")

        # Validate using a regex that it contains only letters, numbers, underscores,
        # dashes and spaces
        if not re.match(r"^[a-zA-Z0-9_\- ]*$", name):
            raise ValueError(
                f"Experiment name contains invalid characters (got {name}). "
                f"Only letters, numbers, underscores, dashes and spaces are allowed."
            )

        return name

    @model_validator(mode="after")
    def validate_3D(self: Self) -> Self:
        """
        Validate algorithm dimensions to match data dimensions.

        Returns
        -------
        Self
            Validated configuration.
        """
        if self.data_config.is_3D() != self.algorithm_config.model.is_3D():
            raise ValueError(
                f"Mismatch between data ({'3D' if self.data_config.is_3D() else '2D'}) "
                f"and algorithm ("
                f"{'3D' if self.algorithm_config.model.is_3D() else '2D'}). Data "
                f"dimensionality is determined by the axes ({self.data_config.axes}), "
                f"as well as patch size (if applicable) and data type (if data type "
                f"is 'czi', which uses 3D when 'T' axis is specified)."
            )

        return self

    def __str__(self) -> str:
        """
        Pretty string reprensenting the configuration.

        Returns
        -------
        str
            Pretty string.
        """
        return pformat(self.model_dump())

    def get_algorithm_friendly_name(self) -> str:
        """
        Get the algorithm name.

        Returns
        -------
        str
            Algorithm name.
        """
        return self.algorithm_config.get_algorithm_friendly_name()

    def get_algorithm_description(self) -> str:
        """
        Return a description of the algorithm.

        This method is used to generate the README of the BioImage Model Zoo export.

        Returns
        -------
        str
            Description of the algorithm.
        """
        return self.algorithm_config.get_algorithm_description()

    def get_algorithm_citations(self) -> list[CiteEntry]:
        """
        Return a list of citation entries of the current algorithm.

        This is used to generate the model description for the BioImage Model Zoo.

        Returns
        -------
        List[CiteEntry]
            List of citation entries.
        """
        return self.algorithm_config.get_algorithm_citations()

    def get_algorithm_references(self) -> str:
        """
        Get the algorithm references.

        This is used to generate the README of the BioImage Model Zoo export.

        Returns
        -------
        str
            Algorithm references.
        """
        return self.algorithm_config.get_algorithm_references()

    def get_algorithm_keywords(self) -> list[str]:
        """
        Get algorithm keywords.

        Returns
        -------
        list[str]
            List of keywords.
        """
        return self.algorithm_config.get_algorithm_keywords()

    def model_dump(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Override model_dump method in order to set default values.

        As opposed to the parent model_dump method, this method sets exclude none by
        default.

        Parameters
        ----------
        **kwargs : Any
            Additional arguments to pass to the parent model_dump method.

        Returns
        -------
        dict
            Dictionary containing the model parameters.
        """
        if "exclude_none" not in kwargs:
            kwargs["exclude_none"] = True

        return super().model_dump(**kwargs)
