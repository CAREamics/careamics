"""Pydantic CAREamics configuration."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Literal, Union

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
    model_validator,
)

from .algorithm_model import AlgorithmModel
from .data_model import DataModel
from .support import SupportedAlgorithm, SupportedTransform
from .training_model import Training
from .transformations.n2v_manipulate_model import (
    N2VManipulationModel,
)


# TODO what do we expect in terms of model dump, with or without the defaults?
class Configuration(BaseModel):
    """
    CAREamics configuration.

    To change the configuration from 2D to 3D, we recommend using the following method:
    >>> set_3D(is_3D, axes)

    Attributes
    ----------
    experiment_name : str
        Name of the experiment.
    working_directory : Union[str, Path]
        Path to the working directory.
    algorithm : Algorithm
        Algorithm configuration.
    training : Training
        Training configuration.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        set_arbitrary_types_allowed=True,
    )

    # version
    version: Literal["0.1.0"] = "0.1.0"

    # required parameters
    experiment_name: str

    # Sub-configurations
    algorithm: AlgorithmModel
    data: DataModel
    training: Training

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
    def validate_3D(self) -> Configuration:
        """
        Check 3D flag validity.

        Check that the algorithm is_3D flag is compatible with the axes in the
        data configuration.

        Returns
        -------
        Configuration
            Validated configuration.

        Raises
        ------
        ValueError
            If the algorithm is 3D but the data axes are not, or if the algorithm is
            not 3D but the data axes are.
        """
        # check that is_3D and axes are compatible
        if self.algorithm.model.is_3D() and "Z" not in self.data.axes:
            raise ValueError(
                f"Algorithm is 3D but data axes are not (got axes {self.data.axes})."
            )
        elif not self.algorithm.model.is_3D() and "Z" in self.data.axes:
            raise ValueError(
                f"Algorithm is not 3D but data axes are (got axes {self.data.axes})."
            )

        return self

    @model_validator(mode="after")
    def validate_algorithm_and_data(self: Configuration) -> Configuration:
        """Validate the algorithm and data configurations.

        In particular, the choice of algorithm will influantiate the potential
        transforms that can be applied to the data.

        - (struct)N2V(2): requires ManipulateN2V to be the last transform.

        Returns
        -------
        Configuration
            Validated configuration
        """
        if self.algorithm.algorithm == SupportedAlgorithm.N2V:

            # if we have a list of transform (as opposed to Compose)
            if isinstance(self.data.transforms, list):
                transform_list = [t.name for t in self.data.transforms]

                # whether we use n2v2
                use_n2v2 = self.algorithm.model.n2v2

                # missing N2V_MANIPULATE
                if SupportedTransform.N2V_MANIPULATE not in transform_list:
                    self.data.transforms.append(
                        N2VManipulationModel(
                            name=SupportedTransform.N2V_MANIPULATE.value,
                        )
                    )

                # multiple N2V_MANIPULATE
                elif transform_list.count(SupportedTransform.N2V_MANIPULATE) > 1:
                    raise ValueError(
                        f"Multiple {SupportedTransform.N2V_MANIPULATE} transforms are "
                        f"not allowed."
                    )

                # N2V_MANIPULATE not the last transform
                elif transform_list[-1] != SupportedTransform.N2V_MANIPULATE:
                    index = transform_list.index(SupportedTransform.N2V_MANIPULATE)
                    transform = self.data.transforms.pop(index)
                    self.data.transforms.append(transform)

                # check that N2V_MANIPULATE has the right n2v2 parameter
                n2v_manipulate = self.data.transforms[-1]
                if use_n2v2:
                    if n2v_manipulate.parameters.strategy != "median":
                        n2v_manipulate.parameters.strategy = "median"
                else:
                    if n2v_manipulate.parameters.strategy != "uniform":
                        n2v_manipulate.parameters.strategy = "uniform"

        return self

    def set_3D(self, is_3D: bool, axes: str) -> None:
        """
        Set 3D flag and axes.

        Parameters
        ----------
        is_3D : bool
            Whether the algorithm is 3D or not.
        axes : str
            Axes of the data.
        """
        # set the flag and axes (this will not trigger validation at the config level)
        self.algorithm.model.set_3D(is_3D)
        self.data.axes = axes

        # cheap hack: trigger validation
        self.algorithm = self.algorithm

    def model_dump(
        self,
        exclude_defaults: bool = False,  # TODO is this what we want?
        exclude_none: bool = True,
        **kwargs: Dict,
    ) -> Dict:
        """
        Override model_dump method in order to set default values.

        Parameters
        ----------
        exclude_defaults : bool, optional
            Whether to exclude fields with default values or not, by default
            True.
        exclude_none : bool, optional
            Whether to exclude fields with None values or not, by default True.
        **kwargs : Dict
            Keyword arguments.

        Returns
        -------
        dict
            Dictionary containing the model parameters.
        """
        dictionary = super().model_dump(
            exclude_none=exclude_none, exclude_defaults=exclude_defaults, **kwargs
        )

        # change Path into str
        # dictionary["working_directory"] = str(dictionary["working_directory"])

        return dictionary


def load_configuration(path: Union[str, Path]) -> Configuration:
    """
    Load configuration from a yaml file.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the configuration.

    Returns
    -------
    Configuration
        Configuration.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    """
    # load dictionary from yaml
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Configuration file {path} does not exist in " f" {Path.cwd()!s}"
        )

    dictionary = yaml.load(Path(path).open("r"), Loader=yaml.SafeLoader)

    return Configuration(**dictionary)


def save_configuration(config: Configuration, path: Union[str, Path]) -> Path:
    """
    Save configuration to path.

    Parameters
    ----------
    config : Configuration
        Configuration to save.
    path : Union[str, Path]
        Path to a existing folder in which to save the configuration or to an existing
        configuration file.

    Returns
    -------
    Path
        Path object representing the configuration.

    Raises
    ------
    ValueError
        If the path does not point to an existing directory or .yml file.
    """
    # make sure path is a Path object
    config_path = Path(path)

    # check if path is pointing to an existing directory or .yml file
    if config_path.exists():
        if config_path.is_dir():
            config_path = Path(config_path, "config.yml")
        elif config_path.suffix != ".yml":
            raise ValueError(
                f"Path must be a directory or .yml file (got {config_path})."
            )
    else:
        if config_path.suffix != ".yml":
            raise ValueError(f"Path must be a .yml file (got {config_path}).")

    # save configuration as dictionary to yaml
    with open(config_path, "w") as f:
        # dump configuration
        yaml.dump(config.model_dump(), f, default_flow_style=False)

    return config_path
