"""Pydantic CAREamics configuration."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Union

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
    model_validator,
)

# ignore typing-only-first-party-import in this file (flake8)
from .algorithm import Algorithm  # noqa: TCH001
from .config_filter import paths_to_str
from .data import Data  # noqa: TCH001
from .training import Training  # noqa: TCH001


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

    model_config = ConfigDict(validate_assignment=True)

    # required parameters
    experiment_name: str
    working_directory: Path

    # Sub-configurations
    algorithm: Algorithm
    data: Data
    training: Training

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
        self.algorithm.is_3D = is_3D
        self.data.axes = axes

        # cheap hack: trigger validation
        self.algorithm = self.algorithm

    @field_validator("experiment_name")
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

    @field_validator("working_directory")
    def parent_directory_exists(cls, workdir: Union[str, Path]) -> Path:
        """
        Validate working directory.

        A valid working directory is a directory whose parent directory exists. If the
        working directory does not exist itself, it is then created.

        Parameters
        ----------
        workdir : Union[str, Path]
            Working directory to validate.

        Returns
        -------
        Path
            Validated working directory.

        Raises
        ------
        ValueError
            If the working directory is not a directory, or if the parent directory does
            not exist.
        """
        path = Path(workdir)

        # check if it is a directory
        if path.exists() and not path.is_dir():
            raise ValueError(f"Working directory is not a directory (got {workdir}).")

        # check if parent directory exists
        if not path.parent.exists():
            raise ValueError(
                f"Parent directory of working directory does not exist (got {workdir})."
            )

        # create directory if it does not exist already
        path.mkdir(exist_ok=True)

        return path

    @model_validator(mode="after")
    def validate_3D(cls, config: Configuration) -> Configuration:
        """
        Check 3D flag validity.

        Check that the algorithm is_3D flag is compatible with the axes in the
        data configuration.

        Parameters
        ----------
        config : Configuration
            Configuration to validate.

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
        if config.algorithm.is_3D and "Z" not in config.data.axes:
            raise ValueError(
                f"Algorithm is 3D but data axes are not (got axes {config.data.axes})."
            )
        elif not config.algorithm.is_3D and "Z" in config.data.axes:
            raise ValueError(
                f"Algorithm is not 3D but data axes are (got axes {config.data.axes})."
            )

        return config

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
            Whether to exclude optional fields with default values or not, by default
            True.
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

        # remove paths
        dictionary = paths_to_str(dictionary)

        dictionary["algorithm"] = self.algorithm.model_dump(
            exclude_optionals=exclude_optionals
        )
        dictionary["data"] = self.data.model_dump()

        dictionary["training"] = self.training.model_dump(
            exclude_optionals=exclude_optionals
        )

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
        yaml.dump(config.model_dump(), f, default_flow_style=False)

    return config_path
