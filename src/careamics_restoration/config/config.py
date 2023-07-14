from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union

import yaml
from pydantic import BaseModel, FieldValidationInfo, field_validator, model_validator

from .algorithm import Algorithm
from .config_filter import paths_to_str
from .data import Data
from .prediction import Prediction
from .training import Training

# TODO: test if parameter parent_config at the top of the config could work
# TODO: is stage necessary? it seems to bring a lot of compelxity for little gain
# TODO: test configuration mutability and whether the validators are called when
# changing a field
# TODO: config version?
# TODO: for the working directory to work it should probably be set globally when
# starting the engine
# TODO: option to dump the whole configuration with all the defaults


class Configuration(BaseModel):
    """Main configuration class.

    The minimum configuration is composed of the following fields:
    - experiment_name:
        name of the experiment, composed solely of letters, numbers, underscores,
        dashes and spaces.
    - working_directory:
        path to the working directory, its parents folders must exist. If the working
        directory does not exist itself, it is then created.
    - algorithm:
        algorithm configuration
    - training or prediction:
        training or prediction configuration, one of the two configuration must be
        provided.

    Attributes
    ----------
    experiment_name : str
        Name of the experiment.
    working_directory : Union[str, Path]
        Path to the working directory.
    trained_model : Optional[str]
        Path to the trained model.
    algorithm : Algorithm
        Algorithm configuration.
    training : Optional[Training]
        Training configuration.
    prediction : Optional[Prediction]
        Prediction configuration.
    """

    # required parameters
    experiment_name: str
    working_directory: Path

    # Optional field
    trained_model: Optional[str] = None

    # Sub-configurations
    algorithm: Algorithm
    data: Data

    # Optional sub-configurations
    training: Optional[Training] = None
    prediction: Optional[Prediction] = None

    @field_validator("experiment_name")
    def no_symbol(cls, name: str) -> str:
        """Validate experiment name.

        A valid experiment name is a non-empty string with only contains letters,
        numbers, underscores, dashes and spaces.
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
        """Validate working directory.

        A valid working directory is a directory whose parent directory exists. If the
        working directory does not exist itself, it is then created.
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

    @field_validator("trained_model")
    def trained_model_exists(
        cls, model_path: str, values: FieldValidationInfo
    ) -> Union[str, Path]:
        """Validate trained model path.

        The model path must point to an existing .pth file, either relative to
        the working directory or with an absolute path.

        Parameters
        ----------
        model_path : str
            Path to the trained model.
        values : FieldValidationInfo
            Information about the other fields.

        Returns
        -------
        Union[str, Path]
            Path to the trained model.


        Raises
        ------
        ValueError
            If the path does not point to an existing .pth file.
        """
        if "working_directory" not in values.data:
            raise ValueError(
                "Working directory is not defined, check if was is correctly entered."
            )

        workdir = values.data["working_directory"]
        relative_path = Path(workdir, model_path)
        absolute_path = Path(model_path)

        # check suffix
        if absolute_path.suffix != ".pth":
            raise ValueError(f"Path to model must be a .pth file (got {model_path}).")

        # check if relative or absolute
        if absolute_path.exists() or relative_path.exists():
            return model_path
        else:
            raise ValueError(
                f"Path to model does not exist. "
                f"Tried absolute ({absolute_path}) and relative ({relative_path})."
            )

    @model_validator(mode="after")
    def at_least_training_or_prediction(cls, config: Configuration) -> Configuration:
        """Checks training/prediction config validity.

        Check that at least one of training or prediction is defined, and that
        the corresponding data path is as well.

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
            If neither training nor prediction is defined, and if their corresponding
            paths are not defined.
        """
        # check that at least one of training or prediction is defined
        if config.training is None and config.prediction is None:
            raise ValueError("At least one of training or prediction must be defined.")

        # check that the corresponding data path is defined as well
        if config.training is not None and config.data.training_path is None:
            raise ValueError(
                "Training configuration is defined but training data path is not."
            )
        if config.prediction is not None and config.data.prediction_path is None:
            raise ValueError(
                "Prediction configuration is defined but prediction data path is not."
            )

        return config

    @model_validator(mode="after")
    def validate_3D(cls, config: Configuration) -> Configuration:
        """Checks 3D flag validity.

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

    def model_dump(self, exclude_optionals: bool = True, *args, **kwargs) -> dict:
        """Override model_dump method.

        The purpose is to ensure export smooth import to yaml. It includes:
            - remove entries with None value
            - remove optional values if they have the default value

        Parameters
        ----------
        exclude_optionals : bool, optional
            Whether to exclude optional fields with default values or not, by default
            True.

        Returns
        -------
        dict
            Dictionary containing the model parameters
        """
        dictionary = super().model_dump(exclude_none=True)

        # remove paths
        dictionary = paths_to_str(dictionary)

        dictionary["algorithm"] = self.algorithm.model_dump(
            exclude_optionals=exclude_optionals
        )
        dictionary["data"] = self.data.model_dump(exclude_optionals=exclude_optionals)

        # same for optional fields
        if self.training is not None:
            dictionary["training"] = self.training.model_dump(
                exclude_optionals=exclude_optionals
            )
        if self.prediction is not None:
            dictionary["prediction"] = self.prediction.model_dump(
                exclude_optionals=exclude_optionals
            )

        return dictionary


def load_configuration(path: Union[str, Path]) -> Configuration:
    """Load configuration from a yaml file.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the configudation.

    Returns
    -------
    Configuration
        Configuration.
    """
    # load dictionary from yaml
    dictionary = yaml.load(Path(path).open("r"), Loader=yaml.SafeLoader)

    return Configuration(**dictionary)


def save_configuration(config: Configuration, path: Union[str, Path]) -> Path:
    """Save configuration to path.

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
    if config_path.is_dir():
        config_path = Path(config_path, "config.yml")
    elif config_path.is_file() and config_path.suffix != ".yml":
        raise ValueError(f"Path must be a directory or .yml file (got {config_path}).")

    # save configuration as dictionary to yaml
    with open(config_path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False)

    return config_path
