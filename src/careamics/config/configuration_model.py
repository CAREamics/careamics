"""Pydantic CAREamics configuration."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Literal, List, Union
from pprint import pformat

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
    model_validator,
    Field
)

from .algorithm_model import AlgorithmModel
from .data_model import DataModel
from .support import SupportedAlgorithm, SupportedTransform, SupportedPixelManipulation
from .training_model import TrainingModel
from .transformations.n2v_manipulate_model import (
    N2VManipulationModel,
)


class Configuration(BaseModel):
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

    Methods
    -------
    set_3D(is_3D: bool, axes: str, patch_size: List[int]) -> None
        Switch configuration between 2D and 3D.
    set_N2V2(use_n2v2: bool) -> None
        Switch N2V algorithm between N2V and N2V2.
    set_structN2V(
        mask_axis: Literal["horizontal", "vertical", "none"], mask_span: int) -> None
        Set StructN2V parameters.
    model_dump(
        exclude_defaults: bool = False, exclude_none: bool = True, **kwargs: Dict
        ) -> Dict
        Export configuration to a dictionary.

    Notes
    -----
    We provide convenience methods to create standards configurations, for instance
    for N2V, in the `careamics.config.configuration_factory` module.
    >>> from careamics.config.configuration_factory import create_N2V_configuration
    >>> config = create_N2V_configuration(...)

    The configuration can be exported to a dictionary using the model_dump method:
    >>> config.model_dump()

    Configurations can also be exported or imported from yaml files:
    >>> from careamics.config import save_configuration, load_configuration
    >>> save_configuration(config, "config.yml")
    >>> config = load_configuration("config.yml")
    
    Examples
    --------
    Minimum example:
    >>> from careamics.config import Configuration
    >>> config_dict = {
    >>>         "experiment_name": "LevitatingFrog",
    >>>         "algorithm": {
    >>>             "algorithm": "custom",
    >>>             "loss": "n2v",
    >>>             "model": {
    >>>                 "architecture": "UNet",
    >>>             }, 
    >>>         },
    >>>         "training": {
    >>>             "num_epochs": 666,
    >>>         },
    >>>         "data": {
    >>>             "data_type": "tiff",
    >>>             "patch_size": [64, 64],
    >>>             "axes": "SYX",
    >>>         },
    >>>     }
    >>> config = Configuration(**config_dict)
    >>> print(config)
    """

    model_config = ConfigDict(
        validate_assignment=True,
        set_arbitrary_types_allowed=True,
    )

    # version
    version: Literal["0.1.0"] = Field(
        default="0.1.0", 
        description="Version of the CAREamics configuration."
    )

    # required parameters
    experiment_name: str = Field(
        ..., description="Name of the experiment, used to name logs and checkpoints."
    )

    # Sub-configurations
    algorithm: AlgorithmModel
    data: DataModel
    training: TrainingModel

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
    def validate_3D(self: Configuration) -> Configuration:
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
        """Validate algorithm and data compatibility.

        In particular, the validation does the following:

        - If N2V is used, it enforces the presence of N2V_Maniuplate in the transforms
        - If N2V2 is used, it enforces the correct manipulation strategy

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

                # make sure that N2V_MANIPULATE has the correct strategy
                median = SupportedPixelManipulation.MEDIAN.value
                uniform = SupportedPixelManipulation.UNIFORM.value
                strategy = median if use_n2v2 else uniform
                self.data.set_N2V2_strategy(strategy)

        return self

    def __str__(self) -> str:
        """Pretty string reprensenting the configuration.

        Returns
        -------
        str
            Pretty string.
        """
        return pformat(self.model_dump())


    def set_3D(self, is_3D: bool, axes: str, patch_size: List[int]) -> None:
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
        self.data.set_3D(axes, patch_size)

        # cheap hack: trigger validation
        self.algorithm = self.algorithm

    def set_N2V2(self, use_n2v2: bool) -> None:
        """Switch N2V algorithm between N2V and N2V2.

        Parameters
        ----------
        use_n2v2 : bool
            Whether to use N2V2 or not.

        Raises
        ------
        ValueError
            If the algorithm is not N2V.
        """
        if self.algorithm.algorithm == SupportedAlgorithm.N2V:
            self.algorithm.model.n2v2 = use_n2v2
            strategy = SupportedPixelManipulation.MEDIAN.value \
                if use_n2v2 else SupportedPixelManipulation.UNIFORM.value
            self.data.set_N2V2_strategy(strategy)            
        else:
            raise ValueError("N2V2 can only be set for N2V algorithm.")
        
    def set_structN2V(
            self, 
            mask_axis: Literal["horizontal", "vertical", "none"],
            mask_span: int
    ) -> None:
        """Set StructN2V parameters.

        Parameters
        ----------
        mask_axis : Literal["horizontal", "vertical", "none"]
            Axis of the structural mask.
        mask_span : int
            Span of the structural mask.
        """
        self.data.set_structN2V_mask(mask_axis, mask_span)

    def model_dump(
        self,
        exclude_defaults: bool = False, 
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
        elif config_path.suffix != ".yml" and config_path.suffix != ".yaml":
            raise ValueError(
                f"Path must be a directory or .yml or .yaml file (got {config_path})."
            )
    else:
        if config_path.suffix != ".yml" and config_path.suffix != ".yaml":
            raise ValueError(
                f"Path must be a directory or .yml or .yaml file (got {config_path})."
            )

    # save configuration as dictionary to yaml
    with open(config_path, "w") as f:
        # dump configuration
        yaml.dump(config.model_dump(), f, default_flow_style=False)

    return config_path
