"""I/O functions for Configuration objects."""

from pathlib import Path
from typing import Union

import yaml

from careamics.config import Configuration


def load_configuration(path: Union[str, Path]) -> Configuration:
    """
    Load configuration from a yaml file.

    Parameters
    ----------
    path : str or Path
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
    path : str or Path
        Path to a existing folder in which to save the configuration, or to a valid
        configuration file path (uses a .yml or .yaml extension).

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
        yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)

    return config_path
