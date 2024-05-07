"""Functions used to create a README.md file for BMZ export."""

from pathlib import Path
from typing import Optional

import yaml

from careamics.config import Configuration
from careamics.utils import cwd, get_careamics_home


def _yaml_block(yaml_str: str) -> str:
    """Return a markdown code block with a yaml string.

    Parameters
    ----------
    yaml_str : str
        YAML string.

    Returns
    -------
    str
        Markdown code block with the YAML string.
    """
    return f"```yaml\n{yaml_str}\n```"


def readme_factory(
    config: Configuration,
    careamics_version: str,
    data_description: Optional[str] = None,
) -> Path:
    """Create a README file for the model.

    `data_description` can be used to add more information about the content of the
    data the model was trained on.

    Parameters
    ----------
    config : Configuration
        CAREamics configuration.
    careamics_version : str
        CAREamics version.
    data_description : Optional[str], optional
        Description of the data, by default None.

    Returns
    -------
    Path
        Path to the README file.
    """
    algorithm = config.algorithm_config
    training = config.training_config
    data = config.data_config

    # create file
    # TODO use tempfile as in the bmz_io module
    with cwd(get_careamics_home()):
        readme = Path("README.md")
        readme.touch()

        # algorithm pretty name
        algorithm_flavour = config.get_algorithm_flavour()
        algorithm_pretty_name = algorithm_flavour + " - CAREamics"

        description = [f"# {algorithm_pretty_name}\n\n"]

        # algorithm description
        description.append("Algorithm description:\n\n")
        description.append(config.get_algorithm_description())
        description.append("\n\n")

        # algorithm details
        description.append(
            f"{algorithm_flavour} was trained using CAREamics (version "
            f"{careamics_version}) with the following algorithm "
            f"parameters:\n\n"
        )
        description.append(
            _yaml_block(yaml.dump(algorithm.model_dump(exclude_none=True)))
        )
        description.append("\n\n")

        # data description
        description.append("## Data description\n\n")
        if data_description is not None:
            description.append(data_description)
            description.append("\n\n")

        description.append("The data was processed using the following parameters:\n\n")

        description.append(_yaml_block(yaml.dump(data.model_dump(exclude_none=True))))
        description.append("\n\n")

        # training description
        description.append("## Training description\n\n")

        description.append("The model was trained using the following parameters:\n\n")

        description.append(
            _yaml_block(yaml.dump(training.model_dump(exclude_none=True)))
        )
        description.append("\n\n")

        # references
        reference = config.get_algorithm_references()
        if reference != "":
            description.append("## References\n\n")
            description.append(reference)
            description.append("\n\n")

        # links
        description.append(
            "## Links\n\n"
            "- [CAREamics repository](https://github.com/CAREamics/careamics)\n"
            "- [CAREamics documentation](https://careamics.github.io/latest/)\n"
        )

        readme.write_text("".join(description))

        return readme.absolute()
