"""Functions used to create a README.md file for BMZ export."""

from pathlib import Path

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
    data_description: str,
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
    data_description : str
        Description of the data.

    Returns
    -------
    Path
        Path to the README file.
    """
    # create file
    # TODO use tempfile as in the bmz_io module
    with cwd(get_careamics_home()):
        readme = Path("README.md")
        readme.touch()

        # algorithm pretty name
        algorithm_flavour = config.get_algorithm_friendly_name()
        algorithm_pretty_name = algorithm_flavour + " - CAREamics"

        description = [f"# {algorithm_pretty_name}\n\n"]

        # data description
        description.append("## Data description\n\n")
        description.append(data_description)
        description.append("\n\n")

        # algorithm description
        description.append("## Algorithm description:\n\n")
        description.append(config.get_algorithm_description())
        description.append("\n\n")

        # configuration description
        description.append("## Configuration\n\n")

        description.append(
            f"{algorithm_flavour} was trained using CAREamics (version "
            f"{careamics_version}) using the following configuration:\n\n"
        )

        description.append(_yaml_block(yaml.dump(config.model_dump(exclude_none=True))))
        description.append("\n\n")

        # validation
        description.append("# Validation\n\n")

        description.append(
            "In order to validate the model, we encourage users to acquire a "
            "test dataset with ground-truth data. Comparing the ground-truth data "
            "with the prediction allows unbiased evaluation of the model performances. "
            "This can be done for instance by using metrics such as PSNR, SSIM, or"
            "MicroSSIM. In the absence of ground-truth, inspecting the residual image "
            "(difference between input and predicted image) can be helpful to identify "
            "whether real signal is removed from the input image.\n\n"
        )

        # references
        reference = config.get_algorithm_references()
        if reference != "":
            description.append("## References\n\n")
            description.append(reference)
            description.append("\n\n")

        # links
        description.append(
            "# Links\n\n"
            "- [CAREamics repository](https://github.com/CAREamics/careamics)\n"
            "- [CAREamics documentation](https://careamics.github.io/)\n"
        )

        readme.write_text("".join(description))

        return readme.absolute()
