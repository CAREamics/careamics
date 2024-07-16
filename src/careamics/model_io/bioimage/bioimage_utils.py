"""Bioimage.io utils."""

from pathlib import Path
from typing import Union


def _format_bmz_name(name: str) -> str:
    """Format the bioimage.io model name.

    Parameters
    ----------
    name : str
        Name of the bioimage.io model.

    Returns
    -------
    str
        Formatted name.
    """
    return name.replace(" ", "_").replace("(", "_").replace(")", "_").lower()


def format_bmz_path(path: Path, name: str) -> Path:
    """Format the bioimage.io model filename.

    Parameters
    ----------
    path : pathlib.Path
        Path to the bioimage.io model.
    name : str
        Name of the bioimage.io model.

    Returns
    -------
    pathlib.Path
        Path to the bioimage.io model with the formatted filename.
    """
    if path.suffix == "":
        # file represents a directory, we make sure it exists
        path.mkdir(parents=True, exist_ok=True)

        # add the name to the path
        path = path / (_format_bmz_name(name) + ".zip")
    else:
        # path has a suffix, we make sure its prents exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # make sure it has the correct suffix
        if path.suffix != ".zip":
            path = path.with_suffix(".zip")

    return path


def get_unzip_path(zip_path: Union[Path, str]) -> Path:
    """Generate unzipped folder path from the bioimage.io model path.

    Parameters
    ----------
    zip_path : Path
        Path to the bioimage.io model.

    Returns
    -------
    Path
        Path to the unzipped folder.
    """
    zip_path = Path(zip_path)

    return zip_path.parent / (str(zip_path.name) + ".unzip")


def create_env_text(pytorch_version: str) -> str:
    """Create environment yaml content for the bioimage model.

    This installs an environemnt with the specified pytorch version and the latest
    changes to careamics.

    Parameters
    ----------
    pytorch_version : str
        Pytorch version.

    Returns
    -------
    str
        Environment text.
    """
    env = (
        f"name: careamics\n"
        f"dependencies:\n"
        f"  - python=3.10\n"
        f"  - pytorch={pytorch_version}\n"
        f"  - torchvision={pytorch_version}\n"
        f"  - pip\n"
        f"  - pip:\n"
        f"    - git+https://github.com/CAREamics/careamics.git\n"
    )

    return env
