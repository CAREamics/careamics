"""Utilities for folder versioning."""

from pathlib import Path


def get_run_version(root_folder: str | Path, folder_name: str) -> int:
    """Get the version run based on experiment folder name.

    Versioned folder starts at `<folder_name>_0` and is incremented based on the content
    of the root folder. This method is intended to be used for creating checkpoint
    folders for each training run, without mixing up checkpoints. It should mimick the
    csv logger.

    Parameters
    ----------
    root_folder : str | Path
        The root folder where the versioned folders are located.
    folder_name : str
        The name of the versioned folder.

    Returns
    -------
    int
        The version number for the new run, which is one higher than the highest
        existing version number. If no versioned folders exist, it returns 0.
    """
    path = Path(root_folder)

    if not path.exists():
        return 0

    # get all folders that start with the folder name
    versioned_folders = [
        f for f in path.iterdir() if f.is_dir() and f.name.startswith(folder_name)
    ]

    versions = sorted(
        [
            int(f.name.split(folder_name + "_")[-1])
            for f in versioned_folders
            if f.name.split(folder_name + "_")[-1].isdigit()
        ]
    )
    return versions[-1] + 1 if versions else 0
