"""
Context submodule.

A convenience function to change the working directory in order to save data.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Union


def get_careamics_home() -> Path:
    """Return the CAREamics home directory.

    CAREamics home directory is a hidden folder in home.

    Returns
    -------
    Path
        CAREamics home directory path.
    """
    home = Path.home() / ".careamics"

    if not home.exists():
        home.mkdir(parents=True, exist_ok=True)

    return home


@contextmanager
def cwd(path: Union[str, Path]) -> Iterator[None]:
    """
    Change the current working directory to the given path.

    This method can be used to generate files in a specific directory, once out of the
    context, the working directory is set back to the original one.

    Parameters
    ----------
    path : Union[str,Path]
        New working directory path.

    Returns
    -------
    Iterator[None]
        None values.

    Examples
    --------
    The context is whcnaged within the block and then restored to the original one.

    >>> with cwd(my_path):
    ...     pass # do something
    """
    path = Path(path)

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    old_pwd = Path(".").absolute()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_pwd)
