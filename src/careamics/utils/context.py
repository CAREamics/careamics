import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Union


@contextmanager
def cwd(path: Union[str, Path]) -> Iterator[None]:
    """Change the current working directory to the given path.

    Can be useful to generate files in a specific directory:
    ```
    with cwd(path):
        // do something
    ```
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
