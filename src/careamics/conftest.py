"""File used to discover python modules and run doctest.

See https://sybil.readthedocs.io/en/latest/use.html#pytest
"""

from pathlib import Path

import pytest
from pytest import TempPathFactory
from sybil import Sybil
from sybil.parsers.codeblock import PythonCodeBlockParser
from sybil.parsers.doctest import DocTestParser


@pytest.fixture(scope="module")
def my_path(tmpdir_factory: TempPathFactory) -> Path:
    return tmpdir_factory.mktemp("my_path")


pytest_collect_file = Sybil(
    parsers=[
        DocTestParser(),
        PythonCodeBlockParser(future_imports=["print_function"]),
    ],
    pattern="*.py",
    fixtures=["my_path"],
).pytest()
