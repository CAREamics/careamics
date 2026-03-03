import enum
import runpy
from pathlib import Path

import pytest

# discover all the code snippet scripts
path_to_guides = Path(__file__).parent
scripts = path_to_guides.rglob("*.py")

# exclude all scripts that end with "errors" to keep a reference of failing examples
# used in the documentation
# TODO: is there a better way to show errors to not have to exclude these scripts?
exclude = ["_errors.py"]
scripts = [
    script
    for script in scripts
    if not any(pattern in script.name for pattern in exclude)
]

# create dynamic enum to hold name and path of each script
# this will allow the name of the script to appear in the pytest summary
dict_scripts = {}
for script in scripts:
    dict_scripts[script.stem] = script

script_enum = enum.Enum("Scripts", dict_scripts)


@pytest.mark.parametrize("script", script_enum)
def test_script_execution(script):
    """Test each the execution of each script."""
    print(f"Script: {script!s}")

    runpy.run_path(str(script.value))
