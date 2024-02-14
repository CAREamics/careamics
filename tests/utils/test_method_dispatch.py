from pathlib import Path
from dataclasses import dataclass

import numpy as np

from careamics.utils import method_dispatch

@dataclass
class TestObject:
    """Simple data class"""
    value: int     


class TestClass:
    """Test class with dispatched methods."""

    @method_dispatch
    def main_function(self, arg: TestObject):
        return arg.value

    @main_function.register
    def _main_function_with_path(self, arg1: Path, arg2: Path):
        min_len = min(len(arg1.name), len(arg2.name))
        return self.main_function(TestObject(min_len))

    @main_function.register
    def _main_function_with_array(self, arg: np.ndarray):
        mean = np.mean(arg)
        return self.main_function(TestObject(mean))



def test_method_dispatch():
    """Test that method dispatch works on an instance method, dispatching the call
    based on the second argument of the method.
    """
    test_class = TestClass()

    # test with TestObject
    test_object = TestObject(5)
    assert test_class.main_function(test_object) == 5

    # test with Path
    test_path1 = Path("test-longer")
    test_path2 = Path("test-short")
    assert test_class.main_function(test_path1, test_path2) == len(test_path2.name)

    # test with np.ndarray
    test_array = np.array([1, 2, 3])
    assert test_class.main_function(test_array) == 2
