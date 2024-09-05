from typing import Union

import numpy as np
import pytest
import torch
from pydantic import BaseModel, ConfigDict

from careamics.config.likelihood_model import Tensor
from careamics.config.nm_model import Array


class MyArray(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    arr: Array


class MyTensor(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    arr: Tensor


@pytest.mark.parametrize("arr", [np.array([1, 2]), torch.tensor([1, 2])])
def test_serialize_array(arr: Union[np.ndarray, torch.Tensor]):
    """Test array_to_json function."""
    arr_model = MyArray(arr=arr)
    assert arr_model.model_dump() == {"arr": "[1, 2]"}


@pytest.mark.parametrize("arr", [np.array([1, 2]), torch.tensor([1, 2])])
def test_serialize_tensor(arr: Union[np.ndarray, torch.Tensor]):
    """Test array_to_json function."""
    arr_model = MyTensor(arr=arr)
    assert arr_model.model_dump() == {"arr": "[1, 2]"}


def test_deserialize_array():
    """Test list_to_numpy function."""
    config = {"arr": [1, 2]}
    arr_model = MyArray(**config)
    assert np.array_equal(arr_model.arr, np.array([1, 2]))


def test_deserialize_tensor():
    """Test list_to_tensor function."""
    config = {"arr": [1, 2]}
    arr_model = MyTensor(**config)
    assert torch.equal(arr_model.arr, torch.tensor([1, 2]))
