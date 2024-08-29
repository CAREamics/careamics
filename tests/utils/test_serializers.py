import json
from typing import Union
from typing_extensions import Annotated

import numpy as np
import pytest
import torch
from pydantic import BaseModel, ConfigDict, PlainSerializer

from careamics.utils import array_to_json

Array = Annotated[
    Union[np.ndarray, torch.Tensor], 
    PlainSerializer(array_to_json, return_type=str)
]

class MyArray(BaseModel):
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    arr: Array   

@pytest.mark.parametrize(
    "arr", 
    [np.array([1, 2]), torch.tensor([1, 2])]
)
def test_array_to_json(arr: Union[np.ndarray, torch.Tensor]):
    """Test array_to_json function."""
    arr_model = MyArray(arr=arr)
    assert arr_model.model_dump() == {"arr": "[1, 2]"}