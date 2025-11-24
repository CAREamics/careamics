import numpy as np

from careamics.dataset_ng.image_stack import InMemoryImageStack


def test_hashing_source():
    """Test that the source attribute of InMemoryImageStack uniquely identifies the
    source array."""

    rng = np.random.default_rng()

    array1 = 50 * rng.random((10, 3, 64, 64)) + 3
    array2 = 50 * rng.random((10, 3, 64, 64)) + 3
    array3 = array1.copy()

    stack1 = InMemoryImageStack.from_array(data=array1, axes="SCYX")
    stack2 = InMemoryImageStack.from_array(data=array2, axes="SCYX")
    stack3 = InMemoryImageStack.from_array(data=array3, axes="SCYX")

    assert stack1.source != stack2.source
    assert stack1.source == stack3.source
