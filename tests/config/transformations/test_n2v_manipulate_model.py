import pytest

from careamics.config.transformations import N2VManipulateModel
from careamics.transforms import N2VManipulate


def test_odd_roi_and_mask():
    """Test that errors are thrown if we pass even roi and mask sizes."""
    # no error
    model = N2VManipulateModel(roi_size=3, struct_mask_span=7)
    assert model.roi_size == 3
    assert model.struct_mask_span == 7

    # errors
    with pytest.raises(ValueError):
        N2VManipulateModel(roi_size=4, struct_mask_span=7)

    with pytest.raises(ValueError):
        N2VManipulateModel(roi_size=3, struct_mask_span=6)


def test_extra_parameters():
    """Test that errors are thrown if we pass extra parameters."""
    with pytest.raises(ValueError):
        N2VManipulateModel(roi_size=3, struct_mask_span=7, extra_param=1)


def test_comptatibility_with_transform():
    """Test that the model allows instantiating a transform."""
    model = N2VManipulateModel(roi_size=3, struct_mask_span=7, strategy="median")

    # instantiate transform
    N2VManipulate(**model.model_dump())
