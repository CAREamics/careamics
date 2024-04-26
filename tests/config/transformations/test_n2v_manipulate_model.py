import pytest

from careamics.config.transformations.n2v_manipulate_model import N2VManipulateModel


def test_odd_roi_and_mask():
    """Test that errors are thrown if we pass even roi and mask sizes."""
    # no error
    model = N2VManipulateModel(name="N2VManipulate", roi_size=3, struct_mask_span=7)
    assert model.roi_size == 3
    assert model.struct_mask_span == 7

    # errors
    with pytest.raises(ValueError):
        N2VManipulateModel(name="N2VManipulate", roi_size=4, struct_mask_span=7)

    with pytest.raises(ValueError):
        N2VManipulateModel(name="N2VManipulate", roi_size=3, struct_mask_span=6)


def test_extra_parameters():
    """Test that errors are thrown if we pass extra parameters."""
    with pytest.raises(ValueError):
        N2VManipulateModel(
            name="N2VManipulate", roi_size=3, struct_mask_span=7, extra_param=1
        )
