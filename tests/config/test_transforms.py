import pytest

from careamics.config.transforms import Transform


@pytest.mark.parametrize("name, parameters", 
    [
        ("flip", {}),
        ("flip", {"p": 0.5}),
        ("DefaultManipulateN2V", {"masked_pixel_percentage": 0.2, "roi_size": 11}),
        ("DefaultManipulateN2V", {}),
    ]
)
def test_transform(name, parameters):
    Transform(name=name, parameters=parameters)

