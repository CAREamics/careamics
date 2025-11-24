from careamics.config.transformations import XYRandomRotate90Config
from careamics.transforms import XYRandomRotate90


def test_comptatibility_with_transform():
    """Test that the model allows instantiating a transform."""
    model = XYRandomRotate90Config(p=0.3)

    # instantiate transform
    XYRandomRotate90(**model.model_dump())
