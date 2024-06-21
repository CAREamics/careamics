from careamics.config.transformations import XYRandomRotate90Model
from careamics.transforms import XYRandomRotate90


def test_comptatibility_with_transform():
    """Test that the model allows instantiating a transform."""
    model = XYRandomRotate90Model(p=0.3)

    # instantiate transform
    XYRandomRotate90(**model.model_dump())
