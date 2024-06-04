from careamics.config.transformations import XYFlipModel
from careamics.transforms import XYFlip


def test_comptatibility_with_transform():
    """Test that the model allows instantiating a transform."""
    model = XYFlipModel(flip_y=False, p=0.3)

    # instantiate transform
    XYFlip(**model.model_dump())
