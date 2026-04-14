from careamics.config.augmentations import XYFlipConfig
from careamics.dataset.transforms import XYFlip


def test_comptatibility_with_transform():
    """Test that the model allows instantiating a transform."""
    model = XYFlipConfig(flip_y=False, p=0.3)

    # instantiate transform
    XYFlip(**model.model_dump())
