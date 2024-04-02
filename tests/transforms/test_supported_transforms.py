from careamics.config.support import SupportedTransform
from careamics.transforms import get_all_transforms


def test_supported_transforms_in_accepted_transforms():
    """Test that all the supported transforms are in the accepted transforms."""
    for transform in SupportedTransform:
        assert transform in get_all_transforms()
