"""Pre-processing factory."""

from typing import Optional

from torch.nn import Identity

from careamics.config.transformations import N2VManipulateModel
from careamics.transforms import N2VManipulateTorch


def preprocess_factory(transform_configs: Optional[N2VManipulateModel]):
    """Create a preprocessing transform from N2V.

    Parameters
    ----------
    transform_configs : N2VManipulateModel or None
        N2V manipulation configuration.

    Returns
    -------
    N2VManipulateTorch or Identity
        N2V manipulation or Identity transform.
    """
    if transform_configs is not None:
        return N2VManipulateTorch(n2v_manipulate_config=transform_configs)
    else:
        return Identity()
