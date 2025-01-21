from typing import Union

from torch.nn import Identity

from careamics.config.transformations import N2VManipulateModel
from careamics.transforms import N2VManipulateTorch


def preprocess_factory(transform_configs: Union[list, N2VManipulateModel]):
    """Create a preprocessing transform from N2V.

    Parameters
    ----------
    transform_configs : Union[list, N2VManipulateModel]
        N2V manipulation configuration.

    Returns
    -------
    N2VManipulateTorch or Identity
        N2V manipulation transform or Identity transform.
    """
    if transform_configs:
        return N2VManipulateTorch(n2v_manipulate_config=transform_configs)
    else:
        return Identity()
