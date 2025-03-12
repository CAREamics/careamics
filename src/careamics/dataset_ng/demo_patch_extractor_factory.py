# %%
import numpy as np

from careamics.config import create_n2n_configuration
from careamics.config.support import SupportedData
from careamics.dataset_ng.patch_extractor.patch_extractor_factory import (
    create_patch_extractors,
)

rng = np.random.default_rng()

# %%
# define example data
array1 = np.arange(36).reshape(1, 6, 6)
array2 = np.arange(50).reshape(2, 5, 5)
target1 = rng.integers(0, 1, size=array1.shape, endpoint=True)
target2 = rng.integers(0, 1, size=array2.shape, endpoint=True)

# %%
config = create_n2n_configuration(
    "test_exp",
    data_type="array",
    axes="SYX",
    patch_size=[8, 8],
    batch_size=1,
    num_epochs=1,
)
data_config = config.data_config

# %%
data_type = SupportedData(data_config.data_type)
train_inputs, train_targets = create_patch_extractors(
    [array1, array2], [target1, target2], axes=data_config.axes, data_type=data_type
)

# %%
train_inputs.extract_patch(data_idx=0, sample_idx=0, coords=(2, 2), patch_size=(3, 3))
