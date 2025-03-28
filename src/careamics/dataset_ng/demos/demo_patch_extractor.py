# %%
import numpy as np

from careamics.dataset_ng.patch_extractor.image_stack import InMemoryImageStack

# %%
from careamics.dataset_ng.patch_extractor.patch_extractor_factory import (
    create_array_extractor,
)
from careamics.dataset_ng.patching_strategies import RandomPatchingStrategy

# %%
array = np.arange(36).reshape(6, 6)
image_stack = InMemoryImageStack.from_array(data=array, axes="YX")
image_stack.extract_patch(sample_idx=0, coords=(2, 2), patch_size=(3, 3))

# %%
rng = np.random.default_rng()

# %%
# define example data
array1 = np.arange(36).reshape(1, 6, 6)
array2 = np.arange(50).reshape(2, 5, 5)
target1 = rng.integers(0, 1, size=array1.shape, endpoint=True)
target2 = rng.integers(0, 1, size=array2.shape, endpoint=True)

# %%
print(array1)
print(array2)
print(target1)
print(target2)

# %%
# define example readers
input_patch_extractor = create_array_extractor([array1, array2], axes="SYX")
target_patch_extractor = create_array_extractor([target1, target2], axes="SYX")

# %%
# generate random patch specification
data_shapes = [
    image_stack.data_shape for image_stack in input_patch_extractor.image_stacks
]
patch_specs_generator = RandomPatchingStrategy(data_shapes, patch_size=(2, 2))
patch_specs = patch_specs_generator.get_patch_spec(18)

# %%
# extract a subset of patches
input_patch_extractor.extract_patch(**patch_specs)

# %%
target_patch_extractor.extract_patch(**patch_specs)
