This is an advanced tutorial for creating a custom image stack. A custom image stack useful for extending CAREamics to train on not natively supported chunked or memory-mapped formats, when all the data cannot fit into memory.  See the [Image Stack & Loader Tutorial](custom_data.md#custom-image-stack--loader) for a full example on how to train on custom memory-mapped data.

If the data can fit into memory, see the [Custom Read Function Tutorial](custom_data.md#custom-read-function) for a description of an alternative simpler mechanism for training on custom data that can be loaded into memory.

The [ImageStack][careamics.dataset.image_stack.ImageStack] protocol provides an interface, so that CAREamics can interact with data stored in different formats.

## Required Attributes

Any class that implements the `ImageStack` protocol is required to provide the following attributes. They can be implemented as simple instance attributes or as properties.

### `source`: `str`

The source is a string that will be passed through the prediction pipeline to identify the input that the prediction was produced from. Ideally, it should be unique for each image stack instance, usually a natural choice will be a path to the data.

### `data_shape`: `Sequence[int]`

This is the the shape of the data after it has been transformed to the `SC(Z)YX` axes order. The `AxesTransform` class provides a convenient way to calculate this.

```python title="Data Shape Transformation"
--8<-- "tutorials/implementing_an_image_stack.py:transform-data-shape"
```
```python title="Output"
(1, 2, 512, 620)
```

### `data_dtype`: `numpy.typing.DTypeLike`

This is the data type of the data as it's equivalent NumPy representation.

### `original_data_shape`: `Sequence[int]`

This is the original shape of the data, before any transformations.

### `original_axes`: `str`

This is the original axes order of the data, before the transformation. The image stack should be initialized with an axes argument and save that value as an attribute.

## Required Methods

### `extract_patch`

The full signature can be seen in the API reference [extract_patch][careamics.dataset.image_stack.ImageStack.extract_patch].

The extract patch method needs to return a patch that is specified by the input parameters. The patch needs to be transformed to have `C(Z)YX` axes.

If the patch is out of bounds of the image, it should be padded with zeros, for the feature of predicting with a tile size that is larger than the image. This feature is not used during training.

Some useful utility functions are:

- [get_patch_slices][careamics.utils.reshape_array.get_patch_slices]: It returns NumPy-style slice objects in the original axis order.
- [reshape_patch][careamics.utils.reshape_array.reshape_patch]: It will transform the patch from its original axes order to `C(Z)YX`.
- [pad_patch][careamics.dataset.image_stack.image_utils.pad_patch]: For padding patches that are queried from outside of the image bounds.

## Example implementation

All the natively supported image stack implementations can be found in the [image_stack][careamics.dataset.image_stack] package.

This is an additional example for HDF5 data.

```python title="Data Shape Transformation"
--8<-- "tutorials/implementing_an_image_stack.py:hdf5-image-stack"
```

1. We decided to make the source the file path followed by a `#` followed by the internal dataset path, e.g. `/data/hdf5_dataset.h5#/image_0`.
2. HDF5 data can be sliced just like NumPy arrays.