As mentioned in the [Data Preparation Guide](../current/data.md#custom-data-formats) CAREamics provides two mechanisms for training and predicting on custom data types. There is:

1. what we call using a ["read function"](#custom-read-function), which can be used to read all the data to memory; and
2. what we call using an ["image stack loader"](#custom-image-stack--loader), which is more advanced but can be used for chunked or memory-mapped file formats.

## Custom Read Function

Any function that loads image data from a path and outputs a numpy array can be used.

This example will show how data saved in the `.npy` format can be loaded for training and prediction. 

First, we will save some toy data and create a CAREamics configuration object.

```python title="Custom data configuration"
--8<-- "tutorials/data_custom_read_func.py:data-config"
```

1. The `data_type` must be set to `"custom"`.
2. The axes of each file.

To train and predict on the data we need to define a function to read the data that matches the protocol described by [`ReadFunc`][careamics.file_io.ReadFunc]. That is, the first argument MUST be named `file_path` and the return type must be a numpy array. Here we just make a simple wrapper around `numpy.load` to have the correct function signature.

For training with [`CAREamist`][careamics.careamist.CAREamist] we pass our custom loading function to the `loading` argument of [`CAREamist.train`][careamics.CAREamist.train], it needs to be contained in the [`ReadFuncLoading`][careamics.ReadFuncLoading] dataclass.

```python title="Custom data training"
--8<-- "tutorials/data_custom_read_func.py:training"
```

1. The input works the same as for tiff, it can be a single file, a list of files or a directory. Here we demonstrate passing a directory.
2. The arguments for custom loading are wrapped in a data class [`ReadFuncLoading`][careamics.ReadFuncLoading].
3. Our custom read function.
4. An extension filter, it uses glob-style pattern matching. This allows us to pass a directory as input.

Prediction works very similarly to training. [`CAREamist.predict`][careamics.CAREamist.predict] outputs the source of the predictions which we can verify are the paths of our data.

```python title="Custom data prediction"
--8<-- "tutorials/data_custom_read_func.py:prediction"
```

1. The same arguments can also be passed to [`CAREamist.predict_to_disk`][careamics.CAREamist.predict_to_disk].

```python title="Output"
['data/image_0.npy',
 'data/image_1.npy',
 'data/image_2.npy',
 'data/image_3.npy',
 'data/image_4.npy']
```

## Custom Image Stack & Loader

Training and predicting on a custom memory-mapped or chunked file format is more complex, but it enables training without loading an entire image file into memory at once. It involves implementing an [ImageStack][careamics.dataset.image_stack.ImageStack] class and an [ImageStackLoader][careamics.dataset.image_stack_loader] function to load the image stacks.

This example will demonstrated how data from a HDF5 file can be loaded for training and prediction.

First, we will save some toy data and create a CAREamics configuration object.

```python title="Custom data configuration"
--8<-- "tutorials/data_custom_image_stack.py:data-config"
```

1. The `data_type` must be set to `"custom"`.
2. The axes of each HDF5 dataset.

Now we will define our custom `HDF5ImageStack` and a `load_hd5fs` function. See the [Implementing an Image Stack Tutorial](implementing_an_image_stack.md) for a more in depth explanation of how to create an image stack class.

To adhere to the [ImageStackLoader][careamics.dataset.image_stack_loader] protocol the `load_hdf5s` function MUST have a `source` argument and an `axes` argument. The `source` argument can have any type, and the `axes` argument Must be a string - a subset of `"SCTZYX"`. The return type MUST be a sequence of `ImageStack` objects. Additional arguments are allowed.

!!! note "Supervised Algorithms, e.g. CARE"

    It is up to the loading function to always load images in a deterministic order so the inputs are matched to their corresponding targets. In our example, `h5py` will return the group keys in alpha-numeric order; so we know that if we provide a target `.h5` file which has the same structure as our input file, then the target images are returned in the same order as our input images.

```python title="Creating a Custom Image Stack & Loader"
--8<-- "tutorials/data_custom_image_stack.py:image-stack-loader"
```

1. The source property is used track the data, and will be returned alongside the predictions. It should be unique for each image stack.
2. Adheres [ImageStackLoader][careamics.dataset.image_stack_loader] protocol call signature.

Now training and prediction is relatively simple, we simply pass our loading function to [`CAREamist.train`][careamics.CAREamist.train] and [`CAREamist.predict`][careamics.CAREamist.predict]. The loading function needs to be wrapped in the [ImageStackLoading][careamics.ImageStackLoading] dataclass, where additional arguments to the function can also be included, if required.

```python title="Training and Prediction"
--8<-- "tutorials/data_custom_image_stack.py:train-pred"
```

1. The input type corresponds to the `source` type in our loading function, a `h5py.File` object.
2. Our loading function wrapped in the [ImageStackLoading][careamics.ImageStackLoading] dataclass.
3. These will match the format we that defined in `HDF5ImageStack.source`.

```python title="Output"
['data/dataset.h5#/image_0',
 'data/dataset.h5#/image_1',
 'data/dataset.h5#/image_2',
 'data/dataset.h5#/image_3',
 'data/dataset.h5#/image_4']
```