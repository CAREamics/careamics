---
icon: lucide/database
description: Data preparation
---

# Data preparation

CAREamics supports by default data stored in memory as numpy arrays, but also data
stored on disk in the form of TIFF, CZI and Zarr files. Each format comes with
particular constraints.

## Arrays

Arrays are the simplest and fastest way to train and predict with CAREamics, they can be
passed as is to the `CAREamist`.

## TIFF

As TIFF are widely used, they are the most common use case of CAREamics. TIFF loading
is compatible with [in-memory training](./configuration.md#training-in-memory). If the
data is too large to fit in memory, CAREamics will train by loading files from disk,
one at a time and cycle through them to extract patches. While slower, this ensure that
training is performed over the entire set of files.

To train on TIFF files, you can either pass a single path to a TIFF, a list of paths to
TIFF files, or a path to a directory containing TIFF files. In the latter case, all
TIFF files in the directory will be used for training.

## CZI

The CZI format is used by Zeiss microscopes, and has constraints on the axes that should
be specified:
- `S`, `C`, `Y` and `X` are always present and in this order. Even in the absence of
channels, the `C` axis should be present.
- If `T` or `Z` is specified (`axes=SCTYX` or `axes=SCZYX`), then they will be used
as depth axis (as `Z`).
- `T` and `Z` are mutually exclusive.

```python title="Using CZI""
--8<-- "current/data.py:czi"
```

1. We set the data type to `czi`.
2. Axes must be one of `SCYX`, `SCTYX` or `SCZYX`.
3. If `Z` or `T` is specified, we need to pass a 3D `patch_size`.
4. The number of channels must be specified, and can be a singleton if there is a single
channel.

Only paths to CZI files can be used as input to CAREamics. Passing a directory containing
multiple CZI files is not accepted as input, the list of files should be passed
explicitly.


!!! note "CARE and Noise2Noise"

    This example is valid for CARE and Noise2Noise, albeit with their respective
    function to create a configuration and the difference in number of channels
    parameter naming. See the [configuration](./configuration.md#channels) section for
    more details.


## Zarr

Zarr is a chunked format allowing to train on very large data without having to load
it in memory. Because a Zarr file can hold multiple arrays, and can have arbitrary
organization, we defined a flexible way to specify which data should be used.

There are three ways to specify which array(s) should be used for training or
prediction:

- Pointing to a Zarr file (`path/to/file.zarr`)
- Pointing to a single Zarr group using a URI (`file://path/to/file.zarr/group_name`)
- Pointing to a single Zarr array using a URI (`file://path/to/file.zarr/group_name/array_name`)

All these options are valid, and multiple can be bundled together in a list. The only constraints 
is that the list must contain only URIs or only paths to Zarr files, but not a mix of both. Zarr 
URIs can be constructed by getting a reference to a group or array, and calling
`group.store_path` or `array.store_path`.

In the following example, we construct a Zarr file with arrays in different hierarchy
levels, and showcase various ways to specify which array should be used for training.

```python title="Using Zarr""
--8<-- "current/data.py:zarr"
```

1. Only `array_1` and `array_2` will be loaded.
2. Only arrays in `others`, i.e. `array_2` and `array_3`, will be loaded.
3. Only `array_1` will be loaded.
4. All arrays will be loaded, as they are all specified in the list.

!!! note "OME-Zarr"

    Currently, we are ignoring whether a file is an OME-Zarr or not. As a result,
    simply passing a path to the Zarr file will fail, since CAREamics will expect
    arrays in the root of the file. 

    Therefore, to use an OME-Zarr file, you need to specify the URI to the array
    you want to train on.

    In the near future, we will add full OME-Zarr support.


!!! warning "Multiscales OME-Zarr and Noise2Void"

    Noise2Void is very sensitive to the noise distribution in the data, if a an image
    has been downscaled, correlations may have been introduced in the noise, causing
    Noise2Void to perform poorly. We advise training on the raw unprocessed data if
    available.


## Custom data formats

CAREamics allows reading formats not natively supported using two mechanisms:

- Simple loading using a python function. All files with the expected file extension will be loaded in memory.
- Advanced loading using a custom `ImageStack` implementation, useful for more complex formats such as chunked or memory-mapped ones.

### Custom Read Function

Any function that loads image data from a path and outputs a numpy array can be used.

This example will show how data saved in the `.npy` format can be loaded for training and prediction. 

First, we will save some toy data and create a CAREamics configuration object.

```python title="Custom data configuration"
--8<-- "current/data_custom_read_func.py:data-config"
```

1. The `data_type` must be set to `"custom"`.
2. The axes of each file.

To train and predict on the data we need to define a function to read the data that matches the protocol described by [`ReadFunc`][careamics.file_io.ReadFunc]. That is, the first argument MUST be named `file_path` and the return type must be a numpy array. Here we just make a simple wrapper around `numpy.load` to have the correct function signature.

For training with [`CAREamist`][careamics.careamist.CAREamist] we pass our custom loading function to the `loading` argument of [`CAREamist.train`][careamics.CAREamist.train], it needs to be contained in the [`ReadFuncLoading`][careamics.ReadFuncLoading] dataclass.

```python title="Custom data training"
--8<-- "current/data_custom_read_func.py:training"
```

1. The input works the same as for tiff, it can be a single file, a list of files or a directory. Here we demonstrate passing a directory.
2. The arguments for custom loading are wrapped in a data class [`ReadFuncLoading`][careamics.ReadFuncLoading].
3. Our custom read function.
4. An extension filter, it uses glob-style pattern matching. This allows us to pass a directory as input.

Prediction works very similarly to training. [`CAREamist.predict`][careamics.CAREamist.predict] outputs the source of the predictions which we can verify are the paths of our data.

```python title="Custom data prediction"
--8<-- "current/data_custom_read_func.py:prediction"
```

1. The same arguments can also be passed to [`CAREamist.predict_to_disk`][careamics.CAREamist.predict_to_disk].

```python title="Output"
['data/image_0.npy',
 'data/image_1.npy',
 'data/image_2.npy',
 'data/image_3.npy',
 'data/image_4.npy']
```

### Custom Image Stack & Loader

Training on custom file format that is memory-mapped or chunked, enabling not all of the data to be loaded into memory at one time, is more complex. It involves defining a class  that follows the [ImageStack][careamics.dataset.image_stack.ImageStack] protocol to represent the image data and a function that follows the [ImageStackLoader][careamics.dataset.image_stack_loader] protocol to load those image stacks.

This example will demonstrated how data from a HDF5 file can be loaded for training and prediction.

First, we will save some toy data and create a CAREamics configuration object.

```python title="Custom data configuration"
--8<-- "current/data_custom_image_stack.py:data-config"
```

1. The `data_type` must be set to `"custom"`.
2. The axes of each HDF5 dataset.

Now we will define our custom `HDF5ImageStack` and a `load_hd5fs` function. See the tutorials section for a more in depth explanation of how to create an image stack class.

To adhere to the [ImageStackLoader][careamics.dataset.image_stack_loader] protocol the `load_hdf5s` function MUST have a `source` argument and an `axes` argument. The `source` argument can have any type, and the `axes` argument Must be a string - a subset of `"SCTZYX"`. The return type MUST be a sequence of `ImageStack` objects. Additional arguments are allowed.

```python title="Creating a Custom Image Stack & Loader"
--8<-- "current/data_custom_image_stack.py:image-stack-loader"
```

1. The source property is used track the data, and will be returned alongside the predictions. It should be unique for each image stack.
2. Adheres [ImageStackLoader][careamics.dataset.image_stack_loader] protocol call signature.

Now training and prediction is relatively simple, we simply pass our loading function to [`CAREamist.train`][careamics.CAREamist.train] and [`CAREamist.predict`][careamics.CAREamist.predict]. The loading function needs to be wrapped in the [ImageStackLoading][careamics.ImageStackLoading] dataclass, where additional arguments to the function can also be included, if required.

```python title="Training and Prediction"
--8<-- "current/data_custom_image_stack.py:train-pred"
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