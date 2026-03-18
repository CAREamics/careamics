---
icon: lucide/database
description: Data handling
---

# Handling data

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

To train on TIFF files, you can either passa single path to a TIFF, a list of paths to
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
--8<-- "data.py:czi"
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
--8<-- "data.py:zarr"
```

1. Only `array_1` and `array_2` will be loaded.
2. Only arrays in `others`, i.e. `array_2` and `array_3`, will be loaded.
3. Only `array_1` will be loaded.
4. All arrays will be loaded, as they are all specified in the list.


## Custom data formats

CAREamics allows reading formats not natively supported using two mechanisms:

- Simple loading using a python function. If passing a dictionary, all files with the
expectected file extension will be loaded in memory.
- Advanced loading using a custom `ImageStack` implementation, useful for more complex
formats such as chunked or memory-mapped ones.

(soon)

### Read functions
### `ImageStack` loader