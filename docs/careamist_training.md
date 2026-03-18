---
icon: lucide/network
description: Quick start
---

# Training CAREamics

After having [created a configuration](./configuration.md) and 
[assembled the training data](./data.md), you are ready to train CAREamics. The preferred
way to train with CAREamics is to create `CAREamist` object.

```python
--8<-- "careamist_training.py:careamist_from_cfg"
```

1. Here the configuration can either be passed as we have seen in the
[configuration](./configuration.md) section, or as a path to configuration [saved to the
disk](./configuration.md#saving-and-loading).

### Working directory

By default CAREamics will save the training logging data and the checkpoints in the
root directory from which it is called. However, you can pass `work_dir` to the 
`CAREamist` to specify a different directory.

```python
--8<-- "careamist_training.py:careamist_workdir"
```

### Disabling progress bar

The PyTorch Lightning progress bar is be verbose, and can be disabled by passing
`enable_progress_bar=False` to the `CAREamist`.

```python
--8<-- "careamist_training.py:careamist_pb"
```

## Training basics

### Without validation

Once you have a `CAREamist` object, you can train CAREamics with the `train` method. Data 
is expected to be coherent with the [choice in the configuration](./configuration.md#simple-configuration)
and the [data section](./data.md).

Without validation, CAREamics will split the data into training and validation internally.
The amount of validation data can be set [in the configuration](./configuration.md#validation).

=== "Noise2Void"
    
    ```python title="Training Noise2Void"
    --8<-- "careamist_training.py:train_n2v_no_val"
    ```
 
    1. `train_data` should be an array, a path to a file, a path to a folder, or a list
    of array/paths.

=== "CARE/N2N"

    ```python title="Training CARE"
    --8<-- "careamist_training.py:train_care_no_val"
    ```

    1. `train_data` should be an array, a path to a file, a path to a folder, or a list
    of array/paths.
    2. For CARE and N2N, a target should be provided. It needs to be of the same type as
    `train_data` and pairs are formed by matching the order of the data and target lists.

!!! note "Passing a directory"

    If you are passing a path to a directory and the data type if `tiff` (or a custom
    type with the required reading utilities, see [custom data](#custom-data)), then
    all the files with the expected file extension in that directory will be used for
    training.
    
    Passing a dictionary is not compatible with CZI or Zarr data.

### With validation

When passing validation, the only constraint is that the validation data is of the same
type as the training data. The amount of validation data is determined by the size of
the validation data.


=== "Noise2Void"
    
    ```python title="Training Noise2Void with validation"
    --8<-- "careamist_training.py:train_n2v_val"
    ```
 
    1. Validation is passed to `val_data`.

=== "CARE/N2N"

    ```python title="Training CARE with validation"
    --8<-- "careamist_training.py:train_care_val"
    ```
    
    1. Validation is passed to `val_data`.
    2. Target validation should be provided as well.


### Custom data

In the [data](./data.md) section we have seen two ways of specifying how to load custom
data, `ReadFuncLoading` and `ImageStackLoading`. Once either of these classes have been
defined and instantiated, they can be passed to the `CAREamist` to train on custom data.

=== "Noise2Void"
    
    ```python title="Training on custom data"
    --8<-- "careamist_training.py:train_n2v_custom"
    ```
 
    1. Both `ReadFuncLoading` and `ImageStackLoading` can be passed to `loading`.

=== "CARE/N2N"

    ```python title="Training on custom data"
    --8<-- "careamist_training.py:train_care_custom"
    ```
    
    1. Both `ReadFuncLoading` and `ImageStackLoading` can be passed to `loading`.


## Advanced training

### Masking

CAREamics supports providing a mask of the training data to define from which region
should the training patches be sampled. This can be useful to exclude certain regions
from training, for example areas with no signal or with zero values.

=== "Noise2Void"
    
    ```python title="Specifying a mask for Noise2Void training"
    --8<-- "careamist_training.py:train_n2v_mask"
    ```
 
    1. The mask is passed alongside the data.

=== "CARE/N2N"

    ```python title="Specifying a mask for CARE training"
    --8<-- "careamist_training.py:train_care_mask"
    ```
    
    1. The mask is passed alongside the data.

!!! note "What is masked?"

    The mask is a binary set of images with the same size as the training data and
    should have value `1` for pixels that should be included in the training and `0`
    for pixels that should be excluded.
    

### Passing callbacks

PyTorch Lightning provide different callbacks, and a callback interface, that can be
useful to further tune the training process. You can pass callbacks directly upon
instantiating the `CAREamist` object. Two callbacks are already specified in the
configuration ([ModelCheckpoint](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html)
and [EarlyStopping](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html)),
but you can also pass additional callbacks.

```python title="Passiong callbacks"
--8<-- "careamist_training.py:callbacks"
```

1. Early stopping callback is currently not defined via the convenience functions, but
it can be accessed directly. 
2. The configuration already contains the `ModelCheckpoint` and `EarlyStopping` callbacks.
3. Any additional callback can be passed via the `callbacks` argument.

!!! note "`ModelCheckpoint` and `EarlyStopping`"

    `ModelCheckpoint` and `EarlyStopping` are already specified in the configuration
    and instantiated by the `CAREamist`. If you want to set their parameters, use
    the configuration, otherwise an error will be raised.
