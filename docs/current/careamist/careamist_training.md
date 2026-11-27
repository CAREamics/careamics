---
icon: lucide/network
description: Quick start
---

# Training CAREamics

After having [created a configuration](../configuration/) and 
[assembled the training data](../data/data.md), you are ready to train CAREamics. The preferred
way to train with CAREamics is to create a `CAREamist`.

```python
--8<-- "current/careamist/careamist_training.py:careamist_from_cfg"
```

1. Here the configuration can either be passed as we have seen in the
[configuration](../configuration/) section, or as a path to configuration [saved to the
disk](../configuration/n2v.md#saving-and-loading).

### Working directory

By default CAREamics will save the training logging data and the checkpoints in the
root directory from which it is called. However, you can pass `work_dir` to the 
`CAREamist` to specify a different directory.

```python
--8<-- "current/careamist/careamist_training.py:careamist_workdir"
```

1. Pass a relative or absolute path.

### Resume training

To resume training from a checkpoint, pass its path when constructing `CAREamist` and
call `train` again. The checkpoint restores the model and trainer state, including the
optimizer state when it is present.

```python
careamist = CAREamist(checkpoint_path="path/to/experiment_last.ckpt")
careamist.train(train_data=train_data)
```


### Disabling progress bar

The PyTorch Lightning progress bar is verbose, and can be disabled by passing
`enable_progress_bar=False` to the `CAREamist`.

```python
--8<-- "current/careamist/careamist_training.py:careamist_pb"
```

## Training basics

Once you have a `CAREamist` object, you can train CAREamics with the `train` method. The
data (input and/or targets) needs to be coherent with the [configuration](../configuration/index.md).

Depending on the algorithm, you may have to pass [targets and additional data](../data/data.md#algorithms) (e.g. 
noise model). Refer to the [data section](../data/data.md#algorithms) for more details on
each algorithm. 

When passing validation, the only constraint is that the validation data is of the same
type as the training data. The amount of validation data is determined by the size of
the validation data itself. Note that not all algorithms are compatible with passing
validation data.


=== "Noise2Void"
    
    ```python title="Training Noise2Void with validation"
    --8<-- "current/careamist/careamist_training.py:train_n2v_val"
    ```
 
    1. Validation is passed to `val_data`.

=== "CARE"

    ```python title="Training CARE with validation"
    --8<-- "current/careamist/careamist_training.py:train_care_val"
    ```
    
    1. Validation is passed to `val_data`.
    2. Target validation should be provided as well.

=== "HDN"

    ```python title="Training HDN with validation"
    --8<-- "current/careamist/careamist_training.py:train_hdn_val"
    ```

    1. Validation is passed to `val_data`.

=== "MicroSplit"

    ```python title="Training MicroSplit with validation"
    --8<-- "current/careamist/careamist_training.py:train_microsplit_val"
    ```

    1. MicroSplit requires explicit validation data and its paired target.

=== "cryoCARE"

    ```python title="Training cryoCARE with validation"
    --8<-- "current/careamist/careamist_training.py:train_cryocare_val"
    ```

    1. Input and targets are reconstructions from split odd and even angles. cryoCARE is
    not compatible with passing explicit validation data.

=== "Segmentation"

    ```python title="Training segmentation with validation"
    --8<-- "current/careamist/careamist_training.py:train_segmentation_val"
    ```

    1. Validation data and validation label masks are passed separately.


!!! note "Data format"

    Data should be an array, a path to a file, a path to a folder, or a list
    of array/paths. See [data formats section](../data/data.md#formats).

!!! note "Passing a directory"

    If you are passing a path to a directory and the data type is `tiff` (or a custom
    type with the required reading utilities, see [custom data](#custom-data)), then
    all the files with the expected file extension in that directory (including sub-directories) will be used for
    training.
    
    Passing a directory is not compatible with CZI data.

!!! info "cryoCARE validation"

    cryoCARE validation is automatically split from the training data along the `tilt_axis`
    and validation cannot be passed to the `CAREamist`.


### Split train and validation data


Without validation, CAREamics will split the data into training and validation internally.
The number of validation patches can be set [in the configuration](../configuration/index.md).

=== "Noise2Void"
    
    ```python title="Training Noise2Void"
    --8<-- "current/careamist/careamist_training.py:train_n2v_no_val"
    ```

    1. Simply pass the training data.

=== "CARE"

    ```python title="Training CARE"
    --8<-- "current/careamist/careamist_training.py:train_care_no_val"
    ```

    1. Simply pass the training data.
    2. And its target.

=== "HDN"

    ```python title="Training HDN"
    --8<-- "current/careamist/careamist_training.py:train_hdn_no_val"
    ```

    1. HDN does not require a target in its unsupervised mode.

=== "MicroSplit"

    MicroSplit requires explicit validation data when using the `CAREamist` API. See the
    validation example above.

=== "cryoCARE"

    ```python title="Training cryoCARE"
    --8<-- "current/careamist/careamist_training.py:train_cryocare_no_val"
    ```

=== "Segmentation"

    ```python title="Training segmentation"
    --8<-- "current/careamist/careamist_training.py:train_segmentation_no_val"
    ```

    1. Segmentation requires a label mask as `train_data_target`.



### Custom data

In the [data](../data/data.md) section we have seen two ways of specifying how to load custom
data, `ReadFuncLoading` and `ImageStackLoading`. Once either of these classes have been
defined and instantiated, they can be passed to the `CAREamist` to train on custom data.

=== "Noise2Void"
    
    ```python title="Training on custom data"
    --8<-- "current/careamist/careamist_training.py:train_n2v_custom"
    ```
 
    1. Both `ReadFuncLoading` and `ImageStackLoading` can be passed to `loading`.

=== "CARE"

    ```python title="Training on custom data"
    --8<-- "current/careamist/careamist_training.py:train_care_custom"
    ```
    
    1. Both `ReadFuncLoading` and `ImageStackLoading` can be passed to `loading`.


=== "HDN"

    ```python title="Training HDN on custom data"
    --8<-- "current/careamist/careamist_training.py:train_hdn_custom"
    ```

    1. Both `ReadFuncLoading` and `ImageStackLoading` can be passed to `loading`.

=== "MicroSplit"

    ```python title="Training MicroSplit on custom data"
    --8<-- "current/careamist/careamist_training.py:train_microsplit_custom"
    ```

    1. Both `ReadFuncLoading` and `ImageStackLoading` can be passed to `loading`.

=== "cryoCARE"

    ```python title="Training cryoCARE on custom data"
    --8<-- "current/careamist/careamist_training.py:train_cryocare_custom"
    ```

    1. Both `ReadFuncLoading` and `ImageStackLoading` can be passed to `loading`.

=== "Segmentation"

    ```python title="Training segmentation on custom data"
    --8<-- "current/careamist/careamist_training.py:train_segmentation_custom"
    ```

    1. Both `ReadFuncLoading` and `ImageStackLoading` can be passed to `loading`.


## Advanced training

### Noise model

HDN and MicroSplit can use a noise model during training. See the
[noise-model guide](../noise_model/noise_model.md) for how to train and save the model
files used below.

=== "HDN"

    ```python title="Training HDN with a noise model"
    --8<-- "current/careamist/careamist_training.py:train_hdn_noise_model"
    ```

=== "MicroSplit"

    ```python title="Training MicroSplit with a noise model"
    --8<-- "current/careamist/careamist_training.py:train_microsplit_noise_model"
    ```


### Masking

Masking can be used to exclude certain regions from training, for example areas with no signal or with zero values.

CAREamics supports two methods of masking data during training:

- providing a mask of the training data to define which regions the training patches should be sampled from, or
- built-in filtering functions. See the full [tutorial](../../tutorials/patch_filtering.md#filtering-functions).

=== "Noise2Void"
    
    ```python title="Specifying a mask for Noise2Void training"
    --8<-- "current/careamist/careamist_training.py:train_n2v_mask"
    ```
 
    1. The mask is passed alongside the data.

=== "CARE"

    ```python title="Specifying a mask for CARE training"
    --8<-- "current/careamist/careamist_training.py:train_care_mask"
    ```
    
=== "HDN"

    ```python title="Specifying a mask for HDN training"
    --8<-- "current/careamist/careamist_training.py:train_hdn_mask"
    ```

=== "MicroSplit"

    Mask-based patch filtering is not currently implemented for MicroSplit.

=== "cryoCARE"

    ```python title="Specifying a mask for cryoCARE training"
    --8<-- "current/careamist/careamist_training.py:train_cryocare_mask"
    ```

=== "Segmentation"

    ```python title="Specifying a mask for segmentation training"
    --8<-- "current/careamist/careamist_training.py:train_segmentation_mask"
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

```python title="Passing callbacks"
--8<-- "current/careamist/careamist_training.py:callbacks"
```

1. Early stopping callback is currently not defined via the convenience functions, but
it can be accessed directly. 
2. The configuration already contains the `ModelCheckpoint` and `EarlyStopping` callbacks.
3. Any additional callback can be passed via the `callbacks` argument.

!!! note "`ModelCheckpoint` and `EarlyStopping`"

    `ModelCheckpoint` and `EarlyStopping` are already specified in the configuration
    and instantiated by the `CAREamist`. If you want to set their parameters, use
    the configuration, otherwise an error will be raised.
