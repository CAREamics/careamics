---
icon: lucide/sliders-horizontal
description: Configuring CAREamics
---

# Configuring CAREamics

To start with CAREamics, we need to create a configuration object that holds most of the
useful parameters. The configuration ensures cross-validation and coherence of the
parameters, in particular avoiding sets of parameters that could trigger errors 
deep in the library.

Configuration can be created using any of the algorithm-specific convenience functions
below. We provide a simple function with a minimum set of parameters, and an advanced 
function giving access to many more.

```python
--8<-- "configuration_simple.py:all_configs"
```

!!! note "CARE and Noise2Noise"

    CARE and Noise2Noise configurations have the exact same set of parameters, contrary
    to Noise2Void. In this section, we only show the CARE configuration, but the same
    applies to Noise2Noise by simply swapping `create_care_config` with
    `create_n2n_config`.


## Simple configuration

The simple configuration functions are designed to only expose the parameters most
commonly used. This is a good starting point for most experiments.

=== "Noise2Void"
    
    ```python title="Configure Noise2Void"
    --8<-- "configuration_simple.py:config_n2v"
    ```
 
    1. The length of the patch size is conditioned on the presence of the `Z` axis.

=== "CARE/N2N"

    ```python title="Configure CARE"
    --8<-- "configuration_simple.py:config_care"
    ```

    1. The length of the patch size is conditioned on the presence of the `Z` axis.

- `experiment_name`: The experiment name is used in the logging and automatic model
saving. It should only contain letters, numbers, underscores, dashes and spaces.
- `data_type`: The data type impacts other parameters and which features may be available.
CAREamics supports `array` (when passing `numpy` arrays directly), `tiff`, `zarr`, `czi`
and `custom`. Refer to the [data section]() for more details.
- `axes`: The axes of the data, in the order they have on disk (or in memory). This is
important to identify correctly the spatial and channel dimensions. Refer to the
[data section]() for tips on how to identify axes.
- `patch_size`: The size of the patches to extract from the data during training. Note
that the patch size only refers to spatial axes (`X`, `Y` and optionally `Z`). Patch sizes
are power of 2 and are greater than 8. They are also usually the same for `X` and `Y` axes.
`patch_size` should be 2D for axes without `Z`, and 3D for axes with `Z`.
- `batch_size`: The number of patches to use in each training batch.
- `num_epochs`: The number of epochs to train for. Note that in the case of large
datasets, you might want to also set the [number of steps parameter](#reducing-the-number-of-steps).


### Reducing the number of steps


Each training epoch cycles through all patches. Therefore, for large datasets, an epoch
can be lengthy and validation happening rarely. In this case, it is useful to set the
number of steps `num_steps`.

=== "Noise2Void"
    
    ```python title=""
    --8<-- "configuration_simple.py:config_n2v_steps"
    ```

    1. Use a number smaller than the total number of steps (given the batch size), see
    notes below.

=== "CARE/N2N"

    ```python title=""
    --8<-- "configuration_simple.py:config_care_steps"
    ```

    1. Use a number smaller than the total number of steps (given the batch size), see
    notes below.


!!! note "How many steps per epoch?"

    Each epoch consists of `n_patchs / batch_size` steps. The total number of steps is
    shown in the console during training:

    ```sh
    Epoch 1: 12%|█████████████                                     | 36/300
    ```

    While there is a programmaticaly way to know how many patches would CAREamics
    extract from the data, it is easier to simply run a training for an epoch and
    check the console output.



!!! note "Advanced `num_epochs` and `num_steps`"

    The `num_epochs` and `num_steps` correspond to the `max_epochs` and
    `limit_train_batches` parameters of the Pytorch Lightning `Trainer`. Refer to the
    [Trainer API](https://lightning.ai/docs/pytorch/stable/common/trainer.html#init) for
    details about these parameters.


### Augmentations

CAREamics applies augmentations to the training patches, by default random flips in X or
Y, and random rotations by 90 degrees. In certain cases, these augmentations may not be
desirable, for example when the result of the augmentation is not a possible occurence
in the data. In microscopy, this can happen when there are structures that have always
the same orientation, or noise with a spatial correlation. To set them,
use the `augmentations` parameter.

=== "Noise2Void"
    
    ```python title=""
    --8<-- "configuration_simple.py:"
    ```

    1. These are all the possible occurences.

=== "CARE/N2N"

    ```python title=""
    --8<-- "configuration_simple.py:"
    ```

    1. These are all the possible occurences.


To disable augmentations, set `augmentations=[]`.


!!! note "How are augmentations applied?"

    Each augmentation has a 0.5 of being applied to each patch. The random 90 degree
    rotations applies either a 90, 180 or 270 rotations (if applied). The augmentations
    are applied sequentially, such that a patch can be flipped in X, not flipped in Y,
    and then rotated by 180 degrees.


### Channels

Channels are a particular type of axes, and they influence the way the deep-learning
model is build. As a result, when `C` is present in the `axes`, additional parameters
need to be set depending on the algorithm.


=== "Noise2Void"
    
    ```python title=""
    --8<-- "configuration_simple.py:config_n2v_channels"
    ```

    1. Channels are considered to be present as soon as `C` is in `axes`.
    2. For Noise2Void, the number of input and output channels are the same, so we only
    need to set `n_channels`.

=== "CARE/N2N"

    ```python title=""
    --8<-- "configuration_simple.py:config_care_channels"
    ```

    1. Channels are considered to be present as soon as `C` is in `axes`.
    2. For CARE/N2N, the number of input and output channels can be different, so we
    need to set `n_channels_in` and optionally `n_channels_out`.

    Note that if `n_channels_out` is not set, it will be set to the same value as
    `n_channels_in`.

!!! note "Advanced channels parameters"

    The advanced CAREamics configuration gives access to more channel related parameters,
    such as sub-setting or channel independence during training. Refer to the
    [advanced configuration](#advanced-configuration) section for more details.


### Validation

When no validation data is provided, CAREamics will automatically split some patches
from the training data to use as validation. The number of validation patches is set 
then governed by the `num_val_patches` parameter. By default, it is set to `8`.

=== "Noise2Void"
    
    ```python title=""
    --8<-- "configuration_simple.py:config_n2v_val"
    ```

    1. Choose an appropriate number of validation patches, depending on the size of the
    training data, to avoid pulling too many patches from the training data.

=== "CARE/N2N"

    ```python title=""
    --8<-- "configuration_simple.py:config_care_val"
    ```

    1. Choose an appropriate number of validation patches, depending on the size of the
    training data, to avoid pulling too many patches from the training data, while
    maintaining meaningful validation.


!!! note "What happens when validation data is passed?"

    In the presence of validation data, the `num_val_patches` parameter is ignored and
    the effective number of validation patches is determined by the size of the
    validation data.

    You can however limit the number of validation steps using PyTorch Lightning 
    parameters, refer to the [advanced training parameters]() section.


## Advanced configuration

More parameters are available by using the advanced configuration convenience functions.
In this section, we explore these additional parameters.

### Training in memory

Where the training data resides influences the speed at which patches can be extracted,
and in turn total training time. The faster way to train is to hold all the data in
memory. However, this is only possible when the data is small enough to fit in the RAM.
Data can be loaded in memory by setting the `in_memory` parameter to `True` in the
configuration.


=== "Noise2Void"
    
    ```python title=""
    --8<-- "configuration_intermediate.py:adv_config_n2v_in_memory"
    ```

    1. Only `array`, `tiff` and `custom` are compatible with in-memory training.

=== "CARE/N2N"

    ```python title=""
    --8<-- "configuration_intermediate.py:adv_config_care_in_memory"
    ```

    1. Only `array`, `tiff` and `custom` are compatible with in-memory training.


!!! note "`data_type` and `in_memory` parameters"

    Only `tiff` and `custom` are compatible with `in_memory=True`. For `array`, this is
    automatically set to `True` and cannot be set to `False`. For `czi` and `zarr`,
    training is done by using random access to the data on disk and currently in-memory
    is not implemented.

    For more details on `custom` data type, refer to the [data]() section.


### Subsetting channels

When the data has channels, it is possible to train from a subset of them only by
passing list of channel indices to the `channels` parameter.


=== "Noise2Void"
    
    ```python title=""
    --8<-- "configuration_intermediate.py:adv_config_n2v_subchannels"
    ```

    1. For channels to be considered present, `C` needs to be in `axes`.
    2. Training would only be performed using two channels, the first and third, since 
    channels are indexed starting from 0.

=== "CARE/N2N"

    ```python title=""
    --8<-- "configuration_intermediate.py:adv_config_care_subchannels"
    ```

    1. For channels to be considered present, `C` needs to be in `axes`.
    2. Training would only be performed using two channels, the first and third, since 
    channels are indexed starting from 0.


!!! note "Number of channels"

    In these examples, you might notice that `n_channels`/`n_channels_in` are not set,
    although they are required when `C` is in `axes`. The reason is that when `channels`
    is set, the number of channels is automatically inferred from `channels`. 

    In the case of CARE/N2N, if `n_channels_out` is also set automatically to the size
    of `channels`, but can also be set to a different value.


### Channel independence

By default, channels are trained independently. This means that the model 


=== "Noise2Void"
    
    ```python title=""
    --8<-- "configuration_intermediate.py:adv_config_n2v_in_memory"
    ```

    1. 

=== "CARE/N2N"

    ```python title=""
    --8<-- "configuration_intermediate.py:adv_config_care_in_memory"
    ```

    1. 



=== "Noise2Void"
    
    ```python title=""
    --8<-- "configuration_intermediate.py:adv_config_n2v_in_memory"
    ```

    1. 

=== "CARE/N2N"

    ```python title=""
    --8<-- "configuration_intermediate.py:adv_config_care_in_memory"
    ```

    1. 



=== "Noise2Void"
    
    ```python title=""
    --8<-- "configuration_intermediate.py:adv_config_n2v_in_memory"
    ```

    1. 

=== "CARE/N2N"

    ```python title=""
    --8<-- "configuration_intermediate.py:adv_config_care_in_memory"
    ```

    1. 



=== "Noise2Void"
    
    ```python title=""
    --8<-- "configuration_intermediate.py:adv_config_n2v_in_memory"
    ```

    1. 

=== "CARE/N2N"

    ```python title=""
    --8<-- "configuration_intermediate.py:adv_config_care_in_memory"
    ```

    1. 
