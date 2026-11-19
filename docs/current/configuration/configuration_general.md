---
icon: lucide/sliders-horizontal
description: Configuring CAREamics
---

# Configuring CAREamics

To start with CAREamics, create a configuration object containing the parameters for
your experiment. The configuration validates individual parameters and their
coherence, avoiding combinations that could trigger errors later in the workflow.

CAREamics provides algorithm-specific convenience functions. Each algorithm has a
simple factory for common experiments and an advanced factory for more control.

=== "Noise2Void"

    ```python title="Noise2Void configuration factories"
    --8<-- "current/configuration/n2v/general_n2v.py:all_configs"
    ```

=== "CARE"

    ```python title="CARE and Noise2Noise configuration factories"
    --8<-- "current/configuration/care/general_care.py:all_configs"
    ```

=== "HDN"

    ```python title="HDN configuration factories"
    --8<-- "current/configuration/hdn/general_hdn.py:all_configs"
    ```

=== "MicroSplit"

    ```python title="MicroSplit configuration factories"
    --8<-- "current/configuration/microsplit/general_microsplit.py:all_configs"
    ```

=== "cryoCARE"

    ```python title="cryoCARE configuration factories"
    --8<-- "current/configuration/config_general/configuration_cryocare.py:all_configs"
    ```

=== "Segmentation"

    ```python title="Segmentation configuration factories"
    --8<-- "current/configuration/segmentation/general_seg.py:all_configs"
    ```


## Simple configuration

The simple configuration functions expose the parameters most commonly used to start
an experiment.

=== "Noise2Void"

    ```python title="Configure Noise2Void"
    --8<-- "current/configuration/n2v/general_n2v.py:config_n2v"
    ```

    1. Patch size concerns only spatial axes, must be in order `(Z)YX`, and must agree
    with `axis` about dimensionality (2D or 3D).


=== "CARE"

    ```python title="Configure CARE"
    --8<-- "current/configuration/care/general_care.py:config_care"
    ```

    1. Patch size concerns only spatial axes, must be in order `(Z)YX`, and must agree
    with `axis` about dimensionality (2D or 3D).


=== "HDN"

    ```python title="Configure HDN"
    --8<-- "current/configuration/hdn/general_hdn.py:config_hdn"
    ```
    
    1. Patch size concerns only spatial axes, must be in order `(Z)YX`, and must agree
    with `axis` about dimensionality (2D or 3D).


=== "MicroSplit"

    ```python title="Configure MicroSplit"
    --8<-- "current/configuration/microsplit/general_microsplit.py:config_microsplit"
    ```

    1. Patch size concerns only spatial axes, must be in order `(Z)YX`, and must agree
    with `axis` about dimensionality (2D or 3D).
    2. The number of channels to be unmixed.

=== "cryoCARE"

    ```python title="Configure cryoCARE"
    --8<-- "current/configuration/config_general/configuration_cryocare.py:config_cryocare"
    ```

    1. Patch size concerns only spatial axes, in order `ZYX`, and must be 3D.
    2. Rotational axis around which the specimen was tilted inside the microscope.
  
=== "Segmentation"

    ```python title="Configure segmentation"
    --8<-- "current/configuration/segmentation/general_seg.py:config_seg"
    ```

    1. Patch size concerns only spatial axes, must be in order `(Z)YX`, and must agree
    with `axis` about dimensionality (2D or 3D).
    2. Number of foreground classes to be segmented.


- `experiment_name`: experiment name used in logging and automatic model
    saving. It should only contain letters, numbers, underscores, dashes, and spaces.
- `data_type`: input data type. CAREamics
    supports `array` (when passing NumPy arrays directly), `tiff`, `zarr`, `mrc`, `czi`,
    and `custom` (see [data](../data/)).
- `axes`: axes of the input data, in the order they have on disk or in memory. Accepted
    axes are `(S)(T)(C)(Z)YX` in any order (with `YX` being contiguous).
- `patch_size`: size of the spatial patches extracted during training, in `(Z)YX` order. 
    It should be 3D when `Z` is in `axes` and 2D otherwise. See note below to use `T` as
    a third dimension.
- `batch_size`: number of patches used in each training step.
- `num_epochs`: optional number of epochs to train for. For large datasets, also
    consider setting [`num_steps`](#reducing-the-number-of-steps).
- `num_steps`: optional number of batches per epoch. If omitted, the full training
    dataset is used. See [Reducing the number of steps](#reducing-the-number-of-steps).
- `augmentations`: optional list of `x_flip`, `y_flip`, and `rotate_90`. If omitted,
    all three are applied. Use an empty list to disable them.
- `n_val_patches`: number of training patches reserved for validation when separate
    validation data is not provided. See [Number of validation
    patches](#number-of-validation-patches).
- `seed`: fix a seed to make data ordering, augmentations, and patch sampling
    reproducible.

!!! info "Algorithm-specific parameters"

    Each algorithm may have specific parameters that are not shared with the others. They
    are described in the [corresponding section](#algorithm-specific-parameters).

!!! info "Noise2Noise"

    Noise2Noise's implementation relies heavily on CARE and is omitted in this pages,
    but all CARE-related descriptions hold for Noise2Noise and its configuration factory
    functions `create_n2n_config` and `create_advanced_n2n_config`.

!!! info "structN2V"

    We provide a convenience factory for structN2V: `create_structn2v_config`.


!!! note "Training with `T` as depth axis"

    To use `T` as the depth axis, relabel it as `Z`. For `data_type="czi"`, `T` can
    also be used directly as a depth axis, for example `axes="SCTYX"`. See the
    [data section](../data/data.md) for details.


!!! note "cryoCARE and augmentations"

    cryoCARE augmentation are automatically set and cannot be specified by users.

### Channels

Channels are a particular type of axes, and they influence the way the deep-learning
model is built. As a result, when `C` is present in the `axes`, additional parameters
need to be set. These parameters vary from algorithm to algorithm.

=== "Noise2Void"
    
    ```python title="Setting channels parameters"
    --8<-- "current/configuration/n2v/general_n2v.py:config_n2v_channels"
    ```

    1. Channels are considered to be present as soon as `C` is in `axes`.
    2. Number of input and output channels are the same, so we only
    need to set `n_channels`.

=== "CARE"

    ```python title="Setting channels parameters"
    --8<-- "current/configuration/care/general_care.py:config_care_channels"
    ```

    1. Channels are considered to be present as soon as `C` is in `axes`.
    2. Number of input and output channels can be different, so we
    need to set `n_channels_in` and optionnally `n_channels_out`.

    !!! note

        If `n_channels_out` is not set and `target_axes` has a "C" dimension, then it is
        set to the same value as `n_channels_in`, otherwise it is set to 1.

=== "Segmentation"

    ```python title="Setting segmentation input channels"
    --8<-- "current/configuration/segmentation/general_seg.py:config_seg_channels"
    ```

    1. Channels are considered to be present as soon as `C` is in `axes`.
    2. Number of input need to be set in `n_channels`.


!!! info "Other algorithms and channels"

    - **HDN** is currently not compatible with channels.
    - **MicroSplit** works on a single input channel (overlapping structures) and unmixes
    channels in output, therefore it is not compatible with `C` in `axes`.
    - **cryoCARE** deals with cryoET tomograms which do not have channels.


!!! note "Advanced channels parameters"

    The advanced CAREamics configuration factories give access to more channel related
    parameters, such as sub-setting or channel independence during training. Refer to the
    [advanced configuration](#advanced-configuration) section for more details.


### Reducing the number of steps

Each training epoch cycles through all patches. For large datasets, set `num_steps` to
limit the number of batches processed in an epoch.

=== "Noise2Void"

    ```python title="Setting the number of steps"
    --8<-- "current/configuration/n2v/general_n2v.py:config_n2v_steps"
    ```

    1. Use a number smaller than the total number of steps, based on the batch size.

=== "CARE"

    ```python title="Setting the number of steps"
    --8<-- "current/configuration/care/general_care.py:config_care_steps"
    ```

    1. Choose a value smaller than the total number of available batches when shorter
       epochs are desired.

=== "HDN"

    ```python title="Setting the number of steps"
    --8<-- "current/configuration/hdn/general_hdn.py:config_hdn_steps"
    ```

    1. Choose a value smaller than the total number of available batches when shorter
       epochs are desired.

=== "MicroSplit"

    ```python title="Setting the number of steps"
    --8<-- "current/configuration/microsplit/general_microsplit.py:config_msplit_steps"
    ```

    1. Choose a value smaller than the total number of available batches when shorter
       epochs are desired.

=== "cryoCARE"

    ```python title="Setting the number of steps"
    --8<-- "current/configuration/config_general/configuration_cryocare.py:config_cryocare_steps"
    ```

    1. Choose a value smaller than the total number of available batches when shorter
       epochs are desired.

=== "Segmentation"

    ```python title="Setting the number of steps"
    --8<-- "current/configuration/segmentation/general_seg.py:config_seg_steps"
    ```

    1. Choose a value smaller than the total number of available batches when shorter
       epochs are desired.


!!! note "How many steps per epoch?"

    Each epoch consists of `n_patchs / batch_size` steps. The total number of steps is
    shown in the console during training (here `300` steps):

    ```sh
    Epoch 1: 12%|█████████████                                     | 36/300
    ```

    While there is a programmaticaly way to know how many patches would CAREamics
    extract from the data, it is easier to simply run a training for a single epoch and
    check the console output.

!!! note "Why very long epochs may be an issue?"

    The longer, in terms of steps, an epoch lasts, the more the network has learned in between
    two validation steps. Since one of the main reporting mechanism for deep learning
    training is to look at the total training and validation losses over the epochs, that
    means that long epochs lead to a larger sampling of the training performances.


### Augmentations

CAREamics applies random X/Y flips and 90-degree rotations by default. The same
spatial transformation is applied consistently to the training data and targets.

=== "Noise2Void"

    ```python title="Setting augmentations"
    --8<-- "current/configuration/n2v/general_n2v.py:config_n2v_augs"
    ```

    1. The available choices are `x_flip`, `y_flip`, and `rotate_90`.

=== "CARE"

    ```python title="Setting augmentations"
    --8<-- "current/configuration/care/general_care.py:config_care_augs"
    ```

    1. The available choices are `x_flip`, `y_flip`, and `rotate_90`.

=== "HDN"

    ```python title="Setting augmentations"
    --8<-- "current/configuration/hdn/general_hdn.py:config_hdn_augs"
    ```

    1. The available choices are `x_flip`, `y_flip`, and `rotate_90`.

=== "MicroSplit"

    ```python title="Setting augmentations"
    --8<-- "current/configuration/microsplit/general_microsplit.py:config_microsplit_augs"
    ```

    1. The available choices are `x_flip`, `y_flip`, and `rotate_90`.


=== "Segmentation"

    ```python title="Setting augmentations"
    --8<-- "current/configuration/segmentation/general_seg.py:config_seg_augs"
    ```

    1. The available choices are `x_flip`, `y_flip`, and `rotate_90`.

By default, all the augmentations are used. Pass `augmentations=[]` to disable
augmentations.

!!! info "cryoCARE augmentations"

    cryoCARE relies on specific augmentations (rotation perpendicular to the tilt-axis,
    input/target swapping) that are automatically inferred from `tilt_axis` and
    `patch_size`. These cannot be changed by users.

!!! note "When to disable augmentations"

    As soon as your data has a particular symmetry, because objects or features are
    always oriented in a particular way (e.g. patterned illumination), then it may be
    necessary to turn off the augmentations that break this symmetry.

### Number of validation patches

When no validation data is provided, CAREamics can reserve `n_val_patches` patches
from the training data for validation.

=== "Noise2Void"

    ```python title="Setting the number of validation patches"
    --8<-- "current/configuration/n2v/general_n2v.py:config_n2v_val"
    ```

=== "CARE"

    ```python title="Setting the number of validation patches"
    --8<-- "current/configuration/care/general_care.py:config_care_val"
    ```

=== "HDN"

    ```python title="Setting the number of validation patches"
    --8<-- "current/configuration/hdn/general_hdn.py:config_hdn_val"
    ```

=== "MicroSplit"

    ```python title="Setting the number of validation patches"
    --8<-- "current/configuration/microsplit/general_microsplit.py:config_microsplit_val"
    ```

=== "Segmentation"

    ```python title="Setting the number of validation patches"
    --8<-- "current/configuration/segmentation/general_seg.py:config_seg_val"
    ```

If separate validation data is supplied, `n_val_patches` is ignored and the validation
data determines the number of validation patches automatically.

!!! info "cryoCARE validation"

    cryoCARE does not define a `n_val_patches`, but a `validation_fraction` corresponding
    to the percentage of slices to be used as validation.


## Algorithm-specific parameters

### Noise2Void
#### N2V2

Noise2Void has multiple flavours that were developed to overcome some of its limitations,
such as checkerboard artefacts (N2V2) or correlated noise (structN2V). 

To use N2V2, simply set `use_n2v2` to `True`.

```python title="Using N2V2"
--8<-- "current/configuration/n2v/general_n2v.py:config_n2v2"
```

#### structN2V

structN2V has its own configuration factory function. It allows setting 

```python title="Using structN2V"
--8<-- "current/configuration/n2v/general_n2v.py:config_structn2v"
```

1. Orientation of the noise correlation, can be set to `"horizontal"`, `"vertical"`,
`"cross"`, or `"square"`.
2. Size of the mask in pixels.

!!! note "Advanced structN2V"

    In the [advanced configuration section](#advanced-configuration), the Noise2Void
    factory also allow using structN2V using the same two parameters.


### CARE

The configuration factory also allows CARE targets to have different axes or axis order,
provided that they are specified using `target_axes`.

```python title="Setting augmentations"
--8<-- "current/configuration/care/general_care.py:target_axes"
```

!!! note "Assumptions parameters are not passed"

    If `target_axes` is not set, it is assumed to be the same as `axes`.


### HDN

HDN simple configuration factory gives access to two additional parameters: 
`use_noise_model` and `supervised`.

#### Noise model

A noise model guides training towards realistic output values and can be obtained from
noisy images (see the [noise model](../noise_model/) section).

```python title="Training with noise model"
--8<-- "current/configuration/hdn/advanced_hdn.py:hdn_noise_model"
```

!!! warning "Using a noise model"

    If `use_noise_model` is set to `True`, then a trained noise model must be supplied
    at training time.


#### Supervised HDN


If you happen to have ground-truth data, you can also trained HDN with supervision by
setting 

```python title="Training with supervision"
--8<-- "current/configuration/hdn/advanced_hdn.py:hdn_supervised"
```

!!! warning "Supervised HDN"

    If `supervised` is set to `True`, then target images must be supplied at training
    time.

!!! info "Supervised with noise model"

    The two parameters `supervised` and `use_noise_model` cannot be both set to `True`.


### MicroSplit

The MicroSplit simple configuration factory includes the `multiscale_count` parameter,
which controls how many spatial scales MicroSplit uses as input. A value of 1 disables
lateral context and uses only the original patch. Larger values provide progressively
wider image context at the same patch size. Increasing it can help the model use
surrounding structures, but also increases memory and computation.


```python title="Setting the number of lateral-context scales"
--8<-- "current/configuration/microsplit/advanced_microsplit.py:adv_config_microsplit_multiscale"
```


### cryoCARE

In addition to the mandatory `tilt_axis` parameter (see [simple configuration](#simple-configuration)),
cryCARE also expose a `validation_fraction` parameter. The validation fraction is the
percentage of continuous slice in the input data that is reserved for validation.

```python title="Validation fraction"
--8<-- "current/configuration/config_algorithm/configuration_specific_cryocare.py:cryocare_validation"
```

### Segmentation

The `prediction_output` parameter takes one of two values: `masks` or `probabilities`. It
controls the type of output after prediction, either direclty the masks (class with the
highest probability) or the `[0.1]` probability map.


```python title="Prediction output type"
--8<-- "current/configuration/config_algorithm/configuration_specific_seg.py:seg_prediction"
```

## Advanced configuration

### Training in memory

Loading data into memory can speed up patch extraction when the dataset fits in RAM.

=== "Noise2Void"

    ```python title="Setting in-memory training"
    --8<-- "current/configuration/n2v/advanced_n2v.py:adv_config_n2v_in_memory"
    ```

    1. Only `array`, `tiff`, and `custom` are compatible with in-memory training.

=== "CARE"

    ```python title="Setting in-memory training"
    --8<-- "current/configuration/care/advanced_care.py:adv_config_care_in_memory"
    ```

    1. Only `array`, `tiff`, and `custom` are compatible with in-memory training.

=== "HDN"

    HDN supports in-memory training for `array`, `tiff`, and `custom` data. The
    `in_memory` option is available through its advanced factory.

=== "MicroSplit"

    MicroSplit supports in-memory training for `array`, `tiff`, and `custom` data. The
    `in_memory` option is available through its advanced factory.

=== "cryoCARE"

    cryoCARE does not expose an `in_memory` parameter; its data configuration controls
    loading for the supported cryoCARE formats.

=== "Segmentation"

    ```python title="Setting in-memory training"
    --8<-- "current/configuration/segmentation/advanced_seg.py:adv_config_seg_in_memory"
    ```

    1. In-memory training is supported for `array`, `tiff`, and `custom` data.

Only `array`, `tiff`, and `custom` data types support in-memory training. Array data is
kept in memory automatically; `zarr` and `czi` use random access from disk.

### Subsetting channels

To train on only some input channels, pass their zero-based indices through `channels`.
When `channels` is set, the number of input channels is inferred automatically. The
target is not subsetted.

=== "Noise2Void"

    ```python title="Selecting a subset of channels"
    --8<-- "current/configuration/n2v/advanced_n2v.py:adv_config_n2v_subchannels"
    ```

    1. `C` must be present in `axes` for channels to be considered.
    2. This trains using the first and third channels, because channel indices start at
       zero.

=== "CARE"

    ```python title="Selecting a subset of channels"
    --8<-- "current/configuration/care/advanced_care.py:adv_config_care_subchannels"
    ```

    1. `C` must be present in `axes` for channels to be considered.
    2. This trains using the first and third channels, because channel indices start at
       zero.

=== "HDN"

    HDN accepts zero-based input channel indices through `channels`; the number of
    selected channels is inferred automatically.

=== "MicroSplit"

    MicroSplit accepts zero-based input channel indices through `channels`; the number
    of selected channels is inferred automatically.

=== "cryoCARE"

    cryoCARE does not support channel subsetting because its input axes cannot contain
    `C`.

=== "Segmentation"

    ```python title="Selecting a subset of channels"
    --8<-- "current/configuration/segmentation/advanced_seg.py:adv_config_seg_subchannels"
    ```

    1. This example uses the first and third input channels. The label mask is not
       channel-subsetted.

`C` must be present in `axes`.

### Channel independence

By default, channels are trained independently. Set `independent_channels=False` to
allow information to flow between input channels.

=== "Noise2Void"

    ```python title="Training non-independent channels"
    --8<-- "current/configuration/n2v/advanced_n2v.py:adv_config_n2v_ind_channels"
    ```

    1. `C` must be present in `axes`.
    2. The number of channels must be specified.
    3. Setting `independent_channels=False` allows channels to inform each other.

=== "CARE"

    ```python title="Training non-independent channels"
    --8<-- "current/configuration/care/advanced_care.py:adv_config_care_ind_channels"
    ```

    1. `C` must be present in `axes`.
    2. The number of input channels must be specified.
    3. Setting `independent_channels=False` allows channels to inform each other.

=== "HDN"

    Channel independence is not exposed as an HDN parameter; HDN uses its LVAE channel
    handling.

=== "MicroSplit"

    Channel independence is not exposed as a MicroSplit parameter; channel mixing is
    controlled by its synthetic-input parameters.

=== "cryoCARE"

    Channel independence is not applicable because cryoCARE does not support `C` in
    its input axes.

=== "Segmentation"

    Segmentation fixes channel independence to `False`; it does not expose this
    parameter.

`C` must be present in `axes`. Segmentation fixes channel independence to `False` and
therefore does not expose this parameter.

### Normalization

Normalization is applied during training and prediction. The available strategies are
`mean_std`, `quantile`, `min_max`, and `none`.

=== "Noise2Void"

    ```python title="Specifying the normalization method"
    --8<-- "current/configuration/n2v/advanced_n2v.py:adv_config_n2v_norm"
    ```

    1. Select the normalization method with `normalization`.
    2. `normalization_params` changes the method's default parameters.

=== "CARE"

    ```python title="Specifying the normalization method"
    --8<-- "current/configuration/care/advanced_care.py:adv_config_care_norm"
    ```

    1. Select the normalization method with `normalization`.
    2. `normalization_params` changes the method's default parameters.

=== "HDN"

    HDN supports `mean_std`, `min_max`, `quantile`, and `none` normalization through
    `normalization` and `normalization_params`.

=== "MicroSplit"

    MicroSplit supports `mean_std`, `min_max`, `quantile`, and `none` normalization
    through `normalization` and `normalization_params`.

=== "cryoCARE"

    cryoCARE supports the same normalization strategies. Its normalization is applied
    to the reconstructed volumes used for training and validation.

=== "Segmentation"

    ```python title="Specifying the normalization method"
    --8<-- "current/configuration/segmentation/advanced_seg.py:adv_config_seg_norm"
    ```

    1. Select the normalization method with `normalization`.
    2. `normalization_params` customizes the quantile bounds; target normalization is
       skipped automatically.

If statistics are not provided, CAREamics computes them from the training data. They
can also be passed explicitly:

=== "Noise2Void"

    ```python title="Passing pre-computed normalization parameters"
    --8<-- "current/configuration/n2v/advanced_n2v.py:adv_config_n2v_norm_params"
    ```

    1. This selects zero-mean, unit-variance normalization, which is the default.
    2. The normalization parameters are pre-computed and passed explicitly.

=== "CARE"

    ```python title="Passing pre-computed normalization parameters"
    --8<-- "current/configuration/care/advanced_care.py:adv_config_care_norm_params"
    ```

    1. This selects zero-mean, unit-variance normalization, which is the default.
    2. The input normalization parameters are pre-computed and passed explicitly.
    3. CARE and N2N can also receive pre-computed target statistics.

=== "HDN"

    `normalization_params` can contain pre-computed input statistics for HDN, avoiding
    recomputation over the training data.

=== "MicroSplit"

    `normalization_params` can contain pre-computed input statistics for MicroSplit,
    avoiding recomputation over the training data.

=== "cryoCARE"

    `normalization_params` can contain pre-computed statistics for cryoCARE volumes.

When channels are present, normalization parameters normally contain one value per
channel. Set `per_channel=False` when a shared value is intended.

=== "Noise2Void"

    ```python title="Normalizing multi-channel data"
    --8<-- "current/configuration/n2v/advanced_n2v.py:adv_config_n2v_norm_ch"
    ```

    1. The example uses multiple channels.
    2. The number of channels must be specified.
    3. Each normalization parameter has one value per channel.
    4. With `per_channel=False`, each parameter instead expects one shared value.

=== "CARE"

    ```python title="Normalizing multi-channel data"
    --8<-- "current/configuration/care/advanced_care.py:adv_config_care_norm_ch"
    ```

    1. The example uses multiple channels.
    2. The number of input channels must be specified.
    3. Each normalization parameter has one value per channel.
    4. With `per_channel=False`, each parameter instead expects one shared value.

=== "HDN"

    HDN accepts one normalization value per selected channel. Use `per_channel=False`
    when a shared value is intended.

=== "MicroSplit"

    MicroSplit accepts one normalization value per selected channel. Use
    `per_channel=False` when a shared value is intended.

=== "cryoCARE"

    cryoCARE does not support channel axes, so its normalization parameters are not
    channel-indexed.

### Choosing a logger

CSV logging is always enabled. An additional Weights & Biases or TensorBoard logger can
be selected with `logger`.

=== "Noise2Void"

    ```python title="Choosing a logger"
    --8<-- "current/configuration/n2v/advanced_n2v.py:adv_config_n2v_logger"
    ```

    1. This selects Weights & Biases in addition to the CSV logger.

=== "CARE"

    ```python title="Choosing a logger"
    --8<-- "current/configuration/care/advanced_care.py:adv_config_care_logger"
    ```

    1. This selects Weights & Biases in addition to the CSV logger.

=== "HDN"

    Set `logger` to `wandb` or `tensorboard` to add the selected logger to CSV logging.

=== "MicroSplit"

    Set `logger` to `wandb` or `tensorboard` to add the selected logger to CSV logging.

=== "cryoCARE"

    Set `logger` to `wandb` or `tensorboard` to add the selected logger to CSV logging.

### Number of workers

`num_workers` controls data-loading workers. The default `-1` selects a value based on
the available CPUs.

=== "Noise2Void"

    ```python title="Setting the number of workers"
    --8<-- "current/configuration/n2v/advanced_n2v.py:adv_config_n2v_num_workers"
    ```

    1. This sets the number of workers used for data loading.

=== "CARE"

    ```python title="Setting the number of workers"
    --8<-- "current/configuration/care/advanced_care.py:adv_config_care_num_workers"
    ```

    1. This sets the number of workers used for data loading.

=== "HDN"

    Set `num_workers` to control HDN data loading; `-1` selects a value automatically.

=== "MicroSplit"

    Set `num_workers` to control MicroSplit data loading; `-1` selects a value
    automatically.

=== "cryoCARE"

    Set `num_workers` to control cryoCARE data loading; `-1` uses the platform default.

=== "Segmentation"

    ```python title="Setting the number of workers"
    --8<-- "current/configuration/segmentation/advanced_seg.py:adv_config_seg_num_workers"
    ```

    1. Set this according to the available CPU resources and storage performance.


## PyTorch and Lightning parameters

The following PyTorch Lightning and model parameters are shared by all algorithms.

### Model parameters

The U-Net model parameters can be passed through `model_params`.

=== "Noise2Void"

    ```python title="Passing model parameters"
    --8<-- "current/configuration/n2v/lightning_n2v.py:adv_config_n2v_model"
    ```

=== "CARE"

    ```python title="Passing model parameters"
    --8<-- "current/configuration/care/lightning_care.py:adv_config_care_model"
    ```

=== "HDN"

    HDN uses LVAE model parameters rather than the U-Net parameters shown above. Use
    `model_params` for non-structural LVAE settings; structural settings are fixed by
    the factory.

=== "MicroSplit"

    MicroSplit uses LVAE model parameters. Use `model_params` for non-structural LVAE
    settings; structural settings are fixed by the factory.

=== "cryoCARE"

    cryoCARE does not expose `model_params`; its model architecture is fixed by the
    algorithm.

=== "Segmentation"

    ```python title="Setting U-Net parameters"
    --8<-- "current/configuration/segmentation/advanced_seg.py:adv_config_seg_model"
    ```

    1. Increasing `depth` adds U-Net levels and generally increases model capacity and
       memory use.

Common model parameters include `depth`, `num_channels_init`, `residual`, and
`use_batch_norm`.

### Trainer parameters

Pass additional PyTorch Lightning `Trainer` options through `trainer_params`.

=== "Noise2Void"

    ```python title="Passing Trainer parameters"
    --8<-- "current/configuration/n2v/lightning_n2v.py:adv_config_n2v_trainer"
    ```

    1. Pass additional Trainer parameters as a dictionary through `trainer_params`.

=== "CARE"

    ```python title="Passing Trainer parameters"
    --8<-- "current/configuration/care/lightning_care.py:adv_config_care_trainer"
    ```

    1. Pass additional Trainer parameters as a dictionary through `trainer_params`.

=== "HDN"

    Pass additional PyTorch Lightning `Trainer` options through `trainer_params`.

=== "MicroSplit"

    Pass additional PyTorch Lightning `Trainer` options through `trainer_params`.

=== "cryoCARE"

    Pass additional PyTorch Lightning `Trainer` options through `trainer_params`.

`num_epochs` and `num_steps` take precedence over corresponding
`max_epochs` and `limit_train_batches` values in `trainer_params`.

### Optimizer parameters

CAREamics uses `Adam` by default. Select another supported optimizer and pass its
parameters through `optimizer_params`.

=== "Noise2Void"

    ```python title="Passing optimizer parameters"
    --8<-- "current/configuration/n2v/lightning_n2v.py:adv_config_n2v_opt"
    ```

    1. Select the optimizer with `optimizer`.
    2. Pass optimizer parameters as a dictionary through `optimizer_params`.

=== "CARE"

    ```python title="Passing optimizer parameters"
    --8<-- "current/configuration/care/lightning_care.py:adv_config_care_opt"
    ```

    1. Select the optimizer with `optimizer`.
    2. Pass optimizer parameters as a dictionary through `optimizer_params`.

=== "HDN"

    Select `Adam`, `Adamax`, or `SGD` with `optimizer` and customize it through
    `optimizer_params`. HDN defaults to `Adamax`.

=== "MicroSplit"

    Select `Adam`, `Adamax`, or `SGD` with `optimizer` and customize it through
    `optimizer_params`. MicroSplit defaults to `Adamax`.

=== "cryoCARE"

    Select `Adam`, `Adamax`, or `SGD` with `optimizer` and customize it through
    `optimizer_params`. cryoCARE defaults to `Adam`.

Supported optimizers are `Adam`, `SGD`, and `Adamax`.

### Learning-rate schedulers

Configure the learning-rate scheduler with `lr_scheduler` and
`lr_scheduler_params`.

=== "Noise2Void"

    ```python title="Passing learning-rate scheduler parameters"
    --8<-- "current/configuration/n2v/lightning_n2v.py:adv_config_n2v_lr"
    ```

    1. Select the scheduler with `lr_scheduler`.
    2. Pass scheduler parameters as a dictionary through `lr_scheduler_params`.

=== "CARE"

    ```python title="Passing learning-rate scheduler parameters"
    --8<-- "current/configuration/care/lightning_care.py:adv_config_care_lr"
    ```

    1. Select the scheduler with `lr_scheduler`.
    2. Pass scheduler parameters as a dictionary through `lr_scheduler_params`.

=== "HDN"

    Select `ReduceLROnPlateau` or `StepLR` with `lr_scheduler` and pass options through
    `lr_scheduler_params`.

=== "MicroSplit"

    Select `ReduceLROnPlateau` or `StepLR` with `lr_scheduler` and pass options through
    `lr_scheduler_params`.

=== "cryoCARE"

    Select `ReduceLROnPlateau` or `StepLR` with `lr_scheduler` and pass options through
    `lr_scheduler_params`.

Supported schedulers are `ReduceLROnPlateau` and `StepLR`.

### Dataloader parameters

Use `train_dataloader_params` and `val_dataloader_params` to customize the PyTorch
dataloaders.

=== "Noise2Void"

    ```python title="Passing dataloader parameters"
    --8<-- "current/configuration/n2v/lightning_n2v.py:adv_config_n2v_train_dm"
    ```

=== "CARE"

    ```python title="Passing dataloader parameters"
    --8<-- "current/configuration/care/lightning_care.py:adv_config_care_train_dm"
    ```

=== "HDN"

    Use `train_dataloader_params` and `val_dataloader_params` to customize HDN
    dataloaders.

=== "MicroSplit"

    Use `train_dataloader_params` and `val_dataloader_params` to customize MicroSplit
    dataloaders.

=== "cryoCARE"

    Use `train_dataloader_params` and `val_dataloader_params` to customize cryoCARE
    dataloaders.

When overriding `train_dataloader_params`, specify the desired `shuffle` value.

### Checkpoint callback

Override the default checkpoint behavior with `checkpoint_params`.

=== "Noise2Void"

    ```python title="Passing checkpoint parameters"
    --8<-- "current/configuration/n2v/lightning_n2v.py:adv_config_n2v_checkpoint"
    ```

=== "CARE"

    ```python title="Passing checkpoint parameters"
    --8<-- "current/configuration/care/lightning_care.py:adv_config_care_checkpoint"
    ```

=== "HDN"

    Pass callback options through `checkpoint_params` to configure HDN checkpoints.

=== "MicroSplit"

    Pass callback options through `checkpoint_params` to configure MicroSplit
    checkpoints.

=== "cryoCARE"

    Pass callback options through `checkpoint_params` to configure cryoCARE
    checkpoints.

CARE saves the best validation checkpoints and the last checkpoint by default. N2V
uses a different default because validation loss is not necessarily correlated with
denoising quality.

### Early stopping callback

Configure early stopping with `early_stopping_params`.

=== "Noise2Void"

    Early stopping is disabled by default for Noise2Void and is not exposed by its
    configuration factory.

=== "CARE"

    ```python title="Passing early-stopping parameters"
    --8<-- "current/configuration/care/lightning_care.py:adv_config_care_early_stop"
    ```

=== "HDN"

    Configure early stopping through `early_stopping_params`.

=== "MicroSplit"

    Configure early stopping through `early_stopping_params`.

=== "cryoCARE"

    Early stopping is not exposed by the cryoCARE configuration factory.

=== "Segmentation"

    Configure early stopping through `early_stopping_params`.

Early stopping is generally useful for supervised algorithms. It is disabled by
default for Noise2Void.

### Algorithm-specific configuration

For algorithm-specific options, see:

- [Advanced N2V](./n2v.md)
- [Advanced CARE](./care.md)
- [HDN](./hdn.md)
- [MicroSplit](./microsplit.md)
- [cryoCARE](./cryocare.md)
- [Segmentation](./segmentation.md)

## Saving and loading

Configurations are automatically saved with checkpoints, but they can also be saved
and loaded manually.

```python title="Save and load configurations"
--8<-- "current/configuration/config_general/configuration_io.py:save_load"
```
