---
icon: lucide/sticky-notes
description: Advanced CARE configuration
---

# Configuring CARE

To start with CAREamics, create a configuration object containing the parameters for
your experiment. The configuration validates individual parameters and their
coherence, avoiding combinations that could trigger errors later in the workflow.

CAREamics provides a simple factory for common CARE and Noise2Noise denoising experiments and
an advanced factory for more complex parameters.

```python title="CARE and Noise2Noise configuration factories"
--8<-- "current/configuration/care/general_care.py:all_configs"
```

1. Parameters in CARE and Noise2Noise are equivalent and we omit Noise2Noise in these
pages.

## Simple configuration

The simple configuration functions expose the parameters most commonly used to start
an experiment.

```python title="Configure CARE"
--8<-- "current/configuration/care/general_care.py:config_care"
```

<!-- break down the block to allow the list to be generated -->

- `experiment_name`: used during logging and automatic model saving. It
  should only contain letters, numbers, underscores, dashes, and spaces.
- `data_type`: input data type. CAREamics supports `array` (when passing NumPy arrays
  directly), `tiff`, `zarr`, `mrc`, `czi`, and `custom` (see [data](../data/)).
- `axes`: axes of the input data, in the order they have on disk or in memory. Accepted
  axes are `(S)(T)(C)(Z)YX` in any order, with `YX` being contiguous.
- `patch_size`: size of the spatial patches extracted during training, in `(Z)YX`
  order. It should be 3D when `Z` is in `axes` and 2D otherwise. See the note below
  to use `T` as a third dimension.
- `batch_size`: number of patches used in each training step.
- `num_epochs`: optional number of epochs. For large datasets, also
  consider setting `num_steps`, see [Reducing the number of steps](#reducing-the-number-of-steps).
- `augmentations`: optional list of `x_flip`, `y_flip`, and `rotate_90`. If omitted,
  all three are applied. Use an empty list to disable them, see [Augmentations](#augmentations).
- `n_val_patches`: number of training patches reserved for validation when separate
  validation data is not provided. See [Number of validation patches](#number-of-validation-patches).
- `seed`: optional, fix a seed to make data ordering, augmentations, and patch sampling
  reproducible.


!!! info "Noise2Noise"

    Noise2Noise's implementation relies heavily on CARE. CARE-related descriptions
    also apply to Noise2Noise and its configuration factory functions
    `create_n2n_config` and `create_advanced_n2n_config`.

!!! note "Training with `T` as depth axis"

    To use `T` as the depth axis, relabel it as `Z`. For `data_type="czi"`, `T` can
    also be used directly as a depth axis, for example `axes="SCTYX"`. See the
    [data section](../data/data.md) for details.


### Channels

Channels are a particular type of axis and influence the way the deep-learning model
is built. When `C` is present in `axes`, set the input and output channel counts.

```python title="Setting channels parameters"
--8<-- "current/configuration/care/general_care.py:config_care_channels"
```

1. Channels are considered to be present as soon as `C` is in `axes`.
2. The number of input and output channels can be different, so set `n_channels_in`
   and, optionally, `n_channels_out`.

!!! note "Target axes"

    If `n_channels_out` is not set and `target_axes` has a `C` dimension, it is set to
    the same value as `n_channels_in`, otherwise it is set to 1.

!!! warning "CARE and Noise2Noise defaults"

    By default CARE and Noise2Noise use channels differently: in Noise2Noise they are
    independent and are trained separately, while they are mixed by CARE. See the [channel independence parameter](#channel-independence)
    to change that behaviour.

### Reducing the number of steps

Each training epoch cycles through all patches. For large datasets, set `num_steps` to
limit the number of batches processed in an epoch.

```python title="Setting the number of steps"
--8<-- "current/configuration/care/general_care.py:config_care_steps"
```

1. Choose a value smaller than the total number of available batches when shorter
   epochs are desired.

!!! note "How many steps per epoch?"

    Each epoch consists of `n_patches / batch_size` steps. The total number of steps
    is shown in the console during training (for example, `300` steps):

    ```sh
    Epoch 1: 12%|█████████████                                     | 36/300
    ```

### Augmentations

CAREamics applies random X/Y flips and 90-degree rotations by default. The same
spatial transformation is applied to a training patch and its corresponding target.

```python title="Setting augmentations"
--8<-- "current/configuration/care/general_care.py:config_care_augs"
```

1. The available choices are `x_flip`, `y_flip`, and `rotate_90`.

By default, all augmentations are used. Pass `augmentations=[]` to disable them.

!!! note "When to disable augmentations"

    If your data has a particular symmetry, because objects or features are always
    oriented in a particular way, it may be necessary to turn off augmentations that
    break this symmetry.

### Number of validation patches

When no validation data is provided, CAREamics can reserve `n_val_patches` patches
from the training data for validation.

```python title="Setting the number of validation patches"
--8<-- "current/configuration/care/general_care.py:config_care_val"
```

If separate validation data is supplied, `n_val_patches` is ignored and the validation
data determines the number of validation patches automatically.

### Target axes

The configuration factory allows CARE targets to have different axes or axis order,
provided that they are specified using `target_axes`.

```python title="Setting target axes"
--8<-- "current/configuration/care/general_care.py:target_axes"
```

!!! note "When target axes are not passed"

    If `target_axes` is not set, it is assumed to be the same as `axes`.

## Advanced configuration

### Training in memory

Loading data into memory can speed up patch extraction when the dataset fits in RAM.

```python title="Setting in-memory training"
--8<-- "current/configuration/care/advanced_care.py:adv_config_care_in_memory"
```

1. Only `array`, `tiff`, and `custom` are compatible with in-memory training.

Only `array`, `tiff`, and `custom` data types support in-memory training. Array data is
kept in memory automatically. Other formats, such as `zarr`, `czi` or `mrc` use random
access from disk.

### Subsetting channels

To train on only some input channels, pass their zero-based indices through `channels`.
When `channels` is set, the number of input channels is inferred automatically.

```python title="Selecting a subset of channels"
--8<-- "current/configuration/care/advanced_care.py:adv_config_care_subchannels"
```

1. `C` must be present in `axes` for channels to be considered.
2. Train using the first and third channels.

!!! info "`n_channels_in` and `channels`"

    If `C` is in `axes`, one of `n_channels` and `channels` must be specified. If
    only `channels` is specified, then `n_channels_in` will be set to `len(channels)`.

    If `n_channels_in` and `channels` are incompatible (`n_channels_in!=len(channels)`),
    an error will be raised.

!!! warning "Targets"

    `channels` is not applied to target images.

### Channel independence

By default, channels are trained together. Set `independent_channels=True` to
allow information to flow between input channels.

```python title="Training non-independent channels"
--8<-- "current/configuration/care/advanced_care.py:adv_config_care_ind_channels"
```

1. `C` must be present in `axes`.
2. The number of input channels must be specified.
3. Allow channels to inform each other.

!!! warning "CARE and Noise2Noise defaults"

    By default CARE and Noise2Noise use channels differently: in Noise2Noise they are
    independent and are trained separately, while they are mixed by CARE.


### Normalization

Normalization is applied during training and prediction. The available strategies are
`mean_std`, `quantile`, `min_max`, and `none`.

```python title="Specifying the normalization method"
--8<-- "current/configuration/care/advanced_care.py:adv_config_care_norm"
```

1. Select the normalization method with `normalization`.
2. `normalization_params` changes the method's default parameters.

If statistics are not provided, CAREamics computes them from the training data. They
can also be passed explicitly:

```python title="Passing pre-computed normalization parameters"
--8<-- "current/configuration/care/advanced_care.py:adv_config_care_norm_params"
```

1. This selects zero-mean, unit-variance normalization, which is the default.
2. The input normalization parameters are pre-computed and passed explicitly.
3. CARE and Noise2Noise can also receive pre-computed target statistics.

When channels are present, normalization parameters normally contain one value per
channel. Set `per_channel=False` when a shared value is intended.

```python title="Normalizing multi-channel data"
--8<-- "current/configuration/care/advanced_care.py:adv_config_care_norm_ch"
```

1. The example uses multiple channels.
2. The number of input channels must be specified.
3. Each normalization parameter has one value per channel.
4. With `per_channel=False`, each parameter instead expects one shared value.

### Choosing a logger

CSV logging is always enabled. An additional Weights & Biases or TensorBoard logger can
be selected with `logger`.

```python title="Choosing a logger"
--8<-- "current/configuration/care/advanced_care.py:adv_config_care_logger"
```

1. This selects Weights & Biases in addition to the CSV logger.

### Number of workers

`num_workers` controls data-loading workers. The default `-1` selects a value based on
the available CPUs.

```python title="Setting the number of workers"
--8<-- "current/configuration/care/advanced_care.py:adv_config_care_num_workers"
```

1. This sets the number of workers used for data loading.

## PyTorch and Lightning parameters

### Model parameters

The U-Net model parameters can be passed through `model_params`.

```python title="Passing model parameters"
--8<-- "current/configuration/care/lightning_care.py:adv_config_care_model"
```

Common model parameters include `depth`, `num_channels_init`, `residual`, and
`use_batch_norm`.

### Trainer parameters

Pass additional PyTorch Lightning `Trainer` options through `trainer_params`.

```python title="Passing Trainer parameters"
--8<-- "current/configuration/care/lightning_care.py:adv_config_care_trainer"
```

1. Pass additional Trainer parameters as a dictionary through `trainer_params`.

`num_epochs` and `num_steps` take precedence over corresponding `max_epochs` and
`limit_train_batches` values in `trainer_params`.

### Optimizer parameters

CAREamics uses `Adam` by default. Select another supported optimizer and pass its
parameters through `optimizer_params`.

```python title="Passing optimizer parameters"
--8<-- "current/configuration/care/lightning_care.py:adv_config_care_opt"
```

1. Select the optimizer with `optimizer`.
2. Pass optimizer parameters as a dictionary through `optimizer_params`.

Supported optimizers are `Adam`, `SGD`, and `Adamax`.

### Learning-rate schedulers

Configure the learning-rate scheduler with `lr_scheduler` and `lr_scheduler_params`.

```python title="Passing learning-rate scheduler parameters"
--8<-- "current/configuration/care/lightning_care.py:adv_config_care_lr"
```

1. Select the scheduler with `lr_scheduler`.
2. Pass scheduler parameters as a dictionary through `lr_scheduler_params`.

Supported schedulers are `ReduceLROnPlateau` and `StepLR`.

### Dataloader parameters

Use `train_dataloader_params` and `val_dataloader_params` to customize the PyTorch
dataloaders.

```python title="Passing dataloader parameters"
--8<-- "current/configuration/care/lightning_care.py:adv_config_care_train_dm"
```

When overriding `train_dataloader_params`, specify the desired `shuffle` value.

### Checkpoint callback

Override the default checkpoint behavior with `checkpoint_params`.

```python title="Passing checkpoint parameters"
--8<-- "current/configuration/care/lightning_care.py:adv_config_care_checkpoint"
```

CARE saves the best validation checkpoints and the last checkpoint by default.

### Early stopping callback

Configure early stopping with `early_stopping_params`.

```python title="Passing early-stopping parameters"
--8<-- "current/configuration/care/lightning_care.py:adv_config_care_early_stop"
```

## Saving and loading

Configurations are automatically saved with checkpoints, but they can also be saved
and loaded manually.

```python title="Save and load configurations"
--8<-- "current/configuration/config_general/configuration_io.py:save_load"
```
