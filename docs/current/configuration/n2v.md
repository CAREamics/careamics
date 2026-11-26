---
icon: lucide/eye-off
description: Advanced Noise2Void configuration
---

# Configuring Noise2Void

To start with CAREamics, create a configuration object containing the parameters for
your experiment. The configuration validates individual parameters and their
coherence, avoiding combinations that could trigger errors later in the workflow.

CAREamics provides algorithm-specific convenience functions. Noise2Void has a simple
factory for common denoising experiments and an advanced factory for more complex
parameters.

```python title="Noise2Void configuration factories"
--8<-- "current/configuration/n2v/general_n2v.py:all_configs"
```

## Simple configuration

The simple configuration functions expose the parameters most commonly used to start
an experiment.

```python title="Configure Noise2Void"
--8<-- "current/configuration/n2v/general_n2v.py:config_n2v"
```

<!-- break down the block to allow the list to be generated -->

- `experiment_name`: used during logging and automatic model saving. It
  should only contain letters, numbers, underscores, dashes, and spaces.
- `data_type`: input data type. CAREamics supports `array` (when passing NumPy arrays
  directly), `tiff`, `zarr`, `czi`, and `custom` (see [data](../data/data.md)).
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

!!! info "structN2V"

    We provide a convenience factory for structN2V:
    `create_structn2v_config`. See the [structN2V](#structn2v) section for more details.

!!! note "Training with `T` as depth axis"

    To use `T` as the depth axis, relabel it as `Z`. For `data_type="czi"`, `T` can
    also be used directly as a depth axis, for example `axes="SCTYX"`. See the
    [data section](../data/data.md) for details.

### Channels

Channels are a particular type of axis and influence the way the deep-learning model
is built. When `C` is present in `axes`, set `n_channels`.

```python title="Setting channels parameters"
--8<-- "current/configuration/n2v/general_n2v.py:config_n2v_channels"
```

1. Channels are considered to be present as soon as `C` is in `axes`.
2. Number of input and output channels.

!!! info "Independent channels"

    By defaults, channels are trained independently from each other. See the
    [channel independence parameter](#channel-independence) to change the behaviour.

### Reducing the number of steps

Each training epoch cycles through all patches. For large datasets, set `num_steps` to
limit the number of batches processed in an epoch.

```python title="Setting the number of steps"
--8<-- "current/configuration/n2v/general_n2v.py:config_n2v_steps"
```

1. Use a number smaller than the total number of steps, based on the batch size.

!!! note "How many steps per epoch?"

    Each epoch consists of `n_patches / batch_size` steps. The total number of steps
    is shown in the console during training (for example, `300` steps):

    ```sh
    Epoch 1: 12%|█████████████                                     | 36/300
    ```

### Augmentations

CAREamics applies random X/Y flips and 90-degree rotations by default.

```python title="Setting augmentations"
--8<-- "current/configuration/n2v/general_n2v.py:config_n2v_augs"
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
--8<-- "current/configuration/n2v/general_n2v.py:config_n2v_val"
```

If separate validation data is supplied, `n_val_patches` is ignored and the validation
data determines the number of validation patches automatically.

## Algorithm-specific parameters

Noise2Void has multiple flavours developed to overcome some of its limitations, such
as checkerboard artefacts (N2V2) or correlated noise (structN2V).

### N2V2

To use N2V2, set `use_n2v2` to `True`.

```python title="Using N2V2"
--8<-- "current/configuration/n2v/general_n2v.py:config_n2v2"
```

### structN2V

structN2V has its own configuration factory function. It allows setting the orientation
of the noise correlation and the mask size.

```python title="Using structN2V"
--8<-- "current/configuration/n2v/general_n2v.py:config_structn2v"
```

1. Orientation of the noise correlation can be set to `"horizontal"`, `"vertical"`,
   `"cross"`, or `"square"`.
2. Size of the mask in pixels.

!!! note "Advanced structN2V"

    In the [advanced configuration section](#advanced-configuration), the Noise2Void
    factory also allows using structN2V with the same two parameters.

## Advanced configuration

### Training in memory

Loading data into memory can speed up patch extraction when the dataset fits in RAM.

```python title="Setting in-memory training"
--8<-- "current/configuration/n2v/advanced_n2v.py:adv_config_n2v_in_memory"
```

1. Only `array`, `tiff`, and `custom` are compatible with in-memory training.

Only `array`, `tiff`, and `custom` data types support in-memory training. Array data is
kept in memory automatically. Other formats, such as `zarr` or `czi`, use random
access from disk.

### Subsetting channels

To train on only some input channels, pass their zero-based indices through `channels`.

```python title="Selecting a subset of channels"
--8<-- "current/configuration/n2v/advanced_n2v.py:adv_config_n2v_subchannels"
```

1. `C` must be present in `axes` for channels to be considered.
2. Train using the first and third channels.


!!! info "`n_channels` and `channels`"

    If `C` is in `axes`, one of `n_channels` and `channels` must be specified. If
    only `channels` is specified, then `n_channels` will be set to `len(channels)`.

    If `n_channels` and `channels` are incompatible (`n_channels!=len(channels)`),
    an error will be raised.


### Channel independence

By default, channels are trained independently. Set `independent_channels=False` to
allow information to flow between input channels.

```python title="Training non-independent channels"
--8<-- "current/configuration/n2v/advanced_n2v.py:adv_config_n2v_ind_channels"
```

1. `C` must be present in `axes`.
2. The number of channels must be specified.
3. Setting `independent_channels=False` allows channels to inform each other.

### Normalization

Normalization is applied during training and prediction. The available strategies are
`mean_std`, `quantile`, `min_max`, and `none`.

```python title="Specifying the normalization method"
--8<-- "current/configuration/n2v/advanced_n2v.py:adv_config_n2v_norm"
```

1. Select the normalization method with a string or a Pydantic normalization
   configuration such as `QuantileConfig`.
2. The Pydantic configuration customizes the method's parameters.

If statistics are not provided, CAREamics computes them from the training data. They
can also be passed explicitly:

```python title="Passing pre-computed normalization parameters"
--8<-- "current/configuration/n2v/advanced_n2v.py:adv_config_n2v_norm_params"
```

1. This selects zero-mean, unit-variance normalization, which is the default.
2. The pre-computed statistics are passed through `MeanStdConfig`.

When channels are present, normalization parameters normally contain one value per
channel. Set `per_channel=False` when a shared value is intended.

```python title="Normalizing multi-channel data"
--8<-- "current/configuration/n2v/advanced_n2v.py:adv_config_n2v_norm_ch"
```

1. The example uses multiple channels.
2. The number of channels must be specified.
3. Each normalization parameter has one value per channel.
4. With `per_channel=False`, each parameter instead expects one shared value.

### Choosing a logger

CSV logging is always enabled. An additional Weights & Biases or TensorBoard logger can
be selected with `logger`.

```python title="Choosing a logger"
--8<-- "current/configuration/n2v/advanced_n2v.py:adv_config_n2v_logger"
```

1. This selects Weights & Biases in addition to the CSV logger.

### Number of workers

`num_workers` controls data-loading workers. The default `-1` selects a value based on
the available CPUs.

```python title="Setting the number of workers"
--8<-- "current/configuration/n2v/advanced_n2v.py:adv_config_n2v_num_workers"
```

1. This sets the number of workers used for data loading.

## PyTorch and Lightning parameters

### Model parameters

The U-Net model parameters can be passed through `model_params`.

```python title="Passing model parameters"
--8<-- "current/configuration/n2v/lightning_n2v.py:adv_config_n2v_model"
```

Common model parameters include `depth`, `num_channels_init`, `residual`, and
`use_batch_norm`.

### Trainer parameters

Pass additional PyTorch Lightning `Trainer` options through `trainer_params`.

```python title="Passing Trainer parameters"
--8<-- "current/configuration/n2v/lightning_n2v.py:adv_config_n2v_trainer"
```

1. Pass additional Trainer parameters as a dictionary through `trainer_params`.

`num_epochs` and `num_steps` take precedence over corresponding `max_epochs` and
`limit_train_batches` values in `trainer_params`.

### Optimizer parameters

CAREamics uses `Adam` by default. Select another supported optimizer and pass its
parameters through `optimizer_params`.

```python title="Passing optimizer parameters"
--8<-- "current/configuration/n2v/lightning_n2v.py:adv_config_n2v_opt"
```

1. Select the optimizer with `optimizer`.
2. Pass optimizer parameters as a dictionary through `optimizer_params`.

Supported optimizers are `Adam`, `SGD`, and `Adamax`.

### Learning-rate schedulers

Configure the learning-rate scheduler with `lr_scheduler` and `lr_scheduler_params`.

```python title="Passing learning-rate scheduler parameters"
--8<-- "current/configuration/n2v/lightning_n2v.py:adv_config_n2v_lr"
```

1. Select the scheduler with `lr_scheduler`.
2. Pass scheduler parameters as a dictionary through `lr_scheduler_params`.

Supported schedulers are `ReduceLROnPlateau` and `StepLR`.

### Dataloader parameters

Use `train_dataloader_params` and `val_dataloader_params` to customize the PyTorch
dataloaders.

```python title="Passing dataloader parameters"
--8<-- "current/configuration/n2v/lightning_n2v.py:adv_config_n2v_train_dm"
```

When overriding `train_dataloader_params`, specify the desired `shuffle` value.

### Checkpoint callback

Override the default checkpoint behavior with `checkpoint_params`.

```python title="Passing checkpoint parameters"
--8<-- "current/configuration/n2v/lightning_n2v.py:adv_config_n2v_checkpoint"
```

N2V uses a different default checkpoint behavior because validation loss is not
necessarily correlated with denoising quality.

Set `monitor_metric` to choose the metric used for checkpointing and scheduling. The
available values are `val_loss` (the default), `train_loss`, and `train_loss_epoch`.
When training without validation patches (`n_val_patches=0`), use
`monitor_metric="train_loss_epoch"`.

### Early stopping callback

Early stopping is disabled by default for Noise2Void and is not exposed by its
configuration factory.

## Saving and loading

Configurations are automatically saved with checkpoints, but they can also be saved
and loaded manually.

```python title="Save and load configurations"
--8<-- "current/configuration/configuration_io.py:save_load"
```
