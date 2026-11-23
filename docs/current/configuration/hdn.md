---
icon: lucide/sticky-note-off
description: Advanced HDN configuration
---

# Configuring HDN

To start with CAREamics, create a configuration object containing the parameters for
your experiment. The configuration validates individual parameters and their
coherence, avoiding combinations that could trigger errors later in the workflow.

HDN provides a simple factory for common denoising experiments and an advanced factory
for more complex parameters.

```python title="HDN configuration factories"
--8<-- "current/configuration/hdn/general_hdn.py:all_configs"
```

## Simple configuration

The simple configuration function exposes the parameters most commonly used to start
an experiment.

```python title="Configure HDN"
--8<-- "current/configuration/hdn/general_hdn.py:config_hdn"
```

<!-- break down the block to allow the list to be generated -->

The simple factory supports Gaussian-likelihood denoising by default. Set
`use_noise_model=True` to use a trained noise model, or `supervised=True` to train
against target data.

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
- `use_noise_model`: optional, use a supplied noise model for training. See [Noise model](#noise-model).
- `supervised`: optional, train against target images. This cannot be combined with
  `use_noise_model=True`.
- `n_val_patches`: number of training patches reserved for validation when separate
  validation data is not provided. See [Number of validation patches](#number-of-validation-patches).
- `seed`: optional, fix a seed to make data ordering, augmentations, and patch sampling
  reproducible.

!!! note "Training with `T` as depth axis"

    To use `T` as the depth axis, relabel it as `Z`. For `data_type="czi"`, `T` can
    also be used directly as a depth axis, for example `axes="SCTYX"`. See the
    [data section](../data/data.md) for details.

### Channels

Channels are a particular type of axis and influence the way the deep-learning model
is built.

!!! warning "HDN and multi-channel data"

    Currently, HDN is not compatible with multi-channel data.


### Reducing the number of steps

Each training epoch cycles through all patches. For large datasets, set `num_steps` to
limit the number of batches processed in an epoch.

```python title="Setting the number of steps"
--8<-- "current/configuration/hdn/general_hdn.py:config_hdn_steps"
```

### Augmentations

CAREamics applies random X/Y flips and 90-degree rotations by default. The same
spatial transformation is applied to a training patch and its corresponding target.

```python title="Setting augmentations"
--8<-- "current/configuration/hdn/general_hdn.py:config_hdn_augs"
```

1. The available choices are `x_flip`, `y_flip`, and `rotate_90`.


!!! note "When to disable augmentations"

    If your data has a particular symmetry, because objects or features are always
    oriented in a particular way, it may be necessary to turn off augmentations that
    break this symmetry.


### Number of validation patches

When no validation data is provided, CAREamics can reserve `n_val_patches` patches
from the training data for validation.

```python title="Setting the number of validation patches"
--8<-- "current/configuration/hdn/general_hdn.py:config_hdn_val"
```


### Noise model

A noise model guides training towards realistic output values and can be obtained from
noisy images (see the [noise model](../noise_model/) section). If `use_noise_model` is
`True`, you need to supply the trained model at [training time](../careamist/careamist_training.md#noise-model).

```python title="Training with a noise model"
--8<-- "current/configuration/hdn/advanced_hdn.py:hdn_noise_model"
```

### Supervised HDN

If ground-truth data is available, you can train HDN with supervision by setting `supervised`
to `True`. Target images must then be supplied at [training time](../careamist/careamist_training.md#targets).

```python title="Training with supervision"
--8<-- "current/configuration/hdn/advanced_hdn.py:hdn_supervised"
```

!!! warning "Supervised HDN"

    `supervised=True` and `use_noise_model=True` cannot be used together.


## Advanced configuration

### Training in memory

Loading data into memory can speed up patch extraction when the dataset fits in RAM.

```python title="Setting in-memory training"
--8<-- "current/configuration/hdn/advanced_hdn.py:adv_config_hdn_in_memory"
```

Only `array`, `tiff`, and `custom` data types support in-memory training. Array data is
kept in memory automatically. Other formats, such as `zarr`, `czi` or `mrc` use random
access from disk.

### Normalization

Normalization is applied during training and prediction. The available strategies are
`mean_std`, `quantile`, `min_max`, and `none`.

```python title="Specifying the normalization method"
--8<-- "current/configuration/hdn/advanced_hdn.py:adv_config_hdn_norm"
```

Pre-computed normalization parameters can also be supplied explicitly:

```python title="Passing pre-computed normalization parameters"
--8<-- "current/configuration/hdn/advanced_hdn.py:adv_config_hdn_norm_params"
```

### Choosing a logger

CSV logging is always enabled. An additional Weights & Biases or TensorBoard logger can
be selected with `logger`.

```python title="Choosing a logger"
--8<-- "current/configuration/hdn/advanced_hdn.py:adv_config_hdn_logger"
```

### Number of workers

`num_workers` controls data-loading workers. The default `-1` selects a value based on
the available CPUs.

```python title="Setting the number of workers"
--8<-- "current/configuration/hdn/advanced_hdn.py:adv_config_hdn_num_workers"
```

### Loss parameters

The advanced factory exposes the reconstruction and KL weights, as well as the lower
bound for the predicted log-variance.

```python title="Setting HDN loss parameters"
--8<-- "current/configuration/hdn/advanced_hdn.py:adv_config_hdn_loss"
```


## PyTorch and Lightning parameters

### Model parameters

HDN uses an LVAE model. Use `model_params` for non-structural LVAE settings; structural
settings are fixed by the factory.

```python title="Passing model parameters"
--8<-- "current/configuration/hdn/lightning_hdn.py:adv_config_hdn_model"
```

### Trainer parameters

Pass additional PyTorch Lightning `Trainer` options through `trainer_params`.

```python title="Passing Trainer parameters"
--8<-- "current/configuration/hdn/lightning_hdn.py:adv_config_hdn_trainer"
```

### Optimizer parameters

HDN defaults to `Adamax`. Select another supported optimizer and pass its parameters
through `optimizer_params`.

```python title="Passing optimizer parameters"
--8<-- "current/configuration/hdn/lightning_hdn.py:adv_config_hdn_opt"
```

Supported optimizers are `Adam`, `SGD`, and `Adamax`.

### Learning-rate schedulers

Configure the learning-rate scheduler with `lr_scheduler` and `lr_scheduler_params`.

```python title="Passing learning-rate scheduler parameters"
--8<-- "current/configuration/hdn/lightning_hdn.py:adv_config_hdn_lr"
```

Supported schedulers are `ReduceLROnPlateau` and `StepLR`.

### Dataloader parameters

Use `train_dataloader_params` and `val_dataloader_params` to customize the PyTorch
dataloaders. When overriding `train_dataloader_params`, specify the desired `shuffle`
value.

```python title="Passing dataloader parameters"
--8<-- "current/configuration/hdn/lightning_hdn.py:adv_config_hdn_train_dm"
```

### Checkpoint callback

Override the default checkpoint behavior with `checkpoint_params`.

```python title="Passing checkpoint parameters"
--8<-- "current/configuration/hdn/lightning_hdn.py:adv_config_hdn_checkpoint"
```

### Early stopping callback

Configure early stopping through `early_stopping_params`.

```python title="Passing early-stopping parameters"
--8<-- "current/configuration/hdn/lightning_hdn.py:adv_config_hdn_early_stop"
```

## Saving and loading

Configurations are automatically saved with checkpoints, but they can also be saved
and loaded manually.

```python title="Save and load configurations"
--8<-- "current/configuration/config_general/configuration_io.py:save_load"
```
