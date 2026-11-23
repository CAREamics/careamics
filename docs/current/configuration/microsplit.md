---
icon: lucide/image-plus
description: Advanced MicroSplit configuration
---

# Configuring MicroSplit

To start with CAREamics, create a configuration object containing the parameters for
your experiment. The configuration validates individual parameters and their
coherence, avoiding combinations that could trigger errors later in the workflow.

MicroSplit provides a simple factory for common channel-unmixing experiments and an advanced factory
for more complex parameters.

```python title="MicroSplit configuration factories"
--8<-- "current/configuration/microsplit/general_microsplit.py:all_configs"
```

## Simple configuration

The simple configuration function exposes the parameters most commonly used to start
an experiment.

```python title="Configure MicroSplit"
--8<-- "current/configuration/microsplit/general_microsplit.py:config_microsplit"
```

1. Patch size concerns only spatial axes and must be in `(Z)YX` order.
2. `output_channels` is the number of channels to reconstruct.

- `experiment_name`: experiment name used in logging and automatic model saving. It
  should only contain letters, numbers, underscores, dashes, and spaces.
- `data_type`: input data type. CAREamics supports `array`, `tiff`, `zarr`, `czi`, and
  `custom` (see [data](../data/)).
- `axes`: axes of the input data, in the order they have on disk or in memory. Accepted
  axes are `(S)(T)(C)(Z)YX` in any order, with `YX` being contiguous.
- `patch_size`: size of the spatial patches extracted during training, in `(Z)YX`
  order.
- `batch_size`: number of patches used in each training step.
- `output_channels`: number of target channels to split the input into.
- `num_epochs`: optional number of epochs to train for.
- `num_steps`: optional number of batches per epoch. If omitted, the full training
  dataset is used. See [Reducing the number of steps](#reducing-the-number-of-steps).
- `augmentations`: optional list of `x_flip`, `y_flip`, and `rotate_90`. If omitted,
  all three are applied. Use an empty list to disable them.
- `n_val_patches`: number of training patches reserved for validation when separate
  validation data is not provided.
- `multiscale_count`: number of lateral-context scales used by MicroSplit.
- `seed`: fix a seed to make data ordering, augmentations, and patch sampling
  reproducible.

!!! note "Training with `T` as depth axis"

    To use `T` as the depth axis, relabel it as `Z`. For `data_type="czi"`, `T` can
    also be used directly as a depth axis, for example `axes="SCTYX"`. See the
    [data section](../data/data.md) for details.

### Reducing the number of steps

Each training epoch cycles through all patches. For large datasets, set `num_steps` to
limit the number of batches processed in an epoch.

```python title="Setting the number of steps"
--8<-- "current/configuration/microsplit/general_microsplit.py:config_msplit_steps"
```

1. Choose a value smaller than the total number of available batches when shorter
   epochs are desired.

### Augmentations

CAREamics applies random X/Y flips and 90-degree rotations by default. The same
spatial transformation is applied consistently to the training data and targets.

```python title="Setting augmentations"
--8<-- "current/configuration/microsplit/general_microsplit.py:config_microsplit_augs"
```

1. The available choices are `x_flip`, `y_flip`, and `rotate_90`.

Pass `augmentations=[]` to disable all augmentations.

### Number of validation patches

When no validation data is provided, CAREamics can reserve `n_val_patches` patches
from the training data for validation.

```python title="Setting the number of validation patches"
--8<-- "current/configuration/microsplit/general_microsplit.py:config_microsplit_val"
```

If separate validation data is supplied, `n_val_patches` is ignored and the validation
data determines the number of validation patches automatically.

## Algorithm-specific parameters

### Lateral context

`multiscale_count` controls how many spatial scales MicroSplit uses as input. A value
of 1 disables lateral context and uses only the original patch. Larger values provide
progressively wider image context at the same patch size.

```python title="Setting lateral-context scales"
--8<-- "current/configuration/microsplit/advanced_microsplit.py:adv_config_microsplit_multiscale"
```

### Padding mode

When lateral-context patches extend beyond image borders, `padding_mode` controls how
they are padded. Supported modes are `reflect` and `wrap`.

```python title="Setting the padding mode"
--8<-- "current/configuration/microsplit/advanced_microsplit.py:adv_config_microsplit_padding"
```

### Synthetic-input parameters

MicroSplit creates synthetic mixtures of channels during training. `alpha_ranges`
controls the ranges used to sample channel-mixing weights, while
`uncorrelated_channel_prob` controls how often uncorrelated channels are sampled.

```python title="Setting channel-mixing ranges"
--8<-- "current/configuration/microsplit/advanced_microsplit.py:adv_config_microsplit_alpha"
```

```python title="Sampling uncorrelated channels"
--8<-- "current/configuration/microsplit/advanced_microsplit.py:adv_config_microsplit_uncorrelated"
```

### Noise model training

For denoiSplit, set `noise_model_likelihood_weight` above zero and provide a trained
noise model at training time through `CAREamist.train(noise_model=...)` or
`MicroSplitModule.set_noise_model`.

### Loss parameters

The advanced factory exposes the reconstruction, KL, Gaussian-likelihood, and
noise-model-likelihood weights. It can also configure the predicted log-variance.

```python title="Setting MicroSplit loss parameters"
--8<-- "current/configuration/microsplit/advanced_microsplit.py:adv_config_microsplit_loss"
```

## Advanced configuration

### Training in memory

Loading data into memory can speed up patch extraction when the dataset fits in RAM.

```python title="Setting in-memory training"
--8<-- "current/configuration/microsplit/advanced_microsplit.py:adv_config_microsplit_in_memory"
```

Only `array`, `tiff`, and `custom` data types support in-memory training. Array data is
kept in memory automatically; `zarr` and `czi` use random access from disk.

### Subsetting channels

Pass zero-based input channel indices through `channels` to train on only a subset of
the available channels.

```python title="Selecting input channels"
--8<-- "current/configuration/microsplit/advanced_microsplit.py:adv_config_microsplit_channels"
```

### Normalization

Normalization is applied during training and prediction. The available strategies are
`mean_std`, `quantile`, `min_max`, and `none`.

```python title="Specifying the normalization method"
--8<-- "current/configuration/microsplit/advanced_microsplit.py:adv_config_microsplit_norm"
```

Pre-computed normalization parameters can also be supplied explicitly:

```python title="Passing pre-computed normalization parameters"
--8<-- "current/configuration/microsplit/advanced_microsplit.py:adv_config_microsplit_norm_params"
```

When channels are present, normalization parameters normally contain one value per
channel. Set `per_channel=False` when a shared value is intended.

### Choosing a logger

CSV logging is always enabled. An additional Weights & Biases or TensorBoard logger can
be selected with `logger`.

```python title="Choosing a logger"
--8<-- "current/configuration/microsplit/advanced_microsplit.py:adv_config_microsplit_logger"
```

### Number of workers

`num_workers` controls data-loading workers. The default `-1` selects a value based on
the available CPUs.

```python title="Setting the number of workers"
--8<-- "current/configuration/microsplit/advanced_microsplit.py:adv_config_microsplit_num_workers"
```

## PyTorch and Lightning parameters

### Model parameters

MicroSplit uses an LVAE model. Use `model_params` for non-structural LVAE settings;
structural settings such as input shape, output channels, and scale count are fixed by
the factory.

```python title="Passing model parameters"
--8<-- "current/configuration/microsplit/lightning_microsplit.py:adv_config_microsplit_model"
```

### Trainer parameters

Pass additional PyTorch Lightning `Trainer` options through `trainer_params`.
`num_epochs` and `num_steps` take precedence over corresponding `max_epochs` and
`limit_train_batches` values in `trainer_params`.

```python title="Passing Trainer parameters"
--8<-- "current/configuration/microsplit/lightning_microsplit.py:adv_config_microsplit_trainer"
```

### Optimizer parameters

MicroSplit defaults to `Adamax`. Select another supported optimizer and pass its
parameters through `optimizer_params`.

```python title="Passing optimizer parameters"
--8<-- "current/configuration/microsplit/lightning_microsplit.py:adv_config_microsplit_opt"
```

Supported optimizers are `Adam`, `SGD`, and `Adamax`.

### Learning-rate schedulers

Configure the learning-rate scheduler with `lr_scheduler` and `lr_scheduler_params`.

```python title="Passing learning-rate scheduler parameters"
--8<-- "current/configuration/microsplit/lightning_microsplit.py:adv_config_microsplit_lr"
```

Supported schedulers are `ReduceLROnPlateau` and `StepLR`.

### Dataloader parameters

Use `train_dataloader_params` and `val_dataloader_params` to customize the PyTorch
dataloaders. When overriding `train_dataloader_params`, specify the desired `shuffle`
value.

```python title="Passing dataloader parameters"
--8<-- "current/configuration/microsplit/lightning_microsplit.py:adv_config_microsplit_train_dm"
```

### Checkpoint callback

Override the default checkpoint behavior with `checkpoint_params`.

```python title="Passing checkpoint parameters"
--8<-- "current/configuration/microsplit/lightning_microsplit.py:adv_config_microsplit_checkpoint"
```

### Early stopping callback

Configure early stopping through `early_stopping_params`.

```python title="Passing early-stopping parameters"
--8<-- "current/configuration/microsplit/lightning_microsplit.py:adv_config_microsplit_early_stop"
```

## Saving and loading

Configurations are automatically saved with checkpoints, but they can also be saved
and loaded manually.

```python title="Save and load configurations"
--8<-- "current/configuration/config_general/configuration_io.py:save_load"
```
