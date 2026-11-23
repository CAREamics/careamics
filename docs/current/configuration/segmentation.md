---
icon: lucide/cookie
description: Configuring semantic segmentation in CAREamics
---

# Configuring segmentation

To start with CAREamics, create a configuration object containing the parameters for
your experiment. The configuration validates individual parameters and their
coherence, avoiding combinations that could trigger errors later in the workflow.

CAREamics provides a simple factory for common segmentation experiments and an advanced factory
for more complex parameters.

```python title="Segmentation configuration factories"
--8<-- "current/configuration/segmentation/general_seg.py:all_configs"
```

## Simple configuration

The simple configuration function exposes the parameters most commonly used to start
an experiment.

```python title="Configure segmentation"
--8<-- "current/configuration/segmentation/general_seg.py:config_seg"
```

1. Patch size concerns only spatial axes, must be in order `(Z)YX`, and must agree
   with `axis` about dimensionality (2D or 3D).
2. Set the number of foreground classes to be segmented.

<!-- break down the block to allow the list to be generated -->


- `experiment_name`: experiment name used in logging and automatic model saving. It
  should only contain letters, numbers, underscores, dashes, and spaces.
- `data_type`: input data type. CAREamics supports `array` (when passing NumPy arrays
  directly), `tiff`, `zarr`, `mrc`, `czi`, and `custom` (see [data](../data/)).
- `axes`: axes of the input data, in the order they have on disk or in memory. Accepted
  axes are `(S)(T)(C)(Z)YX` in any order, with `YX` being contiguous.
- `patch_size`: size of the spatial patches extracted during training, in `(Z)YX`
  order. It should be 3D when `Z` is in `axes` and 2D otherwise. See the note below
  to use `T` as a third dimension.
- `batch_size`: number of patches used in each training step.
- `num_epochs`: optional number of epochs to train for. For large datasets, also
  consider setting [`num_steps`](#reducing-the-number-of-steps).
- `num_steps`: optional number of batches per epoch. If omitted, the full training
  dataset is used. See [Reducing the number of steps](#reducing-the-number-of-steps).
- `augmentations`: optional list of `x_flip`, `y_flip`, and `rotate_90`. If omitted,
  all three are applied. Use an empty list to disable them.
- `n_val_patches`: number of training patches reserved for validation when separate
  validation data is not provided. See [Number of validation patches](#number-of-validation-patches).
- `seed`: fix a seed to make data ordering, augmentations, and patch sampling
  reproducible.

!!! note "Training with `T` as depth axis"

    To use `T` as the depth axis, relabel it as `Z`. For `data_type="czi"`, `T` can
    also be used directly as a depth axis, for example `axes="SCTYX"`. See the
    [data section](../data/data.md) for details.

### Channels

Channels are a particular type of axis and influence the way the deep-learning model
is built. When `C` is present in `axes`, set `n_channels`.

```python title="Setting segmentation input channels"
--8<-- "current/configuration/segmentation/general_seg.py:config_seg_channels"
```

1. Channels are considered to be present as soon as `C` is in `axes`.
2. The number of input channels must be set in `n_channels_in`.

Segmentation fixes channel independence to `False`; it does not expose an
`independent_channels` parameter.

### Reducing the number of steps

Each training epoch cycles through all patches. For large datasets, set `num_steps` to
limit the number of batches processed in an epoch.

```python title="Setting the number of steps"
--8<-- "current/configuration/segmentation/general_seg.py:config_seg_steps"
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
spatial transformation is applied consistently to the training data and targets.

```python title="Setting augmentations"
--8<-- "current/configuration/segmentation/general_seg.py:config_seg_augs"
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
--8<-- "current/configuration/segmentation/general_seg.py:config_seg_val"
```

If separate validation data is supplied, `n_val_patches` is ignored and the validation
data determines the number of validation patches automatically.

## Algorithm-specific parameters

### Prediction output

The `prediction_output` parameter takes one of two values: `masks` or `probabilities`.
It controls the type of output after prediction: either the mask for the class with
the highest probability or the probability map.

```python title="Prediction output type"
--8<-- "current/configuration/segmentation/general_seg.py:config_seg_prediction"
```

## Advanced configuration

### Training in memory

Loading data into memory can speed up patch extraction when the dataset fits in RAM.

```python title="Setting in-memory training"
--8<-- "current/configuration/segmentation/advanced_seg.py:adv_config_seg_in_memory"
```

1. In-memory training is supported for `array`, `tiff`, and `custom` data.

Only `array`, `tiff`, and `custom` data types support in-memory training. Array data is
kept in memory automatically; `zarr` and `czi` use random access from disk.

### Subsetting channels

To train on only some input channels, pass their zero-based indices through `channels`.
When `channels` is set, the number of input channels is inferred automatically. The
label mask is not channel-subsetted.

```python title="Selecting a subset of channels"
--8<-- "current/configuration/segmentation/advanced_seg.py:adv_config_seg_subchannels"
```

1. This example uses the first and third input channels, because channel indices start
   at zero.

`C` must be present in `axes` for channels to be considered.

### Normalization

Normalization is applied during training and prediction. The available strategies are
`mean_std`, `quantile`, `min_max`, and `none`.

```python title="Specifying the normalization method"
--8<-- "current/configuration/segmentation/advanced_seg.py:adv_config_seg_norm"
```

1. Select the normalization method with `normalization`.
2. `normalization_params` customizes the quantile bounds; target normalization is
   skipped automatically.

If statistics are not provided, CAREamics computes them from the training data. When
channels are present, normalization parameters normally contain one value per channel.
Set `per_channel=False` when a shared value is intended.

### Loss

Segmentation supports the `dice`, `ce`, and `dice_ce` losses. Use `loss_parameters` to
configure class weights and the relative Dice/cross-entropy contributions.

```python title="Setting the segmentation loss"
--8<-- "current/configuration/segmentation/advanced_seg.py:adv_config_seg_loss"
```

### Patch filtering

Advanced segmentation configurations can filter patches before training. This can be
useful when most patches contain little or no foreground signal.

```python title="Filtering low-signal patches"
--8<-- "current/configuration/segmentation/advanced_seg.py:adv_config_seg_filter"
```

### Choosing a logger

CSV logging is always enabled. An additional Weights & Biases or TensorBoard logger can
be selected with `logger`.

```python title="Choosing a logger"
--8<-- "current/configuration/segmentation/advanced_seg.py:adv_config_seg_logger"
```

### Number of workers

`num_workers` controls data-loading workers. Set this according to the available CPU
resources and storage performance.

```python title="Setting the number of workers"
--8<-- "current/configuration/segmentation/advanced_seg.py:adv_config_seg_num_workers"
```

## PyTorch and Lightning parameters

### Model parameters

The U-Net model parameters can be passed through `model_params`.

```python title="Setting U-Net parameters"
--8<-- "current/configuration/segmentation/lightning_segmentation.py:adv_config_seg_model"
```

1. Increasing `depth` adds U-Net levels and generally increases model capacity and
   memory use.

Common model parameters include `depth`, `num_channels_init`, `residual`, and
`use_batch_norm`.

### Trainer parameters

Pass additional PyTorch Lightning `Trainer` options through `trainer_params`.
`num_epochs` and `num_steps` take precedence over corresponding `max_epochs` and
`limit_train_batches` values in `trainer_params`.

```python title="Passing Trainer parameters"
--8<-- "current/configuration/segmentation/lightning_segmentation.py:adv_config_seg_trainer"
```

### Optimizer parameters

CAREamics uses `Adam` by default. Select another supported optimizer and pass its
parameters through `optimizer_params`. Supported optimizers are `Adam`, `SGD`, and
`Adamax`.

```python title="Passing optimizer parameters"
--8<-- "current/configuration/segmentation/lightning_segmentation.py:adv_config_seg_opt"
```

### Learning-rate schedulers

Configure the learning-rate scheduler with `lr_scheduler` and `lr_scheduler_params`.
Supported schedulers are `ReduceLROnPlateau` and `StepLR`.

```python title="Passing learning-rate scheduler parameters"
--8<-- "current/configuration/segmentation/lightning_segmentation.py:adv_config_seg_lr"
```

### Dataloader parameters

Use `train_dataloader_params` and `val_dataloader_params` to customize the PyTorch
dataloaders. When overriding `train_dataloader_params`, specify the desired `shuffle`
value.

```python title="Passing dataloader parameters"
--8<-- "current/configuration/segmentation/lightning_segmentation.py:adv_config_seg_train_dm"
```

### Checkpoint callback

Override the default checkpoint behavior with `checkpoint_params`.

```python title="Passing checkpoint parameters"
--8<-- "current/configuration/segmentation/lightning_segmentation.py:adv_config_seg_checkpoint"
```

### Early stopping callback

Configure early stopping through `early_stopping_params`.

```python title="Passing early-stopping parameters"
--8<-- "current/configuration/segmentation/lightning_segmentation.py:adv_config_seg_early_stop"
```

## Saving and loading

Configurations are automatically saved with checkpoints, but they can also be saved
and loaded manually.

```python title="Save and load configurations"
--8<-- "current/configuration/config_general/configuration_io.py:save_load"
```
