---
icon: lucide/layers-arrow-up
description: Advanced cryoCARE configuration
---

# Configuring cryoCARE

To start with cryoCARE, create a configuration object containing the parameters for
your experiment. The configuration validates individual parameters and their
coherence, avoiding combinations that could trigger errors later in the workflow.

cryoCARE provides a simple factory for common tomogram denoising and an advanced factory
for more complex parameters.

```python title="cryoCARE configuration factories"
--8<-- "current/configuration/cryocare/general_cryocare.py:all_configs"
```

## Simple configuration

The simple configuration function exposes the parameters most commonly used to start
an experiment.

```python title="Configure cryoCARE"
--8<-- "current/configuration/cryocare/general_cryocare.py:config_cryocare"
```

1. cryoCARE requires a 3D stack.
2. And 3D patches.
3. `tilt_axis` is the rotational axis around which the specimen was tilted.

<!-- break down the block to allow the list to be generated -->

- `experiment_name`: used during logging and automatic model saving. It
  should only contain letters, numbers, underscores, dashes, and spaces.
- `data_type`: input data type. CAREamics supports `array` (when passing NumPy arrays
  directly), `tiff`, `zarr`, `mrc`, `czi`, and `custom` (see [data](../data/)).
- `axes`: axes of the input data, in the order they have on disk or in memory. Accepted
  axes are `ZYX` in any order, with `YX` being contiguous.
- `patch_size`: size of the spatial patches extracted during training, must be 3D
- `batch_size`: number of patches used in each training step.
- `tilt_axis`: tilt axis, one of `X`, `Y`, or `Z`.
- `num_epochs`: optional number of epochs. For large datasets, also
  consider setting `num_steps`, see [Reducing the number of steps](#reducing-the-number-of-steps).
- `validation_fraction`: optional, fraction of continuous slices along `tilt_axis` reserved for validation.
- `n_val_patches`: optional, number of training patches reserved for validation when separate
  validation data is not provided. See [Number of validation patches](#number-of-validation-patches).
- `seed`: optional, fix a seed to make data ordering, augmentations, and patch sampling
  reproducible.


### Channels

cryoCARE is not compatible with a channel dimension.


### Reducing the number of steps

Each training epoch cycles through the available patches. For large datasets, set
`num_steps` to limit the number of batches processed in an epoch.

```python title="Setting the number of steps"
--8<-- "current/configuration/cryocare/general_cryocare.py:config_cryocare_steps"
```

### Augmentations

cryoCARE derives its augmentations from `tilt_axis` and `patch_size`. These
transformations cannot be configured independently by users.


### Validation fraction

Use `validation_fraction` to control the percentage of continuous slices reserved for
validation.

```python title="Setting the validation fraction"
--8<-- "current/configuration/cryocare/general_cryocare.py:cryocare_validation"
```

!!! info "Number of patches"

    `validation_fraction` is distinct from `n_val_patches`, which controls how many
    patches are extracted from the reserved continuous slices.

## Advanced configuration

### Normalization

Normalization is applied to the reconstructed volumes used for training and validation.
The available strategies are `mean_std`, `quantile`, `min_max`, and `none`.

```python title="Specifying the normalization method"
--8<-- "current/configuration/cryocare/advanced_cryocare.py:cryocare_normalization"
```

Pre-computed statistics can be passed through a Pydantic normalization configuration
in the advanced factory.

### Choosing a logger

CSV logging is always enabled. An additional Weights & Biases or TensorBoard logger can
be selected with `logger`.

```python title="Choosing a logger"
--8<-- "current/configuration/cryocare/advanced_cryocare.py:adv_config_cryocare_logger"
```

### Number of workers

`num_workers` controls data-loading workers. The default `-1` selects a value based on
the available CPUs.

```python title="Setting the number of workers"
--8<-- "current/configuration/cryocare/advanced_cryocare.py:adv_config_cryocare_num_workers"
```

## PyTorch and Lightning parameters

cryoCARE uses a fixed model architecture and does not expose `model_params`.

### Trainer parameters

Pass additional PyTorch Lightning `Trainer` options through `trainer_params`.

```python title="Passing Trainer parameters"
--8<-- "current/configuration/cryocare/lightning_cryocare.py:adv_config_cryocare_trainer"
```

### Optimizer parameters

cryoCARE defaults to `Adam`. Select another supported optimizer and pass its parameters
through `optimizer_params`.

```python title="Passing optimizer parameters"
--8<-- "current/configuration/cryocare/lightning_cryocare.py:adv_config_cryocare_opt"
```

Supported optimizers are `Adam`, `SGD`, and `Adamax`.

### Learning-rate schedulers

Configure the learning-rate scheduler with `lr_scheduler` and `lr_scheduler_params`.

```python title="Passing learning-rate scheduler parameters"
--8<-- "current/configuration/cryocare/lightning_cryocare.py:adv_config_cryocare_lr"
```

Supported schedulers are `ReduceLROnPlateau` and `StepLR`.

### Dataloader parameters

Use `train_dataloader_params` and `val_dataloader_params` to customize the PyTorch
dataloaders. When overriding `train_dataloader_params`, specify the desired `shuffle`
value.

```python title="Passing dataloader parameters"
--8<-- "current/configuration/cryocare/lightning_cryocare.py:adv_config_cryocare_train_dm"
```

### Checkpoint callback

Override the default checkpoint behavior with `checkpoint_params`.

```python title="Passing checkpoint parameters"
--8<-- "current/configuration/cryocare/lightning_cryocare.py:adv_config_cryocare_checkpoint"
```

### Early stopping callback

cryoCARE does not expose `early_stopping_params` in its configuration factory.

## Saving and loading

Configurations are automatically saved with checkpoints, but they can also be saved
and loaded manually.

```python title="Save and load configurations"
--8<-- "current/configuration/configuration_io.py:save_load"
```
