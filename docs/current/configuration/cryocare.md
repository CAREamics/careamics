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

1. `patch_size` is a three-dimensional size in `ZYX` order.
2. `tilt_axis` is the rotational axis around which the specimen was tilted.

<!-- break down the block to allow the list to be generated -->


- `experiment_name`: experiment name used in logging and automatic model saving.
- `data_type`: input data type. cryoCARE uses tomographic data such as `mrc`.
- `axes`: input axes. cryoCARE requires `Z` and does not support channel axes.
- `patch_size`: size of the three-dimensional patches extracted during training, in
  `ZYX` order.
- `batch_size`: number of patches used in each training step.
- `tilt_axis`: tilt axis, one of `X`, `Y`, or `Z`.
- `num_epochs`: optional number of epochs to train for.
- `num_steps`: optional number of batches per epoch.
- `validation_fraction`: fraction of continuous slices reserved for validation.
- `n_val_patches`: number of validation patches sampled from the validation split.

### Reducing the number of steps

Each training epoch cycles through the available patches. For large datasets, set
`num_steps` to limit the number of batches processed in an epoch.

```python title="Setting the number of steps"
--8<-- "current/configuration/cryocare/general_cryocare.py:config_cryocare_steps"
```

### Augmentations

cryoCARE derives its augmentations from `tilt_axis` and `patch_size`. These
transformations cannot be configured independently by users.

## Algorithm-specific parameters

### Tilt-axis splitting

The tilt axis determines how the tomogram is split into training and validation data.
The validation fraction refers to continuous slices along the reconstruction.

```python title="Setting the tilt axis and validation split"
--8<-- "current/configuration/cryocare/advanced_cryocare.py:cryocare_tilt_split"
```

### Validation fraction

Use `validation_fraction` to control the percentage of continuous slices reserved for
validation. This is distinct from `n_val_patches`, which controls patch sampling.

```python title="Setting the validation fraction"
--8<-- "current/configuration/cryocare/advanced_cryocare.py:cryocare_validation"
```

## Advanced configuration

### In-memory training and channels

cryoCARE does not expose an `in_memory` parameter and does not support channel axes.
Its data configuration controls loading for the supported tomographic formats.

### Normalization

Normalization is applied to the reconstructed volumes used for training and validation.
The available strategies are `mean_std`, `quantile`, `min_max`, and `none`.

```python title="Specifying the normalization method"
--8<-- "current/configuration/cryocare/advanced_cryocare.py:cryocare_normalization"
```

Pre-computed statistics can be passed through `normalization_params` in the advanced
factory.

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
--8<-- "current/configuration/config_general/configuration_io.py:save_load"
```
