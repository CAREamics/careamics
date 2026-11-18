---
icon: lucide/layers
description: CryoCARE-specific configuration
---

# CryoCARE-specific configuration

These examples target the `dev/cryocare` branch. CryoCARE is designed for cryo-electron
tomography volumes and differs from the other algorithms in how it creates training
and validation regions.

## Factory parameters

The simple factory accepts `experiment_name`, `data_type`, `axes`, `patch_size`,
`batch_size`, `num_epochs`, `num_steps`, `tilt_axis`, `validation_fraction`, and
`n_val_patches`.

The advanced factory adds `project_name`, `normalization`,
`normalization_params`, and the shared worker, trainer, optimizer, scheduler,
dataloader, checkpoint, logger, and seed parameters. CryoCARE does not expose channel
selection, user-defined augmentations, or an `in_memory` parameter.

```python title="Advanced CryoCARE configuration"
--8<-- "current/config_advanced/config_advanced_cryocare.py:adv_config_cryocare"
```

## Tilt-axis split

`tilt_axis` identifies the physical tilt axis. `validation_fraction` reserves a
fraction of the reconstructed stack along that axis, and `n_val_patches` controls the
number of validation patches drawn from that region.

```python
--8<-- "current/config_algorithm_specific/configuration_specific_cryocare.py:cryocare_tilt_split"
```

## Data and augmentation constraints

CryoCARE requires three-dimensional axes containing `Z`, does not support `C`, and
requires target axes to match input axes. Training uses stratified patching. The
factory generates the input-target swap and compatible tilt-axis rotation
augmentations automatically; users cannot override them.

The `mrc` data type is supported on `dev/cryocare`. `normalization` and
`normalization_params` remain configurable:

```python
--8<-- "current/config_algorithm_specific/configuration_specific_cryocare.py:cryocare_normalization"
```
