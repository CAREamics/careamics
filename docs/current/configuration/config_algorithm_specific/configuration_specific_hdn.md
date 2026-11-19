---
icon: lucide/layers
description: HDN-specific configuration
---

# HDN-specific configuration

HDN is an LVAE-based denoising algorithm. Its advanced factory supports either a
Gaussian likelihood (DivNoising) or a supplied noise-model likelihood, as well as an
optional supervised mode.

## Factory parameters

The simple factory accepts `experiment_name`, `data_type`, `axes`, `patch_size`,
`batch_size`, `num_epochs`, `num_steps`, `augmentations`, `n_val_patches`,
`use_noise_model`, and `supervised`.

The advanced factory adds `channels`, `normalization`, `normalization_params`,
`in_memory`, `output_channels`, `model_params`, `reconstruction_weight`,
`kl_weight`, `logvar_lowerbound`, and the shared worker, trainer, optimizer,
scheduler, dataloader, checkpoint, early-stopping, logger, and seed parameters.

```python title="Advanced HDN configuration"
--8<-- "current/configuration/config_advanced/configuration_advanced_hdn.py:adv_config_hdn"
```

## Noise model and supervised modes

Set `use_noise_model=True` when a trained noise model will be supplied to
`CAREamist.train(noise_model=...)`. Leave it `False` to use the Gaussian likelihood.
`supervised=True` requires target data and is incompatible with the noise-model path.

```python
--8<-- "current/configuration/config_algorithm_specific/configuration_specific_hdn.py:hdn_noise_model"
```

```python
--8<-- "current/configuration/config_algorithm_specific/configuration_specific_hdn.py:hdn_supervised"
```

## Algorithm-specific parameters

`reconstruction_weight` and `kl_weight` balance the reconstruction and KL terms.
`logvar_lowerbound` applies to the Gaussian pathway, and `output_channels` controls
the number of predicted channels. The LVAE architecture can be adjusted through
non-structural `model_params`.
