---
icon: lucide/layers
description: MicroSplit-specific configuration
---

# MicroSplit-specific configuration

MicroSplit is an LVAE-based denoising algorithm. Its simple factory requires the
number of output channels to split into; the advanced factory additionally exposes
the multiscale, synthetic-input, LVAE, loss, and optimization parameters.

## Factory parameters

The simple factory accepts `experiment_name`, `data_type`, `axes`, `patch_size`,
`batch_size`, `output_channels`, `num_epochs`, `num_steps`, `augmentations`,
`n_val_patches`, and `multiscale_count`.

The advanced factory adds `channels`, `normalization`, `normalization_params`,
`in_memory`, `padding_mode`, `alpha_ranges`, `uncorrelated_channel_prob`,
`model_params`, `predict_logvar`, `logvar_lowerbound`, the four loss weights, and
the shared worker, trainer, optimizer, scheduler, dataloader, checkpoint, early
stopping, logger, and seed parameters.

```python title="Advanced MicroSplit configuration"
--8<-- "current/config_advanced/config_advanced_microsplit.py:adv_config_microsplit"
```

## Algorithm-specific parameters

### Synthetic inputs

`alpha_ranges` controls the ranges used to sample channel-mixing weights, while
`uncorrelated_channel_prob` controls how often uncorrelated synthetic channels are
sampled.

```python
--8<-- "current/config_algorithm_specific/configuration_specific_microsplit.py:microsplit_synthetic_inputs"
```

### LVAE and multiscale model

`multiscale_count` controls the number of lateral-context scales and `padding_mode`
controls how lateral-context patches are padded at image borders. `model_params` can
override non-structural LVAE settings such as latent dimensions, filter count, and
dropout.

```python
--8<-- "current/config_algorithm_specific/configuration_specific_microsplit.py:microsplit_model"
```

### Loss and likelihood terms

MicroSplit combines reconstruction, KL, Gaussian-likelihood, and noise-model
likelihood terms. The likelihood weights select between muSplit-like and denoiSplit-
like behavior. A noise model must be supplied at training time when
`noise_model_likelihood_weight` is non-zero.

```python
--8<-- "current/config_algorithm_specific/configuration_specific_microsplit.py:microsplit_loss"
```
