---
icon: lucide/chart-no-axes-combined
description: Train and use noise models with CAREamics
---

# Noise models

CAREamics can use a trained Gaussian-mixture noise model in the noise-model
likelihoods of [HDN](../config_algorithm_specific/configuration_specific_hdn.md) and
[MicroSplit](../config_algorithm_specific/configuration_specific_microsplit.md).
Noise models are not required for the default Gaussian-likelihood or self-supervised
workflows.

## What a noise model represents

A noise model describes how the observed intensity varies around an underlying signal.
It is trained from two matching arrays:

- `signal`: a clean or denoised estimate, typically obtained from an N2V prediction;
- `observation`: the corresponding raw noisy data.

The arrays must have identical shapes. Their axes are passed explicitly, for example
`SYX` for a stack of single-channel 2D images or `SCYX` for a multi-channel stack.

## Create signal and observation pairs

The following example creates a small synthetic dataset with additive Gaussian noise.
In a real workflow, `signal` would generally be a denoised estimate and `observation`
would be the original noisy acquisition.

```python title="Preparing noise-model training data"
--8<-- "current/noise_model/noise_model.py:noise_model_data"
```

## Train and save a noise model

`train_noise_model` fits one Gaussian-mixture model per channel. The saved directory
contains one `.npz` file per channel. For a quick example, use a small number of
Gaussian components and epochs; realistic experiments generally need more data and
training epochs.

```python title="Training a noise model"
--8<-- "current/noise_model/noise_model.py:noise_model_train"
```

`n_gaussian` controls the number of mixture components, while `n_coeff` controls the
polynomial used to describe how the noise parameters vary with signal intensity.
`min_sigma` can be used to impose a lower bound on the fitted standard deviation.

## Use a noise model with HDN

Set `use_noise_model=True` in the HDN configuration and pass the generated `.npz`
files to `CAREamist.train(noise_model=...)`. The number of files must match the
configured output channels.

```python title="Using a noise model with HDN"
--8<-- "current/noise_model/noise_model.py:noise_model_use_hdn"
```

HDN's noise-model likelihood is incompatible with `supervised=True`. Use the Gaussian
likelihood when training HDN against explicit target images.

## Use a noise model with MicroSplit

For MicroSplit, set `noise_model_likelihood_weight` to a non-zero value and pass the
per-channel model files to `CAREamist.train`.

```python title="Using a noise model with MicroSplit"
--8<-- "current/noise_model/noise_model.py:noise_model_use_microsplit"
```

MicroSplit's noise-model likelihood requires mean-standard-deviation normalization.
Keep the noise model and the training data in the same raw intensity space; CAREamics
normalizes the model internally when required.

## Practical considerations

- Use signal/observation pairs with matching spatial content and axes.
- Train the model on data representative of the images that will be denoised.
- Keep intensity ranges consistent between noise-model training and denoising.
- Train one model per channel when channels have different noise characteristics.
- A noise model is an input to the likelihood; it does not replace the algorithm's
  denoising model or configuration.
