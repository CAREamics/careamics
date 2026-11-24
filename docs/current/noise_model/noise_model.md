---
icon: lucide/chart-bell-curve
description: Train and use noise models with CAREamics
---

# Noise models

CAREamics can use a trained Gaussian-mixture noise model in the loss of [HDN](../configuration/hdn.md) and
[MicroSplit](../configuration/microsplit.md).

## What a noise model represents

A noise model describes how the observed intensity varies around an underlying signal.
It is trained from two matching arrays:

- `signal`: a clean or denoised estimate, typically obtained from an N2V prediction;
- `observation`: the corresponding raw noisy data.

The arrays must have identical shapes. Their axes are passed explicitly, for example
`SYX` for a stack of single-channel 2D images or `SCYX` for a multi-channel stack.

## Create signal and observation pairs

The easiest way to create pairs of signal and observation is to train Noise2Void, use it
to predict denoised images. Together, the raw images and the denoised predictions can
be used to train a noise model.

## Train and save a noise model

Once you have pairs of signal and observation images, you can train the noise model using
`train_noise_model`.

```python title="Training a noise model"
--8<-- "current/noise_model/noise_model.py:noise_model_train"
```

- `n_gaussian` controls the number of mixture components (Gaussians)
- `n_coeff` controls the polynomial used to describe how the noise parameters vary with signal intensity.
- `min_sigma` can be used to impose a lower bound on the fitted standard deviation.

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

For MicroSplit, set `use_noise_model` to `True` and pass the
per-channel model files to `CAREamist.train`.

```python title="Using a noise model with MicroSplit"
--8<-- "current/noise_model/noise_model.py:noise_model_use_microsplit"
```

MicroSplit's noise-model likelihood requires mean-standard-deviation normalization.

## Practical considerations

- Use signal/observation pairs with matching spatial content and axes.
- Train the model on data representative of the images that will be denoised.
- Keep intensity ranges consistent between noise-model training and denoising.
