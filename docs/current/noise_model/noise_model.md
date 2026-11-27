---
icon: lucide/chart-bell-curve
description: Train and use noise models with CAREamics
---

# Noise models

Some algorithms in CAREamics may require a noise model, such as [HDN](../configuration/hdn.md) and
[MicroSplit](../configuration/microsplit.md). Noise models are representation of the expected pixel value distributions and consist of a mixture of Gaussians for each potential real pixel value.


## Create signal and observation pairs

The easiest way to create pairs of signal and observation is to train [Noise2Void](../configuration/n2v.md) and to use the trained model to predict clean images. Together, the raw images and the denoised predictions can be used to train a noise model.

## Train and save a noise model

Once you have pairs of signal and observation images, you can train the noise model using
`train_noise_model`.

```python title="Training a noise model"
--8<-- "current/noise_model/noise_model.py:noise_model_train"
--8<-- "current/noise_model/noise_model.py:noise_model_train_2"
```

The `train_noise_model` has multiple optional parameters:
- `n_gaussian` controls the number of mixture components (Gaussians)
- `n_coeff` controls the polynomial used to describe how the noise parameters vary with signal intensity.
- `n_epochs` is the number of epoch to train the noise model for.
- `min_sigma` can be used to impose a lower bound on the fitted standard deviation.

Trained noise model are then passed to the train function, see the [noise model section](../careamist/careamist_training.md#noise-model).