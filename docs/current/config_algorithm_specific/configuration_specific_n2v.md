---
icon: lucide/eye-off
description: Advanced N2V configuration
---

# Advanced N2V

Before configuring N2V-specific parameters, check the [general configuration
page](../configuration.md) for shared options.

This page documents parameters and mechanisms specific to Noise2Void.

## Channels

Noise2Void uses the same number of input and output channels. When `C` is present in
`axes`, set `n_channels` to the number of input channels.

```python title="Setting Noise2Void channel parameters"
--8<-- "current/config_general/configuration_n2v.py:config_n2v_channels"
```

1. `C` indicates that channels are present in `axes`.
2. Noise2Void uses the same number of input and output channels, so set `n_channels`.

## Noise2Void variants and parameters

### N2V2

N2V2 modifies the U-Net to reduce checkerboard artifacts. Enable it with
`use_n2v2=True`.

```python title="Using N2V2"
--8<-- "current/config_general/configuration_n2v.py:config_n2v2"
```

### structN2V

structN2V masks lines of pixels instead of a single pixel and can help with structured
noise. Configure the line orientation with `struct_n2v_axes` and its span with
`struct_n2v_span`.

```python title="Using structN2V"
--8<-- "current/config_general/configuration_n2v.py:adv_config_structn2v"
```

1. Choices are `horizontal`, `vertical`, `cross`, and `square`.
2. `struct_n2v_span` is the number of pixels masked on each side of the central pixel.

### Noise2Void parameters

Noise2Void-specific masking is controlled by `roi_size` and
`masked_pixel_percentage`.

```python title="Setting Noise2Void parameters"
--8<-- "current/config_general/configuration_n2v.py:adv_config_n2v_params"
```

1. `roi_size` defines the region around masked pixels from which replacement values are
   drawn.
2. `masked_pixel_percentage` controls the percentage of pixels masked in each training
   patch.

### Training without validation

Validation is not strictly necessary for Noise2Void. To avoid reserving validation
patches, set `n_val_patches=0` and monitor the training loss instead.

```python title="Training Noise2Void without validation"
--8<-- "current/config_advanced/configuration_lightning.py:adv_config_n2v_no_val"
```

1. Set the validation split to zero.
2. Monitor `train_loss_epoch` so the scheduler does not require a missing validation
   metric.
