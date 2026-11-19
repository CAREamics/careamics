---
icon: lucide/cookie
description: Configuring semantic segmentation in CAREamics
---

# Segmentation

Before configuring segmentation-specific parameters, check the [general configuration
page](../configuration.md) for shared options.

CAREamics supports supervised semantic segmentation with a U-Net. A segmentation
configuration describes the input images and the corresponding label masks. The labels
must contain integer class indices; `n_classes` is the number of foreground classes,
with the background class added automatically by CAREamics.

## Factory parameters

The simple factory accepts `experiment_name`, `data_type`, `axes`, `patch_size`,
`batch_size`, `n_classes`, `num_epochs`, `num_steps`, `augmentations`,
`n_val_patches`, `n_channels_in`, and `prediction_output`.

The advanced factory adds `channels`, `normalization`, `normalization_params`,
`in_memory`, `loss`, `loss_parameters`, `model_params`, and the shared worker,
trainer, optimizer, scheduler, dataloader, checkpoint, early-stopping, logger, and
seed parameters.

Segmentation uses the same data formats, axes conventions, patch extraction, and
augmentation choices as the other CAREamics algorithms. See the [data
section](../../data/data.md) for details on axes and data formats.

## Shared configuration

Segmentation uses the shared data, patching, augmentation, normalization, and training
parameters described in the [main configuration guide](../configuration.md). The
segmentation tab in each shared section includes the relevant examples.

## Advanced configuration

Use the advanced factory when selecting input channels, normalization, loss, or model
parameters explicitly.

```python title="Advanced segmentation configuration"
--8<-- "current/configuration/config_advanced/configuration_advanced_seg.py:adv_config_seg"
```

### Prediction output

Segmentation prediction can return a class mask or per-class probabilities. The
default is `mask`.

```python title="Selecting the prediction output"
--8<-- "current/configuration/config_algorithm_specific/configuration_specific_seg.py:seg_prediction"
```

1. Set `prediction_output` to `probabilities` when per-class probabilities are needed.

### Loss function and class weights

Segmentation supports Dice, cross-entropy, and a weighted combination of both. Class
weights must include one value for the background and one value for every foreground
class.

```python title="Configuring the segmentation loss"
--8<-- "current/configuration/config_algorithm_specific/configuration_specific_seg.py:seg_loss"
```

1. Supported losses are `dice`, `ce`, and `dice_ce`.
2. For `n_classes=2`, `class_weights` must contain three values: background and two
   foreground classes.

For `dice_ce`, `dice_weight` and `ce_weight` control the contribution of each loss.

### Filtering background patches

Patch filtering can reduce the probability of sampling patches that contain little
useful signal. This can be helpful when label masks contain a large background region.

```python title="Filtering patches"
--8<-- "current/configuration/config_algorithm_specific/configuration_specific_seg.py:seg_filter"
```

1. `threshold` is the minimum value used by the max filter. Tune it to the intensity
   scale of the input data.

## Differences from CARE and Noise2Void

Segmentation shares many data and training parameters with CARE and Noise2Void, but it
should not be treated as a drop-in variant of either configuration:

- `n_classes` describes foreground label classes, while `n_channels_in` describes
  image channels. Background is implicit and adds one model output class.
- Segmentation targets are integer masks. They are not intensity targets and must not
  be normalized.
- Segmentation derives target axes by removing `C` from the input axes. Unlike CARE,
  users do not provide an arbitrary `target_axes` value.
- Segmentation uses `dice`, `ce`, and `dice_ce` losses, with optional class weights;
  CARE/N2N and N2V expose different loss parameters.
- Segmentation prediction output is controlled by `prediction_output` and can be a
  mask or class probabilities.

These differences are documented here while the shared configuration options are
covered by the main configuration guide.
