---
icon: lucide/sticky-notes
description: Advanced CARE configuration
---

# Advanced CARE

Before configuring CARE-specific parameters, check the [general configuration
page](../configuration.md) for shared options.

CARE and Noise2Noise use the same algorithm-specific configuration parameters. The
examples below use CARE; replace `create_care_config` with `create_n2n_config`, or use
the corresponding advanced factory, for Noise2Noise.

For shared configuration options—including augmentations, normalization, logging,
workers, seeds, and Lightning parameters—see the [main configuration
page](../configuration.md).

## CARE and Noise2Noise: differences

CARE and Noise2Noise use the same configuration structure and both require a target
image for supervised training. Their main difference is the training data: CARE uses
a clean target, whereas Noise2Noise uses a second noisy realization of the same
underlying signal as its target. The input and target must therefore be paired in both
cases, but the target does not need to be noise-free for Noise2Noise.

The simple and advanced factories expose the same data, model, and optimization
parameters, with two differences in their defaults and available training controls:

| Parameter | CARE | Noise2Noise |
| --- | --- | --- |
| `independent_channels` | `False` by default: channels can inform one another | `True` by default: channels are trained independently |
| `early_stopping_params` | Available in `create_advanced_care_config` | Not exposed by `create_advanced_n2n_config` |

Set `independent_channels` explicitly when reproducing a training setup across the two
algorithms. The remaining training parameters—such as `num_epochs`, `num_steps`,
`batch_size`, `optimizer`, `lr_scheduler`, and dataloader settings—have the same
meaning and defaults. See [Training parameters](../configuration.md#pytorch-and-lightning-parameters)
for examples.

For example, these configurations use the same training budget but retain each
algorithm's default channel behavior:

```python title="CARE and Noise2Noise training defaults"
from careamics.config import create_advanced_care_config, create_advanced_n2n_config

care_config = create_advanced_care_config(
    experiment_name="care_training",
    data_type="array",
    axes="CYX",
    patch_size=[64, 64],
    batch_size=8,
    n_channels_in=3,
    num_epochs=30,
)

n2n_config = create_advanced_n2n_config(
    experiment_name="n2n_training",
    data_type="array",
    axes="CYX",
    patch_size=[64, 64],
    batch_size=8,
    n_channels_in=3,
    num_epochs=30,
)
```

## Channels and targets

CARE and Noise2Noise can have different numbers of input and output channels. Set
`n_channels_in` for the input and use `n_channels_out` together with `target_axes` when
the target has a different channel layout.

```python title="Setting input channel parameters"
--8<-- "current/configuration/config_general/configuration_care.py:config_care_channels"
```

1. `C` indicates that channels are present in `axes`.
2. `n_channels_in` is the number of input channels.

```python title="Setting output channel parameters"
--8<-- "current/configuration/config_general/configuration_care.py:config_care_out_channels"
```

1. `target_axes` specifies the target axis order and can differ from `axes`.
2. `n_channels_out` can differ from the number of input channels.

If `target_axes` is not set, it defaults to `axes`. If the target axes contain `C` and
`n_channels_out` is omitted, it defaults to `n_channels_in`; otherwise it defaults to
one output channel.

### Different target axes

The target can have a different axis layout from the input, for example when the input
has channels but the target does not.

```python title="Setting target axes"
--8<-- "current/configuration/config_general/configuration_care.py:target_axes"
```

1. The target in this example does not have a channel dimension.
2. `n_channels_in` is still required because the input has three channels.
