---
icon: lucide/sliders-horizontal
description: Configuring CAREamics
---

# Configuring CAREamics

To start with CAREamics, create a configuration object containing the parameters for
your experiment. The configuration validates individual parameters and their
coherence, avoiding combinations that could trigger errors later in the workflow.

CAREamics provides algorithm-specific convenience functions. Each algorithm has a
simple factory for common experiments and an advanced factory for more control.

=== "Noise2Void"

    ```python title="Noise2Void configuration factories"
    --8<-- "current/config_general/configuration_n2v.py:all_configs"
    ```

=== "CARE/N2N"

    ```python title="CARE and Noise2Noise configuration factories"
    --8<-- "current/config_general/configuration_care.py:all_configs"
    ```

=== "HDN"

    ```python title="HDN configuration factories"
    --8<-- "current/config_general/configuration_hdn.py:all_configs"
    ```

=== "MicroSplit"

    ```python title="MicroSplit configuration factories"
    --8<-- "current/config_general/configuration_microsplit.py:all_configs"
    ```

=== "cryoCARE"

    ```python title="cryoCARE configuration factories"
    --8<-- "current/config_general/configuration_cryocare.py:all_configs"
    ```

=== "Segmentation"

    ```python title="Segmentation configuration factories"
    --8<-- "current/config_general/configuration_seg.py:all_configs"
    ```


## Simple configuration

The simple configuration functions expose the parameters most commonly used to start
an experiment. The starting parameters differ slightly between algorithms, so the
algorithm-specific descriptions are kept in tabs.

=== "Noise2Void"

    ```python title="Configure Noise2Void"
    --8<-- "current/config_general/configuration_n2v.py:config_n2v"
    ```

    1. The length of `patch_size` is conditioned on the presence of the `Z` axis.

    - `experiment_name`: The experiment name is used in logging and automatic model
      saving. It should only contain letters, numbers, underscores, dashes, and spaces.
    - `data_type`: The data type impacts which features are available. CAREamics
      supports `array` (when passing NumPy arrays directly), `tiff`, `zarr`, `czi`, and
      `custom`.
    - `axes`: The axes of the data, in the order they have on disk or in memory. This
      identifies the spatial and channel dimensions; see the [data
      section](./data/data.md) for guidance.
    - `patch_size`: The size of the patches extracted during training. It refers only
      to spatial axes (`X`, `Y`, and optionally `Z`), and should be 2D without `Z` and
      3D with `Z`.
    - `batch_size`: The number of patches used in each training step.
    - `num_epochs`: The number of epochs to train for. For large datasets, also
      consider setting [`num_steps`](#reducing-the-number-of-steps).
    - `num_steps`: Optional number of batches per epoch. If omitted, the full training
      dataset is used; see [Reducing the number of steps](#reducing-the-number-of-steps).
    - `augmentations`: Optional list of `x_flip`, `y_flip`, and `rotate_90`. If omitted,
      all three are applied; use an empty list to disable them.
    - `n_val_patches`: Number of training patches reserved for validation when separate
      validation data is not provided; see [Number of validation
      patches](#number-of-validation-patches).
    - `use_n2v2`: Whether to use the N2V2 variant instead of standard N2V.
    - `n_channels`: Number of input and output channels when `C` is present in `axes`.

=== "CARE/N2N"

    ```python title="Configure CARE"
    --8<-- "current/config_general/configuration_care.py:config_care"
    ```

    1. The length of `patch_size` is conditioned on the presence of the `Z` axis.

    - `experiment_name`: The experiment name is used in logging and automatic model
      saving. It should only contain letters, numbers, underscores, dashes, and spaces.
    - `data_type`: The data type impacts which features are available. CAREamics
      supports `array` (when passing NumPy arrays directly), `tiff`, `zarr`, `czi`, and
      `custom`.
    - `axes`: The axes of the input data, in the order they have on disk or in memory.
      This identifies the spatial and channel dimensions; see the [data
      section](./data/data.md) for guidance.
    - `patch_size`: The size of the patches extracted during training. It refers only
      to spatial axes (`X`, `Y`, and optionally `Z`), and should be 2D without `Z` and
      3D with `Z`.
    - `batch_size`: The number of patches used in each training step.
    - `num_epochs`: The number of epochs to train for. For large datasets, also
      consider setting [`num_steps`](#reducing-the-number-of-steps).
    - `num_steps`: Optional number of batches per epoch. If omitted, the full training
      dataset is used; see [Reducing the number of steps](#reducing-the-number-of-steps).
    - `augmentations`: Optional list of `x_flip`, `y_flip`, and `rotate_90`. If omitted,
      all three are applied; use an empty list to disable them.
    - `n_val_patches`: Number of training patches reserved for validation when separate
      validation data is not provided; see [Number of validation
      patches](#number-of-validation-patches).
    - `n_channels_in`: Number of input channels when `C` is present in `axes`.
    - `n_channels_out`: Number of output channels. It is inferred from `target_axes` or
      defaults to one when the target has no channel axis.
    - `target_axes`: Optional axes of the target data. If omitted, it defaults to
      `axes`. This can be used when input and target axes differ.

=== "HDN"

    ```python title="Configure HDN"
    --8<-- "current/config_general/configuration_hdn.py:config_hdn"
    ```

    - `experiment_name`: The experiment name is used in logging and automatic model
      saving. It should only contain letters, numbers, underscores, dashes, and spaces.
    - `data_type`: The data type impacts which features are available. HDN supports
      `array`, `tiff`, `zarr`, `czi`, and `custom`.
    - `axes`: The axes of the input data, in the order they have on disk or in memory;
      see the [data section](./data/data.md) for guidance.
    - `patch_size`: The spatial size of extracted patches. Use two values for 2D data
      and three values when `Z` is present.
    - `batch_size`: The number of patches used in each training step.
    - `num_epochs`: The number of epochs to train for. For large datasets, also
      consider setting [`num_steps`](#reducing-the-number-of-steps).
    - `num_steps`: Optional number of batches per epoch. If omitted, the full training
      dataset is used; see [Reducing the number of steps](#reducing-the-number-of-steps).
    - `augmentations`: Optional list of `x_flip`, `y_flip`, and `rotate_90`. If omitted,
      all three are applied; use an empty list to disable them.
    - `n_val_patches`: Number of training patches reserved for validation when separate
      validation data is not provided; see [Number of validation
      patches](#number-of-validation-patches).
    - `use_noise_model`: Selects the noise-model likelihood when enabled; otherwise
      HDN uses the Gaussian likelihood pathway.
    - `supervised`: Enables supervised training when target data is available.

=== "MicroSplit"

    ```python title="Configure MicroSplit"
    --8<-- "current/config_general/configuration_microsplit.py:config_microsplit"
    ```

    - `experiment_name`: The experiment name is used in logging and automatic model
      saving. It should only contain letters, numbers, underscores, dashes, and spaces.
    - `data_type`: The data type impacts which features are available. MicroSplit
      supports `array`, `tiff`, `zarr`, `czi`, and `custom`.
    - `axes`: The axes of the input data, in the order they have on disk or in memory;
      see the [data section](./data/data.md) for guidance.
    - `patch_size`: The spatial size of extracted patches. Use two values for 2D data
      and three values when `Z` is present.
    - `batch_size`: The number of patches used in each training step.
    - `num_epochs`: The number of epochs to train for. For large datasets, also
      consider setting [`num_steps`](#reducing-the-number-of-steps).
    - `num_steps`: Optional number of batches per epoch. If omitted, the full training
      dataset is used; see [Reducing the number of steps](#reducing-the-number-of-steps).
    - `augmentations`: Optional list of `x_flip`, `y_flip`, and `rotate_90`. If omitted,
      all three are applied; use an empty list to disable them.
    - `n_val_patches`: Number of training patches reserved for validation when separate
      validation data is not provided; see [Number of validation
      patches](#number-of-validation-patches).
    - `output_channels`: Number of target channels to split into.
    - `multiscale_count`: Number of lateral-context scales.

=== "cryoCARE"

    ```python title="Configure cryoCARE"
    --8<-- "current/config_general/configuration_cryocare.py:config_cryocare"
    ```

    - `experiment_name`: The experiment name is used in logging and automatic model
      saving. It should only contain letters, numbers, underscores, dashes, and spaces.
    - `data_type`: The input format. cryoCARE supports `array`, `tiff`, `zarr`, `czi`,
      `mrc`, and `custom` on the `dev/cryocare` branch.
    - `axes`: The input axes. cryoCARE requires a three-dimensional permutation of
      `ZYX`, including `Z` and without `C`.
    - `patch_size`: A three-dimensional patch size in `ZYX` order.
    - `batch_size`: The number of patches used in each training step.
    - `num_epochs`: The number of epochs to train for. For large datasets, also
      consider setting [`num_steps`](#reducing-the-number-of-steps).
    - `num_steps`: Optional number of batches per epoch. If omitted, the full training
      dataset is used; see [Reducing the number of steps](#reducing-the-number-of-steps).
    - `tilt_axis`: The physical tilt axis used for the train/validation split and the
      automatically generated compatible augmentation.
    - `validation_fraction`: The fraction of the reconstructed stack reserved for
      validation along `tilt_axis`.
    - `n_val_patches`: The number of validation patches drawn from the validation
      region. It must be positive.
    - `augmentations`: CryoCARE generates input-target swapping and compatible
      tilt-axis rotations automatically; this parameter is not user-configurable.

=== "Segmentation"

    ```python title="Configure segmentation"
    --8<-- "current/config_general/configuration_seg.py:config_seg"
    ```

    1. The length of `patch_size` must match the spatial dimensions.
    2. `n_classes` counts foreground classes; the background class is added
       automatically.

    - `experiment_name`: The experiment name is used in logging and automatic model
      saving. It should only contain letters, numbers, underscores, dashes, and spaces.
    - `data_type`: The data type impacts which features are available. CAREamics
      supports `array` (when passing NumPy arrays directly), `tiff`, `zarr`, `czi`, and
      `custom`.
    - `axes`: The axes of the input data, in the order they have on disk or in memory.
      This identifies the spatial and channel dimensions; see the [data
      section](./data/data.md) for guidance.
    - `patch_size`: The size of the patches extracted during training. It refers only
      to spatial axes (`X`, `Y`, and optionally `Z`), and should be 2D without `Z` and
      3D with `Z`.
    - `batch_size`: The number of patches used in each training step.
    - `n_classes`: The number of foreground classes in the label mask. The background
      class is added automatically.
    - `num_epochs`: The number of epochs to train for. For large datasets, also
      consider setting [`num_steps`](#reducing-the-number-of-steps).
    - `num_steps`: Optional number of batches per epoch. If omitted, the full training
      dataset is used; see [Reducing the number of steps](#reducing-the-number-of-steps).
    - `augmentations`: Optional list of `x_flip`, `y_flip`, and `rotate_90`. All three
      are applied by default; use an empty list to disable augmentations.
    - `n_val_patches`: Number of training patches reserved for validation when separate
      validation data is not provided; see [Number of validation
      patches](#number-of-validation-patches).
    - `n_channels_in`: Number of input channels when `C` is present in `axes`.
    - `prediction_output`: Whether prediction returns a `mask` or per-class
      `probabilities`.

For more algorithm-specific parameters, see the [Advanced N2V](./config_algorithm_specific/configuration_specific_n2v.md),
[Advanced CARE](./config_algorithm_specific/configuration_specific_care.md), [HDN](./config_algorithm_specific/configuration_specific_hdn.md),
[MicroSplit](./config_algorithm_specific/configuration_specific_microsplit.md), [cryoCARE](./config_algorithm_specific/configuration_specific_cryocare.md),
or [Segmentation](./config_algorithm_specific/configuration_seg.md) pages.


!!! note "Training with `T` as depth axis"

    To use `T` as the depth axis, relabel it as `Z`. For `data_type="czi"`, `T` can
    also be used directly as a depth axis, for example `axes="SCTYX"`; see the
    [data section](./data/data.md) for details.


### Reducing the number of steps

Each training epoch cycles through all patches. For large datasets, set `num_steps` to
limit the number of batches processed in an epoch.

=== "Noise2Void"

    ```python title="Setting the number of steps"
    --8<-- "current/config_general/configuration_n2v.py:config_n2v_steps"
    ```

    1. Use a number smaller than the total number of steps, based on the batch size.

=== "CARE/N2N"

    ```python title="Setting the number of steps"
    --8<-- "current/config_general/configuration_care.py:config_care_steps"
    ```

    1. Use a number smaller than the total number of steps, based on the batch size.

=== "HDN"

    `num_steps` limits the number of batches in each HDN epoch. The parameter has the
    same meaning as for N2V and CARE/N2N.

=== "MicroSplit"

    `num_steps` limits the number of batches in each MicroSplit epoch. The parameter
    has the same meaning as for N2V and CARE/N2N.

=== "cryoCARE"

    `num_steps` limits the number of batches in each cryoCARE epoch. The parameter has
    the same meaning as for the other algorithms.

=== "Segmentation"

    ```python title="Setting the number of steps"
    --8<-- "current/config_general/configuration_seg.py:config_seg_steps"
    ```

    1. Choose a value smaller than the total number of available batches when shorter
       epochs are desired.

The total number of steps is approximately `n_patches / batch_size`. The
`num_epochs` and `num_steps` parameters correspond to `max_epochs` and
`limit_train_batches` in the PyTorch Lightning `Trainer`.

### Augmentations

CAREamics applies random X/Y flips and 90-degree rotations by default. The same
spatial transformation is applied consistently to the training data and targets.

=== "Noise2Void"

    ```python title="Setting augmentations"
    --8<-- "current/config_general/configuration_n2v.py:config_n2v_augs"
    ```

    1. These are all the possible augmentation choices.

=== "CARE/N2N"

    ```python title="Setting augmentations"
    --8<-- "current/config_general/configuration_care.py:config_care_augs"
    ```

    1. These are all the possible augmentation choices.

=== "HDN"

    HDN accepts the shared `x_flip`, `y_flip`, and `rotate_90` augmentations. Pass an
    empty list to disable them.

=== "MicroSplit"

    MicroSplit accepts the shared `x_flip`, `y_flip`, and `rotate_90` augmentations.
    Pass an empty list to disable them.

=== "cryoCARE"

    cryoCARE generates input-target swapping and compatible tilt-axis rotations
    automatically; its augmentations are not user-configurable.

=== "Segmentation"

    ```python title="Setting augmentations"
    --8<-- "current/config_general/configuration_seg.py:config_seg_augs"
    ```

    1. The available choices are `x_flip`, `y_flip`, and `rotate_90`.

Use `augmentations=[]` to disable augmentations.

### Number of validation patches

When no validation data is provided, CAREamics can reserve `n_val_patches` patches
from the training data for validation.

=== "Noise2Void"

    ```python title="Setting the number of validation patches"
    --8<-- "current/config_general/configuration_n2v.py:config_n2v_val"
    ```

=== "CARE/N2N"

    ```python title="Setting the number of validation patches"
    --8<-- "current/config_general/configuration_care.py:config_care_val"
    ```

=== "HDN"

    `n_val_patches` reserves training patches for validation when separate validation
    data is not supplied. Separate validation data takes precedence.

=== "MicroSplit"

    `n_val_patches` reserves training patches for validation when separate validation
    data is not supplied. Separate validation data takes precedence.

=== "cryoCARE"

    `n_val_patches` is the number of validation patches drawn from the tilt-axis
    validation region. It must be positive.

=== "Segmentation"

    ```python title="Setting the number of validation patches"
    --8<-- "current/config_general/configuration_seg.py:config_seg_val"
    ```

    1. Choose a number appropriate for the size and diversity of the training data.

If separate validation data is supplied, `n_val_patches` is ignored and the validation
data determines the number of validation patches.

### Input channels

When `C` is present in `axes`, segmentation requires `n_channels_in` to describe the
image channels. This does not change `n_classes`, which describes the label mask.

=== "Segmentation"

    ```python title="Setting segmentation input channels"
    --8<-- "current/config_general/configuration_seg.py:config_seg_channels"
    ```

    1. `C` indicates that the input contains channels.
    2. `n_channels_in` is the number of input channels.

## Algorithm-specific parameters

Additional parameters may be algorithm-specific. Click on the following pages for more
details on them.

<div class="grid cards" markdown>

-   :lucide-eye-off:{ .lg .middle } __Advanced N2V__

    ---

    Configure N2V, including N2V2, structN2V, and N2V-specific training options.

    [:octicons-arrow-right-24: Advanced N2V parameters](./config_algorithm_specific/configuration_specific_n2v.md)

-   :lucide-sticky-notes:{ .lg .middle } __Advanced CARE__

    ---

    Configure CARE and Noise2Noise experiments, including target axes and output
    channels.

    [:octicons-arrow-right-24: Advanced CARE parameters](./config_algorithm_specific/configuration_specific_care.md)

-   :lucide-layers:{ .lg .middle } __HDN__

    ---

    Configure HDN denoising with Gaussian or noise-model likelihoods.

    [:octicons-arrow-right-24: HDN parameters](./config_algorithm_specific/configuration_specific_hdn.md)

-   :lucide-grid-2x2-plus:{ .lg .middle } __MicroSplit__

    ---

    Configure MicroSplit multiscale denoising and likelihood terms.

    [:octicons-arrow-right-24: MicroSplit parameters](./config_algorithm_specific/configuration_specific_microsplit.md)

-   :lucide-layers-arrow-up:{ .lg .middle } __cryoCARE__

    ---

    Configure cryoCARE experiments, including tilt-axis validation and MRC inputs.

    [:octicons-arrow-right-24: cryoCARE parameters](./config_algorithm_specific/configuration_specific_cryocare.md)

-   :lucide-cookie:{ .lg .middle } __Segmentation__

    ---

    Configure semantic segmentation experiments.

    [:octicons-arrow-right-24: Segmentation parameters](./config_algorithm_specific/configuration_seg.md)

</div>

## Advanced configuration

### Training in memory

Loading data into memory can speed up patch extraction when the dataset fits in RAM.

=== "Noise2Void"

    ```python title="Setting in-memory training"
    --8<-- "current/config_advanced/config_advanced_n2v.py:adv_config_n2v_in_memory"
    ```

    1. Only `array`, `tiff`, and `custom` are compatible with in-memory training.

=== "CARE/N2N"

    ```python title="Setting in-memory training"
    --8<-- "current/config_advanced/config_advanced_care.py:adv_config_care_in_memory"
    ```

    1. Only `array`, `tiff`, and `custom` are compatible with in-memory training.

=== "HDN"

    HDN supports in-memory training for `array`, `tiff`, and `custom` data. The
    `in_memory` option is available through its advanced factory.

=== "MicroSplit"

    MicroSplit supports in-memory training for `array`, `tiff`, and `custom` data. The
    `in_memory` option is available through its advanced factory.

=== "cryoCARE"

    cryoCARE does not expose an `in_memory` parameter; its data configuration controls
    loading for the supported cryoCARE formats.

=== "Segmentation"

    ```python title="Setting in-memory training"
    --8<-- "current/config_general/configuration_seg.py:adv_config_seg_in_memory"
    ```

    1. In-memory training is supported for `array`, `tiff`, and `custom` data.

Only `array`, `tiff`, and `custom` data types support in-memory training. Array data is
kept in memory automatically; `zarr` and `czi` use random access from disk.

### Subsetting channels

To train on only some input channels, pass their zero-based indices through `channels`.
When `channels` is set, the number of input channels is inferred automatically. The
target is not subsetted.

=== "Noise2Void"

    ```python title="Selecting a subset of channels"
    --8<-- "current/config_advanced/config_advanced_n2v.py:adv_config_n2v_subchannels"
    ```

    1. `C` must be present in `axes` for channels to be considered.
    2. This trains using the first and third channels, because channel indices start at
       zero.

=== "CARE/N2N"

    ```python title="Selecting a subset of channels"
    --8<-- "current/config_advanced/config_advanced_care.py:adv_config_care_subchannels"
    ```

    1. `C` must be present in `axes` for channels to be considered.
    2. This trains using the first and third channels, because channel indices start at
       zero.

=== "HDN"

    HDN accepts zero-based input channel indices through `channels`; the number of
    selected channels is inferred automatically.

=== "MicroSplit"

    MicroSplit accepts zero-based input channel indices through `channels`; the number
    of selected channels is inferred automatically.

=== "cryoCARE"

    cryoCARE does not support channel subsetting because its input axes cannot contain
    `C`.

=== "Segmentation"

    ```python title="Selecting a subset of channels"
    --8<-- "current/config_general/configuration_seg.py:adv_config_seg_subchannels"
    ```

    1. This example uses the first and third input channels. The label mask is not
       channel-subsetted.

`C` must be present in `axes`.

### Channel independence

By default, channels are trained independently. Set `independent_channels=False` to
allow information to flow between input channels.

=== "Noise2Void"

    ```python title="Training non-independent channels"
    --8<-- "current/config_advanced/config_advanced_n2v.py:adv_config_n2v_ind_channels"
    ```

    1. `C` must be present in `axes`.
    2. The number of channels must be specified.
    3. Setting `independent_channels=False` allows channels to inform each other.

=== "CARE/N2N"

    ```python title="Training non-independent channels"
    --8<-- "current/config_advanced/config_advanced_care.py:adv_config_care_ind_channels"
    ```

    1. `C` must be present in `axes`.
    2. The number of input channels must be specified.
    3. Setting `independent_channels=False` allows channels to inform each other.

=== "HDN"

    Channel independence is not exposed as an HDN parameter; HDN uses its LVAE channel
    handling.

=== "MicroSplit"

    Channel independence is not exposed as a MicroSplit parameter; channel mixing is
    controlled by its synthetic-input parameters.

=== "cryoCARE"

    Channel independence is not applicable because cryoCARE does not support `C` in
    its input axes.

=== "Segmentation"

    Segmentation fixes channel independence to `False`; it does not expose this
    parameter.

`C` must be present in `axes`. Segmentation fixes channel independence to `False` and
therefore does not expose this parameter.

### Normalization

Normalization is applied during training and prediction. The available strategies are
`mean_std`, `quantile`, `min_max`, and `none`.

=== "Noise2Void"

    ```python title="Specifying the normalization method"
    --8<-- "current/config_advanced/config_advanced_n2v.py:adv_config_n2v_norm"
    ```

    1. Select the normalization method with `normalization`.
    2. `normalization_params` changes the method's default parameters.

=== "CARE/N2N"

    ```python title="Specifying the normalization method"
    --8<-- "current/config_advanced/config_advanced_care.py:adv_config_care_norm"
    ```

    1. Select the normalization method with `normalization`.
    2. `normalization_params` changes the method's default parameters.

=== "HDN"

    HDN supports `mean_std`, `min_max`, `quantile`, and `none` normalization through
    `normalization` and `normalization_params`.

=== "MicroSplit"

    MicroSplit supports `mean_std`, `min_max`, `quantile`, and `none` normalization
    through `normalization` and `normalization_params`.

=== "cryoCARE"

    cryoCARE supports the same normalization strategies. Its normalization is applied
    to the reconstructed volumes used for training and validation.

=== "Segmentation"

    ```python title="Specifying the normalization method"
    --8<-- "current/config_general/configuration_seg.py:adv_config_seg_norm"
    ```

    1. Select the normalization method with `normalization`.
    2. `normalization_params` customizes the quantile bounds; target normalization is
       skipped automatically.

If statistics are not provided, CAREamics computes them from the training data. They
can also be passed explicitly:

=== "Noise2Void"

    ```python title="Passing pre-computed normalization parameters"
    --8<-- "current/config_advanced/config_advanced_n2v.py:adv_config_n2v_norm_params"
    ```

    1. This selects zero-mean, unit-variance normalization, which is the default.
    2. The normalization parameters are pre-computed and passed explicitly.

=== "CARE/N2N"

    ```python title="Passing pre-computed normalization parameters"
    --8<-- "current/config_advanced/config_advanced_care.py:adv_config_care_norm_params"
    ```

    1. This selects zero-mean, unit-variance normalization, which is the default.
    2. The input normalization parameters are pre-computed and passed explicitly.
    3. CARE and N2N can also receive pre-computed target statistics.

=== "HDN"

    `normalization_params` can contain pre-computed input statistics for HDN, avoiding
    recomputation over the training data.

=== "MicroSplit"

    `normalization_params` can contain pre-computed input statistics for MicroSplit,
    avoiding recomputation over the training data.

=== "cryoCARE"

    `normalization_params` can contain pre-computed statistics for cryoCARE volumes.

When channels are present, normalization parameters normally contain one value per
channel. Set `per_channel=False` when a shared value is intended.

=== "Noise2Void"

    ```python title="Normalizing multi-channel data"
    --8<-- "current/config_advanced/config_advanced_n2v.py:adv_config_n2v_norm_ch"
    ```

    1. The example uses multiple channels.
    2. The number of channels must be specified.
    3. Each normalization parameter has one value per channel.
    4. With `per_channel=False`, each parameter instead expects one shared value.

=== "CARE/N2N"

    ```python title="Normalizing multi-channel data"
    --8<-- "current/config_advanced/config_advanced_care.py:adv_config_care_norm_ch"
    ```

    1. The example uses multiple channels.
    2. The number of input channels must be specified.
    3. Each normalization parameter has one value per channel.
    4. With `per_channel=False`, each parameter instead expects one shared value.

=== "HDN"

    HDN accepts one normalization value per selected channel. Use `per_channel=False`
    when a shared value is intended.

=== "MicroSplit"

    MicroSplit accepts one normalization value per selected channel. Use
    `per_channel=False` when a shared value is intended.

=== "cryoCARE"

    cryoCARE does not support channel axes, so its normalization parameters are not
    channel-indexed.

### Choosing a logger

CSV logging is always enabled. An additional Weights & Biases or TensorBoard logger can
be selected with `logger`.

=== "Noise2Void"

    ```python title="Choosing a logger"
    --8<-- "current/config_advanced/config_advanced_n2v.py:adv_config_n2v_logger"
    ```

    1. This selects Weights & Biases in addition to the CSV logger.

=== "CARE/N2N"

    ```python title="Choosing a logger"
    --8<-- "current/config_advanced/config_advanced_care.py:adv_config_care_logger"
    ```

    1. This selects Weights & Biases in addition to the CSV logger.

=== "HDN"

    Set `logger` to `wandb` or `tensorboard` to add the selected logger to CSV logging.

=== "MicroSplit"

    Set `logger` to `wandb` or `tensorboard` to add the selected logger to CSV logging.

=== "cryoCARE"

    Set `logger` to `wandb` or `tensorboard` to add the selected logger to CSV logging.

### Number of workers

`num_workers` controls data-loading workers. The default `-1` selects a value based on
the available CPUs.

=== "Noise2Void"

    ```python title="Setting the number of workers"
    --8<-- "current/config_advanced/config_advanced_n2v.py:adv_config_n2v_num_workers"
    ```

    1. This sets the number of workers used for data loading.

=== "CARE/N2N"

    ```python title="Setting the number of workers"
    --8<-- "current/config_advanced/config_advanced_care.py:adv_config_care_num_workers"
    ```

    1. This sets the number of workers used for data loading.

=== "HDN"

    Set `num_workers` to control HDN data loading; `-1` selects a value automatically.

=== "MicroSplit"

    Set `num_workers` to control MicroSplit data loading; `-1` selects a value
    automatically.

=== "cryoCARE"

    Set `num_workers` to control cryoCARE data loading; `-1` uses the platform default.

=== "Segmentation"

    ```python title="Setting the number of workers"
    --8<-- "current/config_general/configuration_seg.py:adv_config_seg_num_workers"
    ```

    1. Set this according to the available CPU resources and storage performance.

### Setting a seed

Set `seed` to make data ordering, augmentations, and patch sampling reproducible.

=== "Noise2Void"

    ```python title="Setting a seed"
    --8<-- "current/config_advanced/config_advanced_n2v.py:adv_config_n2v_seed"
    ```

=== "CARE/N2N"

    ```python title="Setting a seed"
    --8<-- "current/config_advanced/config_advanced_care.py:adv_config_care_seed"
    ```

=== "HDN"

    Pass an integer through `seed` to make HDN sampling and augmentations reproducible.

=== "MicroSplit"

    Pass an integer through `seed` to make MicroSplit sampling and augmentations
    reproducible.

=== "cryoCARE"

    Pass an integer through `seed` to make cryoCARE sampling reproducible.

=== "Segmentation"

    ```python title="Setting a seed"
    --8<-- "current/config_general/configuration_seg.py:adv_config_seg_seed"
    ```


## PyTorch and Lightning parameters

The following PyTorch Lightning and model parameters are shared by all algorithms.

### Model parameters

The U-Net model parameters can be passed through `model_params`.

=== "Noise2Void"

    ```python title="Passing model parameters"
    --8<-- "current/config_advanced/configuration_lightning.py:adv_config_n2v_model"
    ```

=== "CARE/N2N"

    ```python title="Passing model parameters"
    --8<-- "current/config_advanced/configuration_lightning.py:adv_config_care_model"
    ```

=== "HDN"

    HDN uses LVAE model parameters rather than the U-Net parameters shown above. Use
    `model_params` for non-structural LVAE settings; structural settings are fixed by
    the factory.

=== "MicroSplit"

    MicroSplit uses LVAE model parameters. Use `model_params` for non-structural LVAE
    settings; structural settings are fixed by the factory.

=== "cryoCARE"

    cryoCARE does not expose `model_params`; its model architecture is fixed by the
    algorithm.

=== "Segmentation"

    ```python title="Setting U-Net parameters"
    --8<-- "current/config_general/configuration_seg.py:adv_config_seg_model"
    ```

    1. Increasing `depth` adds U-Net levels and generally increases model capacity and
       memory use.

Common model parameters include `depth`, `num_channels_init`, `residual`, and
`use_batch_norm`.

### Trainer parameters

Pass additional PyTorch Lightning `Trainer` options through `trainer_params`.

=== "Noise2Void"

    ```python title="Passing Trainer parameters"
    --8<-- "current/config_advanced/configuration_lightning.py:adv_config_n2v_trainer"
    ```

    1. Pass additional Trainer parameters as a dictionary through `trainer_params`.

=== "CARE/N2N"

    ```python title="Passing Trainer parameters"
    --8<-- "current/config_advanced/configuration_lightning.py:adv_config_care_trainer"
    ```

    1. Pass additional Trainer parameters as a dictionary through `trainer_params`.

=== "HDN"

    Pass additional PyTorch Lightning `Trainer` options through `trainer_params`.

=== "MicroSplit"

    Pass additional PyTorch Lightning `Trainer` options through `trainer_params`.

=== "cryoCARE"

    Pass additional PyTorch Lightning `Trainer` options through `trainer_params`.

`num_epochs` and `num_steps` take precedence over corresponding
`max_epochs` and `limit_train_batches` values in `trainer_params`.

### Optimizer parameters

CAREamics uses `Adam` by default. Select another supported optimizer and pass its
parameters through `optimizer_params`.

=== "Noise2Void"

    ```python title="Passing optimizer parameters"
    --8<-- "current/config_advanced/configuration_lightning.py:adv_config_n2v_opt"
    ```

    1. Select the optimizer with `optimizer`.
    2. Pass optimizer parameters as a dictionary through `optimizer_params`.

=== "CARE/N2N"

    ```python title="Passing optimizer parameters"
    --8<-- "current/config_advanced/configuration_lightning.py:adv_config_care_opt"
    ```

    1. Select the optimizer with `optimizer`.
    2. Pass optimizer parameters as a dictionary through `optimizer_params`.

=== "HDN"

    Select `Adam`, `Adamax`, or `SGD` with `optimizer` and customize it through
    `optimizer_params`. HDN defaults to `Adamax`.

=== "MicroSplit"

    Select `Adam`, `Adamax`, or `SGD` with `optimizer` and customize it through
    `optimizer_params`. MicroSplit defaults to `Adamax`.

=== "cryoCARE"

    Select `Adam`, `Adamax`, or `SGD` with `optimizer` and customize it through
    `optimizer_params`. cryoCARE defaults to `Adam`.

Supported optimizers are `Adam`, `SGD`, and `Adamax`.

### Learning-rate schedulers

Configure the learning-rate scheduler with `lr_scheduler` and
`lr_scheduler_params`.

=== "Noise2Void"

    ```python title="Passing learning-rate scheduler parameters"
    --8<-- "current/config_advanced/configuration_lightning.py:adv_config_n2v_lr"
    ```

    1. Select the scheduler with `lr_scheduler`.
    2. Pass scheduler parameters as a dictionary through `lr_scheduler_params`.

=== "CARE/N2N"

    ```python title="Passing learning-rate scheduler parameters"
    --8<-- "current/config_advanced/configuration_lightning.py:adv_config_care_lr"
    ```

    1. Select the scheduler with `lr_scheduler`.
    2. Pass scheduler parameters as a dictionary through `lr_scheduler_params`.

=== "HDN"

    Select `ReduceLROnPlateau` or `StepLR` with `lr_scheduler` and pass options through
    `lr_scheduler_params`.

=== "MicroSplit"

    Select `ReduceLROnPlateau` or `StepLR` with `lr_scheduler` and pass options through
    `lr_scheduler_params`.

=== "cryoCARE"

    Select `ReduceLROnPlateau` or `StepLR` with `lr_scheduler` and pass options through
    `lr_scheduler_params`.

Supported schedulers are `ReduceLROnPlateau` and `StepLR`.

### Dataloader parameters

Use `train_dataloader_params` and `val_dataloader_params` to customize the PyTorch
dataloaders.

=== "Noise2Void"

    ```python title="Passing dataloader parameters"
    --8<-- "current/config_advanced/configuration_lightning.py:adv_config_n2v_train_dm"
    ```

=== "CARE/N2N"

    ```python title="Passing dataloader parameters"
    --8<-- "current/config_advanced/configuration_lightning.py:adv_config_care_train_dm"
    ```

=== "HDN"

    Use `train_dataloader_params` and `val_dataloader_params` to customize HDN
    dataloaders.

=== "MicroSplit"

    Use `train_dataloader_params` and `val_dataloader_params` to customize MicroSplit
    dataloaders.

=== "cryoCARE"

    Use `train_dataloader_params` and `val_dataloader_params` to customize cryoCARE
    dataloaders.

When overriding `train_dataloader_params`, specify the desired `shuffle` value.

### Checkpoint callback

Override the default checkpoint behavior with `checkpoint_params`.

=== "Noise2Void"

    ```python title="Passing checkpoint parameters"
    --8<-- "current/config_advanced/configuration_lightning.py:adv_config_n2v_checkpoint"
    ```

=== "CARE/N2N"

    ```python title="Passing checkpoint parameters"
    --8<-- "current/config_advanced/configuration_lightning.py:adv_config_care_checkpoint"
    ```

=== "HDN"

    Pass callback options through `checkpoint_params` to configure HDN checkpoints.

=== "MicroSplit"

    Pass callback options through `checkpoint_params` to configure MicroSplit
    checkpoints.

=== "cryoCARE"

    Pass callback options through `checkpoint_params` to configure cryoCARE
    checkpoints.

CARE saves the best validation checkpoints and the last checkpoint by default. N2V
uses a different default because validation loss is not necessarily correlated with
denoising quality.

### Early stopping callback

Configure early stopping with `early_stopping_params`.

=== "Noise2Void"

    Early stopping is disabled by default for Noise2Void and is not exposed by its
    configuration factory.

=== "CARE/N2N"

    ```python title="Passing early-stopping parameters"
    --8<-- "current/config_advanced/configuration_lightning.py:adv_config_care_early_stop"
    ```

=== "HDN"

    Configure early stopping through `early_stopping_params`.

=== "MicroSplit"

    Configure early stopping through `early_stopping_params`.

=== "cryoCARE"

    Early stopping is not exposed by the cryoCARE configuration factory.

=== "Segmentation"

    Configure early stopping through `early_stopping_params`.

Early stopping is generally useful for supervised algorithms. It is disabled by
default for Noise2Void.

### Algorithm-specific configuration

For algorithm-specific options, see:

- [Advanced N2V](./config_algorithm_specific/configuration_specific_n2v.md)
- [Advanced CARE](./config_algorithm_specific/configuration_specific_care.md)
- [HDN](./config_algorithm_specific/configuration_specific_hdn.md)
- [MicroSplit](./config_algorithm_specific/configuration_specific_microsplit.md)
- [cryoCARE](./config_algorithm_specific/configuration_specific_cryocare.md)
- [Segmentation](./config_algorithm_specific/configuration_seg.md)

## Saving and loading

Configurations are automatically saved with checkpoints, but they can also be saved
and loaded manually.

```python title="Save and load configurations"
--8<-- "current/config_general/configuration_io.py:save_load"
```
