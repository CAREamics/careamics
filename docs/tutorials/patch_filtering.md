# Patch Filtering

Patch filtering is useful if your data contains large areas with no signal. These areas can be filtered from the training process which can speed up the convergence of the model.

### How does it work?

CAREamics will perform a first pass through all the data before training starts to determine regions of background and regions of signal. Background regions will not be completely be excluded from training, instead their probability of being selected during an epoch will be reduced.

There are two options for the filtering function, either:

- [pre-computed masks](#pre-computed-masks) can be provided, or
- one of the [built-in filtering functions](#filtering-functions) can be selected and parametrized.

## Pre-computed masks

Using precomputed masks is relatively simple, the masks in the same format as the data — either as an array or saved in files — can be provided during training.

=== "Noise2Void"
    
    ```python title="Specifying a mask for Noise2Void training"
    --8<-- "current/careamist_training.py:train_n2v_mask"
    ```
 
    1. The mask is passed alongside the data.

=== "CARE/N2N"

    ```python title="Specifying a mask for CARE training"
    --8<-- "current/careamist_training.py:train_care_mask"
    ```
    
    1. The mask is passed alongside the data.

!!! note "What is masked?"

    The mask is a binary set of images with the same size as the training data and
    should have value `1` for pixels that should be included in the training and `0`
    for pixels that should be excluded.

## Filtering functions

CAREamics has 3 built-in filtering functions, which work by filtering out patches using thresholds on different metrics:

- [`MaxPatchFilter`][careamics.dataset.patch_filter.MaxPatchFilter]: that filters the data based on the max value of each region.
- [`MeanStdPatchFilter`][careamics.dataset.patch_filter.MeanStdPatchFilter]: that filters the data based on the mean and optionally the standard deviation of regions of the data.
- [`ShannonPatchFilter`][careamics.dataset.patch_filter.ShannonPatchFilter]: that filters the data based on the shannon entropy of regions of the data.

!!! note "Multi-channel data"

    For multi-channel data the filtering function is only applied to a single channel of your choosing.

### Finding appropriate thresholds

Finding appropriate thresholds requires manually inspecting some examples. The patch filter classes provide `filter_map` and `plot_filter_map` which can be used to visualize at what threshold a region will be considered background.

---

For demonstration purposes we will use the Hagen dataset which is used in other CAREamics examples; however, it doesn't have enough background area to typically require patch filtering.

```python title="Download the data"
--8<-- "tutorials/patch_filtering.py:download-data"
```

---

Now we inspect the filter maps to decide on a patch filtering function and threshold. For data with multiple samples it is generally a good idea to inspect the filter maps of a few different samples; and for 3D data one should look at multiple z-slices. 

```python title="Plot Filter Maps"
--8<-- "tutorials/patch_filtering.py:filter-maps"
```

!!! info "3D data"

    For 3D data `plot_filter_map` has the `z_idx` argument to control which z-slice is displayed.

![Max filter map](../../../images/tutorials/patch_filtering/max_filter_map.png)

![Shannon filter map](../../../images/tutorials/patch_filtering/shannon_filter_map.png)

![Mean-Std filter map](../../../images/tutorials/patch_filtering/mean_std_filter_map.png)

---

We will choose the shannon patch filter, with a threshold of 7.5, and to confirm that this is a good choice we will look at the resulting mask, by using the `ShannonPatchFilter.apply_filter` method.

```python title="Plot Filter Maps"
--8<-- "tutorials/patch_filtering.py:mask"
```

![Filter mask](../../../images/tutorials/patch_filtering/filter_mask.png)

### Training

Next, we have to build the configuration.

Each of the patch filter classes has a corresponding configuration class, where the threshold parameters can be set:

- [`MaxPatchFilterConfig`][careamics.config.MaxPatchFilterConfig]
- [`MeanStdPatchFilterConfig`][careamics.config.MeanStdPatchFilterConfig]
- [`ShannonPatchFilterConfig`][careamics.config.ShannonPatchFilterConfig]

We will create the configuration using `create_advanced_n2v_config` and passing `ShannonPatchFilterConfig` with our selected threshold to the `patch_filter_config` argument.


```python title="Create Config and Train"
--8<-- "tutorials/patch_filtering.py:config"
```

1. Using shannon filtering with a threshold of 7.5

!!! info "Multi-channel data"

    For multi-channel data set the `ref_channel` parameter in the patch filter configs to the index of your desired channel.

!!! info "Other algorithms"

    The configuration factory functions for other algorithms, such as CARE and N2N also have a `patch_filter_config` argument.

!!! success Success

    If patch filtering was correctly applied during training, you should see a log similar to:

    ```
    Filtering background patches with filtering function shannon: 100%|██████████| 79/79 [00:06<00:00, 12.79it/s]
    Found 6345 background regions. Number of patches has been reduced to 14553 from 20224.
    ```