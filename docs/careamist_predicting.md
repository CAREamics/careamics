---
icon: lucide/trending-up-down
description: Quick start
---

# Predicting with CAREamics

## Returning predictions

The simplest form of prediction is to call `predict`, which returns the predictions.

```python title="Prediction"
--8<-- "careamist_predicting.py:pred"
```

### Tiling

For odd-sized or large images, tiling should be used. Tiling is enabled by passing
a `tile_size` to the prediction method.

```python title="Prediction"
--8<-- "careamist_predicting.py:pred_tiled"
```

1. The `tile_size` need not be equal to the training patch size.
2. Overlaps are optional, and default to `(48, 48)`


!!! note "Why use tiling?"

    If the images have dimensions that are not comaptible with the network architecture
    (`shape % 2**model_depth != 0`), then tiling is required to ensure that the network
    can process the images.

    If the images are large, then they might simply not fit in memory and need to be
    processed as tiles.

!!! note "What tiling size to choose?"

    The tiling size should be chosen based on the size of the images and the available
    memory. A good starting point is to use a patch size that is a multiple of
    `2**model_depth`. You can then play with `batch_size` to find the optimal memory
    usage.

    Note that the overlaps must be larger than the receptive field of the network.

!!! warning "Dimensions"

    Obviously, the tiling and overlaps must respect the dimensions of the images (2D
    or 3D).

### Changing dataloading parameters

During prediction, you can change the dataloading parameters by passing them to the
`predict` method. Currently, we only expose `batch_size` and `num_workers`.

```python title="Prediction"
--8<-- "careamist_predicting.py:pred_dataloader"
```

### Data parameters

The data we want to predict on might be different from the training data, in terms of
axes or format. The `predict` method allows changing these parameters (`axes`, 
`data_type`, `channels`, `in_memory`).

Here, let's say we trained from arrays of axes `YX`, and now want to predict with the
trained model on a TIFF file (we have a path `pred_data_path`) that has channels. Since
we did not trained on channels, we need to pass `channels` to specify a single channel
we want to train on. We also need to specify the new axes and data_type. Finally, we
do not want to train in-memory.

```python title="Prediction"
--8<-- "careamist_predicting.py:pred_data_params"
```

1. Now, data is a path.
2. We specify only a single channel.

!!! warning "Coherence of the data parameters"

    Prediction data must have the same type content as the training data (we talk about
    the data being "in distribution"). That means that it cannot have different spatial
    axes or suddenly have more channels.


## Predicting to disk

(soon)

!!! note "CZI format"

    Prediction directly to disk is not available for CZI data. 

!!! note "Zarr format"

    Prediction directly to disk is with Zarr requires tiling to be enabled.