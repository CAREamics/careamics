---
icon: lucide/trending-up-down
description: Quick start
---

# Predicting with CAREamics

## Returning predictions

The simplest form of prediction is to call `predict`, which returns the predictions and
their sources.

```python title="Prediction"
--8<-- "current/careamist_predicting.py:pred"
```

The `predict` method returns both a list of arrays corresponding to the predictions and
a list of their sources. If `pred_data` was a list of path to files, then `sources` will
be a list of path to the file corresponding to each array. If the arrays were
otherwise passed as `numpy` arrays, then the source can be ignored, since it is an empty
list:

```python title="Prediction from arrays"
--8<-- "current/careamist_predicting.py:pred_empty_source"
```

!!! note "Checkpoint"

    By default, CAREamics uses a checkpoint callback that saves multiple checkpoints
    during training. Depending on the algorithm, the prediction method will either
    use the best checkpoint (the one with the lowest validation loss) or the last
    checkpoint (the one from the last epoch).

    Noise2Void and Noise2Noise will use the last checkpoint, while the other algorithms
    will use the best checkpoint.

### Choosing the checkpoint

Checkpoints can be specified by passing a `checkpoint` argument to the `predict` method.
It can be either a path to a checkpoint file or one of the keywords specified by
PyTorch Lightning (typically `"best"` or `"last"`, see [documentation](https://lightning.ai/docs/pytorch/stable/common/trainer.html#predict)).

```python title="Prediction with checkpoint"
--8<-- "current/careamist_predicting.py:pred_checkpoint"
```

!!! warning "Noise2Void and Noise2Noise"

    Noise2Void and Noise2Noise models do not have a well-defined "best" checkpoint
    based on validation loss. Specify an explicit path if you want to use a
    specific checkpoint.


### Tiling

For odd-sized or large images, tiling should be used. Tiling is enabled by passing
a `tile_size` to the prediction method.

```python title="Prediction"
--8<-- "current/careamist_predicting.py:pred_tiled"
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
`predict` method. The parameters `batch_size` and `num_workers` can be set through the `predict` function arguments, while any other parameters have to be changed through the configuration.

```python title="Prediction"
--8<-- "current/careamist_predicting.py:pred_dataloader"
```

### Data parameters

The data we want to predict on might be different from the training data, in terms of
axes or format. The `predict` method allows changing these parameters (`axes`, 
`data_type`, `channels`, `in_memory`).

Here, let's say we trained from arrays of axes `YX`, and now want to predict with the
trained model on a TIFF file (we have a path `pred_data_path`) that has multiple
time-points. We need to set `new_axes` to `SYX` to specify the new axes order. We also
need to specify the new axes and `data_type`. Finally, we do not want to train in-memory.

```python title="Prediction"
--8<-- "current/careamist_predicting.py:pred_data_params"
```

1. Now, data is a path.

!!! warning "Coherence of the data parameters"

    Prediction data must have the same type content as the training data (we talk about
    the data being "in distribution"). That means that it cannot have different spatial
    axes or suddenly have more channels.

!!! note "New data has channels"

    If the new data has channels, but the model was not trained on multiple channels,
    then the `channels` parameter can be used to specify which channels to use for
    prediction. For example, if the new data has 3 channels, but the model was
    trained on single-channel data, then `channels=[1]` can be used to specify that
    only the second channel should be used for prediction.


## Predicting to disk

Prediction can also be saved directly to disk. This is particularly useful for large
number of files that may not fit in memory.

```python title="Predict directly to the disk"
--8<-- "current/careamist_predicting.py:pred_to_disk"
```

1. Optional, if not provided the predictions will be saved in the working directory,
   in a folder called `predictions`.

!!! important "Other parameters"

    `predict_to_disk` accepts similar parameters as `predict`, including tiling,
    dataloader parameters or checkpoints. Refer to the previous sections for more
    details.

!!! note "Default format"

    Predictions are currently by default written as TIFF files.

!!! note "CZI format"

    Prediction directly to disk is not available for CZI data. 

!!! note "Zarr format"

    Prediction directly to disk is with Zarr requires tiling to be enabled.


### Change the write type

You can also select a different write type that the default or from the source data.

```python title="Predict with a different type"
--8<-- "current/careamist_predicting.py:pred_write_type"
```

1. The write type can be changed. Currently, only `tiff` and `zarr` are supported.


!!! important "Zarr as write type"

    Zarr can be selected as write type only if tiling is enabled.

### Use a custom write function

We support passing a custom write function to save the predictions. The function must
implement a particular `Protocol` ([`WriteFunc`][careamics.image_io.write.WriteFunc]),
meaning that the function signature must strictly be that of the `WriteFunc` protocol.
The function must accept a `path`, an `array` and allows passing additional keyword
arguments. Here is an example of writing to a custom type (`.npy`):

```python title="Custom write function"
--8<-- "current/careamist_predicting.py:pred_write_func"
```

1. First define your write function.
2. We need to select `custom` as the write type.
3. And specify the extension of the file to write.
4. Pass the custom write function to the `write_func` argument.
5. Optional, to pass additional parameters to underlying functions.

