---
icon: lucide/trending-up-down
description: Quick start
---

# Predicting with CAREamics

Prediction can be performed right after training, re-using the same `CAREamist` object,
or by creating a new one with a path to a checkpoint:

```python title="Starting from a checkpoint"
--8<-- "current/careamist/careamist_predicting.py:pred_start"
```

## Returning predictions

The simplest form of prediction is to call `predict`, which returns the predictions and
their sources.

```python title="Prediction"
--8<-- "current/careamist/careamist_predicting.py:pred"
```

The `predict` method returns both a list of arrays corresponding to the predictions and
a list of their sources. If `pred_data` was a list of path to files, then `sources` will
be a list of path to the file corresponding to each array. If the arrays were
otherwise passed as `numpy` arrays, then the source can be ignored, since it is an empty
list:

```python title="Prediction from arrays"
--8<-- "current/careamist/careamist_predicting.py:pred_empty_source"
```

!!! info "Prediction output"

    By default, `predict` returns `(predictions, sources)`.
    
    For HDN and MicroSplit, which are variational-autoencoder-based algorithms,
    passing `n_predictions` returns `(predictions, uncertainty, sources)`, where
    `uncertainty` is the per-pixel standard deviation across the sampled predictions.

    For segmentation models, use `segmentation_output="mask"` or
    `segmentation_output="probabilities"` to choose the output representation.

!!! note "Dimensions of the prediction"
    
    Predictions are restored to the configured target/original output shape and axis
    order. The output axes can differ from the input axes when `target_axes` or
    `output_axes` specifies a different order or channel layout.

!!! note "Checkpoint"

    By default, CAREamics saves multiple checkpoints during training. If saving the
    best checkpoint (the one with the lowest validation loss) is enabled, it will be
    loaded for prediction. Otherwise, prediction uses the last checkpoint saved.
    Supervised algorithms save the best checkpoint by default. For self-supervised
    algorithms such as Noise2Void and Noise2Noise, saving the best checkpoint is
    disabled by default because the validation loss may not accurately reflect model
    performance.

### Prediction return values and output shapes

| Model/workflow | Return values | Prediction content |
|---|---|---|
| Deterministic model (UNet)| `(predictions, sources)` | One prediction per input image, restored to the requested output axes |
| HDN or MicroSplit with `n_predictions` | `(predictions, uncertainty, sources)` | Mean prediction and per-pixel standard deviation across stochastic samples |
| Segmentation with `segmentation_output="mask"` | `(predictions, sources)` | Class mask selected from the highest-probability class |
| Segmentation with `segmentation_output="probabilities"` | `(predictions, sources)` | Per-class probability map |
| Tiled prediction | Same return structure as above | Tiles are stitched back into the requested output shape |


### Sampling-based prediction

HDN and MicroSplit are generative models and can draw multiple predictions for the same
input. Pass `n_predictions` to average the sampled predictions and obtain a per-pixel
standard-deviation estimate. It must be at least `2`.

```python title="HDN/MicroSplit Sampling"
--8<-- "current/careamist/careamist_predicting.py:pred_sampling_hdn"
```

The return values are `(predictions, uncertainty, sources)`. The uncertainty is the
standard deviation across the sampled predictions.

### cryoCARE prediction

cryoCARE requires both odd and even reconstructions at prediction time. Pass one reconstruction as
`pred_data` and its paired odd/even reconstruction as `pred_data_target`. cryoCARE runs
the model on both and averages the two predictions.

```python title="cryoCARE prediction"
--8<-- "current/careamist/careamist_predicting.py:pred_cryocare"
```

### Tiling

For odd-sized or large images, tiling should be used. Tiling is enabled by passing
a `tile_size` to the prediction method.

```python title="Prediction"
--8<-- "current/careamist/careamist_predicting.py:pred_tiled"
```

1. `tile_size` needs to be specified and may not be equal to the patch size (see notes).
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
    usage. For large receptive fields (UNet with depth `>2` or LVAE), it is better to
    use the training patch as tile since otherwise the tiles will be out of distribution.

    Note that the overlaps must be larger than the receptive field of the network.


!!! warning "Dimensions"

    Tile sizes and overlaps must respect the dimensions of the images (2D
    or 3D).

### Changing dataloading parameters

During prediction, you can change the dataloading parameters by passing them to the
`predict` method. The parameters `batch_size` and `num_workers` can be set through the `predict` function arguments, while any other parameters have to be changed through the configuration.

For example, set other PyTorch `DataLoader` options through
`config.data_config.pred_dataloader_params`:

```python
config.data_config.pred_dataloader_params = {
    "pin_memory": True,
    "persistent_workers": True,
}
```

```python title="Prediction"
--8<-- "current/careamist/careamist_predicting.py:pred_dataloader"
```

### Data parameters

The data we want to predict on might be different from the training data, in terms of
axes or format. The `predict` method allows changing these parameters (`axes`, 
`data_type`, `channels`, `in_memory`).

Here, let's say we trained from arrays of axes `YX`, and now want to predict with the
trained model on a TIFF file (we have a path `pred_data_path`) that has multiple
time-points. We set `axes="SYX"` to specify the input axes and `data_type="tiff"`.
Finally, we do not want to load the data in memory.

```python title="Prediction"
--8<-- "current/careamist/careamist_predicting.py:pred_data_params"
```

1. Now, data is a path.

!!! warning "Coherence of the data parameters"

    Prediction data must have the same type of content as the training data (we talk about
    the data being "in distribution"). That means that it cannot have different spatial
    axes or suddenly have more channels.

!!! note "New data has channels"

    If the new data has channels, but the model was not trained on multiple channels,
    then the `channels` parameter can be used to specify which channels to use for
    prediction. For example, if the new data has 3 channels, but the model was
    trained on single-channel data, then `channels=[1]` can be used to specify that
    only the second channel should be used for prediction.

!!! note "New target axes"

    `target_axes` is the configuration-level default for the prediction output axes.
    The `predict(output_axes=...)` argument overrides it. For example,
    `predict(..., axes="SYX", output_axes="CYX")` requests channel-first output.

### Choosing the checkpoint

Checkpoints can be specified by passing a `checkpoint` argument to the `predict` method.
It can be either a path to a checkpoint file or one of the keywords specified by
PyTorch Lightning (`"best"` or `"last"`).

```python title="Prediction with checkpoint"
--8<-- "current/careamist/careamist_predicting.py:pred_checkpoint"
```

1. `careamist.get_checkpoints()` returns a checkpoint record containing the available
   checkpoints and the checkpoint selected by default for inference.


!!! warning "Noise2Void and Noise2Noise"

    Noise2Void and Noise2Noise models do not have a well-defined "best" checkpoint
    based on validation loss. Specify an explicit path if you want to use a
    specific checkpoint.


## Predicting to disk

Prediction can also be saved directly to disk. This is particularly useful for large
numbers of files that may not fit in memory.

```python title="Predict directly to the disk"
--8<-- "current/careamist/careamist_predicting.py:pred_to_disk"
```

1. Optional, if not provided the predictions will be saved in the working directory,
   in a folder called `predictions`.

!!! important "Other parameters"

    `predict_to_disk` accepts similar parameters as `predict`, including tiling,
    dataloader parameters or checkpoints. Refer to the previous sections for more
    details.

    It also accepts `n_predictions` for stochastic HDN/MicroSplit prediction and
    `segmentation_output="mask"` or `"probabilities"` for segmentation models.

!!! note "Default format"

    Predictions are currently by default written as the input format for TIFF, Zarr or
    MRC files, otherwise as TIFF files.

!!! note "CZI format"

    Prediction directly to disk is not available for CZI data. 


### Change the write type

You can also select a different write type from the default or from the source data.

```python title="Predict with a different type"
--8<-- "current/careamist/careamist_predicting.py:pred_write_type"
```

1. The supported write types are `tiff`, `zarr`, and `custom`. Custom output requires
   `write_extension` and `write_func`.


### Use a custom write function

We support passing a custom write function to save the predictions. The function must
implement a particular `Protocol` ([`WriteFunc`][careamics.image_io.write.WriteFunc]),
meaning that the function signature must strictly be that of the `WriteFunc` protocol.
The function must accept a path, an array and allow passing additional keyword
arguments. Here is an example of writing to a custom type (`.npy`):

```python title="Custom write function"
--8<-- "current/careamist/careamist_predicting.py:pred_write_func"
```

1. First define your write function.
2. We need to select `custom` as the write type.
3. And specify the extension of the file to write.
4. Pass the custom write function to the `write_func` argument.
5. Optional, to pass additional parameters to underlying functions.
