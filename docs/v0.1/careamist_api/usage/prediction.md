# Prediction

!!! warning "Legacy documentation"
    This documentation is for the legacy version of CAREamics (v0.1), which is
    accessible through the `careamics.compat` module. It is kept here for reference, but
    we recommend using the latest version of CAREamics (v0.2) for new projects. Head to the [v0.2 guides](../v0.2/index.md).

Prediction is done by calling `careamist.predict` on either path or arrays. By default,
the prediction function will expect the same type of data (e.g. array or path) as it
was trained on, but it is possible to predict on a different type of data.

Prediction is performed using the current weights.


## Predict on arrays or paths


=== ":material-application-array-outline: On numpy arrays"

    ```python
    --8<-- "v0.1/careamist_api/usage/prediction.py:array"
    ```

=== ":octicons-rel-file-path-16: On Paths"

    ```python
    --8<-- "v0.1/careamist_api/usage/prediction.py:path"
    ```

## Tiling

Often, an image will be too large to fit on the GPU memory, or will have dimensions that
are incompatible with the model (e.g. odd dimensions). To solve this, the image can be
tiled into smaller overlapping patches, predicted upon and ultimately recombined.

In `careamist.predict`, this is done by passing two parameters:

``` python
--8<-- "v0.1/careamist_api/usage/prediction.py:tiling"
```

1. The tile sizes correspond to each spatial dimensions. A good start is the patch size
used during training.

2. The overlap is the number of pixels that each patch will overlap with its neighbors.

After prediction, each tile is cropped to half of the overlap in each of the directions
it overlaps with a neighboring tile. The reason is to minimize edge artifacts.

!!! warning "Tile and overlap model constraints"
    Some models, e.g. UNet model, impose constraints on the tile and overlap sizes. This
    is a direct consequence of the model architecture.

    For instance, when using a UNet model, the pooling and upsampling operations are
    not compatible with any tile size:

    - tile sizes must be equal to $k2^n$, where $n$ is the number of pooling layers
    (equal to the model depth) and $k$ is an integer.
    - overlaps must be even and larger than twice the receptive field.


## Test time augmentation

Test-time augmentation applies augmentations to the prediction input and averages the 
(de-augmented) predictions. This can improve the quality of the prediction. The TTA
generates all possible flipped and rotated versions of the image.

By default, test-time augmentation is applied by CAREamics. In order to deactivate TTA, 
you can set `tta` to `False`.

``` python
--8<-- "v0.1/careamist_api/usage/prediction.py:tta"
```

1. By default, TTA is activated!

!!! warning "Transforms and TTA"
    If you have turned off transforms, or used the non default ones, then you should 
    turn off TTA, as it would otherwise create images that do not correspond to your
    training data.

## Using batches

To potentially predict faster, you can predict on batches of images.

``` python
--8<-- "v0.1/careamist_api/usage/prediction.py:batches"
```

1. Each prediction step will be performed on 2 images or tiles.

!!! info "Batch and TTA"
    Having `batch_size>1` is compatible with the TTA.


## Changing the data type

You can use a different type (in the sense `path` vs `array`) of data at prediction time
by changing the `data_type` parameter.

``` python
--8<-- "v0.1/careamist_api/usage/prediction.py:diff_type"
```

1. As in the rest of CAREamics, the supported value are `tiff`, `array` and `custom`. 

## Changing the axes

Similarly, if you want to predict on data that has different axes than the training data,
as long as those have the same number of channels and spatial dimensions, then you can
change the `axes` parameter.

``` python
--8<-- "v0.1/careamist_api/usage/prediction.py:diff_axes"
```

1. Obviously, this need to match `source`.

## (Advanced) Predict on custom data type

As for the training, one can predict on custom data types by providing a function that
reads the data from a path and a function to filter the requested extension.

``` python
--8<-- "v0.1/careamist_api/usage/prediction.py:custom_type"
```

