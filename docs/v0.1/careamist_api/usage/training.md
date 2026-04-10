---
description: CAREamics training
---

# Training

!!! warning "Legacy documentation"
    This documentation is for the legacy version of CAREamics (v0.1), which is
    accessible through the `careamics.compat` module. It is kept here for reference, but
    we recommend using the latest version of CAREamics (v0.2) for new projects. Head to the [v0.2 guides](../v0.2/index.md).

You can provide data in various way to train your model: as a `numpy` array, using a
path to a folder or files, or by using CAREamics data module class for more control
(advanced).

The details of how CAREamics deals with the loading and patching is detailed in the
[dataset section](datasets.md).


!!! warning "Data type"
    The data type of the source and targets must be the same as the one specified in the configuration.
    That is to say `array` in the case of `np.ndarray`, and `tiff` in the case of paths.


## Training by passing an array

CAREamics can be trained by simply passing numpy arrays.

```python title="Training by passing an array"
--8<-- "v0.1/careamist_api/usage/training.py:array"
```

1. All parameters to the `train` method must be specified by keyword.
2. If you don't provide a validation source, CAREamics will use a fraction of the training data
   to validate the model.


!!! info "Supervised training"
    If you are training a supervised model, you must provide the target data as well.

    ```python
    --8<-- "v0.1/careamist_api/usage/training.py:supervised"
    ```

## Training by passing a path

The same thing can be done by passing a path to a folder or files.

```python title="Training by passing a path"
--8<-- "v0.1/careamist_api/usage/training.py:path"
```

1. The path can point to a single file, or contain multiple files.


!!! info "Training from path"
    To train from a path, the data type must be set to `tiff` or `custom` in the 
    configuration.


## Splitting validation from training data

If you only provide training data, CAREamics will extract the validation data directly
from the training set. There are two parameters controlling that behaviour: `val_percentage`
and `val_minimum_split`.

`val_percentage` is the fraction of the training data that will be used for validation, and
`val_minimum_split` is the minimum number of images used. If the percentage leads to a 
number of patches smaller than `val_minimum_split`, CAREamics will use `val_minimum_split`.

```python title="Splitting validation from training data"
--8<-- "v0.1/careamist_api/usage/training.py:split"
```

1. 10% of the training data will be used for validation.
2. If the number of images is less than 5, CAREamics will use 5 images for validation.


!!! warning "Arrays vs files"
    The behaviour of `val_percentage` and `val_minimum_split` is based different depending
    on whether the source data is an array or a path. If the source is an array, the
    split is done on the patches (`N` patches are used for validation). If the source is a
    path, the split is done on the files (`N` files are used for validation).


## Training by passing a `TrainDataModule` object

CAREamics provides a class to handle the data loading of custom data type. We will dive 
in more details in the next section into what this class can be used for. Here is a 
brief overview of how it is passed to the `train` method.

```python title="Training by passing a TrainDataModule object"
--8<-- "v0.1/careamist_api/usage/training.py:datamodule"
```

1. Here this does the same thing as passing the `train_source` directly into the `train` method.
    In the next section, we will see a more useful example.


## Logging the training

By default, CAREamics simply log the training progress in the console. However, it is 
possible to use either [WandB](https://wandb.ai/site) or [TensorBoard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html).

To decide on the logger, check out the [Configuration section](../configuration/convenience_functions.md).


!!! note "Loggers installation"

    Using WandB or TensorBoard require the installation of `extra` dependencies. Check
    out the [installation section](../../../installation.md#extra-dependencies) to know more about it.


### Plotting loss

To plot the loss curves, you can use the `CAREamist.get_losses` function:

```python title="Plotting losses"
--8<-- "v0.1/careamist_api/usage/training.py:losses"
```