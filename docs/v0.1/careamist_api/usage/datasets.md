# (Intermediate) Datasets

Datasets are the internal classes providing the individual patches for training, 
validation and prediction. In CAREamics, we provide a `TrainDataModule` class that 
creates the datasets for training and validation (there is a class for prediction
as well, which is simpler and shares some parameters with the training one). In most cases,
it is created internally. In this section, we describe what it does and shed light on
some of its parameters that are passed to the [train methods](training.md).

!!! info "Datasets in practice"
    This section contains descriptions of the internal working of CAREamics. In practice,
    most users will never have to instantiate the datasets themselves, as they are created
    from within the `careamist.train` or `careamist.predict` methods.


## Overview

The `TrainDataModule` receives both data configuration and data itself. The data
can be passed a path to a folder, to a file or as `numpy` array. 

```python title="Simplest way to instantiate TrainDataModule"
--8<-- "datasets.py:train_data"
```

It has the following parameters:

- `data_config`: data configuration
- `train_data`: training data (array or path)
- `(optional) val_data`: validation data, if not provided, the validation data is taken from the training data
- `(optional) train_data_target`: target data for training (if applicable)
- `(optional) val_data_target`: target data for validation (if applicable)
- `(optional) read_source_func`: function to read custom data types 
    (see [custom data types](#advanced-custom-data-types))
- `(optional) extension_filter`: filter to select custom types
    (see [custom data types](#advanced-custom-data-types))
- `(optional) val_percentage`: percentage of validation data to extract from the training
    (see [splitting validation](training.md#splitting-validation-from-training-data))
- `(optional) val_minimum_split`: minimum validation split 
    (see [splitting validation](training.md#splitting-validation-from-training-data))
- `(optional) use_in_memory`: whether to use in-memory dataset if possible (Default is `True`), 
    not applicable to mnumpy arrays.

Depending on the type of the data, which is specified in the `data_config` and
is compared to the type of `train_data`, the `TrainDataModule` will create the appropriate
dataset for both training and validation data.

In the absence of validation, validation data is extracted from training data
(see [splitting validation](training.md#splitting-validation-from-training-data)).


## Available datasets

CAREamics currently support two datasets:

- [InMemoryDataset](#in-memory-dataset): used when the data fits in memory.
- [IterableDataset](#iterable-dataset): used when the data is too large to fit in memory.

If the data is a `numpy` array, the `InMemoryDataset` is used automatically. Otherwise,
we list the files contained in the path, compute the size of the data and instantiate
an `InMemoryDataset` **if the data is less than 80% of the total RAM size**. If not,
CAREamics instantiate an `IterableDataset`.

Both datasets work differently, and the main differences can be summarized as follows:

| Feature          | `InMemoryDataset`    | `IterableDataset`   |
| ---------------- | -------------------- | ------------------- |
| Used with arrays | :material-check: Yes | :material-close: No |
| Patch extraction | Sequential           | Random              |
| Data loading     | All in memory        | One file at a time  |


In the next sections, we describe the different steps they perform.


### In-memory dataset

As the name implies, the in-memory dataset loads all the data in memory. It is used when
the data on the disk seems to fit in memory, or when the data is already in memory and 
passed as a numpy array. The advantage of the dataset is that is allows faster access
to the patches, and therefore faster training time.

!!! note "What about supervised training?"

    For supervised training, the steps are the same and are performed for the targets
    alongside the source.


!!! note "What if I have a time (`T`) axis?"

    `T` axes are accepted by the CAREamics configuration, but are treated as a sample
    dimension (`S`). If both `S` and `T` are present, the two axes are concatenated.


### Iterable dataset

The iterable dataset is used to load patches from a single file at a time, one file after
another. This allows training on datasets that are too large to fit in memory. This dataset
is exclusively used with files input (data passed as paths).

!!! warning "Iterable dataset and splitting validation"
    The iterable dataset does not split patches from the training data, but files! 
    (see [splitting validation](./training.md#splitting-validation-from-training-data)).


!!! note "What about supervised training?"

    For supervised training, the steps are the same and are performed for the targets
    alongside the source.


!!! note "What if I have a time (`T`) axis?"

    `T` axes are accepted by the CAREamics configuration, but are treated as a sample
    dimension (`S`). If both `S` and `T` are present, the two axes are concatenated.


## (Intermediate) Transforms

Transforms are augmentations and any operation applied to the patches before feeding them
into the network. CAREamics supports the following transforms (see 
[configuration full spec](../configuration/full_spec.md) for an example on how to configure them):


| Transform               | Description                                  | Notes                                 |
| ----------------------- | -------------------------------------------- | ------------------------------------- |
| `Normalize`             | Normalize (zero mean, unit variance)         | Necessary                             |
| `XYFlip`                | Flip the image along X and Y, one at a time  | Can flip a single axis, optional |
| `XYRandomRotate90Model` | Rotate by 90 degrees the XY axes             | Optional                                   |
| `N2VManipulateModel`    | N2V pixel manipulation                       | Only for N2V, in which case it is necessary|


The `Normalize` transform is always applied, and the rest are optional. The exception is
`N2VManipulateModel`, which is only applied when training with N2V (see [Noise2Void](../../../algorithms/Noise2Void)).

!!! note "When to turn off transforms?"

    The configuration allows turning off transforms. In this case, only normalization
    (and potentially the `N2VManipulateModel` for N2V) is applied. This is useful when
    the structures in your sample are always in the same orientation, and flipping and
    rotation do not make sense.


## (Advanced) Custom data types

To read custom data types, you can set `data_type` to `custom` in `data_config`
and provide a function that returns a numpy array from a path as
`read_source_func` parameter. The function will receive a Path object and
an axies string as arguments, the axes being derived from the `data_config`.

You should also provide a `fnmatch` and `Path.rglob` compatible expression (e.g.
"*.npy") to filter the files extension using `extension_filter`.


```python title="Read custom data types"
--8<-- "datasets.py:custom"
```

1. We define a function that reads the custom data type.

2. It takes a path as argument!

3. But it also need to receive `*args` and `**kwargs` to be compatible with the `read_source_func` signature.

4. It simply returns a `numpy` array.

5. The data type must be `custom`!

6. And we pass a `Path | str`.

7. Simply pass the method by name.

8. We also need to provide an extension filter that is compatible with `fnmatch` and `Path.rglob`.

9. These two lines are necessary to instantiate the training dataset that we call at the end. They are
    called automatically by PyTorch Lightning during training.

10. The dataloader gives access to the dataset, we choose the first element, and since
    we configured CAREamics to use N2V, the output is a tuple whose first element is our
    first patch!

In practice, you should not access the dataloader directly (except for testing). Using 
custom types for training should be done as follows:

```python
--8<-- "datasets.py:train_custom"
```

## Prediction datasets

The prediction data module, `PredictDataModule` works similarly to `TrainDataModule`, albeit
with different parameters:


- `pred_config`: inference configuration
- `pred_data`: prediction data (array or path)
- `(optional) read_source_func`: function to read custom data types 
    (see [custom data types](#advanced-custom-data-types))
- `(optional) extension_filter`: filter to select custom types
    (see [custom data types](#advanced-custom-data-types))


## (Advanced) Subclass TrainDataModule

The data module used in CAREamics have only a limited number of parameters, and they 
make use of the CAREamics datasets. If you need to have a different dataset, then you
can subclass `TrainDataModule` and override the `setup` method to use your own
datasets.