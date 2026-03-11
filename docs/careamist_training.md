---
icon: lucide/network
description: Quick start
---

# Training CAREamics

After having [created a configuration](./configuration.md) and 
[assembled the training data](./data.md), you are ready to train CAREamics. The preferred
way to train with CAREamics is to create `CAREamist` object.

```python
--8<-- "careamist_training.py:careamist_from_cfg"
```

1. Here the configuration can either be passed as we have seen in the
[configuration](./configuration.md) section, or as a path to configuration [saved to the
disk](./configuration.md#saving-and-loading).

CAREamics saves 

```python
--8<-- "careamist_training.py:careamist_workdir"
```


### Working directory


## Training basics

### Without validation

Once you have a `CAREamist` object, you can train CAREamics with the `train` method. Data 
is expected to be coherent with the [choice in the configuration](./configuration.md#simple-configuration)
and the [data section](./data.md).

Without validation, CAREamics will split the data into training and validation internally.
The amount of validation data can be set [in the configuration](./configuration.md#validation).

=== "Noise2Void"
    
    ```python title="Training Noise2Void"
    --8<-- "careamist_training.py:train_n2v_no_val"
    ```
 
    1. `train_data` should be an array, a path to a file, a path to a folder, or a list
    of array/paths.

=== "CARE/N2N"

    ```python title="Training CARE"
    --8<-- "careamist_training.py:train_care_no_val"
    ```

    1. `train_data` should be an array, a path to a file, a path to a folder, or a list
    of array/paths.
    2. For CARE and N2N, a target should be provided. It needs to be of the same type as
    `train_data` and pairs are formed by matching the order of the data and target lists.


### With validation

When passing validation, the only constraint is that the validation data is of the same
type as the training data. The amount of validation data is determined by the size of
the validation data.


=== "Noise2Void"
    
    ```python title="Training Noise2Void with validation"
    --8<-- "careamist_training.py:train_n2v_val"
    ```
 
    1. Validation is passed to `val_data`.

=== "CARE/N2N"

    ```python title="Training CARE with validation"
    --8<-- "careamist_training.py:train_care_val"
    ```
    
    1. Validation is passed to `val_data`.
    2. Target validation should be provided as well.


