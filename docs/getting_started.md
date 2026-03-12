---
icon: lucide/house
description: Quick start
---

# Using CAREamics

The CAREamist API is the recommended way to use CAREamics, it is a two stage process, in
which users first define a configuration and then use a the `CAREamist` to run their 
training and prediction.

More advanced users wishing to have more control over the training and prediction
process can re-use CAREamics module in a Pytorch Lightning script, which we refer
to as the [Lightning API]().

## Quick start

=== "Noise2Void"
    
    ```python
    --8<-- "getting_started.py:quick_start_n2v"
    ```

    1. This is training from arrays in memory, but this can also be done with files 
    on disk.

    2. The only important thing is that the data passed is coherent with the choice
    in the configuration.


=== "CARE"

    ```python
    --8<-- "getting_started.py:quick_start_care"
    ```

    1. This is training from arrays in memory, but this can also be done with files 
    on disk.

    2. The only important thing is that the data passed is coherent with the choice
    in the configuration.

## On the menu

- [Configuring CAREamics](configuration.md)
    - [Simple configuration](configuration.md#simple-configuration)
    - [Advanced configuration](configuration.md#advanced-configuration-configuration)
    - [Noise2Void parameters](configuration.md#noise2void-flavours-and-parameters)
    - [PyTorch and Lightning parameters](configuration.md#pytorch-and-lightning-parameters)
    - [Saving and loading configurations](configuration.md#saving-and-loading)
- [Data](data.md)
    - [Arrays](data.md#arrays)
    - [TIFF](data.md#tiff)
    - [CZI](data.md#czi)
    - [Zarr](data.md#zarr)
    - [Custom data formats](data.md#custom-data-formats)
- [Training CAREamics]()
- [Predicting with CAREamics]()
- [Lightning API]()
- [Frequently asked questions]()
