---
description: CAREamist API
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
    --8<-- "getting_started.py.py:quick_start_n2v"
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

- Convenience functions (beginner)
- Convenience functions (intermediate)
- Convenience functions (advanced)
- Advanced Noise2Void, N2V2 and structN2V (intermediate)
- Putting data together (beginner)
- Training CAREamics (beginner)
- Logging and visualization (beginner)
- Prediction with CAREamics (beginner)
- Exporting models (beginner)
- Saving and loading configurations (beginner)
- Masking and background patch filtering (advanced)
- Custom data formats (advanced)
- Lightning API (advanced)
