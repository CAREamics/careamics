---
description: Data handling
---

# Handling data

CAREamics supports by default data stored in memory as numpy arrays, but also data
stored on disk in the form of TIFF, CZI and Zarr files. Each format comes with
particular constraints.

## Arrays

Arrays are the simplest and fastest way to train and predict with CAREamics, they can be
passed as is to the `CAREamist`.


TODO: tip on axes order?


## Custom data formats