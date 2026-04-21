# Using CAREamics

!!! warning "Legacy documentation"
    This documentation is for the legacy version of CAREamics (v0.1), which is
    accessible through the `careamics.compat` module. It is kept here for reference, but
    we recommend using the latest version of CAREamics (v0.2) for new projects. Head to the [v0.2 guides](../v0.2/index.md).

In this section, we will explore the many facets of the `CAREamist` class, which
allows training and predicting using the various algorithms in CAREamics.

The workflow in CAREamics has five steps: creating a configuration, instantiating a
`CAREamist` object, training, prediction, and model export.


```python title="Basic CAREamics usage"
--8<-- "v0.1/careamist_api/usage/careamics_usage.py:basic_usage"
```

1. Obviously, one should choose a more realistic number of epochs for training.

2. One should use real data for training!

