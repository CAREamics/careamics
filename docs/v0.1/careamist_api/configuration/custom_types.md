# Custom types

The `data_type` parameter of the `DataConfig` class is a string that is used to choose
the data loader within CAREamics. We currently only support `array` and `tiff` explicitly.

However, users can set the `data_type` to `custom` and use their own read function.

```python title="Custom data type"
--8<-- "advanced_configuration.py:data"
```

1. As far as the configuration is concerned, you only set the `data_type` to `custom`. The
    rest happens in the `CAREamist` instance.

!!! info "Full example in other sections"

    A full example of the use of a custom data type is available in the [CAREamist]() and [Applications]() sections.
