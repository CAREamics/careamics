# Configuration errors

Because the CAREamics configuration is validated using Pydantic, all the configuration
errors are delegated to the validation library.

Here we showcase a certain number of confguration errors that can appear when
using the convenience functions. It is impossible to cover all possibilities, so these
examples are here to highlight how to read Pydantic errors.

!!! warning "Under construction"
    This page is under construction and will be updated soon.



### Pydantic error

```python
ValidationError: 1 validation errors for union[function-after[validate_n2v2(), function-after[validate_3D(), N2VConfiguration]],function-after[validate_3D(), N2NConfiguration],function-after[validate_3D(), CAREConfiguration]]
function-after[validate_n2v2(), function-after[validate_3D(), N2VConfiguration]].data_config.data_type
  Input should be 'array', 'tiff' or 'custom' [type=literal_error, input_value='arrray', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/literal_error
...
```

In this case, the input to `data_config.data_type` should be either `'array'`, `'tiff'` or `'custom'`. The error message is telling us that the input value `'arrray'` is not a valid option.

