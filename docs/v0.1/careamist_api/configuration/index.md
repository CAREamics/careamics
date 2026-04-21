---
description: Configuration guide
---

!!! warning "Legacy documentation"
    This documentation is for the legacy version of CAREamics (v0.1), which is
    accessible through the `careamics.compat` module. It is kept here for reference, but
    we recommend using the latest version of CAREamics (v0.2) for new projects. Head to the [v0.2 guides](../v0.2/index.md).


The configuration summarizes all the parameters used internally by CAREamics. It is 
used to create a `CAREamist` instance and is saved together with the checkpoints and 
saved models.

Configurations are created for specific algorithms using convenience functions. They are Python objects, but can be exported to a dictionary, and contain a hierarchy of parameters.

??? note "Configuration example"
    Here is an example of a configuration for the `Noise2Void` algorithm.

    ```python title="Noise2Void configuration"
    --8<-- "v0.1/careamist_api/configuration/config.yml"
    ```

The number of parameters might appear overwhelming, but in practice users only call a function with few parameters. The configuration is designed to hide the complexity of the algorithms and provide a simple interface to the user.


```python
--8<-- "v0.1/careamist_api/configuration/convenience_functions.py:simple"
```

In the next sections, you can dive deeper on how to create a configuration and interact with the configuration with different levels of expertise.

:material-star::material-star-outline::material-star-outline: Beginner

:material-star::material-star::material-star-outline: Intermediate

:material-star::material-star::material-star: Advanced

- :material-star::material-star-outline::material-star-outline: [Convenience functions](convenience_functions.md)
- :material-star::material-star-outline::material-star-outline: [Save and load configurations](save_load.md)
- :material-star::material-star::material-star-outline: [Custom types](custom_types.md)
- :material-star::material-star::material-star-outline: [Understanding configuration errors](configuration_errors.md)
- :material-star::material-star::material-star: [Build the configuration from scratch](build_configuration.md)
- :material-star::material-star::material-star: [Algorithm requirements](algorithm_requirements.md)

