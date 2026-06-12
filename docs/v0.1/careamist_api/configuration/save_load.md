# Save and load

!!! warning "Legacy documentation"
    This documentation is for the legacy version of CAREamics (v0.1), which is
    accessible through the `careamics.compat` module. It is kept here for reference, but
    we recommend using the latest version of CAREamics (v0.2) for new projects. Head to the [v0.2 guides](../v0.2/index.md).

CAREamics configurations can be saved to the disk as `.yml` file and loaded easily to
start similar experiments.

## Save a configuration

```python title="Save a configuration"
--8<-- "v0.1/careamist_api/configuration/save_load.py:save"
```

In the resulting file, you can see all the parameters that are defaults and hidden
from you.

??? Example "resulting config.yml file"

    ```yaml
    --8<-- "v0.1/careamist_api/configuration/config.yml"
    ```

## Load a configuration

```python title="Load a configuration"
--8<-- "v0.1/careamist_api/configuration/save_load.py:load"
```
