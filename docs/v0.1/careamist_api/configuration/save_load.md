# Save and load

CAREamics configurations can be saved to the disk as `.yml` file and loaded easily to
start similar experiments.

## Save a configuration

```python title="Save a configuration"
--8<-- "save_load.py:save"
```

In the resulting file, you can see all the parameters that are defaults and hidden
from you.

??? Example "resulting config.yml file"

    ```yaml
    --8<-- "config.yml"
    ```

## Load a configuration

```python title="Load a configuration"
--8<-- "save_load.py:load"
```
