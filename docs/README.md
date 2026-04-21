# CAREamics guides

These docs are automatically tested in the `.github/workflows/ci_guides.yml` CI and separately imported into the [CAREamics website](https://careamics.github.io/latest/). They consist of a structured organization of python snippets and markdown files.

The structure of the docs on the website is defined in `nav.toml`.

## How to use the snippets

Snippets are stored in python files and automatically aggregated by `test_guides.py` as pytest test cases. A snippet consist of a python code in the following block:

```python
# %%
# --8<-- [start:snippet_id]
...
# --8<-- [end:snippet_id]
```

The `snippet_id` is used to identify the snippet and can be used in the markdown files to include the snippet as follows:

````markdown
```python
--8<-- "path/to/snippet.py:snippet_id"
```
````

> [!NOTE]
> The path to the snippet is relative to the `docs` folder.


## How to add new pages to the guides

The guides are uploaded automatically to the website at build time and organized according to the `nav.toml` file. To add a new page to the guides, simply create a new markdown file in the `docs` folder and add it to the `nav.toml` file.

> [!IMPORTANT]
> Only existing keys are accepted by the website: Using CAREamics, Tutorials, and Legacy (v0.1). For additional sections, the website's `zensical.toml` file has to be updated.