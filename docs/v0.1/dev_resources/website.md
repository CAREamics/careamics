# Github pages

The Github pages are built using [mkdocs](https://www.mkdocs.org/), more specifically the 
[mkdocs-material](https://squidfunk.github.io/mkdocs-material/) theme. Modifications to
the theme were greatly inspired from [pydev-guide](https://github.com/pydev-guide/pydev-guide.github.io).

In this page, we describe some of the technical details on how to maintain this website.

## Environment

The `requirements.txt` file contains all the packages used to generate this website.

### Build the pages locally

In order to build the pages locally, follow these steps:

1. Fork this repository and clone it.
2. Create a new environment and install the dependencies:
    ```bash
    conda create -n careamics-docs python=3.12
    pip install -r requirements.txt
    ```
4. Run the following scripts (which are normally run by the CI):
    ```bash
    sh scripts/check_out_repos.sh
    sh scripts/check_out_examples.sh
    ```

    !!! note

        To run the scripts locally, you will need to install `jq`. For instance with
        `brew install jq`.

3. Build the pages:
    ```bash
    mkdocs serve
    ```
4. Open the local link in your browser.

**Note**: This will not show you the versioning mechanism. For this, check out the 
**Version release** section.


## Code snippets

Code snippets are all automatically tested in [careamics-example](https://github.com/CAREamics/careamics-examples/tree/main/applications)
and added automaticallt to the Github pages in the [guides section](../index.md).

The script `scripts/check_out_examples.sh` clone the examples repository locally and upon
building the pages, the code snippets are automatically added to the markdown by the
[PyMdown Snippets extension](https://facelessuser.github.io/pymdown-extensions/extensions/snippets/).

It works as follows, first create a code snippet in a python file on 
[careamics-example](https://github.com/CAREamics/careamics-examples/tree/main/applications):

```python title="careamics-example/some_example/some_file.py"
# code necessary for running successfully the snippet
# but not shown on the Github pages
import numpy as np

array = np.ones((32, 32))

# we add a snippet section:
# --8<-- [start:my_snippet]
sum_array = np.sum(array) # snippets appearing on the website (can be multi-lines)
# --8<-- [end:my_snippet]

# then more code or snippets sections
...
```

Then, in the [CAREamics Github pages source](https://github.com/CAREamics/careamics.github.io),
the corresponding markdown file has the following content telling `PyMdown.Snippets` to 
include the snippet section:

```markdown title="careamics.github.io/guides/some_example/some_file.py"
    To sum the array, simply do:
    
    ```python
    ;--8<-- "careamics-example/some_example/some_file.py:my_snippet"
    ```
```

The examples in the guide are automatically tested in the CI. To make sure it runs on
the correct branches in the pull request, use the same branch name for the PR to the
CAREamics source code and to the examples repository.

## Jupyter notebook applications

The pages in the application section are automatically generated from the Jupyter
notebooks in [careamics-example](https://github.com/CAREamics/careamics-examples/tree/main/applications) 
using [mkdocs-jupyter](https://github.com/danielfrg/mkdocs-jupyter).
A bash script (`scripts/check_out_examples.sh`) checks out the repository and copies 
all the notebooks referenced into the correct path in the application 
folder. Finally, the script `scripts/gen_jupyter_nav.py` creates entries for each notebook 
in the navigation file of mkdocs.


### Adding a new notebook

1. Add the notebook to `scripts/notebooks.json`. The structure is as follows:
    ```json
    {
        "applications": [
            ...,
            {
                "name": "Name_of_the_Notebook", // (1)!
                "description": "Some description of the notebook.", // (2)!
                "cover": "File_name.jpeg", // (3)!
                "source": "path/in/careamics-examples/repo/File_Name.ipynb", // (4)!
                "destination": "Category_in_Applications", // (5)!
                "tags": "3D, channels, fluorescence" // (6)!
            },
            // (7)!
        ]
        ...
    }
    ```
    1. `name` will be used as the name of the notebook, without the `_` when used as title
    and with `Name_of_the_Notebook.ipynb` as the file name after copy.
    2. `description` will be shown in the cards corresponding to the notebook on the website.
    3. `cover` is the name of the image file that will be used as the cover in the
    notebook card, the actual image should be placed in `docs/assets/notebook_covers`.
    4. `source` is the path to the notebook in the `careamics-examples` repository.
    5. `destination` is the category in the `applications` folder where the notebook will be copied to (e.g. all notebooks using `Noise2Void` are in the `Noise2Void` category).
    6. `tags` are shown on the notebook card and inform on the characteristic of the data. These are comma separated values.
    7. You can add multiple notebooks!
2. You can test that the notebook was correctly added by running `sh scripts/notebooks.sh` then `mkdocs serve`.


!!! info "Cell tags"
    By default, all notebook cell outputs are shown. To hide the output of a particular cell,
    add the tag `remove_output` to the cell. The `mkdocs.ynml` specifies that this 
    tag is used to hide cell outputs.

    For instance, this is useful for the training cell, which output hundreds of lines.


## Code reference

The code reference is generated using [mkdocstrings](https://mkdocstrings.github.io/), 
the script `scripts/checkout_repos.sh` and the page building script `scripts/gen_ref_pages.py`. 
To include a new package, simply add it to the `scripts/git_repositories.txt` file.

```
https://github.com/CAREamics/careamics
https://github.com/CAREamics/careamics-portfolio
<new project here>
```

## Updating the website version


In principle, when a new release of CAREamics is made, the state of the documentation
is saved into the corresponding version, and the documentation is tagged with the
next (ongoing) version.

For instance, the documentation is showing version `0.4`, upon release of version 
`0.4`, the state of the documentation is saved. The latest documentation is then 
tagged with version `0.5` (the next version) until this one is released.

In order to keep track of versions, we use [mike](https://github.com/jimporter/mike). 
We apply the following procedure:

1. Release version MAJOR.MINOR of CAREamics
2. Tag the latest documentation with version MAJOR.(MINOR+1)
  ```bash
  git tag MAJOR.(MINOR+1)
  git push --tags
  ```

To visualize the pages with the versions, you can use:

```bash
mike serve
```

### Correcting a version error

All the versions are stored in the `gh-pages` branch. If you made a mistake in the version
tagging, you can correct it by deleting the tag and pushing the changes.
