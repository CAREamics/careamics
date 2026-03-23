---
description: Docstring conventions
---
# Docstring conventions

CAREamics follows the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) 
docstring conventions. The is enforced by the use of the `numpydoc` 
[pre-commit hook](https://github.com/CAREamics/careamics/blob/955d1e91204fc041736f0716a6b0813c8d6448d0/.pre-commit-config.yaml).

On top of the numpy conventions, we try to build a more human readable docstring by adapting
principles from [Pandas docstring](https://pandas.pydata.org/docs/development/contributing_docstring.html).

## Parameter declaration

``` python
"""
	param : Union[str, int] ❌
	param : str or int ✅
	param : str | int ✅

	choice : Literal["a", "b", "c"] ❌
	choice : {"a", "b", "c"} ✅

	param : Tuple[int, int] ❌
	param : (int, int) ✅

	sequence: List[int] ❌
	sequence: list of int ✅

	param : int 
		The default is 1. ❌
	param : int, default=1 ✅

	param : Optional[int] ❌
	param : int, optional ✅
"""
```

## Third-party types

``` python
"""
	param : pandas.DataFrame
	param : NDArray
	param : torch.Tensor
	param : tensorflow.Tensor
"""
```
