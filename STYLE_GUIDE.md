# Style conventions

## Docstrings

We use `numpydoc` style for docstrings. Please follow the conventions described in the
[numpydoc documentation](https://numpydoc.readthedocs.io/en/latest/format.html).

We write out third party types with the library name, e.g. `numpy.NDArray`.

## Imports

Absolute imports must always be used over relative imports unless the import is from a module a the same directory level, e.g. `from .foo import bar`.

## Protocols vs inheritance

When defining a new class, prefer using Protocols over inheritance when possible.

## Naming

### Protocols and implementations

- Protocols should not have `_protocol` in their module name nor `Protocol` in their class name.
- Protocol implementation should be named `<Implementation><ProtocolName>` and be in a module
named `<implementation>_<protocol_name>.py`.

## Module structure

For source code (i.e. excluding test files and \_\_init\_\_.py files have their own conventions) prefer to have the most important functions/classes at the top of a module and helper/utility functions at the bottom.

If a function 1 calls function 2 from within the same module, function 2 should be preferably be placed below function 2.
This also applies for methods within classes.

For classes, all double underscore methods should be defined first, followed by properties, followed by public methods, followed by private methods.