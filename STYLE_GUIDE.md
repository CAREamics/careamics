# Style conventions

## Docstrings

We use `numpydoc` style for docstrings. Please follow the conventions described in the
[numpydoc documentation](https://numpydoc.readthedocs.io/en/latest/format.html).

We write out third party types with the library name, e.g. `numpy.NDArray`.

## Protocols vs inheritance

When defining a new class, prefer using Protocols over inheritance when possible.

## Naming

### Protocols and implementations

- Protocols should not have `_protocol` in their module name nor `Protocol` in their class name.
- Protocol implementation should be named `<Implementation><ProtocolName>` and be in a module
named `<implementation>_<protocol_name>.py`.