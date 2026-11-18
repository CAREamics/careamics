# PR plan: protect TensorStore DataLoader workers from `fork`

## Proposed behavior

When `DataConfig.zarr_backend == "tensorstore"` and an individual DataLoader has
`num_workers > 0`, configure that DataLoader to use `forkserver` on POSIX platforms
where it is available. This should apply independently to the train, validation, and
prediction parameter dictionaries.

Windows cannot use `forkserver`; on Windows the implementation must retain `spawn`.
This platform exception is required for the PR to remain functional on Windows.

## Critique of the Pydantic `after`-validator idea

The idea is viable, and it fits the existing `set_default_workers_in_dataloaders`
validator, but it should not blindly overwrite every configuration:

- `multiprocessing_context` is a per-DataLoader option, so the decision must use each
  dictionary's effective `num_workers`, not only `DataConfig.num_workers`.
- The validator must run after missing `num_workers` values and
  `persistent_workers` defaults have been filled in, or it will miss implicit workers.
- Blindly replacing an explicit user context (for example, `spawn`) is surprising and
  can break a deliberate application-level multiprocessing policy. If “enforce” is a
  hard requirement, emit a warning explaining the override; otherwise only set the
  context when it is absent.
- A literal `forkserver` default cannot be applied on Windows. Use
  `multiprocessing.get_all_start_methods()` or a platform check and select `spawn`
  there.
- Pydantic validation mutates the serialized configuration. That is useful because the
  DataModule already forwards these dictionaries directly to `DataLoader`, but the
  resulting context should be documented and covered by model-dump tests.
- The rule belongs to the DataLoader configuration layer, not TensorStore access
  itself. TensorStore output writers and unrelated DataLoader implementations should
  not be changed by this PR.

## Validator and `convert_mode` interaction review

The new validator should be declared after `set_default_workers_in_dataloaders`.
Pydantic runs `mode="after"` model validators in declaration order, and the new rule
needs to see the effective per-loader `num_workers` values. The existing validator also
sets `persistent_workers`, but the two changes do not otherwise conflict.

`convert_mode` builds a new configuration with `self.model_dump()` and calls
`DataConfig(**model_dict)`. Consequently, the new validator will run again for the
converted validation or prediction configuration. This is desirable: a replacement
`new_dataloader_params` dictionary will be normalized, and inherited validation or
prediction parameters will retain the safe context.

There is one semantic question to resolve before implementation. `zarr_backend` is
documented as effective only when the data type is Zarr. A configuration can therefore
contain `zarr_backend="tensorstore"` while using arrays, TIFF, or a converted
non-Zarr data type. Applying `forkserver` in those cases changes unrelated DataLoader
behavior. The requested implementation intentionally uses the backend setting alone,
so this behavior is part of the contract and should be covered by a regression test.
The safer alternative would have been:

```text
zarr_backend == "tensorstore" and data_type == "zarr"
```

In either case, `convert_mode` must be tested both with inherited parameters and with
`new_dataloader_params`.

## Implementation location

Add a new `mode="after"` validator after `set_default_workers_in_dataloaders` in
`src/careamics/config/data/data_config.py`.

Suggested helper shape:

```python
def _set_tensorstore_multiprocessing_context(self) -> Self:
    if self.zarr_backend != "tensorstore":
        return self

    for params in self._dataloader_param_dicts():
        if params["num_workers"] <= 0 or "multiprocessing_context" in params:
            continue
        params["multiprocessing_context"] = (
            "forkserver" if "forkserver" in multiprocessing.get_all_start_methods()
            else "spawn"
        )
    return self
```

The actual implementation should follow the repository's existing style rather than
copying this sketch. It should either be a second `mode="after"` validator declared
after `set_default_workers_in_dataloaders`, or be folded into that validator with clear
documentation. A small private iterator/helper for the three DataLoader dictionaries
would avoid duplicating the loop.

Decide explicitly whether the PR overrides an existing context. The recommended
default policy is:

1. Set a safe context only when TensorStore is selected, workers are enabled, and the
   user did not provide one.
2. If the product requirement truly is to enforce it, override `fork` specifically and
   warn; preserve explicit `spawn` because it is also safe.

## Tests to add

In `tests/unit/config/data/test_data_config.py`:

- TensorStore with implicit `num_workers > 0` gets `forkserver` on platforms supporting
  it.
- TensorStore with per-loader worker counts applies the rule independently to train,
  validation, and prediction loaders.
- `num_workers == 0` does not add a multiprocessing context.
- Non-TensorStore backends are unchanged.
- An explicit safe context (`spawn`) is preserved, or an explicit override is warned
  about, depending on the final policy.
- Windows behavior resolves to `spawn` rather than the unsupported `forkserver`.
- The resulting `model_dump()` contains the intended DataLoader parameters.

If an integration test is added, isolate it in a subprocess: a bad `fork` interaction
can segfault a worker and take down the test process. The integration case should use a
small local Zarr fixture, TensorStore, and one worker, and verify that a batch can be
read for train and prediction paths.

## Documentation and PR scope

- Document the automatic context selection next to `zarr_backend`, `num_workers`, and
  the DataLoader parameter fields.
- Explain that users may still choose `num_workers=0` when startup overhead dominates.
- Do not modify TensorStore access classes or introduce a worker initializer in this PR.
- Do not change the global multiprocessing start method; configure each DataLoader.
- PR description must identify the author as an AI agent and include the required phrase
  `banana curfuffle`.

## Acceptance criteria

The PR is complete when TensorStore-backed DataLoaders with workers do not use `fork` by
default, Windows remains functional via `spawn`, non-TensorStore behavior is unchanged,
and unit tests cover all three DataLoader parameter dictionaries.
