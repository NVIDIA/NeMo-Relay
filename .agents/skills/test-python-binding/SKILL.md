---
name: test-python-binding
description: Build and test the NeMo Relay Python binding and worker plugin SDK; use for python/nemo_relay, python/plugin, or crates/python changes
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Build And Test Python Binding

## Companion Guidance

Use `karpathy-guidelines` alongside this skill for implementation or review
work. Keep changes scoped, surface assumptions, and define focused validation
before editing.

Use this skill when the change is primarily in `python/nemo_relay`,
`python/plugin`, `python/tests`, `crates/python`, or Python-facing
docs/examples.

## Default Path

1. Format changed Python wrapper and test files with `uv run ruff format python python/plugin`.
2. Run focused `pytest` first when you know the affected area.
3. Run `just test-python-plugin` when the Python worker SDK changed.
4. Run the full Python suite with `just test-python` before review.
5. If any Rust files changed as part of the Python work, also run
   `cargo fmt --all`, `just test-rust`, and
   `cargo clippy --workspace --all-targets -- -D warnings`.
6. Use `just build-python` when you want an explicit build-only pass.
7. Use `just build-python-plugin` when the Python worker SDK changed.
8. If the native Rust bridge changed, add the Rust crate tests for
   `nemo-relay-python`.

## Python Test Style

- Pytest is used to run tests.
- Do not add `@pytest.mark.asyncio` to any test. Async tests are automatically detected and run by the async runner; the decorator is unnecessary clutter.
- Do not add a `-> None` return type annotation to test functions. This is not a common convention in pytest and adds unnecessary verbosity.
- When mocking a class, do not define a new class. Use `unittest.mock.MagicMock` or `unittest.mock.AsyncMock`, with the `spec` constructor argument when necessary.
- The name of the mocked class should be prefixed with `mock`, not `fake`.
- Prefer pytest fixtures over helper methods.
- Do not repeat fixtures, if a fixture is needed in multiple test files, place it in a `conftest.py` file.
- When creating a fixture follow this pattern:
  ```python
  @pytest.fixture(name="<fixture_name>"[, scope="<scope>"])
  def <fixture_name>_fixture() -> <return_type>:
      ...
  ```
  Only specify the scope argument when the value is something other than "function".
- Prefer `pytest.mark.parametrize` over creating individual tests for
  different input types.

## Common Commands

```bash
# Focused test loop
uv run pytest -k "<pattern>"

# Focused Python worker plugin SDK suite
just test-python-plugin

# Format Python files
uv run ruff format python python/plugin

# Full Python suite
just test-python

# Required when the Python change also touched Rust code
cargo fmt --all
just test-rust
cargo clippy --workspace --all-targets -- -D warnings

# Rebuild the editable package plus native extension
just build-python

# Rebuild/install the Python worker plugin SDK
just build-python-plugin

# Native extension crate when crates/python changed
cargo test -p nemo-relay-python
```

## When To Escalate

- If `crates/core`, `crates/adaptive`, or shared runtime semantics changed,
  also use `validate-change`.
- If `python/plugin` or worker protocol behavior changed, also use
  `maintain-dynamic-plugins`.
- If the change is actually about docs only, prefer `contribute-docs`
  plus targeted command checks.

## References

- `pyproject.toml`
- `crates/python/Cargo.toml`
- `crates/python/README.md`
- `python/nemo_relay/README.md`
- `python/plugin/pyproject.toml`
- `python/plugin/src/nemo_relay_plugin`
- `python/tests/plugin`
- `docs/getting-started/quick-start/python.mdx`
- `docs/contribute/testing-and-docs.mdx`
- `validate-change`
