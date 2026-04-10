<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Plugins

This guide is for two workflows:

- using plugins in an application
- writing a plugin kind that installs NeMo Flow middleware or subscribers

NeMo Flow has one shared plugin host. Adaptive is one plugin kind managed by
that host, and any custom plugin kinds are configured beside it as sibling
top-level components.

## Mental Model

- `PluginConfig.components` is the ordered list of active top-level plugin
  components
- each component has a `kind`, an `enabled` flag, and a component-local `config`
- adaptive uses its own `adaptive.ComponentSpec(...)`, but still lands in the
  same top-level `components` array
- custom plugins are registered once as plugin kinds, then activated by adding
  a matching component to `PluginConfig.components`

If you only need built-in adaptive behavior, configure the `adaptive` component.
If you want to install your own subscribers or intercepts, register a custom
plugin kind and add a component for it.

## Using Plugins

Using a plugin has three steps:

1. Register the plugin kind with the host.
2. Add a matching component to the top-level plugin config.
3. Validate, then initialize the full config.

### Python Example

```python
from nemo_flow import adaptive, plugin
from nemo_flow import JsonObject


class HeaderPlugin(plugin.Plugin):
    def validate(self, plugin_config: JsonObject) -> list[plugin.ConfigDiagnostic]:
        diagnostics: list[plugin.ConfigDiagnostic] = []
        priority = plugin_config.get("priority", 25)
        if not isinstance(priority, int):
            diagnostics.append(
                plugin.ConfigDiagnostic(
                    level="error",
                    code="example.invalid_priority",
                    message="priority must be an integer",
                    field="priority",
                )
            )
        return diagnostics

    def register(
        self,
        plugin_config: JsonObject,
        context: plugin.PluginContext,
    ) -> None:
        priority = int(plugin_config.get("priority", 25))

        def inject_header(name: str, args: JsonObject) -> JsonObject:
            return {
                **args,
                "x_plugin": "enabled",
                "tool": name,
            }

        context.register_tool_request_intercept("inject_header", priority, False, inject_header)


plugin.register("example.header_plugin", HeaderPlugin())

config = plugin.PluginConfig(
    components=[
        adaptive.ComponentSpec(
            adaptive.AdaptiveConfig(
                state=adaptive.StateConfig(
                    backend=adaptive.BackendSpec.in_memory()
                ),
                adaptive_hints=adaptive.AdaptiveHintsConfig(),
            )
        ),
        plugin.ComponentSpec(
            kind="example.header_plugin",
            config={"priority": 25},
        ),
    ]
)

report = plugin.validate(config)
if report.has_errors():
    raise RuntimeError(report.diagnostics)

await plugin.initialize(config)
```

### Operational Rules

- Register plugin kinds during process startup, before validation or
  initialization.
- `plugin.validate(...)` is pure validation. It does not change runtime state.
- `plugin.initialize(...)` replaces the active plugin configuration.
- `plugin.clear()` removes middleware and subscribers installed by the active
  plugin configuration.
- `plugin.list_kinds()` reports registered plugin kinds, not currently active
  components.
- Disabled components are still validated but skipped during registration.

### Composition With Adaptive

Adaptive and custom plugins are peers:

- adaptive contributes one top-level component with `kind = "adaptive"`
- custom plugins contribute separate top-level components with their own `kind`
- adaptive does not own nested hosted-plugin definitions

That means a mixed config looks like this:

```json
{
  "version": 1,
  "components": [
    {
      "kind": "adaptive",
      "enabled": true,
      "config": {
        "version": 1,
        "state": {
          "backend": {
            "kind": "in_memory",
            "config": {}
          }
        }
      }
    },
    {
      "kind": "example.header_plugin",
      "enabled": true,
      "config": {
        "priority": 25
      }
    }
  ]
}
```

## Writing Plugins

A plugin has two responsibilities:

- validate one component's `config`
- register runtime subscribers or intercepts for one enabled component instance

### Plugin Contract

- `validate(plugin_config)`
  Checks the component-local JSON config and returns diagnostics.
- `register(plugin_config, context)`
  Installs runtime behavior for that component.

Validation runs during both `plugin.validate(...)` and `plugin.initialize(...)`.
Registration only runs during `initialize(...)` for enabled components.

### What A Plugin Can Register

The plugin registration context exposes these operations:

- subscriber registration
- LLM request intercept registration
- LLM execution intercept registration
- LLM stream execution intercept registration
- tool request intercept registration
- tool execution intercept registration

The exact method names follow the binding you are using:

- Python: `register_subscriber`, `register_tool_request_intercept`, ...
- Node.js and WASM: `registerSubscriber`, `registerToolRequestIntercept`, ...
- Go: `RegisterSubscriber`, `RegisterToolRequestIntercept`, ...
- Rust: methods on `PluginRegistrationContext`

### Registration Naming

Registration names are component-local.

Use stable names like `inject_header` or `trace_export`, not globally unique
names. The plugin host namespaces them internally, so two different plugin
components can both register `inject_header` without colliding.

### Failure And Rollback

Plugin initialization is replace-with-rollback:

- if a component fails during registration, NeMo Flow removes partial
  registrations from that initialization attempt
- if there was a previous active plugin configuration, NeMo Flow tries to
  restore it

That means plugin handlers should be written as if registration is atomic:

- validate config early
- register through the provided context instead of calling global runtime APIs
  directly
- assume `register(...)` may run more than once across config reloads

### Authoring Guidelines

- Keep `validate(...)` focused on config shape and supported values.
- Return diagnostics with stable `code` values when possible.
- Treat `register(...)` as runtime wiring, not as a place to mutate global app
  state unrelated to NeMo Flow.
- Prefer stable, deterministic registration names inside one component.
- If ordering matters, document the expected `priority` values in the component
  config.

### WASM Caveat

WASM plugin handlers support the same high-level model, but
`registerLlmStreamExecutionIntercept(...)` does not chain downstream stream
handlers the same way the native bindings do. If your plugin depends on stream
intercept composition, prefer Rust, Python, Node.js, or Go.

## Language-Specific Examples

For binding-specific plugin examples and exact callback signatures, see:

- [Language Bindings](language-bindings.md)
- [Adaptive Layer](adaptive-layer.md)
- [API Reference](api-reference.md#plugin-host)
