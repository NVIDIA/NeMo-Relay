<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Register Plugin Behavior

Use this guide when plugin config validation is in place and you need the plugin to install real NeMo Relay runtime behavior.

## What You Build

You will register a plugin kind, initialize a validated config, install subscribers or middleware through `PluginContext`, and clear active plugin configuration during teardown.

## Use PluginContext

`PluginContext` is the component-scoped registration surface passed to the plugin during initialization. Register subscribers, guardrails, and intercepts through this context rather than through global registration calls inside application startup.

That gives the plugin system three important guarantees:

- Runtime names can be qualified for the component instance.
- Partial setup can be rolled back if one registration fails.
- Activation reports can identify which configured component installed behavior.

Use the context only after validation succeeds. Validation should inspect config and return diagnostics; registration should create runtime objects and attach them to the context.

## Activation APIs

Use the plugin APIs in this order:

1. Register the plugin kind.
2. Build a `PluginConfig`.
3. Validate the config.
4. Initialize the config.
5. Inspect the activation report.
6. Clear active config during teardown when needed.

::::{tab-set}
:sync-group: language

:::{tab-item} Python
:sync: python

```python
import nemo_relay

config = nemo_relay.plugin.PluginConfig()
config.components = [
    nemo_relay.plugin.ComponentSpec(
        kind="header-plugin",
        config={"header_name": "x-tenant", "value": "tenant-a"},
    )
]

report = nemo_relay.plugin.validate(config)
active_report = await nemo_relay.plugin.initialize(config)
kinds = nemo_relay.plugin.list_kinds()
nemo_relay.plugin.clear()
```

:::

:::{tab-item} Node.js
:sync: node

```ts
import * as plugin from 'nemo-relay-node/plugin';

const config = plugin.defaultConfig();
config.components = [
  plugin.ComponentSpec(
    'header-plugin',
    { header_name: 'x-tenant', value: 'tenant-a' },
    { enabled: true },
  ),
];

const report = plugin.validate(config);
const activeReport = await plugin.initialize(config);
const kinds = plugin.listKinds();
plugin.clear();
```

:::

:::{tab-item} Rust
:sync: rust

```rust
use nemo_relay::plugin::{
    clear_plugin_configuration, initialize_plugins, list_plugin_kinds, validate_plugin_config,
    PluginComponentSpec, PluginConfig,
};

let mut config = PluginConfig::default();
let mut component = PluginComponentSpec::new("header-plugin");
component.config.insert("header_name".into(), "x-tenant".into());
component.config.insert("value".into(), "tenant-a".into());
config.components.push(component);

let report = validate_plugin_config(&config);
let active_report = initialize_plugins(config).await?;
let kinds = list_plugin_kinds();
clear_plugin_configuration()?;
```

:::

::::

## Header Plugin Example

The same model applies in every binding: validate component-local config, then install middleware through the component-scoped registration context.

::::{tab-set}
:sync-group: language

:::{tab-item} Python
:sync: python

```python
import nemo_relay


class HeaderPlugin:
    def validate(self, plugin_config):
        if "header_name" not in plugin_config or "value" not in plugin_config:
            return [{
                "level": "error",
                "code": "header-plugin.invalid_config",
                "message": "header_name and value are required",
            }]
        return []

    def register(self, plugin_config, context):
        def add_header(name, request, annotated):
            request.headers[plugin_config["header_name"]] = plugin_config["value"]
            return request, annotated

        context.register_llm_request_intercept("inject-header", 100, False, add_header)


nemo_relay.plugin.register("header-plugin", HeaderPlugin())
```

:::

:::{tab-item} Node.js
:sync: node

```ts
import * as plugin from 'nemo-relay-node/plugin';

const headerPlugin: plugin.Plugin = {
  validate(pluginConfig) {
    if (typeof pluginConfig.header_name !== 'string' || typeof pluginConfig.value !== 'string') {
      return [{
        level: 'error',
        code: 'header-plugin.invalid_config',
        message: 'header_name and value are required',
      }];
    }
    return [];
  },
  register(pluginConfig, context) {
    context.registerLlmRequestIntercept('inject-header', 100, false, ({ request, annotated }) => ({
      request: {
        ...request,
        headers: {
          ...(request.headers as Record<string, string>),
          [String(pluginConfig.header_name)]: String(pluginConfig.value),
        },
      },
      annotated,
    }));
  },
};

plugin.register('header-plugin', headerPlugin);
```

:::

:::{tab-item} Rust
:sync: rust

```rust
use nemo_relay::plugin::{
    register_plugin, ConfigDiagnostic, DiagnosticLevel, Plugin, PluginRegistrationContext,
    Result as PluginResult,
};
use serde_json::{Map, Value as Json};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

struct HeaderPlugin;

impl Plugin for HeaderPlugin {
    fn plugin_kind(&self) -> &str {
        "header-plugin"
    }

    fn validate(&self, plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        let mut diagnostics = Vec::new();

        for field in ["header_name", "value"] {
            match plugin_config.get(field) {
                Some(Json::String(_)) => {}
                Some(_) => diagnostics.push(ConfigDiagnostic {
                    level: DiagnosticLevel::Error,
                    code: "header-plugin.invalid_config".into(),
                    component: Some("header-plugin".into()),
                    field: Some(field.into()),
                    message: format!("{field} must be a string"),
                }),
                None => diagnostics.push(ConfigDiagnostic {
                    level: DiagnosticLevel::Error,
                    code: "header-plugin.invalid_config".into(),
                    component: Some("header-plugin".into()),
                    field: Some(field.into()),
                    message: format!("{field} is required"),
                }),
            }
        }

        diagnostics
    }

    fn register<'a>(
        &'a self,
        plugin_config: &Map<String, Json>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = PluginResult<()>> + Send + 'a>> {
        let header_name = plugin_config
            .get("header_name")
            .and_then(Json::as_str)
            .unwrap_or("x-plugin")
            .to_string();
        let header_value = plugin_config
            .get("value")
            .and_then(Json::as_str)
            .unwrap_or("enabled")
            .to_string();

        Box::pin(async move {
            ctx.register_llm_request_intercept(
                "inject-header",
                100,
                false,
                Arc::new(move |_name, mut request, annotated| {
                    request
                        .headers
                        .insert(header_name.clone(), header_value.clone().into());
                    Ok((request, annotated))
                }),
            )?;
            Ok(())
        })
    }
}

register_plugin(Arc::new(HeaderPlugin))?;
```

:::

::::

## Registration Checklist

Before publishing or sharing a plugin:

1. Validate a correct config and confirm no errors are reported.
2. Validate an intentionally invalid config and confirm diagnostics are actionable.
3. Initialize the plugin and verify the expected subscribers or middleware run.
4. Force one registration failure and confirm partial setup is rolled back.
5. Deregister or clear active config during teardown.

## Common Issues

Check these symptoms first when the workflow does not behave as expected.

- **Middleware names collide**: Use component-local names and let the plugin runtime qualify them.
- **Partial registrations remain after failure**: Register through `PluginContext` so rollback can clean up.
- **Registration does validation work**: Move deterministic checks into the validation hook.
- **Global state leaks across component instances**: Create instance-local state during registration or key shared state by component identity.

## Next Steps

Use these links to continue from this workflow into the next related task.

- Add advanced validation and rollout controls with [Design Plugin Configuration](advanced-configuration.md).
- Review concrete authoring patterns in [Code Examples](code-examples.md).
