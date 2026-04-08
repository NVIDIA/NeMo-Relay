<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Optimizer Layer

The optimizer layer is a config-driven runtime. Callers build an
`OptimizerConfig`, optionally attach shared state, select built-in or hosted
components, validate the document, and then manage lifecycle through
`OptimizerRuntime`.

## Public Model

The stable public boundary is:

- `OptimizerConfig`
- `StateConfig`
- `BackendSpec`
- `ComponentSpec`
- `ConfigPolicy`
- `ConfigReport`
- `OptimizerRuntime`
- hosted plugin registration functions in each supported language

Built-in typed helpers are convenience layers over the same dynamic config:

- `TelemetryComponent` / `TelemetryComponentConfig`
- `DynamoHintsComponent` / `DynamoHintsComponentConfig`
- `ToolParallelismComponent` / `ToolParallelismComponentConfig`
- `ExternalComponent` / `ExternalComponentConfig`

## Architecture

```mermaid
flowchart LR
    subgraph Hosts
        PY[Python]
        JS[Node.js]
        GO[Go]
        WA[WASM]
        RS[Rust]
    end

    CFG[OptimizerConfig]
    VAL[validate_config]
    RT[OptimizerRuntime]

    subgraph Registries
        BUILTIN[Built-in component registry]
        HOSTED[Hosted plugin registry]
    end

    STATE[Optional state backend]

    subgraph Components
        TEL[telemetry]
        DYN[dynamo_hints]
        TOOL[tool_parallelism]
        EXT[external_component]
    end

    subgraph Core
        SUB[subscribers]
        LLMREQ[LLM request intercepts]
        LLMEXEC[LLM execution intercepts]
        LLMSTREAM[LLM stream execution intercepts]
        TOOLREQ[tool request intercepts]
        TOOLEXEC[tool execution intercepts]
    end

    PY --> CFG
    JS --> CFG
    GO --> CFG
    WA --> CFG
    RS --> CFG

    CFG --> VAL --> RT
    RT --> BUILTIN
    RT --> HOSTED
    RT --> STATE

    BUILTIN --> TEL
    BUILTIN --> DYN
    BUILTIN --> TOOL
    BUILTIN --> EXT
    HOSTED --> EXT

    TEL --> SUB
    DYN --> LLMREQ
    TOOL --> TOOLEXEC
    EXT --> SUB
    EXT --> LLMREQ
    EXT --> LLMEXEC
    EXT --> LLMSTREAM
    EXT --> TOOLREQ
    EXT --> TOOLEXEC
```

## Built-In Components

The initial built-in component set is:

- `telemetry`
  Registers the event subscriber and drives the learning/drain pipeline.
- `dynamo_hints`
  Registers the LLM request intercept that injects `AgentHints`.
- `tool_parallelism`
  Registers the tool execution intercept for built-in scheduling paths.
- `external_component`
  Activates a previously registered hosted plugin handler.

Components are selected dynamically by `kind`. Unsupported component kinds or
unknown fields yield diagnostics according to `ConfigPolicy`.

## Runtime Lifecycle By Language

### Python

```python
from nat_nexus.optimizer import (
    BackendSpec,
    DynamoHintsComponent,
    OptimizerConfig,
    OptimizerRuntime,
    StateConfig,
    TelemetryComponent,
    ToolParallelismComponent,
)

runtime = OptimizerRuntime(
    OptimizerConfig(
        state=StateConfig(backend=BackendSpec.in_memory()),
        components=[
            TelemetryComponent(learners=["latency_sensitivity"]),
            DynamoHintsComponent(),
            ToolParallelismComponent(),
        ],
    )
)

report = runtime.report()
await runtime.register()
runtime.deregister()
await runtime.shutdown()
```

### Node.js

```javascript
import {
  OptimizerRuntime,
  defaultOptimizerConfig,
  optimizerInMemoryBackend,
  telemetryComponent,
  dynamoHintsComponent,
  toolParallelismComponent,
} from "./typed.js";

const config = defaultOptimizerConfig();
config.state = { backend: optimizerInMemoryBackend() };
config.components = [
  telemetryComponent({ learners: ["latency_sensitivity"] }),
  dynamoHintsComponent(),
  toolParallelismComponent(),
];

const runtime = new OptimizerRuntime(config);
const report = await runtime.report();
await runtime.register();
await runtime.deregister();
await runtime.shutdown();
```

### Go

```go
config := nat_nexus.NewOptimizerConfig()
config.State = &nat_nexus.OptimizerStateConfig{
    Backend: nat_nexus.NewInMemoryOptimizerBackend(),
}
config.Components = []nat_nexus.OptimizerComponentSpec{
    nat_nexus.TelemetryComponent(nat_nexus.TelemetryComponentConfig{
        Learners: []string{"latency_sensitivity"},
    }),
    nat_nexus.DynamoHintsComponent(nat_nexus.NewDynamoHintsComponentConfig()),
    nat_nexus.ToolParallelismComponent(nat_nexus.NewToolParallelismComponentConfig()),
}

runtime, err := nat_nexus.NewOptimizerRuntime(config)
if err != nil {
    panic(err)
}
defer runtime.Close()

report, err := runtime.Report()
if err != nil {
    panic(err)
}
_ = report

if err := runtime.Register(); err != nil {
    panic(err)
}
if err := runtime.Deregister(); err != nil {
    panic(err)
}
if err := runtime.Shutdown(); err != nil {
    panic(err)
}
```

### WebAssembly

```javascript
import init, {
  OptimizerRuntime,
  validateOptimizerConfig,
} from "./pkg/nvidia_nat_nexus_wasm.js";

await init();

const config = {
  version: 1,
  state: {
    backend: { kind: "in_memory", config: {} },
  },
  components: [
    { kind: "telemetry", enabled: true, config: { learners: ["latency_sensitivity"] } },
    { kind: "dynamo_hints", enabled: true, config: {} },
    { kind: "tool_parallelism", enabled: true, config: {} },
  ],
};

const report = validateOptimizerConfig(config);
const runtime = new OptimizerRuntime(config);
await runtime.register();
runtime.deregister();
await runtime.shutdown();
```

### Rust

```rust
use nvidia_nat_nexus_optimizer::{
    BackendSpec, DynamoHintsComponentConfig, OptimizerConfig, OptimizerRuntime, StateConfig,
    TelemetryComponentConfig, ToolParallelismComponentConfig,
};

let mut runtime = OptimizerRuntime::new(OptimizerConfig {
    state: Some(StateConfig {
        backend: BackendSpec::in_memory(),
    }),
    components: vec![
        TelemetryComponentConfig {
            subscriber_name: None,
            learners: vec!["latency_sensitivity".into()],
        }
        .into(),
        DynamoHintsComponentConfig::default().into(),
        ToolParallelismComponentConfig::default().into(),
    ],
    ..OptimizerConfig::default()
})
.await?;

let report = runtime.report().clone();
runtime.register().await?;
runtime.deregister()?;
runtime.shutdown().await?;
```

## Hosted Plugins By Language

Hosted plugins are intentionally narrow. They can register:

- event subscribers
- LLM request intercepts
- LLM execution intercepts
- LLM stream execution intercepts
- tool request intercepts
- tool execution intercepts

They do not receive direct access to optimizer internals such as persistence
backends or hot-cache state.

### Python

```python
from nat_nexus import LLMRequest
from nat_nexus.optimizer import (
    ExternalComponent,
    OptimizerConfig,
    OptimizerRuntime,
    register_optimizer_plugin,
)

class HeaderPlugin:
    def validate(self, instance_id, plugin_config):
        return []

    def register(self, instance_id, plugin_config, context):
        def intercept(_name, request, annotated):
            headers = dict(request.headers)
            headers["x-plugin"] = instance_id
            return LLMRequest(headers, request.content), annotated

        context.register_llm_request_intercept(
            f"{instance_id}.header",
            25,
            False,
            intercept,
        )

register_optimizer_plugin("example.header_plugin", HeaderPlugin())

runtime = OptimizerRuntime(
    OptimizerConfig(
        components=[
            ExternalComponent(
                plugin_kind="example.header_plugin",
                instance_id="plugin-1",
            )
        ]
    )
)
```

### Node.js

```javascript
import {
  OptimizerRuntime,
  defaultOptimizerConfig,
  externalComponent,
  registerOptimizerPlugin,
} from "./typed.js";

registerOptimizerPlugin("example.header_plugin", {
  validate(instanceId, pluginConfig) {
    return [];
  },
  register(instanceId, pluginConfig, context) {
    context.registerLlmRequestIntercept(
      `${instanceId}.header`,
      25,
      false,
      (name, request, annotated) => [
        {
          headers: { ...request.headers, "x-plugin": instanceId },
          content: request.content,
        },
        annotated,
      ],
    );
  },
});

const config = defaultOptimizerConfig();
config.components = [externalComponent("example.header_plugin", "plugin-1", {})];
const runtime = new OptimizerRuntime(config);
```

### Go

```go
pluginKind := "example.header_plugin"
err := nat_nexus.RegisterOptimizerPlugin(pluginKind, nat_nexus.OptimizerPluginHandlerFuncs{
    ValidateFunc: func(instanceID string, pluginConfig map[string]any) ([]nat_nexus.OptimizerConfigDiagnostic, error) {
        return nil, nil
    },
    RegisterFunc: func(instanceID string, pluginConfig map[string]any, ctx *nat_nexus.OptimizerPluginContext) error {
        return ctx.RegisterLlmRequestIntercept(
            instanceID+".header",
            25,
            false,
            func(name string, request map[string]any, annotated map[string]any) (map[string]any, map[string]any, error) {
                headers, _ := request["headers"].(map[string]any)
                if headers == nil {
                    headers = map[string]any{}
                }
                headers["x-plugin"] = instanceID
                request["headers"] = headers
                return request, annotated, nil
            },
        )
    },
})
if err != nil {
    panic(err)
}

config := nat_nexus.NewOptimizerConfig()
config.Components = []nat_nexus.OptimizerComponentSpec{
    nat_nexus.ExternalComponent(nat_nexus.ExternalComponentConfig{
        PluginKind: pluginKind,
        InstanceID: "plugin-1",
    }),
}
```

### WebAssembly

```javascript
import init, {
  OptimizerRuntime,
  registerOptimizerPlugin,
} from "./pkg/nvidia_nat_nexus_wasm.js";

await init();

registerOptimizerPlugin("example.header_plugin", {
  validate(instanceId, pluginConfig) {
    return [];
  },
  register(instanceId, pluginConfig, context) {
    context.registerLlmRequestIntercept(
      `${instanceId}.header`,
      25,
      false,
      (name, request, annotated) => [
        {
          headers: { ...request.headers, "x-plugin": instanceId },
          content: request.content,
        },
        annotated,
      ],
    );
  },
});

const runtime = new OptimizerRuntime({
  version: 1,
  components: [
    {
      kind: "external_component",
      enabled: true,
      config: {
        plugin_kind: "example.header_plugin",
        instance_id: "plugin-1",
      },
    },
  ],
});
```

### Rust

```rust
use std::sync::Arc;

use nvidia_nat_nexus_optimizer::{
    register_hosted_plugin_handler, ConfigDiagnostic, HostedPluginHandler,
    HostedRegistrationContext, Result,
};
use serde_json::{Map, Value as Json};

struct HeaderPlugin;

impl HostedPluginHandler for HeaderPlugin {
    fn plugin_kind(&self) -> &str {
        "example.header_plugin"
    }

    fn validate(&self, _instance_id: &str, _plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        vec![]
    }

    fn register(
        &self,
        instance_id: &str,
        _plugin_config: &Map<String, Json>,
        ctx: &mut HostedRegistrationContext,
    ) -> Result<()> {
        let name = format!("{instance_id}.header");
        ctx.register_llm_request_intercept(
            &name,
            25,
            false,
            Arc::new(|_name, request, annotated| Box::pin(async move { Ok((request, annotated)) })),
        )
    }
}

register_hosted_plugin_handler(Arc::new(HeaderPlugin))?;
```

## Validation By Language

Use validation before registration when you want compatibility warnings without
constructing a running optimizer:

- Python: `validate_optimizer_config(config)`
- Node.js: `validateOptimizerConfig(config)`
- Go: `ValidateOptimizerConfig(config)`
- WebAssembly: `validateOptimizerConfig(config)`
- Rust: `OptimizerRuntime::validate_config(&config)`

Unknown component kinds and unknown fields are expected to remain forward
compatible. They should usually warn rather than break callers, unless the
selected policy makes them strict.
