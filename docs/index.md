<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Overview

NeMo Relay is a portable execution runtime for agent systems that already have a
framework, model provider, policy layer, or observability backend. It gives those
systems one consistent way to describe what is happening when an agent crosses a
request, tool, or LLM boundary.

That layer is useful because agent applications rarely live inside one clean
abstraction. A production stack might combine NeMo Agent Toolkit, LangChain,
LangGraph, provider SDKs, custom harness code, NeMo Guardrails, tracing systems,
and evaluation pipelines. NeMo Relay sits underneath those choices as the shared
runtime contract for scopes, middleware, plugins, lifecycle events, adaptive
behavior, and observability. Under the NeMo Relay scope stack and middleware, the scoped execution path is referred to as work.

The result is a framework-neutral substrate for agent execution. Applications
keep their orchestration model, providers keep their native clients, and
middleware authors get one place to package policy, interception, telemetry, and
adaptive behavior across Rust, Python, and Node.js.

## Benefits

NeMo Relay is designed for teams that need agent runtime behavior to stay
consistent as applications grow across frameworks, languages, and deployment
targets.

- **Instrument once at the execution boundary**: Managed tool and LLM helpers
  attach work to the active scope, emit lifecycle events, and run the same
  middleware pipeline without scattering custom wrappers through every call site.
- **Keep concurrent agents isolated**: Hierarchical scopes preserve parent-child
  event relationships, expose request-local middleware and subscribers, and
  clean up scope-owned registrations when work finishes.
- **Turn policy into reusable runtime components**: Guardrails and intercepts can
  block work, sanitize observability payloads, transform requests, or wrap
  execution. Plugins package that behavior so applications and framework
  integrations can install it from configuration.
- **Export one event stream to many backends**: Subscribers consume the canonical
  lifecycle stream in-process or translate it to Agent Trajectory Interchange
  Format (ATIF) trajectories, OpenTelemetry traces, and OpenInference-compatible
  traces for debugging, evaluation, and production observability.
- **Adopt without replacing the stack**: NeMo Relay can sit below NeMo ecosystem
  components, third-party agent frameworks, provider adapters, or direct
  application code, so teams can add shared runtime semantics without a
  framework migration.
- **Share semantics across primary bindings**: The Rust core, Python wrapper,
  and Node.js binding expose the same execution model, which helps framework
  authors, plugin authors, and application teams reason about behavior
  consistently.

## What Should I Read First?

Use the reading path that matches your task:

| Task | Start With |
|---|---|
| Understand what NeMo Relay adds | [Agent Runtime Primer](getting-started/agent-runtime-primer.md) |
| Run a minimal example | [Quick Start](getting-started/quick-start.md) |
| Install packages | [Installation](getting-started/installation.md) |
| Develop from source | [Development Setup](contribute/development-setup.md) |
| Understand the runtime model | [Concepts](about/concepts/index.md) |
| Instrument an application | [Instrument Applications](instrument-applications/about.md) |
| Use a maintained integration | [Supported Integrations](integrations/about.md) |
| Integrate a framework | [Integrate into Frameworks](integrate-frameworks/about.md) |
| Observe a local coding-agent CLI | [NeMo Relay CLI](nemo-relay-cli/about.md) |
| Package reusable behavior | [Build Plugins](build-plugins/about.md) |
| Export traces or trajectories | [Observability](plugins/observability/about.md) |
| Debug production trace incidents | [Production Incident Runbook](troubleshooting/production-incident-runbook.md) |
| Tune performance with adaptive behavior | [Adaptive](plugins/adaptive/about.md) |
| Look up symbols | [APIs](reference/api/index.md) |

## Conceptual Diagram

The diagram below shows how applications, runtime components, and exporters
relate to each other. Scopes define where work belongs, middleware registries
define what runs around that work, and subscribers consume the lifecycle events
that the core emits.

```{mermaid}
flowchart TB
    Plugin[Plugin]
    App[Application Code / Agent Harness / Agent Framework]
    Framework[Framework Integration]

    subgraph Runtime[Runtime]
        PluginSystem[Plugin System]
        Bindings[Language Bindings]
        Core[Rust Core Runtime]
        Events[Lifecycle Events]

        subgraph RuntimeState[Runtime State]
            Registry[<strong>Middleware Registries</strong><br/>what runs around work]
            Scope[<strong>Scope Stack</strong><br/>where work belongs]
        end

        Subs[Subscribers / Exporters]

        PluginSystem --->|installs| Registry
        PluginSystem ----->|installs| Subs
        Bindings --> Core
        Core -->|emits| Events -->|consumed by| Subs
        Core -->|updates| Scope
        Core -->|resolves| Registry
    end

    App -->|registers| Plugin
    App -->|uses| Framework
    App -->|configures/initializes| PluginSystem
    App -->|uses| Bindings
    Framework -->|calls| Bindings
    Plugin -->|registers with| PluginSystem

    class Runtime grey-lightest;
    class RuntimeState grey-lightest;
    class App purple-lightest;
    class Framework yellow-lightest;
    class Plugin blue-lightest;
    class Bindings green-lightest;
    class PluginSystem green-light;
    class Core green-light;
    class Scope green-light;
    class Registry green-light;
    class Events green-light;
    class Subs green-light;
```


```{toctree}
:hidden:
:caption: About NeMo Relay
:maxdepth: 2

Overview <self>
Architecture <about/architecture>
Ecosystem <about/ecosystem>
about/concepts/index
about/release-notes/index
```

```{toctree}
:hidden:
:caption: Getting Started
:maxdepth: 2

Agent Runtime Primer <getting-started/agent-runtime-primer>
getting-started/prerequisites
getting-started/installation
Configuration / Setup <getting-started/configuration>
Quick Start <getting-started/quick-start>
```

```{toctree}
:hidden:
:caption: NeMo Relay CLI
:maxdepth: 2

About <nemo-relay-cli/about>
Basic Usage <nemo-relay-cli/basic-usage>
Claude Code <nemo-relay-cli/claude-code>
Codex <nemo-relay-cli/codex>
Cursor <nemo-relay-cli/cursor>
Hermes Agent <nemo-relay-cli/hermes>
```

```{toctree}
:hidden:
:caption: Supported Integrations
:maxdepth: 2

About <integrations/about>
OpenClaw Plugin Guide <integrations/openclaw-plugin>
LangChain Integration Guide <integrations/langchain>
LangGraph Integration Guide <integrations/langgraph>
Deep Agents Integration Guide <integrations/deepagents>
```

```{toctree}
:hidden:
:caption: Instrument Applications
:maxdepth: 2

About <instrument-applications/about>
Adding Scopes and Marks <instrument-applications/adding-scopes-and-marks>
Instrument a Tool Call <instrument-applications/instrument-tool-call>
Instrument an LLM Call <instrument-applications/instrument-llm-call>
Add Middleware <instrument-applications/advanced-guide>
Code Examples <instrument-applications/code-examples>
```

```{toctree}
:hidden:
:caption: Observability Plugin
:maxdepth: 2

About <plugins/observability/about>
Configuration <plugins/observability/configuration>
Agent Trajectory Interchange Format (ATIF) <plugins/observability/atif>
Agent Trajectory Observability Format (ATOF) <plugins/observability/atof>
OpenTelemetry <plugins/observability/opentelemetry>
OpenInference <plugins/observability/openinference>
```

```{toctree}
:hidden:
:caption: Adaptive Plugin
:maxdepth: 2

About <plugins/adaptive/about>
Configuration <plugins/adaptive/configuration>
Adaptive Cache Governor (ACG) <plugins/adaptive/acg>
Adaptive Hints <plugins/adaptive/adaptive-hints>
```

```{toctree}
:hidden:
:caption: Integrate into Frameworks
:maxdepth: 2

About <integrate-frameworks/about>
Adding Scopes <integrate-frameworks/adding-scopes>
Wrap Tool Calls <integrate-frameworks/wrap-tool-calls>
Wrap LLM Calls <integrate-frameworks/wrap-llm-calls>
Handle Non-Serializable Data <integrate-frameworks/non-serializable-data>
Using Codecs <integrate-frameworks/using-codecs>
Provider Codecs <integrate-frameworks/provider-codecs>
Provider Response Codecs <integrate-frameworks/provider-response-codecs>
Code Examples <integrate-frameworks/code-examples>
```

```{toctree}
:hidden:
:caption: Build Plugins
:maxdepth: 2

About <build-plugins/about>
Define a Plugin <build-plugins/basic-guide>
Validate Plugin Configuration <build-plugins/validate-configuration>
Plugin Configuration Files <build-plugins/plugin-configuration-files>
Register Plugin Behavior <build-plugins/register-behavior>
Design Plugin Configuration <build-plugins/advanced-configuration>
NeMo Guardrails Example Plugin <build-plugins/nemoguardrails>
Code Examples <build-plugins/code-examples>
```

```{toctree}
:hidden:
:caption: Contribute
:maxdepth: 2

About <contribute/about>
Development Setup <contribute/development-setup>
Workflow and Reviews <contribute/workflow-and-reviews>
Testing and Documentation <contribute/testing-and-docs>
```

```{toctree}
:hidden:
:caption: Reference
:maxdepth: 2

APIs <reference/api/index>
reference/performance
```

```{toctree}
:hidden:
:caption: Troubleshooting
:maxdepth: 2

Troubleshooting Guide <troubleshooting/troubleshooting-guide>
Production Incident Runbook <troubleshooting/production-incident-runbook>
```

```{toctree}
:hidden:
:caption: Resources
:maxdepth: 2

Support and FAQs <resources/support-and-faqs>
resources/glossary
resources/community
resources/legal/index
```
