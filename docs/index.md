<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Overview

NeMo Flow is a portable execution runtime for agent systems that already have a
framework, model provider, policy layer, or observability backend. It gives those
systems one consistent way to describe what is happening when an agent crosses a
request, tool, or LLM boundary.

That layer is useful because agent applications rarely live inside one clean
abstraction. A production stack might combine NeMo Agent Toolkit, LangChain,
LangGraph, provider SDKs, custom harness code, NeMo Guardrails, tracing systems,
and evaluation pipelines. NeMo Flow sits underneath those choices as the shared
runtime contract for scopes, middleware, plugins, lifecycle events, adaptive
behavior, and observability. Under the NeMo Flow scope stack and middleware, the scoped execution path is referred to as work.

The result is a framework-neutral substrate for agent execution. Applications
keep their orchestration model, providers keep their native clients, and
middleware authors get one place to package policy, interception, telemetry, and
adaptive behavior across Rust, Python, and Node.js.

## Benefits

NeMo Flow is designed for teams that need agent runtime behavior to stay
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
  lifecycle stream in-process or translate it to ATIF trajectories,
  OpenTelemetry traces, and OpenInference-compatible traces for debugging,
  evaluation, and production observability.
- **Adopt without replacing the stack**: NeMo Flow can sit below NeMo ecosystem
  components, third-party agent frameworks, provider adapters, or direct
  application code, so teams can add shared runtime semantics without a
  framework migration.
- **Share semantics across primary bindings**: The Rust core, Python wrapper,
  and Node.js binding expose the same execution model, which helps framework
  authors, plugin authors, and application teams reason about behavior
  consistently.

## Use Cases

These paths map common reader goals to the most relevant documentation entry points.

- **End Users**: Start with [Prerequisites](getting-started/prerequisites.md) and [Quick Start](getting-started/quick-start.md).
- **Agent Framework Developers**: Start with [Integrate into Frameworks](integrate-frameworks/about.md).
- **Plugin Writers**: Start with [Build Plugins](build-plugins/about.md), then continue to [Basic Guide: Define a Plugin](build-plugins/basic-guide.md).
- **Contributors**: Start with [Contribute](contribute/about.md) and the repository root `CONTRIBUTING.md` guide.

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
:caption: About NeMo Flow
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

getting-started/prerequisites
getting-started/installation
Configuration / Setup <getting-started/configuration>
Quick Start <getting-started/quick-start>
```

```{toctree}
:hidden:
:caption: Instrument Applications
:maxdepth: 2

About <instrument-applications/about>
Basic Guide: Adding Scopes and Marks <instrument-applications/adding-scopes-and-marks>
Basic Guide: Instrument a Tool Call <instrument-applications/instrument-tool-call>
Basic Guide: Instrument an LLM Call <instrument-applications/instrument-llm-call>
Advanced Guide: Add Middleware <instrument-applications/advanced-guide>
Code Examples <instrument-applications/code-examples>
```

```{toctree}
:hidden:
:caption: Integrate into Frameworks
:maxdepth: 2

About <integrate-frameworks/about>
Basic Guide: Adding Scopes <integrate-frameworks/adding-scopes>
Basic Guide: Wrap Tool Calls <integrate-frameworks/wrap-tool-calls>
Basic Guide: Wrap LLM Calls <integrate-frameworks/wrap-llm-calls>
Advanced Guide: Coding-Agent Gateway Sidecar <integrate-frameworks/coding-agent-sidecar>
Claude Code Sidecar Guide <integrate-frameworks/coding-agent-claude-code>
Codex Sidecar Guide <integrate-frameworks/coding-agent-codex>
Cursor Sidecar Guide <integrate-frameworks/coding-agent-cursor>
Hermes Sidecar Guide <integrate-frameworks/coding-agent-hermes>
Advanced Guide: Handle Non-Serializable Data <integrate-frameworks/non-serializable-data>
Advanced Guide: Using Codecs <integrate-frameworks/using-codecs>
Advanced Guide: Provider Codecs <integrate-frameworks/provider-codecs>
Advanced Guide: Provider Response Codecs <integrate-frameworks/provider-response-codecs>
Code Examples <integrate-frameworks/code-examples>
```

```{toctree}
:hidden:
:caption: Build Plugins
:maxdepth: 2

About <build-plugins/about>
Basic Guide: Define a Plugin <build-plugins/basic-guide>
Basic Guide: Validate Plugin Configuration <build-plugins/validate-configuration>
Basic Guide: Register Plugin Behavior <build-plugins/register-behavior>
Advanced Guide: Design Plugin Configuration <build-plugins/advanced-configuration>
Code Examples <build-plugins/code-examples>
```

```{toctree}
:hidden:
:caption: Export Observability Data
:maxdepth: 2

About <export-observability-data/about>
Basic Guide: Register a Subscriber <export-observability-data/basic-guide>
Advanced Guide: Export OpenTelemetry Data <export-observability-data/opentelemetry>
Advanced Guide: Export OpenInference Data <export-observability-data/advanced-guide>
Advanced Guide: Export ATIF <export-observability-data/atif>
Code Examples <export-observability-data/code-examples>
```

```{toctree}
:hidden:
:caption: Use Adaptive Optimization
:maxdepth: 2

About <use-adaptive-optimization/about>
Basic Guide: Configure Adaptive Optimization <use-adaptive-optimization/configure>
Advanced Guide: Configure Adaptive Components <use-adaptive-optimization/adaptive-components>
Advanced Guide: Tune Adaptive Behavior <use-adaptive-optimization/advanced-guide>
Code Examples <use-adaptive-optimization/code-examples>
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

API <reference/api/index>
reference/performance
```

```{toctree}
:hidden:
:caption: Troubleshooting
:maxdepth: 2

Troubleshooting Guide <troubleshooting/troubleshooting-guide>
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
