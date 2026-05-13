<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# About

Use this section when an agent framework, orchestration layer, SDK, or provider
adapter owns the tool and LLM call sites that need NeMo Flow instrumentation.

Framework integrations differ from direct application instrumentation because the
integration often does not own the full invocation. A framework may control
scheduling, retries, streaming, callback signatures, provider payloads, and
internal object lifetimes. The integration has to choose the best available
boundary without changing framework behavior.

Prefer a managed execution wrapper around a stable tool or LLM callback. When
that is not possible, use explicit lifecycle calls, standalone guardrail or
intercept helpers, or mark events.

## Start Here

Use these signals to decide whether this documentation path matches your current task.

- Maintain a framework integration for NeMo Flow
- Need to instrument calls without rewriting framework internals
- Need to handle provider-specific request or response payloads
- Need to keep non-serializable framework objects outside NeMo Flow payloads
- Are building or reviewing third-party integration patches

If you own the application call sites directly, use [Instrument Applications](../instrument-applications/about.md) first.
If your application uses [LangChain](https://www.langchain.com/langchain) or
[LangGraph](https://www.langchain.com/langgraph), start with
[LangChain Integration](../getting-started/python/langchain.md) or
[LangGraph Integration](../getting-started/python/langgraph.md).

## Guides

Use these guide links to move from the overview into task-specific instructions.

- [Basic Guide: Adding Scopes](adding-scopes.md) shows how framework request and run hooks become NeMo Flow ownership boundaries.
- [Basic Guide: Wrap Tool Calls](wrap-tool-calls.md) explains where to place managed tool wrappers and tool lifecycle fallbacks.
- [Basic Guide: Wrap LLM Calls](wrap-llm-calls.md) explains where to place managed provider wrappers, model names, streaming behavior, and LLM lifecycle fallbacks.
- [Advanced Guide: Coding-Agent Gateway](coding-agent-gateway.md) describes the Rust gateway for observing Codex, Claude Code, Cursor, and Hermes through canonical hooks plus a passthrough LLM gateway.
- [OpenClaw Plugin Guide](openclaw-plugin.md) covers configuring the OpenClaw plugin, mapping OpenClaw hooks to NeMo Flow telemetry, and understanding current LLM replay fidelity boundaries.
- [Claude Code Gateway Guide](coding-agent-claude-code.md) covers transparent Claude Code runs, Anthropic gateway routing, ATIF verification, and unsupported Claude application modes.
- [Codex Gateway Guide](coding-agent-codex.md) covers transparent Codex CLI runs, local GUI/app caveats, model provider routing, and remote-task limits.
- [Cursor Gateway Guide](coding-agent-cursor.md) covers transparent Cursor runs, temporary hook patching, GUI and CLI smoke tests, and gateway routing limits.
- [Hermes Gateway Guide](coding-agent-hermes.md) covers Hermes shell hook installation, dynamic gateway URL handling, session-finalize behavior, and hook consent caveats.
- [Advanced Guide: Handle Non-Serializable Data](non-serializable-data.md) shows how to keep clients, streams, callbacks, and SDK objects outside JSON payloads.
- [Advanced Guide: Using Codecs](using-codecs.md) explains typed value codecs for framework-facing wrappers.
- [Advanced Guide: Provider Codecs](provider-codecs.md) explains provider request and response codecs for normalized middleware and event annotations.
- [Advanced Guide: Provider Response Codecs](provider-response-codecs.md) focuses on response-only annotations for subscribers and exporters.
- [Code Examples](code-examples.md) collects fallback APIs, mark events, and repository patch workflow examples.

Start by identifying the framework's stable tool and LLM boundaries. Prefer
managed execution wrappers wherever the framework exposes a callback that NeMo
Flow can own. Use explicit API calls only when the framework owns invocation
internally but exposes reliable start and finish hooks.

Validate that application-visible framework behavior does not change. Then
confirm that events share the expected root scope, middleware runs exactly once
per managed call, and non-serializable framework objects remain in
framework-owned storage.
