<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# About

Use this section when your application already uses a supported framework or
agent harness and you want the maintained NeMo Flow integration path for that
surface.

Supported integrations are end-user entry points. They use public framework or
plugin APIs where available and document the support level for observability,
security middleware, and optimization features.

## Support Matrix

| Agent / Library | Observability | Security | Optimization | Notes |
|:--|:--:|:--:|:--:|:--|
| LangChain | ✅ Yes | ✅ Yes | ✅ Yes | Wrapped tool and LLM calling. |
| LangGraph | ✅ Yes | ✅ Yes | ✅ Yes | Wrapped tool and LLM calling. |
| Deep Agents | ✅ Yes | ✅ Yes | ✅ Yes | Wrapped tool and LLM calling. |
| OpenClaw | ✅ Yes | ❌ No | ❌ No | Observability support; missing middleware for wrapped execution. |

## Guides

Use these guide links to move from the support matrix into setup and usage
instructions.

- [OpenClaw Plugin Guide](openclaw-plugin.md) covers configuring the OpenClaw
  plugin, mapping OpenClaw hooks to NeMo Flow telemetry, and understanding
  current LLM replay fidelity boundaries.
- [LangChain Integration Guide](langchain.md) covers installing the LangChain
  extra and adding NeMo Flow middleware and callbacks to LangChain agents.
- [LangGraph Integration Guide](langgraph.md) covers installing the LangGraph
  extra and adding NeMo Flow callbacks to LangGraph workflows.
- [Deep Agents Integration Guide](deepagents.md) covers installing the Deep
  Agents extra and capturing Deep Agents-specific marks, skills, subagents, and
  human-in-the-loop lifecycle events.

If you are building a new framework integration or patching framework internals,
use [Integrate into Frameworks](../integrate-frameworks/about.md) instead.
