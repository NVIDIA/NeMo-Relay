<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Guardrails Dynamic Worker

This first-party preview runs NeMo Guardrails 0.23 input rails as an
asynchronous NeMo Relay final-input policy. Relay applies the policy after all
LLM request intercepts and before managed scope creation, cache lookup, routing,
or provider execution. Guardrails checks the request; Relay still owns the
provider call.

## Configure

Add the plugin through the Relay lifecycle so the CLI creates its managed
Python environment:

```bash
nemo-relay plugins add ./plugins/nemo-guardrails/relay-plugin.toml
nemo-relay plugins enable nvidia.nemo_guardrails
```

Configure either a Guardrails path:

```toml
[[plugins.dynamic]]
manifest = "/absolute/path/to/plugins/nemo-guardrails/relay-plugin.toml"

[plugins.dynamic.config]
config_path = "/absolute/path/to/guardrails-config"
timeout_ms = 30000
failure_mode = "fail_closed"
max_concurrency = 16
```

Or inline YAML with optional inline Colang:

```toml
[plugins.dynamic.config]
config_yaml = """
rails:
  config:
    regex_detection:
      input:
        patterns: ["blocked phrase"]
  input:
    flows: ["regex check input"]
"""
```

### Use NVIDIA NIM for LLM-backed rails

Guardrails 0.23 can use an NVIDIA NIM model to back LLM-based rails. The
`self check input` flow is one example. The worker reads `NVIDIA_API_KEY`; it
does not require an OpenAI or Anthropic key:

```bash
export NVIDIA_API_KEY="nvapi-..."
```

```toml
[plugins.dynamic.config]
config_yaml = """
models:
  - type: main
    engine: nim
    model: meta/llama-3.1-8b-instruct
    api_key_env_var: NVIDIA_API_KEY
    parameters:
      temperature: 0

rails:
  input:
    flows: ["self check input"]
"""
```

The rail-model credential and Relay's downstream-provider credential are
separate. When Relay sends the approved request to an OpenAI-compatible NVIDIA
endpoint, the gateway currently reads `OPENAI_API_KEY`; the same NVIDIA key can
be reused without creating an OpenAI key:

```bash
OPENAI_API_KEY="$NVIDIA_API_KEY" nemo-relay
```

## Preview Boundary

- The package pins `nemoguardrails==0.23.0` and uses the public
  `LLMRails.check_async` input-check API.
- OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, and Gemini
  `generateContent` are supported through Relay request codecs.
- Input must be provider-neutral plain text. Multimodal/composite content and
  native tool traffic are rejected explicitly; they are never flattened.
- This preview does not apply Guardrails output, retrieval, dialog, streaming
  output, remote, or tool policies.
- Operational worker errors and timeouts use the configured `failure_mode`.
  Fail-closed errors return a caller-safe terminal rejection while details stay
  in Relay's host logs. Explicit Guardrails blocks and unsupported content
  remain terminal.
