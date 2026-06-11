<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay PII Redaction

`nemo-relay-pii-redaction` is the first-party NeMo Relay plugin crate for
deterministic privacy redaction on tool and LLM observability payloads. It
ships the `pii_redaction` plugin contract, a production-ready `builtin`
backend, and the future `local_model` seam for model-backed detection and
redaction.

The plugin is designed for the common case where teams want a supported,
config-driven privacy policy surface instead of writing custom sanitize
middleware by hand.

## Why Use It

- **Use a supported privacy plugin**: Install deterministic redaction behavior
  through the NeMo Relay plugin system instead of custom sanitize callbacks.
- **Cover tool and LLM observability surfaces**: Sanitize emitted tool
  request/response payloads and supported codec-backed LLM request/response
  payloads through one shared config surface.
- **Choose explicit action semantics**: Use `remove`, `redact`,
  `regex_replace`, `hash`, or `mask` depending on the privacy and debugging
  tradeoff you need.
- **Start deterministic, leave room for model-backed policy later**: Ship the
  lightweight `builtin` backend now and keep `local_model` as the future seam
  for local inference-based detection.

## What You Get

- **`PiiRedactionConfig`**: Canonical config contract for the top-level
  `pii_redaction` plugin component.
- **Deterministic `builtin` backend**: Production-ready action engine for
  remove/redact/replace/hash/mask workflows.
- **Built-in detector presets**: First-party detectors for common PII,
  structured secrets, and cloud credentials.
- **Codec-aware LLM handling**: Overlay support for `openai_chat`,
  `openai_responses`, and `anthropic_messages`.
- **Local backend seam**: `local_model` config contract and provider
  registration surface for future model-backed implementations.

## Plugin Versus Raw Middleware

Use raw middleware when you need bespoke runtime logic. Use
`nemo-relay-pii-redaction` when you want a reusable privacy policy surface.

- **Raw middleware** gives you the generic hook mechanism and full code-level
  control.
- **`pii_redaction`** packages the common privacy policy contract on top of
  those hooks, including typed config, validation, editor support, detector
  presets, and cross-runtime behavior.

This crate does not change real callback arguments or return values. It
sanitizes emitted observability payloads through NeMo Relay sanitize guardrails.

## Installation

Install the plugin crate alongside the core runtime:

```bash
cargo add nemo-relay nemo-relay-pii-redaction
```

For local source development:

```bash
cargo build -p nemo-relay-pii-redaction
cargo test -p nemo-relay-pii-redaction
```

## Getting Started

Register the plugin component before validating or initializing plugin
configuration that includes a `pii_redaction` component:

```rust
nemo_relay_pii_redaction::component::register_pii_redaction_component()?;
```

A minimal config can redact detected emails from emitted tool input payloads:

```toml
[[plugins.components]]
kind = "pii_redaction"

[plugins.components.config]
mode = "builtin"
tool_input = true

[plugins.components.config.builtin]
action = "redact"
detector = "email"
target_paths = []
```

## Built-In Backend

The shipped `builtin` backend supports these actions:

- `remove`
- `redact`
- `regex_replace`
- `hash`
- `mask`

The detector catalog includes:

- Common PII: `email`, `phone`, `ip_address`, `ipv6`, `url`
- Structured secrets: `api_key`, `uuid`, `bearer_token`, `jwt`, `credit_card`
- Cloud credentials: `aws_access_key_id`, `aws_secret_access_key`,
  `gcp_api_key`, `azure_storage_account_key`

Detector-aware masking defaults are available for the relevant detectors. For
high-risk secrets, prefer `redact` over partial `mask` behavior.

## Local Model Seam

`local_model` is included in the plugin contract now, but no runtime
implementation ships in this crate yet.

The seam exists so a future local detector/redactor backend can be added
without redesigning the public plugin surface. If `mode = "local_model"` is
configured today, the runtime expects a registered local backend provider and
fails fast if one is not installed.

## Documentation

[NeMo Relay documentation](https://docs.nvidia.com/nemo/relay)
