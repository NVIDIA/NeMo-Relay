<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay PII Redaction

`nemo-relay-pii-redaction` is the first-party NeMo Relay plugin crate for
privacy redaction on tool and LLM observability payloads. It provides two
independent component kinds:

- `pii_redaction` provides deterministic policies and a `local_model`
  integration seam.
- `pii_rampart` optionally runs the pinned `nationaldesignstudio/rampart`
  ONNX model inside the Relay process.

The plugin is designed for the common case where teams want a supported,
config-driven privacy policy surface instead of writing custom sanitize
middleware by hand.

## Key Features

NeMo Relay PII Redaction allows you to:

- Use `PiiRedactionConfig`, the canonical config contract for the top-level
  `pii_redaction` plugin component.
- Compose multiple ordered built-in or runtime-provided local-model policies
  inside one singleton component.
- Install deterministic redaction behavior through the NeMo Relay privacy
  plugin system instead of custom sanitize callbacks.
- Sanitize emitted tool request or response payloads and supported codec-backed
  LLM request/response payloads through one shared config surface.
- Choose explicit action semantics such as `remove`, `redact`,
  `regex_replace`, `hash`, or `mask`, depending on the privacy and debugging
  tradeoff you need.
- Use built-in detector presets as first-party detectors for common PII,
  structured secrets, and cloud credentials.
- Handle codec-aware LLMs with overlay support for `openai_chat`,
  `openai_responses`, and `anthropic_messages`.
- Remove conversational trajectory content while preserving event structure,
  tool-call identity, model attribution, routing, usage, and cost analytics.
- Use the `local_model` config contract and provider registration surface for
  future model-backed implementations.
- Enable the `rampart` feature to use the separate in-process `pii_rampart`
  component for contextual PII detection.

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

For a Rust application that uses the Rampart component, enable its feature:

```bash
cargo add nemo-relay-pii-redaction --features rampart
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

A profile-array config can apply multiple policies across every supported
sanitization surface:

```toml
[[components]]
kind = "pii_redaction"

[components.config]

[[components.config.profiles]]
mode = "builtin"
priority = 80

[components.config.profiles.builtin]
action = "redact"
detector = "email"

[[components.config.profiles]]
mode = "builtin"
priority = 90

[components.config.profiles.builtin]
action = "redact"
detector = "api_key"
```

Profiles execute by ascending priority, with array order breaking ties. Relay
assigns internal positional names such as `profile_0`; no user-supplied ID is
required. Profile-array mode covers marks, LLM and tool observability, and
scope metadata automatically. The original single-policy surface flags remain
available for backward compatibility but cannot be combined with `profiles`.

### Structure-Preserving Trajectory Export

Use the `trajectory_context` preset when exported trajectories must retain
their analytical structure without retaining chat, reasoning, tool, or
multimodal content. Pair it with a later email profile so email addresses are
also removed from otherwise-preserved metadata and custom marks:

```toml
[[components]]
kind = "pii_redaction"
enabled = true

[components.config]
version = 1

[[components.config.profiles]]
mode = "builtin"
priority = 80

[components.config.profiles.builtin]
preset = "trajectory_context"
custom_mark_payload_policy = "redact_all_leaves"

[[components.config.profiles]]
mode = "builtin"
priority = 90

[components.config.profiles.builtin]
action = "redact"
detector = "email"
```

`custom_mark_payload_policy = "preserve"` is the default and leaves unknown
plugin mark payloads intact for analysis. Use `redact_all_leaves` when opaque
plugins may emit content: scalar leaves in data, metadata, and opaque category
profile fields are replaced while typed category identity remains valid.
Strings become `[REDACTED]`, numbers become `0`, booleans become `false`, and
nulls, keys, arrays, and object shape are retained.
Known Relay marks are sanitized semantically so their structural and analytical
fields remain usable. This choice affects canonical event fields before
subscriber fan-out; exporter-owned resource attributes are outside this
boundary.

For Scope events, the preset retains direct string values for the trusted
low-cardinality classification fields `nemo_relay_scope_role`, `agent_kind`,
`hook_event_name`, `gateway_config_profile`, `gateway_mode`, `turn_source`,
`harness`, `source`, `identity_quality`, `gateway_path`,
`llm_correlation_status`, `llm_correlation_source`, `tool_correlation_status`,
`tool_correlation_source`, `otel.status_code`, and `fidelity_source`. It also
retains the direct boolean `provider_payload_exact`. Do not place PII or
conversational content in these fields. Arbitrary metadata and unexpected value
types continue through the preset's normal semantic redaction.

The preset defines its own action and therefore cannot be combined with
`action`, `detector`, `pattern`, `target_paths`, or mask-specific fields. Its
optional `replacement` defaults to `[REDACTED]`.

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

## Rampart PII Component

`pii_rampart` is a separate component, not an implementation of the
`pii_redaction.local_model` seam. It runs the pinned Rampart ONNX graph through
`tract-onnx` in the Relay Rust process. Relay does not download the model or
make network requests during activation.

### Provision the Model

Install the
[Hugging Face Hub CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli),
then download the files accepted by Relay into a deployment-owned directory:

```bash
hf download nationaldesignstudio/rampart \
  config.json \
  onnx/model_q4.onnx \
  special_tokens_map.json \
  tokenizer.json \
  tokenizer_config.json \
  vocab.txt \
  --revision b1993e4e68b082835b80ffc65acc03325ea2e501 \
  --local-dir /absolute/path/to/rampart
```

Set `model_path` to that absolute directory. During activation, Relay verifies
the SHA-256 digest of every required file and rejects missing files or digest
mismatches before installing sanitizer callbacks.

### Activate the Component

Add a `pii_rampart` component with explicit content selectors. This example
sanitizes normalized LLM request and response content without sending marks,
tool payloads, or provider metadata to the model:

```toml
[[components]]
kind = "pii_rampart"
enabled = true

[components.config]
version = 1
model_path = "/absolute/path/to/rampart"
codec = "openai_chat"
input = true
output = true
mark = false
tool_input = false
tool_output = false
target_paths = ["/message"]
target_path_patterns = [
  "/messages/*/content",
  "/messages/*/content/*/text",
]
```

`target_paths` contains exact JSON pointers. `target_path_patterns` also
accepts `*` as one complete path segment. At least one selector is required.
When a supported codec is active, selectors address the normalized Relay
request or response shape.

### Use a Language Binding

The CLI, Python, Node.js, and FFI host entry points register `pii_rampart`
automatically. Rust applications that initialize plugin configuration directly
must enable the `rampart` feature and call
`register_rampart_pii_component()` first.

Configuration helpers are available through these binding modules:

- Rust: `nemo_relay_pii_redaction::rampart`
- Python: `nemo_relay.pii_rampart`
- Node.js: `nemo-relay-node/pii_rampart`
- Go: `github.com/NVIDIA/NeMo-Relay/go/nemo_relay/pii_rampart`

Go and the raw C FFI remain experimental and source-first. Model loading and
inference run in the shared Rust implementation for every binding.

Rampart admits one sanitizer operation before submitting work to Tokio's
blocking pool. Concurrent operations do not queue for model inference.
They fail closed immediately: tool observability payloads become the configured
replacement, LLM bodies are omitted, and mutable mark or generic scope fields
are omitted. Tool and LLM scope metadata is omitted independently so an
already-sanitized specialized payload remains available. These fallbacks do not
change the arguments or return values seen by the underlying tool or model.

## Documentation

[NeMo Relay documentation](https://docs.nvidia.com/nemo/relay)
