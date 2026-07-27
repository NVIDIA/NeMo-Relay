<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay PII Redaction

`nemo-relay-pii-redaction` is the first-party NeMo Relay plugin crate for
deterministic privacy redaction on tool and LLM observability payloads. It
ships the `pii_redaction` plugin contract, a production-ready `builtin`
backend, and a worker-backed `local_model` seam for model-backed detection and
redaction.

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
- Use an isolated `grpc-v1` worker as the detector behind the `local_model`
  config contract without loading model dependencies into the Relay host.

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

`local_model` delegates bounded detector inference to a manifest-backed
`grpc-v1` worker. The PII component remains responsible for choosing
observability fields, decoding provider payloads, batching text, enforcing the
deadline and failure policy, validating detections, and replacing accepted
spans.

Configure worker inference by its host-qualified name:

```toml
[[components]]
kind = "pii_redaction"
enabled = true

[components.config]
mode = "local_model"
codec = "openai_chat"

[components.config.local]
backend = "acme.pii_worker/detector"
model_id = "acme-pii-v1"
detector_profile = "default"
min_score = 0.4
target_path_patterns = [
  "/messages/*/content",
  "/messages/*/content/*/text",
  "/message",
  "/message/*/text",
]
replacement = "[REDACTED]"
allow_network = false
max_latency_ms = 250
```

The backend name is `<plugin_id>/<registration_name>`. For example, a worker
with plugin ID `acme.pii_worker` that calls
`register_worker_inference("detector", "nemo.relay.pii_detection.v1", ...)`
is selected as `acme.pii_worker/detector`. Relay verifies the PII contract,
installs worker inference before static components initialize, and removes PII
sanitizers before stopping their worker.

Use profiles to compose deterministic and contextual detection. The lower
priority runs first:

```toml
[components.config]
codec = "openai_chat"

[[components.config.profiles]]
mode = "builtin"
priority = 80

[components.config.profiles.builtin]
action = "redact"
detector = "email"

[[components.config.profiles]]
mode = "local_model"
priority = 90

[components.config.profiles.local]
backend = "nemo_relay.pii_rampart/detector"
min_score = 0.4
max_latency_ms = 5000
target_path_patterns = [
  "/messages/*/content",
  "/messages/*/content/*/text",
  "/message",
  "/message/*/text",
]
```

`target_paths` contains exact JSON pointers. `target_path_patterns` also accepts
`*` as one complete path segment, which is useful for message arrays. When a
codec is configured, paths address the normalized request or response shape;
the content-only patterns above cover the `openai_chat` request and response
shape. Without a codec, they address the original JSON payload. When both lists
are empty, Relay inspects every string leaf and reports a configuration warning.

Use content-only paths for contextual classifiers. Do not send model names,
tool identifiers, trace IDs, routing fields, or arbitrary provider metadata to
a classifier unless that is an explicit policy choice.

Relay accepts detections whose confidence is at least `min_score`, which
defaults to `0.4`. `excluded_labels` is an exact, case-sensitive denylist for
detection labels that should remain visible. The host applies both settings
after validating the complete worker response; workers do not own the final
redaction policy.

Worker failures, timeouts, malformed responses, invalid UTF-8 boundaries,
overlapping spans, and input-limit violations fail closed for the affected
batch. If a configured codec cannot decode or safely re-encode an LLM payload,
Relay omits that request or response payload from the emitted event; it does not
retry normalized selectors against the raw provider shape.
`allow_network = true` is rejected; this lane is for same-machine inference.
This setting is a configuration invariant, not a network sandbox: Relay's
worker launcher does not currently prevent a worker process from opening
sockets. Only install workers whose packaging and runtime behavior satisfy
that policy. The default deadline is 250 ms for the complete selected payload,
including every inference batch. Configuration above 60 seconds is rejected.

### PII Detection Contract

The worker receives a versioned JSON request:

```json
{
  "version": 1,
  "model_id": "acme-pii-v1",
  "detector_profile": "default",
  "texts": [
    {"id": 0, "text": "Contact Alice Rivera"}
  ]
}
```

It returns detections using UTF-8 byte offsets:

```json
{
  "version": 1,
  "detections": [
    {
      "text_id": 0,
      "start_utf8": 8,
      "end_utf8": 20,
      "label": "person",
      "score": 0.99
    }
  ]
}
```

The worker performs inference only. It must not choose Relay surfaces,
traverse arbitrary event fields, or apply replacements itself. Rust and Python
workers have SDK helpers for this registration. Other languages can implement
the same `grpc-v1` protobuf contract directly; Rust, Python, and Node hosts all
consume it through the shared core runtime.

### Optional Rampart Worker

The source tree includes an optional
[Rampart worker](./workers/rampart/README.md) that implements this detection
contract with a pinned ONNX token-classification model. It runs in a
Relay-managed Python worker process, keeps ONNX dependencies out of the host,
and complements the built-in deterministic recognizers. The model is prefetched
separately, then resolved and integrity-verified from the local cache at
activation. It is not distributed in the Relay package.

## Documentation

[NeMo Relay documentation](https://docs.nvidia.com/nemo/relay)
