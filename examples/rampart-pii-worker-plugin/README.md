<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Rampart PII Worker Plugin

This package exercises Rampart through NeMo Relay's manifest-backed Python
`grpc-v1` worker convention. It is an optional local observability sanitizer;
it is not loaded unless a user registers and enables it.

This is a standalone dynamic plugin, not a new `pii_redaction.local_model`
provider. It composes with the built-in PII plugin through Relay's ordered
sanitizer registries.

## How It Fits

Relay's built-in PII plugin remains the deterministic policy layer for
structured values and secrets. This worker adds contextual model detection for
names, addresses, and other values whose meaning depends on surrounding text.

The worker also retains Rampart's small deterministic pre-pass for validated
cards, SSNs, emails, URLs, and network addresses. It redacts bare UUID and
long-hex identifiers while preserving them when an adjacent
`trace`/`span`/`request`/`correlation` label identifies an operational ID. That
makes it safe to run by itself without breaking common trace correlation. When
both plugins are enabled, the built-in plugin can cover additional organization
policy such as API keys and tokens before the worker handles contextual PII.

The worker:

- Runs the ONNX model in a Relay-managed local process.
- Loads the model revision pinned by this plugin release.
- Registers LLM and tool observability sanitizers through `PluginContext`.
- Leaves real provider requests, responses, tool arguments, and tool results
  unchanged.
- Defaults to offline activation after model preparation.
- Does not register a generic mark sanitizer because model classification is
  limited to semantic content fields.
- Preserves provider protocol IDs, model names, roles, statuses, and function
  names while scanning known conversational fields and opaque tool payloads.
- Removes provider request headers from the observability copy without
  changing the request sent to the provider.
- Replaces encoded image, audio, and file payloads with a fixed binary-content
  marker instead of sending them through the text model.
- Registers pass-through scope-event sanitizers as a fail-closed backstop.
  Relay clears scope observability fields if the worker becomes unavailable
  instead of retaining the raw payload returned by specialized sanitizer
  fallback behavior.

## Register the Worker

From this directory, run:

```bash
nemo-relay plugins validate ./relay-plugin.toml
nemo-relay plugins add --user ./relay-plugin.toml
nemo-relay plugins edit --user
nemo-relay plugins enable nvidia.rampart_pii
nemo-relay plugins validate nvidia.rampart_pii
```

`plugins add` creates a managed Python environment and installs the worker and
its runtime dependencies. It does not download the model. `plugins edit --user`
uses the manifest's configuration schema to update the user-scoped
`[[plugins.dynamic]]` record created by `plugins add`.

## Prepare the Model

Set `allow_network = true` for the first activation so the approximately 15 MB
model artifact can be added to the local Hugging Face cache:

```toml
[[plugins.dynamic]]
manifest = "/absolute/path/to/rampart-pii-worker-plugin/relay-plugin.toml"

[plugins.dynamic.config]
allow_network = true
```

After the model is cached, set `allow_network = false` for offline activation.
The worker fails activation rather than silently reaching the network when the
model is absent and network access is disabled.

The model repository and immutable revision are owned by the plugin release.
Changing them requires a plugin update so validation and attribution remain
aligned with the implementation.

## Runtime Limits

`max_concurrency` defaults to `2`. `max_content_chars` defaults to `8192`, so
larger aggregate semantic payloads fail closed before ONNX inference.
`max_latency_ms` defaults to `250` and
applies independently to each sanitizer callback, including queue wait and
classification. A managed LLM call can invoke both request and response
sanitizers, so its total observability overhead can exceed one callback budget.

When a deadline expires, the callback returns
`[REDACTED:PII_DETECTION_FAILURE]`. ONNX inference cannot be preempted safely;
the worker retains the occupied concurrency slot until that native call exits
so timeouts cannot create unbounded background inference.

The LLM walker scans known semantic provider fields, including `content`,
`text`, `prompt`, `message`, `query`, `instructions`, and opaque tool
arguments/results. It preserves protocol IDs and tool/function names. Tool
payload sanitizers treat every string value as semantic content.
Encoded media values are replaced with `[REDACTED:BINARY_CONTENT]` and do not
count against the text inference budget.

Raise `max_content_chars` only after benchmarking the target hardware. The
schema allows up to `65536`, but larger values can exceed the default latency
budget and materially increase worker memory.

## Support Boundary

The worker sanitizes emitted LLM and tool observability payloads. It does not
change provider requests, provider responses, tool arguments, or tool results
seen by the application.

Rampart supports English, Spanish, French, German, Italian, Portuguese, and
Dutch text written in Latin script. Do not treat it as complete coverage for
non-Latin names, indirect identifiers, adversarial input, or regulated
government identifiers without additional policy controls. Keep deterministic
Relay profiles enabled for organization-specific secrets and identifiers.

Worker process isolation is not a security sandbox. The worker and model run
locally with the permissions of the Relay user.

## Remove the Worker

```bash
nemo-relay plugins remove nvidia.rampart_pii
```

Removal deletes Relay's managed worker environment. The Hugging Face model
cache remains user-owned and is not deleted by Relay.

## Benchmark

The benchmark compares an empty Relay runtime with the managed worker path.
Pass the environment path reported by `nemo-relay plugins inspect`:

```bash
python benchmark.py \
  --mode worker \
  --manifest ./relay-plugin.toml \
  --environment /path/to/managed/environment \
  --profile comprehensive \
  --output /tmp/rampart-worker.json
```

Each LLM operation includes both request and response sanitization. The
`failure_replacements` field counts replacement occurrences in emitted
lifecycle events, not failed application operations.

Run the model-only quality probe separately:

```bash
PYTHONPATH=. python evaluate.py \
  --output /tmp/rampart-quality.json
```

Refer to [EVALUATION.md](./EVALUATION.md) for the measured quality,
performance, memory, overload, and support-boundary results from the initial
experiment.
