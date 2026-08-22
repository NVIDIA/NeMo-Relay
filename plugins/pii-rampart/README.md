<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Rampart PII Native Plugin

This directory builds Rampart PII redaction as an opt-in Rust native dynamic
plugin. The model and Tract inference engine are linked into the plugin
library, not into the default Relay CLI or language-binding artifacts.

The plugin runs inside the Relay process. Its sanitizer callbacks use Relay's
native ABI v4 typed asynchronous middleware SDK. The SDK drives callbacks on a
per-component Tokio executor, while model inference runs on Rampart's separate
bounded Rayon pool. Relay's runtime workers and the plugin's Tokio executor do
not perform model inference synchronously.

## Build

From this directory, run:

```bash
cargo build --release
```

Copy `relay-plugin.toml` to `relay-plugin.local.toml` and replace
`<platform-library-file>` with the release library for the current platform:

| Platform | Library |
|---|---|
| macOS | `libnemo_relay_pii_rampart_plugin.dylib` |
| Linux | `libnemo_relay_pii_rampart_plugin.so` |
| Windows | `nemo_relay_pii_rampart_plugin.dll` |

Replace `<artifact-sha256>` with the lowercase SHA-256 digest of that library.

## Model

Download the pinned model snapshot into a deployment-owned directory:

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

Relay verifies the expected files and hashes when the plugin is activated.

## Enable

Add and enable the materialized manifest:

```bash
nemo-relay plugins add ./plugins/pii-rampart/relay-plugin.local.toml
nemo-relay plugins enable pii_rampart
nemo-relay plugins edit
```

At minimum, set `model_path` and either `preset = "trajectory_context"` or
explicit `target_paths` or `target_path_patterns`. Native plugins are trusted
code and are not sandboxed.

The SDK-owned Tokio executor defaults to one worker for this plugin. A
component can override that value independently of Rampart's inference pool:

```toml
[plugins.dynamic.config.executor]
worker_threads = 2
```

Increase it only when measurements show async sanitizer callbacks queuing
before they hand work to the bounded inference pool.

## Runtime behavior

Rampart sanitizes copies used for events and exporters. It does not rewrite the
request passed to a provider or tool, or the response returned to the caller.
Managed calls submit copied payload snapshots to Relay's queued observability
publication path without waiting for Rampart inference. Slow sanitization can
delay event delivery to subscribers and exporters, and an explicit subscriber
flush waits for the queued publication lineage to drain, but it does not add
latency to the provider or tool call itself.

The plugin admits at most 16 sanitizer callbacks, waits up to 500 ms for
capacity, and runs model inference on at most three dedicated Rayon workers.
If admission, model execution, codec projection, or a configured payload budget
fails, the affected observable value is omitted or replaced rather than emitted
without sanitization. The original application call continues unchanged.

Successful detections use the configured replacement, which defaults to
`[REDACTED]`. A value that cannot be processed uses a distinct
`[CONTENT OMITTED: ...]` marker so operators can distinguish a detected PII
span from a sanitizer failure. The marker remains in the affected field of the
emitted event and therefore reaches subscribers and exporters.

The `trajectory_context` preset operates directly on selected provider-native
request and response JSON and is the recommended configuration for coding-agent
telemetry. It supports OCI Generative AI without converting its multipart,
tool, candidate, or vendor-specific shapes through a normalized codec.

Explicit normalized LLM paths support Relay's built-in OpenAI Chat, OpenAI
Responses, Anthropic Messages, and Gemini generateContent codec surfaces
through the ABI v4 capability for the payload shapes covered by their
projection tests; normalized projection is not a claim of lossless support for
every multipart or repeated-item provider shape. Runtime request codecs can
also be inspected through that capability. Runtime and opaque response
projections, and explicit
normalized OCI request/response projection, fail closed by omitting the
governed observable value: Relay's response codec is decode-only and OCI's
normalized representation does not yet prove lossless preservation of every
provider-native shape. The provider call and caller-visible result remain
unchanged in every case.
