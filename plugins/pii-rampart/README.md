<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Rampart PII Native Plugin

This directory builds Rampart PII redaction as an opt-in Rust native dynamic
plugin. The model and Tract inference engine are linked into the plugin
library, not into the default Relay CLI or language-binding artifacts.

The plugin runs inside the Relay process. Its sanitizer callbacks use the
native ABI v3 asynchronous middleware contract, then run inference on the
existing bounded Rampart Rayon pool. Relay's Tokio workers do not perform or
wait synchronously on model inference.

## Install a release archive

GitHub releases provide one Rampart plugin archive for each supported Rampart
target. Extract the archive, then register its materialized manifest:

```bash
nemo-relay plugins add ./nemo-relay-pii-rampart-plugin/relay-plugin.toml
nemo-relay plugins enable pii_rampart
nemo-relay plugins edit
```

Each archive contains the platform library, its SHA-256-bound manifest, the
configuration schema, license, Rust dependency attributions, and this README.
The model is not bundled.

## Build from source

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
Install `cargo-about`, then create the same archive produced by CI:

```bash
just package-pii-rampart-plugin \
  ./plugins/pii-rampart/target/release/libnemo_relay_pii_rampart_plugin.dylib \
  aarch64-apple-darwin
```

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

Add and enable a source-built materialized manifest:

```bash
nemo-relay plugins add ./plugins/pii-rampart/relay-plugin.local.toml
nemo-relay plugins enable pii_rampart
nemo-relay plugins edit
```

At minimum, set `model_path` and either `preset = "trajectory_context"` or
explicit `target_paths` or `target_path_patterns`. Native plugins are trusted
code and are not sandboxed.

## Runtime behavior

Rampart sanitizes copies used for events and exporters. It does not rewrite the
request passed to a provider or tool, or the response returned to the caller.
Relay still awaits each sanitizer before continuing the managed lifecycle, so
model inference adds latency and can apply backpressure to the call.

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

The `trajectory_context` preset operates on provider-native request and response
content and is the recommended configuration for coding-agent telemetry.
Explicit normalized LLM paths support Relay's built-in OpenAI Chat, OpenAI
Responses, Anthropic Messages, and Gemini generateContent codecs. The current
native asynchronous ABI does not pass an owned runtime or opaque codec
capability across the callback boundary, so explicit normalized paths fail
closed for other codec kinds.
