<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Rampart PII Worker

This optional manifest-backed Python worker runs the
[`nationaldesignstudio/rampart`](https://huggingface.co/nationaldesignstudio/rampart)
ONNX token classifier behind NeMo Relay's `pii_redaction.local_model` backend.
Relay owns field selection, event sanitization, replacement, deadlines, and
fail-closed behavior. The worker performs detector inference only.

The worker runs as a local child process over Relay's `grpc-v1` protocol. Its
Python, ONNX Runtime, NumPy, tokenizer, and model-cache dependencies remain in a
Relay-managed virtual environment rather than the Relay host process. Process
isolation is not a security sandbox.

## Install

From this directory:

```bash
uvx --from . nemo-relay-pii-rampart-prefetch
nemo-relay plugins add ./relay-plugin.toml
nemo-relay plugins enable nemo_relay.pii_rampart
```

If the worker package is already installed, run
`nemo-relay-pii-rampart-prefetch` directly. `plugins add` creates a separate
Relay-managed Python environment from the same source directory.

Rampart activation is offline-only. It requires the pinned 14.7 MB model
snapshot to already exist in the Hugging Face cache. Model acquisition is an
explicit setup step and never occurs during activation or inference.

`model_id` normally remains `nationaldesignstudio/rampart`. The worker rejects
other repository identifiers and revisions because its integrity manifest pins
one supported snapshot. To load that same snapshot from a local directory, set
`model_id` to an absolute path or use explicit `./`, `../`, or `~/` syntax so a
relative directory cannot shadow the repository identifier. The PII
component's optional `local.model_id` remains the logical
`nationaldesignstudio/rampart` identifier even when worker storage uses a local
path.

On hosts with slow or restricted access to Hugging Face, populate the shared
cache before enabling the plugin:

```bash
nemo-relay-pii-rampart-prefetch
```

Run that command as the same operating-system user that runs Relay. If
`cache_dir` is configured for the plugin, pass the same value with
`--cache-dir`. The prefetch command and activation both verify SHA-256 digests
for every runtime model and tokenizer file. A missing or modified file blocks
activation.

## Configure

Add worker settings to the `[[plugins.dynamic]]` record created by `plugins
add`:

```toml
[plugins.dynamic.config]
local_files_only = true
max_windows_per_request = 128
inference_batch_size = 16
max_pending_requests = 8
```

`inference_batch_size` is an upper bound. The worker batches short token
windows together but automatically reduces the batch width for longer windows
to bound ONNX intermediate memory.

Add the PII component to the same `plugins.toml`:

```toml
[[components]]
kind = "pii_redaction"
enabled = true

[components.config]
codec = "openai_chat"

[[components.config.profiles]]
mode = "builtin"
priority = 70

[components.config.profiles.builtin]
action = "redact"
detector = "email"

[[components.config.profiles]]
mode = "builtin"
priority = 80

[components.config.profiles.builtin]
action = "redact"
detector = "credit_card"

[[components.config.profiles]]
mode = "local_model"
priority = 90

[components.config.profiles.local]
backend = "nemo_relay.pii_rampart/detector"
model_id = "nationaldesignstudio/rampart"
detector_profile = "default"
allow_network = false
max_latency_ms = 5000
min_score = 0.4
replacement = "[REDACTED]"
target_path_patterns = [
  "/messages/*/content",
  "/messages/*/content/*/text",
  "/message",
  "/message/*/text",
]
```

`allow_network = false` means worker inference is local. It does not sandbox
the worker. `local_files_only` must remain `true`; activation-time model
acquisition is not supported.

Rampart is the contextual detector lane, not a replacement for deterministic
recognizers. Configure built-in PII profiles for structured values and the
local-model profile for names and contextual identifiers. Keep the local-model
profile limited to normalized content paths. Classifying every string leaf can
produce false positives on model names, region names, UUIDs, trace IDs, and
other machine identifiers. Relay, not the worker, applies `min_score` and
optional `excluded_labels` policy after validating the worker response.

## Runtime Bounds

- At most 64 texts and 64 KiB of UTF-8 text are accepted per detection request.
- Each text is limited to 16 KiB.
- Long inputs use overlapping 510-token windows, with 64 content tokens of
  overlap.
- ONNX inference batches and total windows are bounded by worker configuration.
- Requests above the worker bounds return an error; the PII component then
  fails closed for the affected batch.
- `max_latency_ms` is one total budget for all inference batches selected from
  one payload.
- CPU inference is serialized per worker process. Host deadlines cancel the
  RPC, while already-running native inference is allowed to finish before its
  admission slot is released.
- Use a `max_latency_ms` of at least 5000 when the selected payload can approach
  the 64 KiB detection-request limit. Smaller content-only payloads normally
  complete much faster. Benchmark representative inputs on deployment
  hardware before lowering the deadline.

The default model supports English, Spanish, French, German, Italian,
Portuguese, and Dutch. Its model card documents weak recall for non-Latin
scripts and government identifiers. Do not treat this detector as a complete
security boundary.

## Attribution

See [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md). The model is downloaded
only by the explicit prefetch command and is not redistributed by this package.
