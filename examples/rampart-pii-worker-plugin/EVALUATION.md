<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Rampart PII Worker Evaluation

Initial experimental evaluation, 23 July 2026

## Recommendation

Rampart is a reasonable optional contextual-detection lane for Relay
observability. Keep it as a disabled-by-default, manifest-backed Python
`grpc-v1` worker alongside the built-in deterministic PII component.

The worker is suitable for short and moderate semantic fields on the
observability hot path. It is not suitable as the only privacy control, a
security boundary, or a high-throughput bulk-document scanner. Do not make it
first-party supported until the model license, dependency set, model
acquisition, and support policy receive NVIDIA approval.

## Evaluated Shape

The experiment uses the current dynamic-plugin conventions:

- Relay owns worker provisioning, activation, health, and teardown.
- The worker uses the public `nemo-relay-plugin` SDK and `grpc-v1` protocol.
- ONNX Runtime, NumPy, Tokenizers, Hugging Face Hub, and model state remain
  outside the Relay host process.
- The manifest pins one immutable model revision.
- Model download is explicit and activation is offline by default.
- The worker changes emitted observability fields only. It does not change
  provider requests, provider responses, tool arguments, or tool results.
- The built-in PII component remains the deterministic policy lane. Rampart
  adds contextual name, address, phone, and identifier detection.

The model lane scans semantic LLM and tool content. It does not model-scan
generic marks or arbitrary scope metadata. Use the built-in deterministic
component for those surfaces and for organization-specific secrets.

## Entry Point and CLI Lifecycle

The real debug CLI validated and exercised the full worker lifecycle:

1. `plugins validate` accepted the manifest, schema, compatibility range, and
   integrity digest.
2. `plugins add --user` resolved
   `nemo_relay_rampart_worker.worker:main` to the integrity-checked
   `worker.py`, provisioned Python 3.14, installed the package, and wrote the
   user-scoped `[[plugins.dynamic]]` record.
3. `plugins inspect` reported the expected `grpc-v1` entry point, managed
   environment, capabilities, digest, and disabled state.
4. The configured record passed `plugins enable` and `plugins validate`.
5. Gateway startup spawned, connected, and registered the managed worker.
6. `plugins remove` deleted the managed environment and configuration record.

The configuration schema is closed, rejects unknown fields, and drives
`plugins edit --user`. Activation is intentionally offline after one explicit
model-preparation run.

The unoptimized debug gateway took about 38 seconds from configuration
resolution to listening while validating and loading this newly provisioned
environment. Three repeated warm activations through the Python binding took
0.32-0.34 seconds; the first comprehensive run took 1.09 seconds. Treat the
debug timing as a developer-experience observation, not a production startup
measurement.

The built wheel is pure Python and contains the worker modules plus
`ATTRIBUTION.md`; it does not contain the model. The managed environment
passes `pip check`. The manifest and configuration schema remain source-tree
registration artifacts rather than wheel contents. That matches the current
`plugins add ./relay-plugin.toml` workflow, but a standalone wheel-only
distribution would need an explicit registry or manifest-packaging design.

## Test Environment

| Item | Value |
|---|---|
| Host | macOS 26.5.2, arm64 |
| CPU availability | 14 logical CPUs |
| Host Python | 3.13.12 |
| Worker Python | 3.14.4 |
| Relay host binding | `nemo-relay 0.7.0` |
| Worker SDK | `nemo-relay-plugin 0.6.0` |
| Model | `nationaldesignstudio/rampart` |
| Model revision | `b1993e4e68b082835b80ffc65acc03325ea2e501` |
| Model artifact | Quantized ONNX, approximately 15 MB |
| Managed environment on disk | Approximately 194 MiB |
| Default worker settings | concurrency 2, 8192 content chars, 250 ms per callback |

Results are single-machine measurements, not portable service-level
objectives. Repeat them on each target platform before selecting production
budgets.

## Detection Quality

### Handcrafted Contract Set

The checked-in `evaluate.py` probe covers supported Latin-script names,
structured identifiers, operational identifiers, long context, source-like
text, adversarial forms, and documented limits.

| Measure | Supported contract | All 50 cases |
|---|---:|---:|
| Cases | 42 | 50 |
| Case pass rate | 97.62% | 82.00% |
| Private terms redacted | 54 / 55 | 54 / 61 |
| Private-term recall | 98.18% | 88.52% |
| Public terms retained | 13 / 13 | 13 / 17 |
| Public-term retention | 100.00% | 76.47% |

The only unexpected supported-contract miss was the Dutch surname particle in
`Daan de Vries`: the model detected `Daan` but not `de Vries`.

Observed documented limits:

- Chinese, Arabic, and Cyrillic names were not reliably detected.
- A base64-encoded email was not decoded and classified.
- An invalid card-like reference was over-redacted.
- `gpt-4.1-mini`, a Kubernetes pod name, and `NVIDIA` produced false
  positives in isolated prose.

These observations are consistent with the model card's non-Latin and
adversarial limitations. The worker reduces some operational false positives
by preserving labeled trace, span, request, correlation, commit, checksum, and
digest identifiers. Unlabeled UUID and long-hex values remain redacted.

### Upstream Public-Case Parity

A Python port of the upstream project's 58 public cases produced:

| Measure | Result |
|---|---:|
| Case pass rate | 94.83% |
| Private terms redacted | 74 / 76 |
| Private-term recall | 97.37% |
| Public terms retained | 78 / 79 |
| Public-term retention | 98.73% |

The three failures matched the upstream suite's documented residuals:

- An agency case number was not detected.
- A Medicare-style identifier was not detected.
- An invalid Luhn card-like number was over-redacted.

This parity probe is evidence that the Python ONNX path follows the upstream
pipeline. It is not a substitute for the model card's 30,000-case held-out
evaluation. The model card reports 98.42% private-term recall and 91.7% public
term retention on that larger corpus.

### Direct Model Latency

The final handcrafted run measured approximately 1.06 ms p50 and 1.51 ms p95
for a single warmed sanitizer call. The upstream public-case port measured
approximately 1.40 ms p50 and 1.80 ms p95. Long 8 KB contexts are materially
slower and are represented in the managed worker measurements below.

## Real Relay Validation

The live probes used the real Relay host binding, real worker SDK, real worker
process, and pinned ONNX model.

Verified behavior:

- LLM start/end and tool start/end events were sanitized.
- Raw names, emails, binary media, and privacy canaries were absent from
  captured event JSON.
- Request headers were removed from the observability copy.
- The provider callback still received its original headers and body.
- Tool callbacks still received and returned their original values.
- Encoded image and tool data became `[REDACTED:BINARY_CONTENT]`.
- A labeled trace ID and the bounded model field were preserved.
- Killing the worker caused affected lifecycle observability fields to clear;
  the application callback result still returned.
- A fresh offline cache failed activation in approximately 244 ms without
  writing model files or leaving a worker process.

The built-in deterministic component and Rampart worker were also activated
together. With deterministic email redaction at priority 50 and Rampart at
priority 100, both redaction markers survived, names and email were absent
from events, headers were absent, and the application request and response
remained unchanged.

## Allocation and Ownership Model

Python does not expose Rust-style borrowing, so the relevant contract is
object ownership and bounded retention:

- Relay serializes each sanitizer payload into the local worker process. The
  worker receives its own decoded Python object graph.
- Sanitizers build a new bounded JSON result instead of mutating the decoded
  input. Immutable scalar values may be reused; dictionaries and lists on
  semantic paths are rebuilt.
- Request headers are replaced with a new empty dictionary in both normal and
  failure paths.
- Tool and LLM content is capped at 8192 characters, 64 semantic strings, 4096
  JSON nodes, and depth 64 by default.
- Stringified tool JSON is parsed, sanitized, and reserialized only after the
  same bounds check.
- Encoded media is replaced without model inference. It has already crossed
  the local gRPC boundary, so large media still causes transient transport and
  decode allocation even though it is not retained.
- One `RampartSanitizer` and one `_SanitizationExecutor` are shared by the four
  registered callbacks. The closures do not create a context cycle.
- The semaphore admits at most two `asyncio.to_thread` native inference tasks.
  A timed-out or cancelled task retains its payload and slot only until ONNX
  returns. Waiting callbacks retain their own bounded payload until their
  250 ms deadline.
- One cached ONNX session is reachable for the single supported worker
  registration. The worker does not allow multiple components, so a second
  config cannot allocate another session in the same activation.
- ONNX owns its native arena and lazily allocated buffers. Python has no useful
  explicit session close; terminating the worker process is the deterministic
  reclamation boundary.
- Per inference, NumPy allocates input arrays plus logits and softmax
  temporaries. Overflow windows are processed sequentially. Two admitted
  native calls bound concurrent inference allocation.

The model runner also creates a character-offset map for each content string.
That allocation grows linearly with content length, which is another reason to
keep the 8192-character default rather than use the schema's 65536 maximum on
the hot path.

## Managed Worker Performance

Each LLM benchmark operation includes request and response sanitization, so
the measured latency includes two worker RPCs and two model passes. Throughput
is managed LLM operations per second, not individual classifier calls.

### Sequential Payload Size

| Semantic chars per request and response | p50 | p95 | Throughput |
|---:|---:|---:|---:|
| 64 | 4.72 ms | 5.28 ms | 202.87 ops/s |
| 1,024 | 17.40 ms | 18.49 ms | 56.60 ops/s |
| 8,192 | 122.52 ms | 127.09 ms | 8.12 ops/s |

### Representative Concurrency

| Semantic chars | Concurrency | p50 | p95 | Throughput | Failure replacements |
|---:|---:|---:|---:|---:|---:|
| 64 | 16 | 22.17 ms | 30.69 ms | 512.32 ops/s | 0 |
| 64 | 64 | 68.67 ms | 113.74 ms | 547.02 ops/s | 0 |
| 1,024 | 16 | 129.35 ms | 180.27 ms | 88.54 ops/s | 0 |
| 1,024 | 64 | 417.05 ms | 698.25 ms | 89.86 ops/s | 0 |
| 8,192 | 4 | 341.43 ms | 391.93 ms | 10.22 ops/s | 0 |

`Failure replacements` counts occurrences in emitted start/end events. It
does not count failed application operations; application callbacks remain
fail-open.

### Tool Field Fanout

| String fields per request and response | p50 | p95 |
|---:|---:|---:|
| 1 | 4.54 ms | 5.23 ms |
| 8 | 20.99 ms | 21.52 ms |
| 32 | 77.57 ms | 78.25 ms |
| 64 | 150.17 ms | 150.51 ms |

The current walker classifies strings individually. Latency therefore grows
with both aggregate characters and field count.

### Overload and Recovery

At 8 KB per request and response:

| Concurrency | p50 | p95 | Replacement occurrences / lifecycle events |
|---:|---:|---:|---:|
| 8 | 484.29 ms | 512.89 ms | 22 / 32 |
| 16 | 512.01 ms | 768.54 ms | 50 / 64 |
| 64 | 1,507.35 ms | 2,560.76 ms | 124 / 128 |

The queue is bounded by two native inference slots. Once the 250 ms callback
deadline expires, observability fails closed. Timed-out native calls retain
their slots until ONNX returns, preventing unbounded background inference.

A 64-character call recovered in 5.27 ms immediately after overload.
Payloads above the configured 8192-character limit and payloads exceeding
field, node, or depth caps failed closed before ONNX inference.

## Activation and Memory

| Measure | Result |
|---|---:|
| Cold activation, download plus ONNX load | 2.64 s |
| Repeated warm activation | 0.32-0.34 s |
| Activation RSS | 128.0 MiB |
| Warm steady-state RSS | 255.6 MiB |
| Post-overload RSS | 258.2 MiB |
| Worker close | 6.11 ms |
| Host RSS before/after full run | 50.3 / 61.3 MiB |
| Managed environment disk | Approximately 194 MiB |
| Model cache disk | Approximately 15 MiB |

The same worker handled 1200 additional managed calls across eight soak
rounds:

- Four rounds of 250 64-character calls at concurrency 16 sustained
  approximately 498-506 ops/s.
- Four rounds of 50 1 KB calls at concurrency 8 sustained approximately
  87.5 ops/s.
- Worker RSS remained between 255.7 and 255.9 MiB after warmup.

This is evidence against per-call memory growth in the measured run. It is not
a multi-hour leak test.

The first model download produced one Python 3.14 Hugging Face/Xet
`resource_tracker` warning. Warm activation and direct ONNX loading were clean,
so the warning appears isolated to the upstream download path rather than
steady worker execution.

## Validation Summary

- `uv run pytest python/tests/plugin/test_rampart_worker_plugin.py -q`:
  46 passed.
- `just test-python`: 584 passed, with 22 existing Python 3.14 tar-extraction
  deprecation warnings from `test_package_build.py`.
- `just docs`: completed with zero errors. Fern skipped the authenticated
  redirect check because no Fern token was present.
- Pre-commit on all changed files: all applicable hooks passed, including
  SPDX, JSON/TOML, Ruff, formatting, `ty`, and Markdown links.
- The current manifest passed the real debug CLI's
  `plugins validate ... --json` integrity and policy checks.
- The worker wheel and source distribution built successfully, the wheel
  contained `ATTRIBUTION.md`, and the installed managed environment passed
  `pip check`.
- The final comprehensive benchmark closed the activation in 6.11 ms and left
  no worker process behind.

## Scale Assessment

The default is appropriate for local observability payloads when most fields
are short. It is not appropriate for repeatedly classifying full 8 KB request
and response histories at high concurrency.

Important scale properties:

- One warmed worker retains roughly 256 MiB RSS on this machine.
- Each Relay process with this plugin has its own worker and model state.
- `max_concurrency = 2` is a deliberate latency/memory compromise.
- Short fields scale well because ONNX releases the GIL and the worker permits
  two concurrent native calls.
- Larger fields saturate the two slots and fail closed under burst load.
- Increasing concurrency increases model memory and should not be done
  without a target-platform sweep.
- The worker must receive bounded event deltas rather than repeated full
  conversation history for predictable hot-path cost.

## Residual Risks

- Rampart is harm reduction, not an adversarial security guarantee.
- Non-Latin names, indirect identifiers, and uncommon regulated identifiers
  require other controls.
- False positives can remove useful observability data.
- Fail-closed behavior intentionally replaces entire affected fields when the
  model, worker, deadline, or payload limits fail.
- Model-backed sanitization does not cover generic mark data in this plugin.
- Protocol IDs, model names, roles, statuses, and tool/function names are
  intentionally preserved.
- Provider headers enter the trusted local worker RPC before the returned
  observability copy clears them.
- Large encoded media enters the local worker before replacement and can
  create transient transport allocation.
- Worker process isolation is not a sandbox.
- Model cache retention is user-owned and survives `plugins remove`.
- CC BY 4.0 attribution and NVIDIA dependency/license approval are required
  before first-party distribution.

## Reproduce

After `plugins add`, prepare the pinned model once with
`allow_network = true`, then restore offline activation.

Run focused correctness checks:

```bash
uv run pytest python/tests/plugin/test_rampart_worker_plugin.py -q
uv run ruff check examples/rampart-pii-worker-plugin \
  python/tests/plugin/test_rampart_worker_plugin.py
uv run ty check examples/rampart-pii-worker-plugin \
  python/tests/plugin/test_rampart_worker_plugin.py
```

Run the model-only quality probe:

```bash
cd examples/rampart-pii-worker-plugin
PYTHONPATH=. python evaluate.py --output /tmp/rampart-quality.json
```

Run the real managed worker benchmark with the environment recorded by
`nemo-relay plugins inspect`:

```bash
python benchmark.py \
  --mode worker \
  --manifest ./relay-plugin.toml \
  --environment /path/to/managed/environment \
  --profile comprehensive \
  --output /tmp/rampart-worker.json
```
