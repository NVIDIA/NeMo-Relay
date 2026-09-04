<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Daemon Transport Benchmark

This standalone Rust/Hyper fixture measures streaming transport independently
of the existing Python latency suite. It supports HTTP/1.1 persistent
connections, cleartext HTTP/2 prior knowledge, remote HTTP/2 through ALPN,
and one client pool per protocol for the entire run.

The benchmark is informational. Integrity failures make the command fail, but
latency and throughput values have no CI threshold until a stable baseline is
established.

## CI Smoke Check

Run the same short, direct-provider check used by CI:

```bash
just daemon-transport-benchmark-smoke
```

The command starts an ephemeral deterministic provider in-process and checks
the load driver against that provider for OpenAI and Anthropic streams over
HTTP/1.1 and HTTP/2. It sends 128 events and 16 KiB per response, verifies the
body hash and trailers, and writes
`target/benchmark-results/daemon-transport-smoke.json`. This fast check does
not launch Relay and therefore does not exercise a daemon or worker hop.

## Authenticated Worker-Only Runs

A production worker accepts only broker-authenticated requests. Its endpoint
and credential are intentionally private, so do not scrape a daemon's state or
invent a worker header value. Supply a Relay binary to let the load driver
create an isolated worker target through the real activation flow:

```bash
just daemon-transport-benchmark-provider --bind 127.0.0.1:48100

# In another terminal:
just daemon-transport-benchmark \
  --config scripts/latency_benchmark/config/daemon-transport-smoke.toml \
  --direct-url http://127.0.0.1:48100 \
  --worker-binary target/release/nemo-relay \
  --output target/benchmark-results/daemon-transport-worker-smoke.json
```

`--worker-binary` adds a `worker-only` target. The benchmark starts a real
daemon, MCP, and worker with isolated temporary state. A benchmark-only control
proxy binds an ephemeral loopback port, permits only Relay control-plane POST
paths, and verifies the authenticated worker registration and readiness flow.
It captures the worker endpoint and credential, forwards load directly to the
worker, and releases the MCP reference during cleanup. Harness credentials
remain in process memory and are never printed or serialized; the endpoint
appears in the report as ordinary target metadata. The provider URL must be an
HTTP origin with a numeric loopback address and explicit port because every
process and connection created by this orchestration is local and ephemeral.

This run exercises the direct baseline and the worker data hop for both API
shapes and both HTTP protocols in the selected matrix. The helper daemon is
used only for worker activation and lifecycle control; it is not in the
`worker-only` request path.

## Full Topology Run

Build both Relay candidates without changing the workspace release profile:

```bash
just daemon-transport-benchmark-build-candidates
```

The normal build is `target/release/nemo-relay`. The `opt-level=3` build is
`target/daemon-benchmark-opt3/release/nemo-relay`. The second build uses the
`CARGO_PROFILE_RELEASE_OPT_LEVEL` environment override and a separate target
directory, so it does not modify `Cargo.toml` or replace the normal release
binary.

Start the deterministic provider on a fixed port before starting the Relay
processes being measured:

```bash
just daemon-transport-benchmark-provider --bind 127.0.0.1:48100
```

Configure each public Relay topology to use `http://127.0.0.1:48100` as its
provider. Keep those processes running and pass their public daemon URLs to the
load driver. A live MCP registration must own each public route for the entire
run. Use a pass-through daemon for `daemon-pass-through` and a normal daemon
with its activated worker for `daemon-worker`.

Add `--worker-binary` to create the broker-authenticated `worker-only` target;
there is no supported direct `--target worker-only=...` form:

```bash
export NEMO_RELAY_CLIENT_TOKEN='...'

just daemon-transport-benchmark \
  --direct-url http://127.0.0.1:48100 \
  --worker-binary target/release/nemo-relay \
  --target daemon-pass-through=http://127.0.0.1:47632 \
  --target daemon-worker=http://127.0.0.1:47633 \
  --header-env daemon-pass-through:x-nemo-relay-client-token=NEMO_RELAY_CLIENT_TOKEN \
  --header-env daemon-worker:x-nemo-relay-client-token=NEMO_RELAY_CLIENT_TOKEN \
  --pid pass-through-daemon=1234 \
  --pid worker-daemon=1235 \
  --binary-metadata release=target/release/nemo-relay \
  --binary-metadata opt-level-3=target/daemon-benchmark-opt3/release/nemo-relay
```

Header values are read from environment variables and are neither serialized
into the report nor printed. Only public daemon targets accept header bindings;
the worker credential comes from the verified activation flow and never enters
the environment or command arguments. The report records configured header
names, not their values.

Run the command once against processes from the normal release build and once
against equivalent processes from the `opt-level=3` build. Use a distinct
`--output` path for each run. Compare runs made on the same otherwise-idle
host.

The full preset performs a 10-second warmup followed by a 60-second measured
interval for each configured topology and each combination of:

- direct provider (always), daemon pass-through (when supplied), worker-only
  (with `--worker-binary`), and daemon-plus-worker (when supplied);
- OpenAI Responses and Anthropic Messages;
- HTTP/1.1 and HTTP/2;
- 16 KiB and 1 MiB streamed responses with 128 events;
- concurrency 1, 16, 64, and 256.

It also starts a separate 1,000-slow-stream capacity scenario for each
configured topology and protocol. One percent of the sustained requests are
cancelled after first content to exercise cancellation propagation. Remove
public targets or omit `--worker-binary` to benchmark a smaller subset; the
direct provider baseline is always present.

## Result Schema

The JSON report records response-head, first-content, per-event forwarding,
and total latency distributions; requests per second; MiB/s goodput; process
CPU and peak RSS samples; RSS growth per active stream; connection attempts;
estimated pool reuse; active HTTP/2 streams; cancellation and reconnect
counts; and missing, duplicate, reordered, cross-stream, corrupt, hash, status,
and trailer errors.

Hyper does not expose queue depth or backpressure-stall counters from outside
the daemon and worker. Those fields are present as `null` with an explicit
reason. Supply daemon/worker telemetry when that instrumentation becomes
available instead of deriving misleading estimates from client timings.

Body delivery is consumed frame-by-frame. The benchmark parser observes SSE
semantics only to calculate integrity and event-delay metrics; it is not part
of Relay's delivery path.
