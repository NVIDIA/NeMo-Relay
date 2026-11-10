<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native Plugin Benchmark

Compare native middleware in the current checkout with a separate source checkout.
The benchmark loads each plugin through its production loader and checks every
returned payload. It measures request middleware and tool execution with downstream
continuations.

## Run the Benchmark

Install the repository's Rust toolchain, `just`, `uv`, and Buf. Fetch the locked
Cargo dependencies for both checkouts before running the benchmark. Then run this
command from the repository root, replacing the path with your comparison checkout:

```bash
just benchmark-native-plugins /path/to/comparison-checkout
```

The runner builds separate host and plugin binaries for each checkout. Both use
the repository's release optimization settings and the active Rust toolchain. Each
build starts with its checkout's lockfile and resolves the benchmark packages
without network access. The output records source commits, working-tree status,
compiler details, and lockfile hashes.

The runner alternates process order between repetitions. By default, each process
runs seven rounds of 200 calls per case. To change these settings, run the script
directly:

```bash
uv run python scripts/native_plugin_benchmark/run.py /path/to/comparison-checkout \
  --iterations 500 --rounds 9 --processes 4
```

Read `target/native-plugin-benchmark/report.md` for the comparison table and
`target/native-plugin-benchmark/results.json` for raw measurements.

## Understand the Measurements

The cases cover small values, nested objects with Unicode and numeric values, and
256 KiB strings. Scheduling cases include ready futures, pending futures, repeated
waker clones, and concurrent continuation calls. The request path has no
continuation, so its fanout case performs the same work as its ready case.

Timing excludes loading and registration. Allocation counting runs in a separate
pass in both the host and plugin. Counts include allocation and reallocation calls
and their requested bytes. They do not measure peak or live memory. The runner
subtracts the cost of the counter-control calls.

The ratio in the report is current latency divided by comparison latency. A value
above one means the current checkout took longer. Check small and object-heavy
payloads as well as large payloads. Run on an idle machine and repeat measurements
before drawing conclusions about small differences. These results describe the
measured middleware paths; they do not predict whole-application throughput.
