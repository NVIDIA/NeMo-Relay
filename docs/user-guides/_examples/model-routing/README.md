<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Evaluate fixed and routed coding tasks

The companion uses pi → Relay gateway + Switchyard native plugin → NVIDIA
inference. The same files run on Brev or an equivalent Linux CPU host.
Use a host without another example's system-wide Relay policy.

## Prerequisites

Python 3.12, Node.js 24.13.1+, pi 0.84.0, Relay 0.10.4, and a trusted
[Switchyard plugin 0.3.2](https://github.com/NVIDIA/NeMo-Relay-Plugins/releases/tag/switchyard-plugin-0.3.2)
bundle. Native code runs inside Relay; the example's integrity-only policy
requires you to trust the artifact's publisher.

Install pi's extension with `nemo-relay install pi`. Then:

```bash
SWITCHYARD_MANIFEST=$(bash install-plugin.sh /tmp/relay-switchyard-bundle)
export SWITCHYARD_MANIFEST
```

The Linux x86_64 helper verifies the downloaded SHA-256 sidecar. On other
supported platforms use Relay's plugin installer and set the installed manifest.

## Configure and run

Export `NVIDIA_API_KEY`, `NVIDIA_BASE_URL`, `EFFICIENT_MODEL`, and
`CAPABLE_MODEL`. Set `JUDGE_MODEL` to an accessible model supporting JSON
verdicts; it otherwise defaults to the efficient target.
The answer models must support tools and the configured output budget.

```bash
python3.12 run.py --output /tmp/relay-routing --repeat 3
python3.12 summarize.py /tmp/relay-routing
```

Use a new output directory. `--prepare-only` writes inspectable configuration
without inference. The startup probe checks native-plugin loading before tasks.

The workload has incident-ID validation, retry delays, and dependency planning.
Three policies across three repetitions produce 27 attempts. Order rotates
between repetitions. All arms receive fresh scaffolds, the same tools and token
budget, and independent acceptance checks outside their working directories.

## Read the evidence

- `results.json`: task status, external acceptance, and elapsed time.
- `summary.json`: model attempts, tokens, truncation, judge errors and recoveries.
- `logs/events.jsonl`: model/tool scopes and Switchyard decision evidence.

A successful HTTP call can still produce an invalid judge verdict. Inspect
classifier recoveries separately from answer fallback flags. A capable selection
caused by recovery does not demonstrate a valid capability forecast.

No monetary savings are inferred. Apply validated input/output prices by target,
including judge usage and failed attempts, before comparing cost per accepted task.
Keep failed trials; the runner returns nonzero if any policy fails acceptance.

## Cleanup

Retain the scorecard, then remove your selected output and dedicated bundle
directories. The runner uses isolated pi/plugin state and stops each gateway.
Leave other users' settings and installed plugins in place.
