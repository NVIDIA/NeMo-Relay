<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Compare pi coding tasks with NVIDIA model routing

This companion uses **pi → Relay CLI gateway + Switchyard native plugin → NVIDIA
inference**. It extends the pi quickstart's installed extension, wrapped run,
OpenAI-compatible request format, and local ATOF output. It uses the native
plugin inside Relay, rather than Switchyard's standalone evaluation server.

The same source runs on a Brev CPU VM and a comparable Linux test host with
external model access. Clone the guide branch and run from this directory. A
premade Launchable and a GPU are unnecessary when inference is remote. This is
a small comparison fixture; use a larger representative workload before making
a production model-policy decision.

## Prerequisites and choices

Install pi 0.84.0 and Relay 0.11.0-rc.1 as in the pi quickstart, with Node.js 24.13.1+ and
Python 3.12. Obtain and verify a compatible Switchyard plugin bundle for the
host architecture, then set `SWITCHYARD_MANIFEST` to its completed
`relay-plugin.toml`. Native plugins run inside Relay: inspect and trust the
bundle before loading it. The public release is
[Switchyard plugin 0.3.2](https://github.com/NVIDIA/NeMo-Relay-Plugins/releases/tag/switchyard-plugin-0.3.2). The script
does not download arbitrary native code or bypass a failed plugin validation.

Supply these variables through the launching environment:

| Variable | Meaning |
| --- | --- |
| `NVIDIA_BASE_URL` | Your NVIDIA endpoint's OpenAI Chat Completions base URL, including `/v1` |
| `NVIDIA_API_KEY` | Credential for that endpoint, used by the Switchyard plugin |
| `EFFICIENT_MODEL` | Tool-capable efficient Nemotron model ID served there |
| `CAPABLE_MODEL` | Tool-capable capable Nemotron model ID served there |
| `JUDGE_MODEL` | Optional; defaults to the efficient model and must support the configured JSON-object verdict |
| `SWITCHYARD_MANIFEST` | Absolute path to the verified native-plugin manifest |

Select exact model IDs from your accessible endpoint. Nano and Super are
candidates, not a tested pairing or a guaranteed quality ranking. Both targets
must support pi's tools and a context of at least 32768 tokens with 16384 output
tokens; adjust the generated `pi/models.json` limits for your actual models.
The judge uses `json_object` and Switchyard's local schema validation. Confirm
judge success in telemetry: failure can route every call to the capable model,
making a successful answer an inadequate routing check.

```bash
# Inspect configuration and task files without loading a plugin or calling a model.
python3.12 run.py --output /tmp/relay-routing-config --prepare-only

# With endpoint access, model IDs, and the verified bundle supplied:
python3.12 run.py --output /tmp/relay-routing-comparison --repeat 3
python3.12 summarize.py /tmp/relay-routing-comparison
```

Use new output directories. The runner isolates pi's configuration with
`PI_CODING_AGENT_DIR` and Relay's user configuration with `XDG_CONFIG_HOME`.
It preserves the host's administrator policy. Secrets remain environment values;
the generated files contain only the key's variable name. The generated trust
override accepts digest-verified unsigned bundles; signed enterprise builds can
use a stricter policy.

## What runs

Three arms use the same pi model metadata, Relay gateway, native plugin, tools,
task inputs, and acceptance checks:

1. `baseline`: a Switchyard passthrough route always selects the capable model.
2. `efficient`: a passthrough route always selects the efficient model.
3. `routed`: a capability classifier selects between them, starting at threshold 0.8. Raising the threshold sends more requests to the capable target.

Each arm starts from the same source scaffold in a fresh task directory. One
task implements an incident-ID normalizer; another implements a capped retry
policy. The agent writes and runs its own tests. Separate acceptance checks run
afterward from this source directory, and `results.json` retains every task's
agent exit status, acceptance result, and elapsed time. This is not a sandboxed
benchmark; run only trusted tasks in the disposable example environment. A
nonzero result remains evidence to investigate, rather than disappearing from
the comparison.

## Inspect the result

Inspect each arm's `logs/events.jsonl` for model/tool activity and Switchyard
routing marks. Compare the requested route, selected target, served model,
fallback flag, routing overhead, and available token usage. The virtual provider
name `relay-nvidia` and virtual model IDs keep pi's route selection distinct from
the actual NVIDIA targets. The plugin owns the NVIDIA credential, so pi's custom
provider uses an inert placeholder key. If the plugin is disabled, that placeholder
must not be mistaken for working NVIDIA authentication.

`results.json` reports task acceptance and latency. It does not infer routing
success from acceptance, calculate savings, or apply prices. Review the trace
evidence separately and repeat with a larger workload. A routing run need not
select both models; if it always selects one, inspect judge errors and task mix
before interpreting that behavior.

## Adapt the setup

pi and NVIDIA-hosted models are the concrete example choices. Other
Relay-supported harnesses, compatible model endpoints, self-hosted NIMs, and
routing algorithms are possible. Keep the baseline, tools, acceptance criteria,
and evidence comparable when changing them. For an administrator-managed daemon,
install the bundle and policy in the worker's system configuration and load the
managed harness integration. Do not deploy this disposable personal configuration
as organization policy.


## Install the public bundle without changing existing user configuration

Install the [Relay CLI](https://docs.nvidia.com/nemo/relay/getting-started/installation)
and pi with the [pi quickstart](https://docs.nvidia.com/nemo/relay/getting-started/quick-start/pi).
On the Ubuntu x86_64 glibc Brev example, run the public release helper:

```bash
SWITCHYARD_MANIFEST=$(bash install-plugin.sh /tmp/relay-switchyard-bundle)
export SWITCHYARD_MANIFEST
```

The helper downloads plugin 0.3.2 and its SHA-256 sidecar, verifies the checksum,
and prints the extracted manifest path. No GitHub account is required. On another
platform, use the Relay plugin installer with an authenticated GitHub CLI:
`nemo-relay plugins install --user github:NVIDIA/NeMo-Relay-Plugins@switchyard-plugin-0.3.2 --no-enable`.
Use a dedicated `XDG_CONFIG_HOME` for that command, then set the manifest path
reported in that directory's `nemo-relay/plugins.toml`.

Use an unused absolute directory in place of `/tmp/relay-switchyard-bundle`.
Installer checksums verify integrity, not independent publisher authenticity.
Native code runs in Relay's process. The runner enables and validates it only
inside the comparison's isolated configuration. `--config` discovers the sibling
`plugins.toml`; keep those two files together when adapting the example.

For the public NVIDIA endpoint, start with `https://integrate.api.nvidia.com/v1`
and candidates `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` (efficient/judge)
and `nvidia/nemotron-3-super-120b-a12b` (capable). These are published IDs, **not a
live-validated pairing** for this workload. Check your account's model availability,
tool support, and JSON-object judge responses before interpreting the comparison.

The runner rotates arm order between trials. Repeats use fresh task directories,
but provider caches and load can still affect timing. This is a small synthetic
comparison, not a statistically controlled benchmark.

## Read the scorecard

`summarize.py` writes `summary.json` and prints task acceptance, elapsed time,
answer-call attempts by target, judge call errors, classifier recoveries, answer fallback flags, and answer/routing
tokens. Counts include observed failed calls; internal HTTP retries may not emit separate call marks. Selected targets describe attempts;
`served_models` in the JSON records the actual model reported in decision marks.
The summary counts only `token_type=total`, never its constituents again. A token
sum is shown only when each observed call of that role has a total-token metric;
missing or partial usage stays `unknown`. `no calls` means no observed calls of
that role, not established zero usage. Inspect telemetry coverage before drawing
conclusions. A missing decision log prevents the runner from reporting completion.

## Cleanup

After the runner has stopped, retain the comparison outputs you need and delete
only its output directory and your dedicated bundle directory. Never remove the
bundle while a comparison is running. No global plugin needs disabling: user
plugin state and pi files used here are isolated. System policy still applies.
The agent can access other files visible to your user; isolation is configuration
isolation, not a security sandbox. Use a disposable VM and trusted synthetic tasks.

The pinned Relay version is a release candidate: use `NEMO_RELAY_VERSION=0.11.0-rc.1`
with the official installer. Stable `0.11.0` assets are not available yet.

Use a clean host for this companion. The two-user coding-agent companion installs
system-wide plugin configuration; if it is present on the same host, its
administrator policy can override this example's exporter or routing settings.
Complete its cleanup before switching scenarios, or use separate VMs.

Classifier recovery is separate from `fallback_used`: an invalid judge verdict
can choose the capable model with `evidence.source=fail_open` (or `fall_open` in
0.3.2), even when the HTTP call succeeded and `fallback_used=false`. The summary
records those recoveries and reason codes separately. Do not add judge errors
and recovery counts as though they were disjoint failures.

The generated `[upstream].openai_base_url` matches `NVIDIA_BASE_URL` in the
Switchyard client and pi custom provider. Keep these aligned: older pi adapters
only redirect providers that the gateway already fronts. A provider response
that bypasses Relay does not verify the routing comparison.

## Output budget and tool diagnostics

The companion uses the same `read`, `write`, and `bash` tools for every arm and
records pi's JSON event stream in each `agent.log`, including tool failures.
This keeps the fixture focused on implementing and testing code with a fixed
toolset. Other toolsets are possible; keep them identical across your comparison.

The default per-call output budget is 16384 tokens, configurable with
`--max-output-tokens`. It includes the provider's reasoning budget where
applicable. An initial 4096-token run exhausted that budget and left code
unfinished even though pi exited zero. The scorecard now counts model
`finish_reason=length` or `max_tokens` as `truncated_calls`; review that count
alongside task acceptance. A larger budget does not guarantee a better answer.
The example caps this setting at 16384 to retain input space in its configured
32768-token context. Adjust model limits together when adapting the fixture.

## Measured run and expected nonzero exit

The Brev comparison used `https://inference-api.nvidia.com/v1`, efficient/judge
`nvidia/nvidia/Nemotron-3-Nano-30B-A3B`, capable
`nvidia/nvidia/nemotron-3-super-v3`, `--repeat 1`, and `--max-output-tokens 16384`.
Use those account-specific values only where accessible; the public catalogue
uses different IDs. Every arm accepted the retry policy and rejected the ticket
normalizer's Unicode-digit handling: 1/2 tasks accepted per arm. All agent exit
codes were zero. The runner correctly returned nonzero for the rejected tasks.

The measured routed arm used seven Nano and eight Super answer calls, with
46,536 answer tokens and 39,105 routing tokens. Six classifier recoveries were
reported (five parse errors, one invalid verdict). No monetary prices were
applied. See the guide for the full scorecard and a real routing mark. A completed
comparison can reject the proposed policy; the example does not guarantee that
routing improves your workload or that any model satisfies every requirement.
