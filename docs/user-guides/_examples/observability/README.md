<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Observe a support-triage agent

This companion to **Connecting to Observability Platforms** uses Hermes Agent's
native Relay integration to read a synthetic incident queue and write a handoff.
The same files run on a Brev CPU VM or a Linux development host. No Launchable is
required. Use a fresh directory for every run; the example creates its own files.

## Prepare a host

Use Python 3.12 and a Hermes installation with native Relay support and
`chat --oneshot`. Install and authenticate Hermes using its installation guide,
select a tool-capable model, and inspect the NeMo Relay Plugins section of
`hermes doctor`. Hermes owns its Relay dependency; installing the Relay CLI is
not required for this example. Use a dedicated Hermes installation/profile with
no unrelated gateway session, and check that administrator plugin policy permits
the selected exporter. A separately running Hermes gateway can retain an earlier
configuration; do not stop a gateway used by other work.

In Brev, create an ordinary CPU VM, clone this guide branch, and use the same
instructions below from this directory. The instance supplies the host; the
workload and Relay configuration do not depend on Brev APIs. Keep provider and
destination keys in the process environment, outside the repository.

## Choose a destination

| Selection | `RELAY_OTLP_ENDPOINT` | Additional environment |
| --- | --- | --- |
| `phoenix` | Defaults to `http://127.0.0.1:6006/v1/traces` for an existing local Phoenix | `PHOENIX_AUTHORIZATION` containing `Bearer <api-key>` only if the intake requires it |
| `langsmith` | Your regional intake, e.g. `https://api.smith.langchain.com/otel/v1/traces` | `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` |
| `weave` | `https://trace.wandb.ai/agents/otel/v1/traces` | `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT` |
| `datadog` | Your running Collector's complete `/v1/traces` URL | Configure the Collector's Datadog exporter and credentials separately |

Provision Phoenix or the destination receiving tier before running. The script
does not install a server. The destination sections in the guide explain the
intake and UI checks. For an authenticated Collector, add its approved
`header_env` mapping to the generated file after `--prepare-only`; run Hermes
with `HERMES_NEMO_RELAY_PLUGINS_TOML` pointing to that file.

```bash
# Inspect a generated configuration without calling a model or exporter.
python3.12 run.py --destination phoenix --output /tmp/relay-triage-config --prepare-only

# After setting provider credentials and starting Phoenix:
python3.12 run.py --destination phoenix --output /tmp/relay-triage-phoenix
```

Select another destination with the same script and a new output directory.
The template renders secrets as environment references, never literal keys.

## Inspect the result

The local check requires all three incident IDs in `handoff.md`, the checkout
incident before the report delay, and correlated agent/model/tool scope events in `logs/events.jsonl`. Review the
actual triage decisions as well: these checks do not grade every recommendation.
In the destination UI, find service `relay-support-triage` at the run time and
check the agent/model/tool parentage against those local events. Local output
alone does not prove remote delivery. Payload capture is disabled; inspect the
exported fields permitted by your policy.

The example fixes the workload so destination differences are easier to inspect.
In an enterprise deployment, replace Hermes with your supported capture path,
the synthetic queue with your approved task, and the endpoint and service name
with your receiving tier. Retain the same trace verification.


## Installation, diagnostics, and cleanup

Follow the [Hermes quickstart](https://docs.nvidia.com/nemo/relay/getting-started/quick-start/hermes)
for installation and provider authentication. Confirm `hermes chat --help` lists
`--oneshot`; in `hermes doctor`, inspect NeMo Relay Plugins for the selected user
file and system-policy conflicts. Stop/restart only your dedicated profile's
gateway after changing its environment; do not restart a shared gateway.

Start a disposable Phoenix with `uvx --from arize-phoenix phoenix serve` in a
separate terminal. On Brev, run the workload on the instance; on your computer,
use `brev port-forward YOUR_INSTANCE --port 6006:6006` (add `--host` if needed).
Use the same clone and run commands as on any Linux CPU host.

Phoenix uses project and service `relay-support-triage`. For a private-network
HTTP Collector, explicitly add `--allow-insecure-endpoint`; secrets and trace
data can then travel unencrypted, so use it only on a trusted network. URLs with
embedded credentials are rejected. The default timeout is 600 seconds; timeout
stops the harness process group. Failures retain `agent.log` and any events.

Stop only your own Phoenix process with Ctrl+C and close your port forward.
Remove your chosen output directories when no longer needed; preserve shared
backends. A local success message still requires a destination UI delivery check.

Use a clean host for this companion. The two-user coding-agent companion installs
system-wide plugin configuration; if it is present on the same host, its
administrator policy can override this example's exporter or routing settings.
Complete its cleanup before switching scenarios, or use separate VMs.

## Measured Phoenix example

On the Brev CPU VM, Hermes with `NVIDIA_BASE_URL=https://inference-api.nvidia.com/v1`
and model `nvidia/nvidia/Nemotron-3-Nano-30B-A3B` completed the task in 70 seconds
including startup. The log had one closed agent scope, four LLM calls, and two
tool calls; Phoenix received all 12 exported spans in project `relay-support-triage`.
This is an account-specific endpoint/model choice; credentials are environment
values and are not part of the example files. Hosted backends require their own
independent delivery verification.
