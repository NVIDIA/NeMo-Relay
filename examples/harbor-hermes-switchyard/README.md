# Harbor + Hermes + Switchyard evaluation

This example runs a Terminal-Bench 2.0 task through Harbor and Hermes while
Hermes owns an in-process NeMo Relay 0.7.0 runtime. Relay initializes static
pricing and observability components and activates Switchyard as a standard
dynamic native plugin.

The example is deliberately a one-task integration reference. It does not
replace Harbor's task lifecycle and it is not the full 89-task coordinator.
The four-task regression command below is the Phase 1 readiness gate.

## Pinned inputs

| Dependency | Input used by this example |
|---|---|
| NeMo Relay | Released Linux/amd64 `nemo-relay==0.7.0` wheel; that exact wheel is installed and its digest is recorded per run. |
| Hermes | `bbednarski9/hermes-agent`, branch `feat/relay-native-plugin-init`, detached commit `efb63e714abc436af88af9b0d6734751c199aa6d` (PR #77915). |
| Switchyard | `bbednarski9/Switchyard`, detached commit `8293936a0f5758aa1a782639d485b8b8948cf03e` (PR #270). |
| Harbor | `harbor==0.18.0`, dataset `terminal-bench@2.0`. |

The branch names make the development inputs discoverable; only the full
commits are authoritative. Every checkout is detached and verified before
execution. The Hermes installer is followed by a final `uv sync --frozen`
against that commit's checked-in lock because its date-relative resolution
guard can otherwise make an older checkout appear stale. The verified Relay
0.7.0 platform wheel is then force-installed by digest without dependencies.
During installation only, the bridge advertises an inert `ffmpeg` command so
the task does not install an unrelated media stack; browser setup and bundled
skills are also disabled for this terminal-only evaluation.

## Request and lifecycle ownership

There is no Switchyard service in this topology:

1. Harbor creates the Terminal-Bench task environment and invokes its built-in
   Hermes lifecycle through the temporary subclass in
   `agents/harbor_hermes_agent.py`.
2. Hermes initializes Relay and asks Relay's public dynamic-plugin loader to
   activate `nvidia.switchyard` from `[[plugins.dynamic]]`.
3. Relay owns the outer managed LLM operation and invokes the native execution
   intercept.
4. The Switchyard plugin selects the route and its `switchyard-llm-client`
   performs the provider HTTP request.
5. Hermes waits for Relay operations, plugin cleanup, subscribers, and
   exporters before returning to Harbor.

That split is important: Relay dispatches into the plugin intercept, while the
pinned Switchyard plugin owns the provider HTTP client. The direct receipt
records both facts without claiming a second routing service exists.

Static components and dynamic plugins are separate concepts. The pricing and
schema-v3 observability components in `config/relay.toml.in` are static Relay
components. Switchyard is a standard dynamic native Relay plugin. Hermes
`[[dynamic_plugins]]` Python workers are not used, and the bridge rejects a
configuration that mixes the two activation models before provider traffic.

## Why the temporary Harbor agent exists

Harbor 0.18.0's built-in Hermes agent accepts a branch-like `version` and
clones the upstream NousResearch repository. It cannot select a fork plus an
immutable arbitrary commit, nor can it project this example's Relay config and
native bundle. `HarborHermesAgent` changes only installation, configuration
projection, and additional artifact framing; the inherited Harbor setup/run,
timeout, task, session export, and ATIF conversion remain in control.

Remove this bridge and use `--agent hermes` once
[hermes-agent#77915](https://github.com/NousResearch/hermes-agent/pull/77915)
is upstream **and** Harbor's built-in agent can install a released, pinned
compatible Hermes revision while projecting the Relay config and plugin
bundle. Merging the Hermes PR alone is not sufficient while Harbor remains
upstream-repository-only and branch-only.

## Prerequisites

- Docker with enough space to build one Linux/amd64 Rust plugin and task image;
- Python 3.11 or newer;
- a provider endpoint compatible with OpenAI Chat Completions;
- a Phoenix endpoint accepting OTLP/HTTP traces; and
- the provider authorization value in an environment variable.

On macOS, place bundle and run roots below a directory shared with Docker
(normally `/Users/...`). Do not assume `$TMPDIR` or `/private/tmp` is shared by
Colima merely because the same path exists inside its VM.

Create a host-side environment for Harbor and the validation tools:

```bash
cd examples/harbor-hermes-switchyard
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export HARBOR_BIN="$PWD/.venv/bin/harbor"
export PHASE1_PYTHON="$PWD/.venv/bin/python"
```

The scripts never put the authorization value into TOML or command-line
configuration. They pass the selected environment variable into the task and
scan direct artifacts, Harbor logs, ATOF, ATIF, and OpenInference evidence for
the exact secret value.

### Provider configuration ownership

The rendered `<run-root>/runtime/plugins.toml` is Relay and Switchyard's
authoritative provider configuration. It contains the target model, protocol,
upstream base URL and endpoint, and the **name** of the environment variable
holding the authorization header. `TARGET_MODEL`, `UPSTREAM_BASE_URL`, and
`UPSTREAM_AUTH_ENV` are preparation inputs used to materialize that immutable
per-run file; they are not an independent provider configuration consumed by
Relay.

Harbor 0.18.0 still requires a `provider/model` value when constructing its
built-in Hermes lifecycle, and Hermes writes that call-side model into its CLI
configuration before Relay intercepts the operation. The runner therefore
passes `openai/<target-model>` to Harbor while Switchyard uses the matching
target from `plugins.toml`. `openai` describes the caller protocol here; it
does not bypass Switchyard. Likewise, the placeholder `OPENAI_API_KEY` only
satisfies Harbor/Hermes provider validation. The real authorization value is
resolved by Switchyard from `header_env` and must remain in the environment,
not in TOML.

## Offline compatibility gate

This is a preflight prerequisite for the first Harbor task run and whenever a
Hermes, Relay, Switchyard, plugin-config, or shutdown-lifecycle input changes.
It is not repeated before every task when those inputs are unchanged, and it
does not replace the single-task or regression gates.

Build the pinned Linux plugin bundle, prepare a fresh run root, and run the
forked Hermes/Relay runtime against local fake provider and OTLP endpoints:

```bash
export EXAMPLE_ROOT="$PWD"
export SPIKE_ROOT="/absolute/new/spike-root"

"$EXAMPLE_ROOT/scripts/build_switchyard_plugin.sh" /absolute/new/switchyard-bundle
"$PHASE1_PYTHON" "$EXAMPLE_ROOT/scripts/prepare_runtime.py" \
  --run-root "$SPIKE_ROOT" \
  --switchyard-bundle /absolute/new/switchyard-bundle \
  --upstream-base-url http://127.0.0.1:8000/v1 \
  --target-model phase1/fake-model \
  --openinference-endpoint http://127.0.0.1:4318/v1/traces \
  --phoenix-project phase1-offline \
  --eval-cohort phase1-offline
"$EXAMPLE_ROOT/scripts/run_offline_compatibility_smoke.sh" "$SPIKE_ROOT"
```

This gate proves the exact detached Hermes checkout, released Relay wheel,
public loader path, one native Switchyard activation, a real fake-provider HTTP
request, routing marks, file sinks, mixed-mode rejection, and clean shutdown.

The review gate targets `linux/amd64`, matching the Terminal-Bench task
environment. On an Apple Silicon Docker host, QEMU may crash while unloading a
native Rust plugin; an ARM control can distinguish that emulator failure from
an integration failure:

```bash
SWITCHYARD_TARGET_ARCHITECTURE=aarch64 \
  "$EXAMPLE_ROOT/scripts/build_switchyard_plugin.sh" /absolute/new/arm64-bundle
"$PHASE1_PYTHON" "$EXAMPLE_ROOT/scripts/prepare_runtime.py" \
  --run-root /absolute/new/arm64-spike-root \
  --switchyard-bundle /absolute/new/arm64-bundle \
  --relay-architecture aarch64 \
  --upstream-base-url http://127.0.0.1:8000/v1 \
  --target-model phase1/fake-model \
  --openinference-endpoint http://127.0.0.1:4318/v1/traces \
  --phoenix-project phase1-offline-arm64 \
  --eval-cohort phase1-offline-arm64
PHASE1_COMPAT_PLATFORM=linux/arm64 \
  "$EXAMPLE_ROOT/scripts/run_offline_compatibility_smoke.sh" \
  /absolute/new/arm64-spike-root
```

That control validates the same source commits and lifecycle on a different
released Relay wheel architecture. It does **not** replace a passing
`linux/amd64` run on native amd64 infrastructure before merge.

`run_terminal_bench.sh` also accepts `RELAY_ARCHITECTURE=aarch64` together
with matching `SWITCHYARD_BUNDLE` and `RELAY_WHEEL` inputs. This is useful for
exercising Harbor's complete bridge and artifact path on a native Apple
Silicon Docker daemon. It remains a diagnostic control; the default and merge
gate stay `x86_64`.

## Run one Terminal-Bench task

Use a new absolute run root on every invocation:

```bash
export TARGET_MODEL="your-provider-model"
export UPSTREAM_BASE_URL="https://your-openai-compatible-endpoint/v1"
export UPSTREAM_AUTH_ENV="SWITCHYARD_PROVIDER_AUTHORIZATION"
export SWITCHYARD_PROVIDER_AUTHORIZATION="Bearer ..."
export PHOENIX_BASE_URL="https://your-phoenix-endpoint"
export PHOENIX_PROJECT="harbor-hermes-switchyard-phase1"
export EVAL_COHORT="harbor-hermes-switchyard-phase1"

./run_terminal_bench.sh /absolute/new/run-root
```

The default task is `adaptive-rejection-sampler`. Override it with
`TASK_NAME`. To avoid rebuilding Switchyard for each task, set
`SWITCHYARD_BUNDLE` to a previously built, immutable bundle. Set `RELAY_WHEEL`
to a downloaded 0.7.0 wheel to avoid a repeated package download.

A task is complete only if both of these files contain `"status": "passed"`:

- `<direct-artifact-root>/validation.json`
- `<direct-artifact-root>/phoenix-upload.json`

`reward.task_passed=false` is a valid completed benchmark observation and is
not retried when both evidence gates pass.

## Phase 1 regression gate

Run all historical risk cases independently:

```bash
./scripts/run_phase1_regressions.sh /absolute/new/regression-root
```

| Task | Assertion |
|---|---|
| `adaptive-rejection-sampler` | Provider/config projection, routing marks, receipt, cleanup, and secret scan. |
| `circuit-fibsqrt` | A deterministic post-response test fault preserves the completed response and records the late failure separately. |
| `gpt2-codegolf` | Harbor's bounded agent timeout applies and no Hermes/plugin process survives the task container. |
| `overfull-hbox` | Streaming validation and bounded Phoenix batching preserve the completed result under a larger export load. |

The deterministic `circuit-fibsqrt` fault is injected only after inherited
Hermes execution returns. It tests the result-framing regression without
corrupting the Relay plugin lifecycle or disabling Phoenix upload.

## Evidence and safety properties

Each run root is immutable and private. Preparation refuses an existing root.
The runtime snapshot contains config and dependency digests; it never contains
credential values. Direct task artifacts include:

- `direct-hermes-result.json`;
- `direct-hermes-receipt.json`;
- `relay/trajectory.atof.jsonl`;
- `relay/atif/trajectory-<session>.atif.json`;
- bounded Hermes diagnostics;
- `validation.json`; and
- `phoenix-upload.json`.

Artifact validation rejects symlinks and canonical paths escaping the declared
root. Phoenix import is streaming, bounded in batches, retry-limited, and runs
only after the task has returned and OpenInference evidence exists.

Phase 2 (a parallel 89-task cohort) and Phase 3 (multiple independent cohorts
and aggregated reporting) intentionally remain outside this first example PR.
