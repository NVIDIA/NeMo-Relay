# Harbor + Hermes + Switchyard evaluation

This example runs one complete Terminal-Bench 2.0 cohort through Harbor and
Hermes. Hermes owns an in-process NeMo Relay runtime satisfying `nemo-relay>=0.7.0`; Relay loads the
Switchyard native plugin, and Switchyard selects and calls the configured
provider route. It runs one resumable 89-task cohort. Multi-cohort execution
and result aggregation are intentionally out of scope for this example.

## 1. Pinned inputs

| Dependency | Input used by this example |
|---|---|
| NeMo Relay | Latest released `nemo-relay>=0.7.0` platform wheel, installed by digest rather than from this source checkout. |
| Hermes | `bbednarski9/hermes-agent`, detached commit `a3d472f0e6bdc376df87b1436a461c4796db6747` from PR #77915. |
| Switchyard | `bbednarski9/Switchyard`, detached commit `8daac03edf8544144833af1fd009b3da737715bc` from PR #270. |
| Harbor | `harbor==0.18.0`, local export of dataset `terminal-bench@2.0`. |

Every source checkout is detached and verified. The Hermes installer is
followed by `uv sync --frozen`, then the selected released Relay wheel is
force-installed without dependencies and verified by digest.

## 2. Request and lifecycle ownership

There is no Switchyard service in this topology:

1. Harbor owns the task container, Hermes lifecycle, timeout, verifier, and
   task artifact collection.
2. The temporary adapter in `agents/harbor_hermes_agent.py` installs the exact
   Hermes commit and projects Relay's config and native bundle.
3. Hermes initializes Relay; Relay's public loader activates
   `nvidia.switchyard` from `[[plugins.dynamic]]`.
4. Relay dispatches the managed operation into the native intercept.
5. Switchyard selects a route and its client owns the provider HTTP request.
6. Hermes waits for operations, plugins, subscribers, and exporters before
   returning to Harbor.

The adapter can be removed after
[hermes-agent#77915](https://github.com/NousResearch/hermes-agent/pull/77915)
is upstream and Harbor can install an immutable compatible Hermes revision
while projecting the Relay configuration and plugin bundle.

## 3. Configuration ownership

The two configuration files have deliberately different responsibilities:

- `.env.example` is copied to an untracked, mode-`0600`
  `.env`. It contains per-machine paths, the run identity, Phoenix
  destination, manually selected capacity, and the real
  `SWITCHYARD_PROVIDER_AUTHORIZATION` header.
- `config/plugins.toml.in` is checked in and non-secret. It is the only source
  of provider URLs, protocols, strong, weak, and judge models, routing/classifier
  policy, native plugin manifest, authorization variable **name**, Relay
  components, and OpenInference export behavior.

The template configures AWS-hosted Opus 5 as the strong route, AWS-hosted
Sonnet 5 as the efficient route, and AWS-hosted Sonnet 4.6 as the classifier
judge, with a `0.5` threshold and
session affinity. The coordinator derives its required route-diversity gates
from the strong and efficient targets, while preflight also verifies the judge.
Environment variables cannot override these settings.
Runtime rendering is limited to the Hermes revision, collector endpoint,
Phoenix project/cohort attributes, and task-owned artifact locations.

For each task, the runner writes only the provider Authorization header to a
mode-`0600` file in the host's canonical private temporary directory,
explicitly outside the run root, and bind-mounts it read-only at
`/run/secrets/switchyard-provider-authorization`.
The Hermes bridge reads and exports it inside the task container immediately
before the agent command. The credential value is therefore absent from Harbor
configuration, Docker Compose arguments, plans, logs, and retained evidence;
the temporary file is removed when the task runner exits.

Harbor still requires a caller model. The example uses the intentionally
unserved `openai/ollama-route-stub` identity and projects a dead local OpenAI
endpoint. If Switchyard is bypassed, the request fails closed instead of
reaching a provider.

## 4. Host prerequisites

- Linux or macOS, Bash, Python 3.11+, Docker, and `tmux`;
- a local, immutable Terminal-Bench 2.0 dataset export containing 89 tasks;
- a Switchyard plugin bundle and released Relay wheel satisfying `nemo-relay>=0.7.0`, matching Docker's
  architecture (`x86_64` or `aarch64`);
- a Phoenix endpoint accepting OTLP/HTTP OpenInference traces; and
- provider and registry access for the full cohort. The all-89 admission uses
  neither; the Docker admission makes no provider calls but may pull its image,
  pinned sources, and packages when they are not cached.

On macOS, keep the dataset, bundle, wheel, admission, and run roots under a
directory shared with Docker (normally `/Users/...`).

Install the exact Harbor-side requirements:

```bash
cd examples/harbor-hermes-switchyard
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Copy and protect the environment file at the example root. Replace every
placeholder, including the complete provider Authorization header. Do not
source this file into the interactive shell used to start `tmux`.

```bash
cp .env.example .env
chmod 0600 .env
./scripts/validate_phase2_environment.sh .env
```

The validator reports names and paths only. It rejects legacy secret-file
variables and never renders or prints the authorization value.

## 5. Prepare and validate a cohort run

Run every stage with the same immutable inputs. If the dataset, concurrency,
architecture, Relay wheel, Switchyard library, plugin template, or Hermes
commit changes, regenerate the affected admission evidence before creating a
plan.

For the commands below, enter a short-lived shell with tracing disabled:

```bash
set +x
set -a
source .env
set +a
set +x
```

## 6. Verify the complete dataset without provider tokens

This all-89 no-token admission loads and uniquely selects all tasks, hashes their instructions and
verifiers, expands the complete Harbor job graph, denies registry/provider
access, and renders the runtime. It starts neither Docker nor an agent.

```bash
mkdir -p "$TERMINAL_BENCH_ADMISSION_ROOT"
chmod 0700 "$TERMINAL_BENCH_ADMISSION_ROOT"
"$EVAL_PYTHON" "$EXAMPLE_ROOT/scripts/smoke_phase2_dataset.py" \
  --dataset-root "$TBENCH_DATASET_PATH" \
  --expected-count 89 \
  --concurrency "$TBENCH_CONCURRENCY" \
  --harbor-bin "$HARBOR_BIN" \
  --switchyard-bundle "$SWITCHYARD_BUNDLE" \
  --relay-wheel "$RELAY_WHEEL" \
  --relay-architecture "$RELAY_ARCHITECTURE" \
  --plugin-config-template "$PLUGIN_CONFIG_TEMPLATE" \
  --output "$TERMINAL_BENCH_SMOKE_EVIDENCE"
```

The passed evidence binds task names, task/instruction/verifier hashes,
concurrency, architecture, Relay wheel, Switchyard library, and plugin config.

## 7. Verify the offline container runtime

The Docker offline runtime admission uses a fresh admission root with test-only
structured overrides. Production
model, URL, and routing values remain owned by `plugins.toml.in`; these flags
exist only to point this closed offline test at its fake endpoints.

```bash
OFFLINE_ROOT="$TERMINAL_BENCH_ADMISSION_ROOT/offline-runtime"
"$EVAL_PYTHON" "$EXAMPLE_ROOT/scripts/prepare_runtime.py" \
  --run-root "$OFFLINE_ROOT" \
  --switchyard-bundle "$SWITCHYARD_BUNDLE" \
  --relay-wheel "$RELAY_WHEEL" \
  --relay-architecture "$RELAY_ARCHITECTURE" \
  --plugin-config-template "$PLUGIN_CONFIG_TEMPLATE" \
  --test-provider-base-url http://127.0.0.1:8000/v1 \
  --test-strong-model phase2/fake-strong \
  --test-weak-model phase2/fake-weak \
  --test-judge-model phase2/fake-judge \
  --openinference-endpoint http://127.0.0.1:4318/v1/traces \
  --phoenix-project phase2-offline \
  --eval-cohort phase2-offline

case "$RELAY_ARCHITECTURE" in
  x86_64) export OFFLINE_COMPAT_PLATFORM=linux/amd64 ;;
  aarch64) export OFFLINE_COMPAT_PLATFORM=linux/arm64 ;;
  *) echo "unsupported architecture" >&2; return 2 ;;
esac
"$EXAMPLE_ROOT/scripts/run_offline_compatibility_smoke.sh" \
  "$OFFLINE_ROOT" "$TERMINAL_BENCH_OFFLINE_EVIDENCE"
```

This performs real Hermes→Relay→Switchyard calls against local fake provider
and OTLP endpoints and proves route selection, authorization injection,
observability, pinned-library loading, and clean shutdown. Its evidence binds
the same Hermes commit, Relay wheel, Switchyard library, architecture, and
plugin template consumed by the cohort.

## 8. Create the immutable run plan

The first command writes `plan.json`; any later invocation with different
immutable inputs is refused. Choose concurrency before this point.

```bash
"$EXAMPLE_ROOT/run_phase2_cohort.sh" "$TERMINAL_BENCH_RUN_ROOT" --plan-only
```

**Optional canary-first scheduling.**

The default `TBENCH_CANARY_TASK=adaptive-rejection-sampler` runs that one real
task first. A completed benchmark result (`validation.benchmark.status=passed`)
opens the parallel lane even when its benchmark reward is a non-pass. Phoenix
upload and other integration evidence remain cohort-level acceptance gates.
This is a conservative production check, not a separate command: launching the
full cohort on a fresh run root automatically starts the canary and then
continues with the remaining tasks.

To skip that one-task checkpoint, set an explicitly blank value in `.env`:

```bash
TBENCH_CANARY_TASK=
```

With the canary disabled, the full cohort starts immediately. Task 1 remains
the first selected task, but it is scheduled in the normal parallel or serial
lane rather than running alone first.

## 9. Check capacity and provider availability

```bash
"$EXAMPLE_ROOT/run_phase2_cohort.sh" "$TERMINAL_BENCH_RUN_ROOT" --preflight-only
```

Preflight authenticates to each configured provider's model catalog and
requires both TOML-owned route models to be present without persisting the
authorization value. It writes `preflight.json` with the verified model IDs,
Docker CPU/memory/architecture, free disk, configured endpoints, selected
concurrency, reserve, and the calculated requirement. It rejects:

```text
max(concurrency × parallel_task_memory_gb, largest_task_memory_gb)
  + docker_reserve_gb > Docker memory
```

It also rejects less than the configured free-disk minimum (100G by default),
concurrency above Docker's CPU count, and an architecture mismatch. The
defaults are a 2G parallel lane and 4G Docker reserve.

## 10. Launch and resume the cohort

Exit the secret-bearing admission shell first. From a shell where the protected
file has **not** been sourced, start one detached supervisor. Only the file path
is placed in the tmux server environment; the child sources it with xtrace
disabled and persists output below the run root.
Before the supervisor starts, the launcher copies only the plan-bound harness
sources into `runtime-harness/` below the run root and verifies their aggregate
hash. Retries execute this snapshot, so later checkout changes cannot alter an
active cohort. No environment file or secret is copied into the snapshot.

```bash
exit  # only when returning from the short-lived admission shell above
./scripts/launch_phase2_tmux.sh harbor-hermes-switchyard-phase2-run-1
```

Operational commands:

```bash
# Detect a live duplicate (success means the session exists).
tmux has-session -t harbor-hermes-switchyard-phase2-run-1

# Attach; detach without stopping the run with Ctrl-b d.
tmux attach-session -t harbor-hermes-switchyard-phase2-run-1

# Inspect durable output and sanitized cohort progress.
tail -F /absolute/path/to/phase2-run-root/supervisor.log
jq '{status,completed_tasks,planned_tasks,benchmark_pass_count,benchmark_nonpass_count}' \
  /absolute/path/to/phase2-run-root/summary.json

# Graceful interruption.
tmux send-keys -t harbor-hermes-switchyard-phase2-run-1 C-c

# After the old session exits, resume from the run-bound snapshot. This also
# works after a checkout update or host reboot.
/absolute/path/to/phase2-run-root/runtime-harness/scripts/launch_phase2_tmux.sh \
  /absolute/path/to/examples/harbor-hermes-switchyard/.env \
  harbor-hermes-switchyard-phase2-run-1
```

The supervisor returns `0` after complete acceptance, `20` for a preserved
integration/harness blocker, and retries other exits with bounded exponential
backoff. The OS advisory lock rejects a second live coordinator. Validated
attempts are immutable and preserved on restart. `tmux` survives terminal
logout, not host reboot; after reboot, launch it again against the same root.
Agent setup also retries transient `apt-get` failures three times locally;
exhausted package-manager failures are classified as infrastructure and remain
subject to the cohort's bounded retry limit.

## 11. Completion gates

A task is complete when `validation.json` records
`benchmark.status=passed`. The top-level validation status mirrors benchmark
completion for compatibility.
A benchmark `reward.task_passed=false` is a valid completed result and is never
retried.

Relay/Switchyard artifact checks, including the Phoenix upload result, are
recorded separately in `validation.integration`. An integration finding is
preserved in the task and cohort report; it does not discard a completed
benchmark result or cause the agent to be run again. It remains a cohort-level
acceptance gate.

The cohort passes only when:

- all 89 tasks have independently completed benchmark results;
- `cohort_gates.integration_validation` passes;
- direct artifacts and logs pass secret scans;
- cache-read evidence is nonzero;
- both models derived from `plugins.toml.in` appear in committed routes; and
- `summary.json.status` is `passed`.

`report.md` is regenerated after each completed attempt and is safe for
progress review. Running multiple cohorts and aggregating their reports are not
part of this runbook.
