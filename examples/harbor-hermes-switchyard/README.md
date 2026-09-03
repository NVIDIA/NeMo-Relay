# Harbor + Hermes + Switchyard evaluation

This example runs one complete Terminal-Bench 2.0 cohort through Harbor and
Hermes. Hermes owns an in-process NeMo Relay runtime satisfying `nemo-relay>=0.8.1,<0.9.0`; Relay loads the
Switchyard native plugin, and Switchyard selects and calls the configured
provider route. The cohort runner operates on one resumable 89-task cohort at
a time. An optional, separate report workflow can compare or aggregate several
completed run roots without mutating them.

## 1. Pinned inputs

| Dependency | Input used by this example |
|---|---|
| NeMo Relay | Official PyPI release `nemo-relay==0.8.3`, installed by digest as a `manylinux2014_x86_64` wheel. |
| Hermes | `NousResearch/hermes-agent`, `main` (commit `48c0c3a873bc5adaf20c632b5b7630a4fac000b4`), which merges PR #96633 upgrading Hermes's own Relay integration from 0.7 to 0.8. No tagged release includes this yet. |
| Switchyard | `NVIDIA-NeMo/Switchyard`, detached commit `7a72c0667774244d66a8b631e375c9d6e393bf57` from `main` (PR #528 merged). |
| Harbor | `harbor==0.20.0`, official registry export of `terminal-bench@2.0`. |

Every source checkout is detached and verified. The Hermes installer is
followed by `uv sync --frozen`, then the selected Relay wheel is
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
  destination, manually selected capacity, and the real bare-token
  `SWITCHYARD_PROVIDER_AUTHORIZATION` credential (Switchyard adds the
  `Bearer ` prefix itself).
- `config/plugins.toml.in` is checked in and non-secret. It is the only source
  of provider URLs, protocols, strong, weak, and judge models, routing/classifier
  policy, native plugin manifest, authorization variable **name**, Relay
  components, and OpenInference export behavior.

The template configures AWS-hosted Claude Opus 4.8 as the strong route, NVIDIA
Nemotron 3 Ultra NVFP4 as the efficient route, and AWS-hosted Claude Sonnet 4.6
as the classifier judge, with a `0.5` threshold and
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

- Linux or macOS, Bash, Python 3.11+, Docker with the Compose v2 plugin, and
  `tmux`;
- an immutable 89-task Terminal-Bench 2.0 export downloaded from Harbor's
  official registry;
- a Switchyard plugin bundle and a Relay wheel satisfying `nemo-relay>=0.8.1,<0.9.0`,
  matching Docker's architecture (`x86_64` or `aarch64`);
- a Phoenix endpoint accepting OTLP/HTTP OpenInference traces; and
- provider and registry access for the full cohort. The all-89 admission uses
  neither; the Docker admission makes no provider calls but may pull its image,
  pinned sources, and packages when they are not cached.

Keep the dataset, bundle, wheel, admission, and run roots on a filesystem
shared with the Docker daemon. On Docker Desktop this normally means a
directory explicitly shared with Docker; on a Linux Docker Engine host, normal
host paths are shared by default.

### Select the active Docker architecture and create matching artifacts

The runtime only supports `x86_64` and `aarch64`. Select the value from the
Docker daemon—not from a previous machine or checkout—and use it consistently
for the Relay wheel and Switchyard bundle:

```bash
case "$(docker info --format '{{.Architecture}}')" in
  x86_64|amd64) export RELAY_ARCHITECTURE=x86_64 ;;
  aarch64|arm64) export RELAY_ARCHITECTURE=aarch64 ;;
  *) echo "unsupported Docker architecture" >&2; exit 2 ;;
esac
```

Download the official Relay wheel from PyPI (the `cp311`/`abi3` tags
satisfy the Hermes runtime):

```bash
python3 -m pip download \
  --only-binary=:all: --no-deps \
  --platform "manylinux2014_${RELAY_ARCHITECTURE}" \
  --implementation cp --python-version 311 --abi abi3 \
  --dest /absolute/path/to/relay-wheel \
  "nemo-relay>=0.8.1,<0.9.0"
```

This produces a `manylinux2014_${RELAY_ARCHITECTURE}` wheel under
`/absolute/path/to/relay-wheel/`; reference it as `RELAY_WHEEL` below.

Build the Switchyard native bundle for that same architecture. With no
`SWITCHYARD_TARGET_ARCHITECTURE` override, the builder detects the Docker
daemon architecture automatically; the explicit setting below makes the
chosen input visible in the command:

```bash
SWITCHYARD_TARGET_ARCHITECTURE="$RELAY_ARCHITECTURE" \
  ./scripts/build_switchyard_plugin.sh /absolute/path/to/switchyard-bundle
```

Set `RELAY_ARCHITECTURE`, `RELAY_WHEEL`, and `SWITCHYARD_BUNDLE` in `.env` to
the resulting architecture-matched artifacts. A wheel or bundle from a
different architecture is rejected during admission.

Install the exact Harbor-side requirements:

```bash
cd examples/harbor-hermes-switchyard
PYTHON_BIN=/absolute/path/to/python3.11-or-newer
"$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3, 11), sys.version'
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Copy and protect the environment file at the example root. Replace every
placeholder, including the complete provider Authorization header. Do not
source this file into the interactive shell used to start `tmux`.

```bash
cp .env.example .env
chmod 0600 .env
```

The validator is run after the registry export below. It reports names and
paths only, rejects legacy secret-file variables, and never renders or prints
the authorization value.

### Download the immutable Terminal-Bench export from the Harbor registry

After setting `TBENCH_DATASET_PATH` in `.env`, download the official dataset
once. Harbor exports it as `<output-dir>/terminal-bench`; therefore the output
directory must be the parent of `TBENCH_DATASET_PATH`. Do not pass
`--overwrite`: a cohort is bound to this export and must use a new run root if
the dataset is downloaded again.

```bash
set +x
set -a
source .env
set +a
set +x

mkdir -p "$(dirname "$TBENCH_DATASET_PATH")"
"$HARBOR_BIN" datasets download terminal-bench@2.0 \
  --output-dir "$(dirname "$TBENCH_DATASET_PATH")" --export
"$EXAMPLE_ROOT/scripts/validate_phase2_environment.sh" "$EXAMPLE_ROOT/.env"
```

Confirm Docker and Compose before continuing:

```bash
docker info >/dev/null
docker compose version
```

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

This all-89 no-token admission loads and uniquely selects the immutable local
export, hashes its instructions and verifiers, expands the complete Harbor job
graph, denies further registry/provider access, and renders the runtime. It
starts neither Docker nor an agent.

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
It is intentionally offline after the registry export.

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
  --switchyard-experiment "$SWITCHYARD_EXPERIMENT" \
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

On Linux, the launcher checks Docker access from inside the detached process.
If an existing tmux server predates the user's `docker` group membership, the
child re-executes under that group with `sg docker`. If the user is not a member
of the group, the launcher fails before starting the cohort instead of entering
an ineffective supervisor retry loop. Docker Desktop hosts continue directly
when `docker info` succeeds.

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
progress review.

## 12. Optional quantitative multi-run report

After every selected cohort passes its completion gates, use the
[`terminal-bench-report`](terminal-bench-report/SKILL.md) workflow to aggregate
repeated configurations or compare router strategies. Reporting is read-only:
give the analyzer arbitrary run-root paths, explicit public labels, a group for
each repeated configuration, and an observed baseline group when one exists.
Do not derive report numbers from progress logs or conversation history.

```bash
python terminal-bench-report/scripts/analyze.py \
  --run-root /absolute/path/to/control-r1 --label control-r1 --group control \
  --run-root /absolute/path/to/control-r2 --label control-r2 --group control \
  --run-root /absolute/path/to/trial-r1 --label trial-r1 --group trial \
  --run-root /absolute/path/to/trial-r2 --label trial-r2 --group trial \
  --output-dir /absolute/path/to/new-report-bundle \
  --mode compare \
  --expected-group-size 2 \
  --baseline-group control \
  --analysis-request "Compare complete configurations with N-run variance"

terminal-bench-report/scripts/render_pdf.sh \
  /absolute/path/to/new-report-bundle
```

The analyzer fails closed on incomplete final inputs unless `--allow-partial`
is explicitly requested. It resolves the run-bound dataset, configuration,
model roles, router parameters, and pricing; reports pass@1 and sample standard
deviation by group; distinguishes observed-control cost from same-workload
counterfactual cost; audits serving-call and routing-only token coverage; and
emits structured JSON/CSV evidence and SVG charts before PDF rendering. Review
`evidence/report-validation.json`, `evidence/pdf-validation.json`, and every PDF
page before distributing the report.
