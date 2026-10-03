<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Harbor 0.20 registry setup status

This is a working handoff record for the Harbor 0.20.0 and Terminal-Bench 2.0
registry migration. The README now documents the proven registry-backed
baseline.

## GPT-5.6 Sol / DeepSeek V4 N=3 experiment (complete)

- Nine isolated Terminal-Bench 2.1 cohorts were launched on 2026-08-09: three
  GPT-5.6 Sol direct controls, three staged capable-first (`CF@0.3`) trials,
  and three staged efficient-first (`EF@0.3`) trials. Each group uses the same
  three independently protected bearer tokens by replicate index, concurrency
  24, the 89-task registry export, and a 32-GB parallel-lane memory ceiling.
- The serving models are the NVIDIA InferenceHub identifiers
  `openai/openai/gpt-5.6-sol` and
  `nvidia/deepseek-ai/deepseek-v4-flash`. DeepSeek is also the staged-router
  judge. Reasoning is disabled for both its serving and structured-judge roles
  so NVIDIA's OpenAI-compatible response carries usable `content` and native
  tool calls rather than reasoning-only output.
- Run-bound OpenRouter prices captured on 2026-08-09 are USD 5.00/M input,
  30.00/M output, 0.50/M cache read, and 6.25/M cache write for GPT-5.6 Sol;
  and USD 0.14/M input, 0.28/M output, and 0.028/M cache read for DeepSeek V4
  Flash. DeepSeek cache write is omitted because it was not advertised; any
  emitted cache-write usage makes cost coverage partial rather than zero-cost.
- The exact NVIDIA Switchyard PR 270 head is
  `5c84c16e84fa781452b1ab9a96a0f12303619824`. The staged bundle is preserved
  at
  `/localhome/local-bbednarski/terminal-bench-artifacts/switchyard-bundle-5c84c16`.
- The shared all-89 Docker setup admission passed 89/89 with no infrastructure
  or integration failures at
  `/localhome/local-bbednarski/terminal-bench-artifacts/admission/tb21-sol56-deepseekv4-stage03-pr270-5c84c16-c24/setup-admission`.
  CF and EF all-89 smoke, offline compatibility, immutable plan, and live
  provider preflight evidence also passed before launch.
- A detached `hhstb21-sol56-monitor` tmux session checked all nine cohorts every
  600 seconds while they were active. It resumed only safe immutable failures,
  refused unreconciled integration blockers, and capped automatic restarts at
  three. Its final machine-readable state is under
  `/localhome/local-bbednarski/terminal-bench-artifacts/monitoring/tb21-sol56-deepseekv4-stage03-pr270-5c84c16`.
- The initial CF/EF routed snapshots exposed a validator-only compatibility
  defect after launch: `inspect_atof` appended `selected_target` from every
  `switchyard.routing.*` mark, so the new routing-only LLM accounting mark's
  valid internal `judge` target was misclassified as a serving destination.
  The six routed supervisors were stopped while the three direct controls
  continued. The validator now derives serving targets exclusively from
  `switchyard.routing.decision`; a regression test proves that an adjacent
  `switchyard.routing.llm_call` for `judge` remains overhead evidence and is
  excluded from the serving-target set.
- The stopped routed roots remain diagnostic `v1` evidence and are excluded
  from analysis. They were not mutated in place. Six fresh `v2` roots proved
  the validator repair, then exposed a separate model-configuration defect:
  DeepSeek serving calls could return long `reasoning_content` with neither
  usable `content` nor a native tool call. Hermes consequently ended some
  tasks cleanly without a normalized final response. This was neither rate
  limiting nor a Switchyard source regression.
- The `v2` roots were stopped and retained as excluded diagnostics. The CF and
  EF templates now disable DeepSeek reasoning for the `weak` serving target as
  well as the `judge` target. Fresh all-89 smoke and offline compatibility
  evidence passed below
  `/localhome/local-bbednarski/terminal-bench-artifacts/admission/tb21-sol56-deepseekv4-stage03-pr270-5c84c16-c24-reasoning-off-v3`.
  Six immutable `v3` roots passed plan and provider/capacity preflight and were
  launched at concurrency 24 on 2026-08-09; their root names contain
  `5c84c16-v3-rN`, and the final analysis manifest points only to these
  replacements.
- The first `v3` lane drain exposed an independent adapter-framing defect. The
  pinned Hermes quiet CLI prints its response to stdout and then prints the
  terminal `session_id` marker to stderr, while `finalize_artifacts.py` had
  attempted to recover response text after that marker. Future task snapshots
  now parse the source-defined order. Existing immutable `v3` tasks are only
  reconcilable when all of the following are present: zero-error agent exit,
  non-empty output preceding the terminal marker, normalized Harbor verifier
  result, no Harbor exception, successful adapter cleanup, passed integration
  validation, and passed Phoenix upload. The repair was validated against a
  preserved 17-call CF task; the complete suite passes 109 tests.
- Report analysis now consumes the commit's
  `switchyard.routing.llm_call` marks. It records routing-only prompt,
  completion, reasoning, cache-read, and cache-write tokens; prices those calls
  from the run-bound catalog; and adds them to serving-call cost without
  double-counting the deliberately excluded successful serving call.
- The isolated final-report manifest is
  `config/sol56_deepseekv4_analysis_runs.json`. All cohorts were strictly
  admitted at 89/89, and `scripts/generate_sol56_deepseekv4_n3_analysis.sh`
  created and validated the Markdown, structured evidence, charts, and PDF
  below `reports/tb21-sol56-deepseekv4-cf03-ef03-n3`.

## Terminal-Bench 2.1 parallel cohort

- The active Terminal-Bench 2.0 cohort remains preserved at
  `/localhome/local-bbednarski/terminal-bench-artifacts-c24-nemotron-ultra-nvfp4-opus48-r3`.
- Harbor 0.20.0 exported all 89 tasks from the official
  `terminal-bench/terminal-bench-2-1` registry dataset into a separate local
  dataset root. The 2.0 export was not overwritten.
- The protected `.env` now selects a new Terminal-Bench 2.1 run, admission
  root, Phoenix project, and evaluation cohort. It keeps setup and provider
  concurrency at 24, disables the canary, and raises the parallel-lane memory
  ceiling to 32 GB for this run.
- The 2.1 cohort must produce new all-89 dataset and offline compatibility
  evidence before its immutable plan is created. Do not update the README to
  make 2.1 the baseline until these gates and provider execution are proven.
- The first 2.1 plan attempt (`r1`) exposed that the coordinator read Harbor's
  older `environment.memory` field but not the `environment.memory_mb` field
  used by the 2.1 registry manifests. It incorrectly projected all tasks as
  2 GB and was abandoned before setup or provider execution. The parser now
  accepts both schemas; the replacement cohort is `r2`.
- The `r2` setup admission exposed a second Harbor 2.1 schema difference:
  task metadata uses canonical names such as `terminal-bench/task-name`, while
  Harbor's local dataset filter accepts the directory basename. The setup
  admission now retains canonical names in evidence but projects basenames to
  `--include-task-name`. The `r2` session was stopped before any provider work;
  its replacement cohort is `r3`.
- The `r3` all-89 admission evidence passed with dataset task-definition digest
  `3ae362b1d108bcb2c9bde58f02365756ed78484dc39969a5d9320f5e3d9527cd`.
  Offline Hermes-to-Relay-to-Switchyard compatibility and provider/Docker
  preflight also passed. Preflight records concurrency 24, a 32-GB parallel
  lane, an 8-GB largest task, a conservative 772-GiB requirement, and 2,267
  GiB exposed by Docker.
- The active Terminal-Bench 2.1 session is
  `harbor-hermes-switchyard-tb21-nemotron-opus48-c24-r3`, with run root
  `/localhome/local-bbednarski/terminal-bench-artifacts-tb21-c24-nemotron-ultra-nvfp4-opus48-r3`.
  Its 24-way setup admission accepted all 89 local task filters and began
  creating task jobs. Provider execution has not started yet.
- Harbor completed all 89 `r3` setup trials successfully, but result import
  initially treated canonical task names as nested paths below `task-results`
  and failed because the namespace directory did not exist. Future runs encode
  the complete canonical name into one collision-safe filename while retaining
  the canonical name inside the JSON evidence. The active immutable `r3`
  snapshot was repaired operationally by creating its expected namespace
  output directory. The supervisor then imported the existing 89 successful
  trial results without rebuilding them, setup admission passed 89/89, and the
  provider lane opened directly at concurrency 24 with no canary.

## Verified baseline

- A fresh `.venv` was created with Python 3.12.13.
- `requirements.txt` installed `harbor==0.20.0`; both the Python package and
  `harbor --version` report `0.20.0`.
- The protected existing `.env` remains unchanged and passes
  `./scripts/validate_phase2_environment.sh .env`.
- Harbor exported `terminal-bench@2.0` from its registry to the missing
  `TBENCH_DATASET_PATH` configured in `.env`. The export contains 89 task
  directories.
- `scripts/verify_harbor_hermes_compat.py` passed with Harbor 0.20.0 and the
  checked-in `HarborHermesAgent` bridge.
- The all-89 no-token admission passed. Its evidence records Harbor 0.20.0,
  89 job trials, no registry requests after the local export, and task
  definition digest
  `75f0ad7de1a56329bba584d44dfdd9b1ef5fea4ff665ec85a93f4507b886a688`.
- The Docker offline Hermes-to-Relay-to-Switchyard smoke passed and wrote
  `harbor-hermes-switchyard.phase2-offline-admission.v1` evidence.
- A new Harbor 0.20 cohort root now has a frozen 89-task plan and a passed,
  content-addressed hermetic Hermes runtime.
- The Docker setup admission passed all 89 tasks with no infrastructure or
  integration failures in the immutable
  `/localhome/local-bbednarski/terminal-bench-artifacts-c24` cohort, whose
  plan records `setup_concurrency: 24`.
- The hermetic runtime builder now sets its container `nofile` limit to 65535
  and restores host ownership of its output on exit. This fixes the host's
  default 1024-descriptor Docker limit and prevents failed builds from leaving
  undeletable runtime directories.

The existing Harbor 0.18.0/local-export cohort is preserved as historical
evidence. Do not reuse its run root for this migration.

## Concurrency finding

The 64-way setup admission exhausted Docker's default address pools before
provider-backed task execution. The failed cohort artifacts were removed at
the user's request. CPU, memory, disk, and the forced Docker builds were not
limiting factors.

The new 24-way admission passed all 89 tasks and drained its task networks
after completion. Use 24 as the current safe setup concurrency. Running at
64 requires a separately reviewed Docker `default-address-pools`
configuration that does not overlap this host or its VPN networks.

## Current code state

The working tree already updates the requirements and runtime version checks
to 0.20.0. It does **not** yet make the registry export a first-class,
reproducible setup stage:

- Phase 2 still requires an existing `TBENCH_DATASET_PATH` and explicitly
  avoids resolving the Harbor registry.
- The all-89 dataset admission treats any registry request as a failure.
- Admission evidence records local file hashes but not the registry dataset
  version or resolved per-task source revisions.
- The registry download remains an explicit documented setup command; Phase 2
  consumes the resulting immutable local export offline.

## Resume checklist

Use the existing `.env` and the fresh `.venv`; do not source `.env` in a
traced shell because it contains the provider authorization header.

1. Confirm the registry export is immutable for this cohort and capture its
   registry version, task-definition digest, and per-task source revisions in
   admission evidence.
2. Change the setup wrapper so it downloads and exports
   `terminal-bench@2.0` when the configured dataset destination is missing,
   without overwriting an existing export unless explicitly requested.
3. Update the no-token admission and its tests to permit the initial registry
   export, then validate the exported task set offline.
4. Run the provider preflight against the new cohort root before starting the
   provider-backed cohort execution.

## Collector repair

Docker is active on the current host. Provider preflight passed for the
24-way cohort, but its provider-backed execution stopped after the first
task with a `harness_or_integration` failure. No task summary or Phoenix
upload was produced, and the coordinator did not open the parallel lane.

The root cause is a collector ownership mismatch. The runner creates the
task's `telemetry` directory with mode `0700` under the host UID/GID, while
`otel/opentelemetry-collector-contrib:0.135.0` runs as `10001:10001`. The
collector therefore exits with code 1 while opening
`/artifacts/trajectory.openinference.json`: `permission denied`.

Three runner behaviors obscure and compound that failure:

- `docker run --detach --rm` reports a successful launch before the
  collector exits and immediately destroys its logs and container state.
- The runner has no collector readiness or liveness check before starting
  Harbor.
- The later unguarded `docker stop` fails because the collector is already
  gone; `set -e` then aborts before validation, Phoenix upload, or the task
  summary.

The active task container also did not resolve `host.docker.internal` on this
Linux Docker Engine host. Harbor's `--allow-agent-host` changes network-policy
allowlisting; it does not create Docker's `host-gateway` hostname mapping.

The runner now:

- runs the collector under the host UID/GID while preserving the private
  `0700` telemetry directory;
- retains and captures collector logs, waits for file and socket readiness,
  and cleans up idempotently;
- publishes OTLP on Docker's bridge gateway instead of host loopback; and
- supplies a Harbor Compose overlay mapping `host.docker.internal` to
  `host-gateway`.

A fresh `adaptive-rejection-sampler` provider canary proved the repair. It
completed benchmark execution, produced a 1,594,163-byte local trajectory,
recorded 15 Switchyard decisions and 425,036 cache-read tokens, and uploaded
38 spans to Phoenix in two batches with no retry. Integration validation
passed with no errors or warnings. The benchmark reward was a valid non-pass,
which remains separate from the integration gate.

## Next stage

The subsequent immutable c24 cohort passed setup admission for all 89 tasks,
but its canary exhausted Hermes response continuation after four attempts.
The collector remained healthy and produced an 804,399-byte trajectory with
22 spans, eight Switchyard decisions, and 190,192 cache-read tokens. The
coordinator stopped because it did not scan the retained `hermes-tail.txt`
diagnostic and therefore misclassified the transient truncation as
`harness_or_integration`.

The coordinator now scans `hermes-tail.txt` and classifies `Response remained
truncated after ... continuation attempts` as retryable infrastructure. Do
not reuse either failed c24 cohort. Create another immutable cohort root with
`setup_concurrency: 24`; a truncated canary response can then retry within the
configured infrastructure-attempt budget instead of stopping the cohort.

## Current Provider Cohort Diagnosis

The earlier replacement cohort at
`/localhome/local-bbednarski/terminal-bench-artifacts-c24-retry-fixed` used
`setup_concurrency: 24` and provider execution concurrency 4. It was stopped
when provider execution was moved to concurrency 24.

The `adaptive-rejection-sampler` canary recovered on attempt 002 and completed
in 2,516 seconds. Integration validation passed with 434 ATOF events, 104
OpenInference spans, 38 Switchyard decisions, and 1,804,820 cache-read tokens.
The observed delay was not a rate-limit failure: no HTTP 429 or provider 5xx
response occurred. A reasoning-heavy model response reached
`finish_reason=length`; Hermes increased the continuation output allowance from
4,096 through 8,192 and 16,384 to 32,768 tokens. The 32,768-token request then
crossed the upstream timeout path and received HTTP 408 responses. Switchyard's
retry and Hermes's stream retry amplified that event into the long stall before
the task recovered.

Relay's shorter post-continuation ATOF request histories are not evidence that
provider context was lost. Relay intentionally projects stale LLM start events
to the current user turn while leaving the request used for provider execution
unchanged. Hermes's persistent state also retained the complete conversation.

The canary's benchmark reward was 0 for a separate task-solution defect. The
generated `ars` function rejected the verifier's supported domain-vector call,
`ars(normal_density, c(-5, 5), n = 1000)`. Eight of nine verifier tests passed;
the provider, Relay, Switchyard, collector, and telemetry path all completed.

The following live tasks initially made successful model calls in approximately
3--56 seconds with no HTTP 408 or 429 responses. This indicates that the canary
stall was triggered by its unusually long completion rather than a generally
misconfigured endpoint. Do not change the endpoint or restart the cohort based
on this incident. For a later repair, bound Hermes continuation escalation to
the provider's practical response-time budget and avoid retrying the same
32,768-token stream through both Switchyard and Hermes.

## Nemotron and Opus 4.8 C24 Reset

The next provider configuration uses NVIDIA
`nvidia/nvidia/nemotron-3-ultra-nvfp4` as the efficient route and
`aws/anthropic/bedrock-claude-opus-4-8` as the strong route. Direct forced-tool
probes against `https://inference-api.nvidia.com/v1` succeeded for both model
IDs. The Nemotron inline pricing snapshot records USD per million tokens as
`0.60` input, `2.40` output, `0.119` cache read, and `0.119` cache write.

The first c24/no-canary plan for these models was created at
`/localhome/local-bbednarski/terminal-bench-artifacts-c24-nemotron-ultra-nvfp4-opus48-r2`.
Its plan and provider preflight passed, but provider execution never started:
the pre-existing tmux server launched the supervisor without the user's
`docker` supplementary group. Every Docker clock preflight failed with socket
permission denied, leaving all 89 tasks pending with zero attempts. The r2 tmux
session was stopped on August 8, 2026, and no cohort containers remained.

`scripts/launch_phase2_tmux.sh` now enters through
`scripts/run_phase2_with_docker_group.sh`. The wrapper uses Docker directly
when it is already accessible and otherwise re-executes through `sg docker`
when the host user belongs to that group. A detached probe against the same
stale tmux server successfully ran `docker info`, and a second probe proved the
new wrapper reached the protected-environment admission boundary.

Both the pricing catalog and tmux wrapper changed immutable runtime inputs. Do
not resume r2. Generate a fresh c24/no-canary cohort root, plan, preflight, and
runtime snapshot before the next provider launch.

## Active R3 Provider Cohort

The fresh c24/no-canary cohort is active in tmux session
`harbor-hermes-switchyard-nemotron-opus48-c24-r3` with run root
`/localhome/local-bbednarski/terminal-bench-artifacts-c24-nemotron-ultra-nvfp4-opus48-r3`.
Its immutable plan records provider and setup concurrency 24, 89 tasks, no
canary, the corrected Nemotron pricing snapshot, and the repaired tmux runtime
source digest.

The r3 all-89 no-token admission passed with Harbor 0.20.0 and no registry
requests. The Docker offline compatibility smoke passed with four provider
requests, OTLP output, no persisted secret, and no surviving shutdown threads.
Provider preflight verified Claude Opus 4.8, Claude Sonnet 4.6, and NVIDIA
Nemotron 3 Ultra NVFP4. Docker, Phoenix, architecture, memory, CPU, and disk
checks passed.

The run-bound setup admission passed all 89 tasks with no failures. The cohort
then started 24 provider-backed task attempts concurrently. At the initial
checkpoint, no attempt log contained an HTTP 429, missing-model response,
traceback, or Docker socket permission error. Keep this cohort running and use
its `summary.json`, `supervisor.log`, and per-task attempt artifacts for later
status checks. At the final launch checkpoint, four tasks had completed and the
scheduler had created 28 attempt logs while keeping the 24-task lane full.

## Turn-Budget Exhaustion Classification Repair

The TB2.0 `polyglot-rust-c` preserved blocker was not an integration failure.
Relay, Switchyard, OpenInference export, Phoenix upload, and secret validation
all passed. Hermes made exactly 90 routed decisions, matching Harbor 0.20.0's
installed Hermes `max_turns` setting, then exited cleanly without a final
response. Harbor independently ran the verifier and recorded reward 0.

`scripts/validate_run.py` now recognizes that exact evidence combination as a
completed benchmark non-pass. It removes only the two direct-response framing
errors and records `terminal_turn_budget_nonpass: true`. Provider exceptions,
early exits, incomplete telemetry, non-normalized verifier results, and runs
with fewer than 90 routed decisions remain failures. Replaying the repaired
validator against the preserved TB2.0 evidence produced passed benchmark
completion and integration validation with `benchmark_task_passed: false`.

This source repair does not modify the immutable runtime snapshot in the active
TB2.1 cohort. A later cohort created from the repaired source will no longer
preserve and stop on this specific turn-budget outcome.

## TB2.1 OpenInference AGENT-Span Diagnosis

The TB2.1 `configure-git-webserver` task completed normally, returned a final
response, received verifier reward 0, and uploaded 42 spans to Phoenix. Its
OpenInference artifact contains 14 LLM, 15 CHAIN, and 13 TOOL spans with valid
lineage and resource attributes, but no AGENT span.

The ATOF lifecycle identifies the cause. The benchmark's primary
`hermes.turn` ended successfully. Sixty milliseconds later, Hermes spawned its
single-session background skill-review agent with the prompt to review the
conversation and update the skill library. That auxiliary turn and its first
LLM call started while the CLI was shutting down, but neither ended; the parent
`hermes.session` end event was consequently absent. The Relay exporter correctly
omitted an AGENT span for the incomplete session. Other completed TB2.1 tasks
have balanced session start/end events and include AGENT spans, so this is a
nondeterministic post-response background-thread race rather than a provider,
Switchyard, collector, or Phoenix failure.

The recommended repair is to disable Hermes background skill review in the
custom Harbor agent's generated evaluation config with
`skills.creation_nudge_interval: 0`. Disabling the periodic curator there as
well keeps the benchmark execution stateless and prevents auxiliary model calls
from entering task telemetry. Keep the AGENT-plus-LLM validation gate strict;
do not hide this incomplete lifecycle as a warning. Apply that source repair to
a new immutable cohort rather than changing the active TB2.1 runtime snapshot.

## TB2.1 OpenInference AGENT-Span Repair and Canary

The custom Harbor Hermes bridge now derives its generated YAML from Harbor
0.20.0's installed base configuration and changes only two evaluation settings:
`skills.creation_nudge_interval: 0` disables the post-answer skill-review
thread, and `curator.enabled: false` disables periodic persistence maintenance.
The 90-turn benchmark budget and every other Harbor setting remain inherited.
The compatibility admission records these overrides and compares the complete
generated object with the installed Harbor base so an unrelated configuration
drift still fails closed.

The first isolated canary root,
`/localhome/local-bbednarski/terminal-bench-artifacts-tb21-configure-git-webserver-curator-disabled-r1`,
stopped before provider execution because the old compatibility guard required
the bridge to inherit `_build_config_yaml` without an override. This was an
admission-policy failure, not a task or provider failure. The guard was narrowed
to accept exactly the two intended persistence overrides, with regression tests
for the resulting YAML and the preserved `max_turns: 90` setting.

The replacement isolated canary at
`/localhome/local-bbednarski/terminal-bench-artifacts-tb21-configure-git-webserver-curator-disabled-r2`
passed `configure-git-webserver` end to end on August 8, 2026. Benchmark
completion, strict integration validation, and Phoenix upload all passed. The
export contained 94 spans: 1 AGENT, 32 CHAIN, 31 LLM, and 30 TOOL. Its ATOF
trajectory has exactly one start and one end for both `hermes.session` and
`hermes.turn`, and neither the ATOF trajectory nor Hermes transcript contains
the background skill-review prompt. Phoenix accepted all 94 spans in five
batches with no retries.

The source-level validation suite passes 67 tests, Ruff format and lint checks
pass for all repaired files, and `git diff --check` is clean. The active
immutable TB2.1 cohort was not changed or interrupted. New cohorts created from
this source will use the repaired stateless Hermes evaluation configuration.

## Parallel TB2.1 Replica Preparation

Two protected, Git-ignored environment files are prepared for independent c24
replicas: `.env.tb21-c24-r4` and `.env.tb21-c24-r5`. Each has mode `0600`, a
unique run root, run ID, Phoenix project, and evaluation cohort. They share the
same TB2.1 export, Harbor 0.20.0 environment, Relay wheel, Switchyard bundle,
bootstrap runtime, smoke evidence, offline evidence, and routing configuration.
Canary scheduling is explicitly disabled. Each file contains a distinct
rejected Bearer placeholder that must be replaced before environment validation
or provider preflight can pass.

The coordinator now supports `TBENCH_REUSE_SETUP_EVIDENCE` and records reused
setup evidence as an immutable plan input. Reuse is fail-closed: it verifies the
source setup plan and summary schemas, the complete 89/89 pass, canonical plan
digest, every per-task binding, dataset records and digest, Harbor version,
Hermes runtime and commit, Relay wheel, architecture, Switchyard library,
setup concurrency, batch size, retry bound, and force-build policy. A verified
reuse writes a run-local passed setup state without invoking the setup-admission
builder.

The r3 setup corpus passed this verifier with 89 task records and evidence
digest `8f4855b04c4a89f4a0f052ecf5d4efe06411d0429347101cfd403fcd17cccf75`.
The full source test suite now passes 69 tests. Ruff, Bash syntax validation,
and `git diff --check` pass. This skips the separate all-89 setup admission;
provider trials retain the existing plan-bound force-build setting and can use
Docker's cached layers.

After the two credentials were populated, both protected environment files
passed local validation. Each credential independently authenticated to the
configured NVIDIA catalog and exposed Claude Opus 4.8, Claude Sonnet 4.6, and
Nemotron 3 Ultra. Fresh r4 and r5 immutable plans were then created with no
canary and setup-evidence mode `reused`; their plan-file SHA-256 digests are
`6ef872f51e168c59ee64643010f2f4da5b0465e358417cb843fa8efbf56159aa`
and `ed8cc09ff3ac1a36fbb65d195d6ddf20a3f96c6eb6755065bef4bc7da3c47eb1`,
respectively.

Both non-task preflights passed provider catalog, Phoenix, Docker, architecture,
CPU, memory, and disk checks. Neither run root contained a task attempt or setup
state. At that checkpoint, concurrent launch remained gated on the host-wide
Docker address-pool repair completed below.

## Docker Address Pool and TB2.1 Runtime Repairs

The host-wide Docker network constraint is now addressed. `/etc/docker/daemon.json`
defines a `172.20.0.0/14` default address pool with `/24` allocations. This does
not overlap the host's `10.57.192.0/18` route or the existing Docker
`172.17.0.0/16`, `172.18.0.0/16`, and `172.19.0.0/16` networks. Docker 29.1.3
accepted the configuration, restarted successfully, and exposed 256 CPUs and
2,267 GiB. A 64-network create/remove probe allocated `172.20.1.0/24` through
`172.20.64.0/24` without failure and left no probe networks behind. The pool
can supply 1,024 `/24` task networks.

The two incomplete r3 tasks were separate deterministic runtime constraints:

- `extract-moves-from-video` was not a provider failure. Both attempts ended
  with Hermes exit 137 in the task's published 2-GiB cgroup. The kernel journal
  confirms repeated `CONSTRAINT_MEMCG` events in each trial container, killing
  ImageMagick/Tesseract workers and ultimately the Hermes process. New plans
  preserve the published 2-GiB value but record and apply a task-specific
  Harbor `--override-memory-mb 8192` runtime override. Capacity calculations
  and scheduling use the 8-GiB effective value.
- `torch-pipeline-parallelism` completed the agent and integration phases, then
  its official verifier reached two of four CPU-only PyTorch tests and timed
  out at exactly 900 seconds. The agent's local reproduction only completed
  after setting PyTorch CPU threads to one; the official verifier lacked that
  cap inside its one-CPU container. New plans record a task-specific verifier
  thread limit and pass `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and
  `VECLIB_MAXIMUM_THREADS` as `1` through Harbor's verifier environment.

An explicit Harbor `VerifierTimeoutError` is now a bounded infrastructure
retry. Generic verifier failures and assertion failures remain fail-closed
integration blockers. Harbor 0.20.0 `--print-config` validation proved the
8192-MB environment override and verifier environment projection. Replaying
the preserved torch attempt now produces the intended infrastructure
classification. Bash syntax checks, `git diff --check`, and the complete
example suite pass with 71 tests.

The earlier r4/r5 plan-only roots were preserved as
`*-superseded-pre-runtime-overrides` because their immutable runtime digest did
not include these repairs. Fresh roots now exist at the paths in
`.env.tb21-c24-r4` and `.env.tb21-c24-r5`. Both contain 89-task immutable plans,
reuse the verified r3 setup corpus, disable the canary, and contain zero task
attempts. Their runtime source digest is
`9a63d1a3e3427f2ab19a2202a03e3d4776b1c2ba65aad84d1e18fdb7433504cd`.
The new plan-file digests are:

- r4: `e6817fe13780cbbde5ce79953c5df8ad6f40020023e1af01c567c921bdfde162`
- r5: `006ef2abc115d13809d02b7cda75255447abdbdc2b99130bd5d61324b93337d8`

Both credentials independently passed provider catalog verification for
Claude Opus 4.8, Claude Sonnet 4.6, and Nemotron 3 Ultra NVFP4. Both non-task
preflights passed with an 8-GiB largest effective task, a 772-GiB per-cohort
conservative requirement, and 2,267 GiB available. The two cohorts are ready
to launch concurrently; no provider task execution has been started.

Both cohorts were launched concurrently at `2026-08-08T20:14:41Z` in detached
tmux sessions `harbor-hermes-switchyard-tb21-c24-r4` and
`harbor-hermes-switchyard-tb21-c24-r5`. Environment validation passed with
secret values withheld. Each coordinator reused the verified setup corpus,
skipped the canary, and opened exactly 24 provider-backed task runners for
tasks 001 through 024. Both tmux sessions and both supervisor/coordinator
process trees remained live at the post-launch checkpoint.

## Quantitative Report Generation Skill

The earlier run-specific report draft is now superseded by the reusable
`terminal-bench-report` skill. Its canonical entrypoint is
`terminal-bench-report/SKILL.md`; `REPORT_GENERATION_SKILL.md` is a forwarding
document so the two instruction sets cannot drift.

The skill accepts arbitrary run-root paths and sanitized labels. It derives a
scientific configuration signature from each run's dataset, task manifest,
runtime and plugin digests, routing configuration, pricing catalog, resource
policy, timeouts, and concurrency. Auto mode aggregates repeated identical
configurations or performs task-paired comparisons when configurations differ.
Each configuration group now enumerates the router algorithm, classifier/judge
model, strong and weak models, thresholds, context window, session and hash
behavior, retry policy, protocol fallback, target transports and weights,
request-body handling, and prices. Values are labelled as explicitly configured
or resolved from the bundled schema. Non-exposed judge controls and a compiled
built-in prompt remain explicit limitations tied to the plugin library digest.
The skill emits Markdown, JSON, CSV, and dependency-free SVG evidence plus a strict
Pandoc/XeLaTeX layout contract. Source roots remain read-only and are omitted
from the generated bundle.

Cost accounting is performed once per ATOF logical LLM call. Raw Switchyard
decisions remain separate retry-attempt evidence, and an explicit fallback mark
sets the effective model. The final ATOF usage chunk is preferred; a
route-scoped OpenInference usage receipt is used only when the final chunk is
absent. All headline costs use the run-bound catalog consistently, while
recorded `llm.cost.total` values are retained as reconciliation receipts.

The first generated analysis is
`reports/tb21-first-run-interim-v1/README.md`. It uses the stopped r3 TB2.1
root because the parallel r4/r5 runs remain out of scope until complete. The
report is deliberately interim: r3 has 87 of 89 benchmark-complete tasks (62
passes, 25 nonpasses), so no final `pass@1` is emitted. Observed accuracy is
62/87 (71.26%), and the planned-task lower bound is 62/89 (69.66%). Estimated
routed cost is USD 96.6637 across 2,811 of 2,832 logical calls, versus a USD
196.7392 all-Opus-4.8 counterfactual on the same covered usage, an estimated
USD 100.0756 (50.87%) reduction.

Reconciliation identifies 2,868 route attempts, 36 retry decisions, eight
fallbacks, and 99.26% usage/cost coverage. Eleven of 54 OpenInference calls
with comparable recorded cost disagree with catalog repricing; the current
examples are Claude cache-creation calls whose `llm.cost.total` omits the
configured cache-write charge. The report keeps catalog-derived costs as the
consistent headline and exposes the receipt mismatch for later integration
repair. Bundle validation passes with no absolute source paths or credential
patterns. The host now provides Pandoc 2.9.2.1, XeLaTeX from TeX Live 2022,
`rsvg-convert` 2.52.5, and Poppler validation utilities. The seven-page
letter-sized PDF renders without clipped tables or figures, passes extracted
text scanning, and is included with its validation receipt in the refreshed
bundle manifest.

## R6 and Direct Opus 4.8 Baseline Cohorts

Four additional TB2.1 evaluations were prepared with the established
89-task corpus, no canary, provider concurrency 24, the reused all-task setup
admission, and a 32-GiB parallel-memory budget. R6 uses the routed
Nemotron-3-Ultra-NVFP4/Claude-Opus-4.8 configuration and a protected copy of
the R4 credential. Its run root is
`/localhome/local-bbednarski/terminal-bench-artifacts-tb21-c24-nemotron-ultra-nvfp4-opus48-r6`.

The three comparison cohorts use `config/plugins.opus48-baseline.toml.in`.
That configuration installs Relay pricing and observability components but no
dynamic Switchyard component. Hermes calls
`aws/anthropic/bedrock-claude-opus-4-8` directly through the NVIDIA
OpenAI-compatible endpoint through Hermes's supported `custom` provider. The
isolated protected environment files and credential
lineage are:

- `.env.tb21-c24-opus48-baseline-r1`: copied from R3
- `.env.tb21-c24-opus48-baseline-r2`: copied from R4
- `.env.tb21-c24-opus48-baseline-r3`: copied from R5

The corresponding run roots end in `opus48-baseline-r1`,
`opus48-baseline-r2`, and `opus48-baseline-r3`. Each immutable plan records
routing mode `direct`, zero canary task, concurrency 24, and the same runtime
source digest `18401fecf1963c044fffba6fd2ce488835d1c5e920de327fdd51cbf0f105150a`.

High parallelism exposed a collector-port time-of-check/time-of-use race. The
runner previously probed a free port, released it, prepared runtime state, and
only then asked Docker to bind it. Concurrent cohorts could select the same
port. Collector startup now asks Docker to allocate and reserve the host port
atomically, reads back that binding, and admits only the resulting
`collector.container-id` and `telemetry/` bootstrap state before immutable
runtime preparation. Hermes's pinned build does not expose an explicit
`openai` provider and its OpenRouter profile pins `openrouter.ai`. The direct
path therefore uses provider `custom`, projects the NVIDIA endpoint, normalizes
Harbor's `openai/`-wrapped model to the endpoint-native model ID, and supplies
the protected key through `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, and the
host-scoped `NVIDIA_API_KEY` expected by Hermes's credential guard.

Prelaunch failures from the template-snapshot omission, fail-closed
OpenRouter transport, collector-port race, collector parent creation, runtime
preparer admission, unsupported explicit provider, OpenRouter's pinned base,
missing NVIDIA-scoped credential, and Harbor model prefix were preserved under
descriptive `*-superseded-*` roots. They contain no accepted benchmark result
for the final cohorts. The final sessions opened 24 tasks per cohort with
unique collectors, native model ID `aws/anthropic/bedrock-claude-opus-4-8`,
provider `custom`, and no 401, 403, connection, or provider-selection errors at
the post-launch checkpoint. At the same checkpoint R6 had completed 69 of 89
tasks with 6,582 uploaded spans. Shell syntax validation, focused Ruff
validation, and the complete example suite pass with 84 tests.

R6 subsequently stopped on task 038, `install-windows-3.11`. The agent ran for
Harbor's exact 17-minute configured deadline and Harbor produced a verifier
reward of zero, but the installed-agent wrapper surfaced the deadline as
`NonZeroAgentExitCodeError` with shell exit 130 rather than the canonical
`CancelledError`/`AgentTimeoutError` pair. Integration validation had otherwise
passed, including 648 ATOF events, 166 OpenInference spans, 58 routing
decisions, both routed models, and secret scanning. The validator now accepts
only the narrow combination of agent-phase `NonZeroAgentExitCodeError`, Harbor
`NonZeroAgentExitCodeError`, an exact `Command failed (exit 130):` marker, and
an independently produced verifier non-pass. Other nonzero exits and
cancellations remain integration failures. The regression suite covers both
the accepted exit-130 deadline and a rejected exit-1 command failure; the full
example suite now passes with 86 tests.

The already-complete task was reconciled rather than rerun. Its previously
stranded 166 spans were uploaded to the original R6 Phoenix project, and
`nonpass-reconciliation.json` records the original and repaired validation
hashes, Harbor trial-result hash, upload-receipt hash, validator hash, and the
fact that the immutable runtime was not modified. The generic
`scripts/reconcile_completed_nonpass.py` helper fails closed unless benchmark
completion, integration validation, the verifier-backed non-pass, the explicit
terminal-timeout marker, and Phoenix upload all pass. R6 was resumed from its
plan-bound staged runtime through
`scripts/resume_staged_phase2_from_env.sh`, which verifies the staged snapshot
digest against the immutable plan before launch. It skipped all 70 accepted
tasks and started only tasks 071 through 089 in parallel.

At the same checkpoint, all three direct Opus 4.8 baselines remained live with
exactly 24 provider tasks in flight per cohort and no failed task states.
Accepted completion counts were 13 for baseline r1, 5 for baseline r2, and 7
for baseline r3. Their staged validators predate the exit-130 normalization;
if that exact deadline representation occurs, the same evidence-preserving
reconciliation and staged-runtime resume procedure applies without changing
their model configuration or rerunning accepted tasks.

At a later live checkpoint, accepted counts had advanced to 27, 22, and 19
for direct baselines r1, r2, and r3. Baseline r1's own
`install-windows-3.11` attempt reached the same exit-130 deadline under its
older staged validator. That validator classified the missing terminal ATIF
and AGENT span as infrastructure and automatically began attempt 002; this is
not a preserved integration blocker, and the retry remains in flight. R2 and
R3 had no failed task states. R6 remained live with 70 accepted tasks and all
19 remaining tasks in flight. All four detached tmux supervisors were live.

## R5 Evidence Repair and Immutable Resume

R5 was preserved at 63/89 because task 011, `cobol-modernization`, had been
classified as a harness/integration blocker. The model execution itself was
complete: Harbor's verifier awarded reward 1, integration validation passed,
the attempt contained 1,201 ATOF events, one ATIF trajectory, 279
OpenInference spans, and no secret findings. Hermes had exhausted its
configured 90 logical-call budget without emitting a normalized final response.
The old validator incorrectly used 92 Switchyard routing decisions as the turn
count; two of those decisions were transport retries.

The validator now counts `hermes.logical_llm_call` start scopes independently
from Switchyard decisions. A missing final response is tolerated only when the
direct adapter has the exact clean turn-budget termination shape, Harbor has no
exception, exactly 90 logical calls are present, and Harbor independently
records a normalized verifier result. This applies to both verifier passes and
nonpasses. Routing attempts remain separate evidence and cannot determine the
turn-budget gate.

Revalidation proved 90 logical calls, 92 routing attempts, benchmark pass,
and integration pass. The previously stranded 279 spans were uploaded to the
original R5 Phoenix project. The reconciliation receipt preserves hashes for
the original validation, repaired validation, Harbor result, uploader receipt,
and repaired validator, and states that the immutable staged runtime was not
modified. R5 then resumed through its plan-bound staged runtime: tasks 001
through 064 were skipped, tasks 065 through 088 started immediately at
concurrency 24, and task 089 remained queued behind the semaphore. No accepted
task was rerun.

## Run-Root Cleanup

After selecting routed R4/R5/R6 and direct-baseline R1/R2/R3 as the six
follow-up analysis inputs, 41 other top-level evaluation roots were deleted.
The removed roots contained superseded prelaunch attempts, collector and
provider-integration probes, earlier incomplete TB2.0/TB2.1 cohorts, and
task-specific curator diagnostics, totaling approximately 22 GiB. Each target
was resolved and checked for live process references before deletion. The
shared `/localhome/local-bbednarski/terminal-bench-artifacts` dataset,
admission, wheel, bundle, and bootstrap cache was explicitly preserved.

The partial routed TB2.1 R3 root is temporarily retained even though it is not
an analysis input. Its `setup-admission` directory is an immutable path-bound
dependency of the still-running R5, R6, and direct-baseline R1 plans. It can be
removed after those supervisors finish and no resume or retry can reference
it. Cleanup did not interrupt any active coordinator; R5 advanced from 64 to
76 accepted tasks during the operation.

## Stage-Router PR 270 Cohort Preparation

Switchyard pull request 270 was verified from the official repository at head
commit `95e25609f414aa0da0a7e77736bd973ec9c58c86`. This commit descends from the
previously pinned `8daac03` integration and adds the NeMo Relay stage-router
configuration and runtime support needed by this example. The example source
pin now targets the full PR-head commit; the existing `switchyard-bundle` and
all active run-local `runtime-harness` snapshots remain unchanged.

The new routed trial configuration is:

- algorithm: `stage_router`;
- picker: `efficient_first`;
- confidence threshold: `0.5`;
- recent-turn window: `3`;
- capable target: Opus 4.8 (`strong`);
- efficient target: Nemotron 3 Ultra NVFP4 (`weak`); and
- ambiguity classifier: Sonnet 4.6 (`judge`), with base threshold `0.5`, zero
  threshold step, a three-turn window, and 4,096 maximum output tokens.

The native plugin was built into the new non-overlapping bundle
`/localhome/local-bbednarski/terminal-bench-artifacts/switchyard-bundle-pr270-95e2560`.
Its manifest SHA-256 is
`65c2658f677522f827b40e51aa64984ad10325b357d0606a60b735ea94e73245`, and
its Linux x86-64 library SHA-256 is
`b003703c4b22f306b016848b142a1db5b3c97488e0345e92c9b2a5cf9ec2feaf`.

Dedicated admission state lives below
`/localhome/local-bbednarski/terminal-bench-artifacts/admission/tb21-stage05-pr270-95e2560-c24`.
The Harbor 0.20.0 all-89, registry-offline dataset smoke passed for the TB2.1
digest `3ae362b1d108bcb2c9bde58f02365756ed78484dc39969a5d9320f5e3d9527cd`.
The offline Hermes-to-Relay-to-Switchyard smoke also passed. It issued four
provider requests in the expected classifier/completion sequence, exercised
both efficient and capable completions, emitted the stage-router decision
marks, and shut down without surviving threads. The all-89 Docker setup
admission was then started at concurrency 24; its final result must pass before
the three provider cohorts are launched.

Three protected run environments were created without exposing their copied
credentials:

- `.env.tb21-c24-stage05-r1`;
- `.env.tb21-c24-stage05-r2`; and
- `.env.tb21-c24-stage05-r3`.

All three files are mode `0600`, use distinct existing provider credentials,
disable the canary, select provider and setup concurrency 24, reserve 32 GiB
per parallel task lane, and name separate result and Phoenix cohort roots. They
all reference the new bundle and dedicated smoke/offline/setup evidence. No
new immutable provider plan had been created at that checkpoint, so
preparation could not overwrite or change any prior result root.

The shared Docker setup admission subsequently passed all 89 tasks with no
infrastructure or integration failures. Plan-only and preflight-only proofs
then passed for each of the three protected environments. Each plan records the
stage-router algorithm, efficient-first picker, threshold 0.5, no canary, 89
tasks, concurrency 24, and reused setup admission. Each provider credential
verified the Opus 4.8, Sonnet 4.6, and Nemotron 3 Ultra model IDs. Docker
reported 2,267 GiB of memory against the 772 GiB per-cohort preflight
requirement, 256 CPUs, and the expected x86-64 architecture.

The first launch exposed one remaining source-only compatibility gate in
`run_terminal_bench.sh`: after preparing a valid staged runtime, it allowed
only the prior `llm_classifier` and `direct` routing-mode labels and therefore
returned exit 2 for `stage_router` before Harbor execution. All three new
sessions were stopped with zero completed benchmark tasks; no prior cohort was
interrupted. Their roots were moved intact to names ending in
`superseded-stage-mode-gate-20260809T0433Z` so the evidence was preserved and
no directory was overwritten.

The launcher now accepts both Switchyard-owned routing algorithms,
`llm_classifier` and `stage_router`, while continuing to reject unknown modes.
Its regression assertion, shell syntax check, focused 78-test validation, and
complete 87-test example suite pass. Fresh plan-only and preflight-only proofs
were generated after the patch, binding all three replacement roots to runtime
source digest
`2de33a0c388cd857c7185a616d1ad181f58e1a92227f0e736d67157ffb2c9dca`.
The three replacement cohorts are live in detached sessions:

- `harbor-hermes-switchyard-tb21-stage05-r1`;
- `harbor-hermes-switchyard-tb21-stage05-r2`; and
- `harbor-hermes-switchyard-tb21-stage05-r3`.

Each opened its first 24 provider tasks and reached Harbor execution without
the earlier routing-mode exit. The pre-existing R5 resume session remained
live throughout preparation, admission, diagnosis, and relaunch.

## Random and Escalation Router Cohorts

The immutable routed R4, R5, and R6 attempt configurations were inspected to
classify the earlier LLM-classifier group. They use `kind = "llm_classifier"`
without a `mode` value, plus `base_threshold = 0.5`; Switchyard PR 270 defines
the omitted mode as `capability`. Those runs therefore are the capability
classifier group and did not exercise trajectory escalation.

Two additional templates complete the four routed groups used by the analysis:

- `config/plugins.random.toml.in` uses entropy-backed `random` routing with an
  equal 1:1 split between Opus 4.8 and Nemotron 3 Ultra. Sonnet remains in the
  shared pricing/target catalog but has weight zero and cannot be selected.
- `config/plugins.escalation.toml.in` uses `llm_classifier` with explicit
  `mode = "escalation"`, Nemotron as weak, Opus as strong, and Sonnet as the
  trajectory judge. It records `confirmations = 1`, a 28-message recent-turn
  window, a 500-character per-message cap, and 4,096 judge output tokens.

The escalation confirmation count is intentionally one. The current Hermes
managed LLM request retains a Relay session scope but does not place an
`x-switchyard-session-id` header in the request decoded by Switchyard. Offline
testing demonstrated that the documented default of two confirmations receives
valid judge verdicts but cannot retain the escalation streak between requests.
The documented one-confirmation setting requires no cross-request streak and
therefore exercises real weak-to-judge-to-strong escalation in this integration.

The source admission contract, runtime provenance, and immutable cohort plans
now retain `classifier_mode`, the full escalation settings, or the random
weights as appropriate. The task launcher admits `random` alongside
`llm_classifier` and `stage_router`. The fake offline provider distinguishes
capability and trajectory verdict schemas and validates the exact expected call
sequence. The complete example suite passes with 93 tests, Ruff passes for all
touched Python files, shell syntax passes, and `git diff --check` passes.

Separate all-89 registry-offline smoke and full offline compatibility evidence
passed under:

- `/localhome/local-bbednarski/terminal-bench-artifacts/admission/tb21-random-pr270-95e2560-c24`; and
- `/localhome/local-bbednarski/terminal-bench-artifacts/admission/tb21-escalation-pr270-95e2560-c24`.

The random offline proof made only completion calls and selected no judge. The
escalation proof exercised weak completion, trajectory judgment, and strong
completion in that order. Both reused the already-passed PR 270 all-89 Docker
setup admission because the dataset, Hermes build, Relay wheel, native plugin
binary, architecture, and setup agent are unchanged.

Six mode-0600 environments and non-overlapping result roots were created: three
`.env.tb21-c24-random50-rN` files and three
`.env.tb21-c24-escalation-c1-rN` files for `N = 1..3`. Each trio preserves the
three credential lineages used by the stage cohort, runs TB2.1 at concurrency
24 with no canary, and uses distinct Phoenix projects. All six plan-only and
preflight-only proofs passed and bind runtime source digest
`13317f506f196858b98c76a878f5ec86a4a118e6a1dc434d396782c6b7e2cc56`.

The six live detached sessions are:

- `harbor-hermes-switchyard-tb21-random50-r1` through `r3`; and
- `harbor-hermes-switchyard-tb21-escalation-c1-r1` through `r3`.

Each opened its first 24-task wave with no immediate task failures. The three
stage-router sessions and any remaining earlier supervisor were not stopped or
modified.

## Partial-Cohort Routing Audit and Resume

After the first supervisors exited, R5 was preserved at 88/89. Its remaining
`write-compressor` attempt had reached the Harbor agent deadline with exit 130
before the verifier ran. The attempt's Relay telemetry and integration checks
were valid, but it had no benchmark reward and could not be reconciled as a
completed result. The failed coordinator state was retained as
`task-state.failed-attempt-001.json`, and R5 was resumed to create attempt 002.
The resume also exposed that the protected environment still referenced the
subsequently changed live capability template. It now references an exact
plan-matching frozen template with SHA-256
`a2681d35d06355b1418957870c8d347d8a55f86588fed0d4711a3a8fa69d9a71`.

The partial PR 270 cohorts were audited from their accepted attempt summaries
and canonical ATOF routing marks before resume. Every accepted task in all nine
roots retained passed integration validation. Random routing is operating as
configured: its three runs recorded 8,541 decisions, split 4,236 weak and
4,305 strong (50.4% strong). Escalation routing also operates as configured:
its accepted results contain 6,843 weak and 79 genuine classifier-selected
strong decisions, including weak-to-strong transitions within tasks.

The stage-router results are not suitable for continuation. Across its three
partial runs, 7,077 of 8,330 decisions (85.0%) used the `fall_open` source,
while only 296 used a valid LLM-classifier verdict. A live provider probe
reproduced the cause. The stage classifier's three-message window can retain an
assistant tool call and its tool result, but its judge request does not include
the corresponding OpenAI `tools` definition. The Bedrock-backed Sonnet 4.6
endpoint rejects that request with HTTP 400 (`Bedrock doesn't support tool
calling without tools= param specified`). The same structured-output request
passes when it contains no tool-call history. The stage roots remain preserved
and stopped; resuming them would add mostly fail-open decisions rather than
valid stage-classifier evidence.

The mass stop was not a host OOM or reboot. The kernel journal contains no OOM
events, the host remained on the same boot, and the resume checkpoint had about
2.2 TiB of available memory and 2.5 TiB of free disk. The supervisors exited on
per-task preserved harness states, principally agent-deadline/final-response
normalization cases, plus isolated verifier timeouts. Every failed task state
for the random and escalation roots was preserved under an attempt-specific
filename, then those six immutable cohorts were resumed. The active detached
sessions are:

- `harbor-hermes-switchyard-tb21-c24-r5-resume2`;
- `harbor-hermes-switchyard-tb21-random50-r1-resume2` through `r3`; and
- `harbor-hermes-switchyard-tb21-escalation-c1-r1-resume2` through `r3`.

All seven sessions reached live coordinator execution and created new task
attempts without rerunning accepted tasks. The three stage sessions were not
resumed and require either a Switchyard judge-history repair or a fresh
immutable configuration that excludes tool history from the stage judge.

## Stage-Router Tool-History Repair Cohorts

Switchyard commit `f78fb0c475cc692fbaaed3fcf86a3e6bfa4098de` from
`bbednarski9/Switchyard` was verified as the direct child of the prior PR 270
head `95e25609f414aa0da0a7e77736bd973ec9c58c86`. The change copies the original
request's tool definitions into classifier-judge requests while deliberately
omitting `tool_choice`. This directly addresses the Bedrock HTTP 400 response
seen when the stage classifier retained assistant tool calls and tool results
without their corresponding definitions. Both focused upstream regression
tests passed: the capability judge retains tool definitions for retained tool
history, and the staged judge receives the tool definitions on an undecided
turn.

The example's future-cohort defaults now pin the full repaired commit and its
fork repository. Existing run-local runtime snapshots and the prior
`switchyard-bundle-pr270-95e2560` directory were not modified. A new native
bundle was built at
`/localhome/local-bbednarski/terminal-bench-artifacts/switchyard-bundle-f78fb0c`.
Its manifest SHA-256 is
`a666ac2feba56a7dcf125b188a0dbbe3188d62b053c3c4d03dc708513bca4719`, and
its Linux x86-64 library SHA-256 is
`e0ad733c75afd5303d637c139000defc199e250abd92b7be1929199686147f0c`.

Fresh admission evidence lives below
`/localhome/local-bbednarski/terminal-bench-artifacts/admission/tb21-stage05-f78fb0c-c24`.
The all-89 registry-offline dataset smoke passed, as did the full
Hermes-to-Relay-to-Switchyard offline smoke with four expected provider calls
and no surviving shutdown threads. The Docker setup admission passed 89/89 at
concurrency 24 with zero infrastructure or integration failures. An initial
new admission directory that incorrectly named the NVIDIA upstream repository
in runtime provenance was preserved with suffix
`superseded-official-provenance-20260809T1332Z`; it is not referenced by any
new plan.

Three protected mode-0600 environments are ready:

- `.env.tb21-c24-stage05-f78fb0c-r1`;
- `.env.tb21-c24-stage05-f78fb0c-r2`; and
- `.env.tb21-c24-stage05-f78fb0c-r3`.

Their result roots are the corresponding
`/localhome/local-bbednarski/terminal-bench-artifacts-tb21-c24-stage05-f78fb0c-nemotron-ultra-opus48-rN`
directories. All three immutable plan-only and provider preflight proofs
passed. Each plan selects all 89 Terminal-Bench 2.1 tasks, disables the canary,
uses provider and setup concurrency 24, reserves 32 GiB per lane, and records
runtime-source SHA-256
`3b5a922cd7d104180a27c8dd74fefb48e7823a9967d20e7f2b26d15c37933668`.
Preflight verified Opus 4.8, Sonnet 4.6, and Nemotron 3 Ultra NVFP4 for every
credential lineage, along with 256 Docker CPUs, 2,267 GiB of Docker memory,
and the 772 GiB per-cohort requirement. At the preparation checkpoint, no new
benchmark task had been launched. The older evaluation sessions continued
independently throughout the build and admission work.

The three repaired staged cohorts were subsequently launched in detached
sessions:

- `harbor-hermes-switchyard-tb21-stage05-f78fb0c-r1`;
- `harbor-hermes-switchyard-tb21-stage05-f78fb0c-r2`; and
- `harbor-hermes-switchyard-tb21-stage05-f78fb0c-r3`.

Before launch, the protected environments were verified to contain three
distinct, non-placeholder Bearer credentials without rendering or persisting
their values. All three sessions passed environment validation, remained live
through the initial checkpoint, and opened exactly 24 attempt-001 directories,
matching provider concurrency 24. No immediate coordinator, Docker, provider,
or permission failure was present. The host retained approximately 2.2 TiB of
available memory at that checkpoint.

An early production-artifact audit at `2026-08-09T13:57:01Z` found 22, 24, and
22 accepted tasks in repaired staged R1, R2, and R3 respectively. All 68
accepted task artifacts passed integration validation, and the exposed wrapper
and validation logs contained no recurrence of the original Bedrock
tool-definition HTTP 400 signature. The repaired path is demonstrably active:
97 classifier verdicts in the first audit sample occurred on requests that
already contained tool history, with tool definitions present.

The stronger routing-distribution gate did not pass, however. Across 1,269
accepted routing decisions in the fixed snapshot, 896 (70.6%) still selected
`fall_open`, 183 (14.4%) used a valid `llm-classifier` verdict, and 190 (15.0%)
were signal-driven. Restricting the comparison to turns left undecided by
signals, 864 of 989 turns (87.4%) whose recent three-message window contained a
tool result fell open, compared with 32 of 90 turns (35.6%) whose recent window
contained no tool result. A valid classifier verdict always selects a tier in
the current Switchyard policy, so these fall-open decisions represent an
unavailable, invalid, or inconsistent judge verdict rather than an intentional
probability abstention. The commit fixes some tool-history requests but has not
removed the production integration blocker. All three sessions remained live
at the snapshot and were not stopped as part of the read-only audit.

### Repaired Stage-Router Stop And Root-Cause Diagnosis

At the user's request, the three `f78fb0c` staged-router sessions were later
stopped gracefully without touching any other experiment. Coordinator shutdown
drained the in-flight work, and a post-stop audit found no matching host
processes or Docker containers. The immutable partial results remain preserved:

- R1: 32 of 89 tasks accepted, 19 passing and 13 non-passing;
- R2: 31 of 89 tasks accepted, 23 passing and 8 non-passing; and
- R3: 32 of 89 tasks accepted, 22 passing and 10 non-passing.

A focused live-endpoint replay isolated the remaining blocker. An exact judge
turn from the accepted `configure-git-webserver` artifact contained system and
user messages, one assistant native tool call, its matching provider-native
tool result, and 17 tool definitions. With the `f78fb0c` behavior—retaining
that native history and copying the tool definitions while leaving
`tool_choice` unset—the Bedrock-backed Sonnet classifier returned HTTP 200 but
selected a new `terminal` tool call. Its response therefore had
`finish_reason: tool_calls` and empty textual content. Switchyard attempted to
parse that empty content as the structured verdict, failed, reported
`classifier_fail_open` with `parse_error`, and routed the turn to the weak
model. Removing only the response schema produced the same tool-call behavior.
Sending OpenAI-style `tool_choice: "none"` also failed to suppress the tool
call on this provider path.

The prior commit thus exchanges one provider failure for another: omitting
tool definitions causes Bedrock to reject native tool-call history with HTTP
400, while including the definitions permits the classification-only judge to
continue the benchmark task by calling a tool. The latter is hidden as a
successful HTTP response followed by a local structured-verdict parse failure.
This explains why the original HTTP 400 signature disappeared while
tool-history turns still fell open at high rates.

The robust repair is to make the judge request provider-neutral rather than
forwarding native tool semantics. Before submitting retained history to the
structured judge:

1. Convert every native `ContentBlock::ToolCall` into bounded plain text that
   records the call ID, tool name, and sanitized arguments.
2. Convert every native `ContentBlock::ToolResult` into bounded, sanitized
   plain text and convert any `Role::Tool` message to a non-tool role such as
   `Role::User`.
3. Preserve ordinary assistant and user text, but assert that no native tool
   call or tool-result blocks remain.
4. Submit no tool definitions and retain no effective tool choice; keep only
   the structured response schema needed for the classifier verdict.

The same extracted turn succeeded against the live endpoint after this
sanitization: it returned `finish_reason: stop` and valid structured classifier
JSON. Because `StructuredJudge` is shared by capability, staged, custom, and
escalation paths, the sanitizer should be applied at that shared boundary (or
all affected inputs must be covered explicitly). Regression tests should
assert both that tool information survives as text and that judge requests
contain no native tool blocks or tool definitions. A provider-level regression
should also assert that the completion contains a parseable verdict and never
returns a tool call.

As a temporary configuration workaround, a zero-length recent-turn window
avoids native tool history and produced a valid verdict in the focused probe,
but it discards useful execution context and is not the recommended permanent
repair. The three partial cohorts should not be resumed with their current
binary. After repairing and rebuilding Switchyard, use fresh immutable cohort
roots so the runtime-source digest and behavior remain auditable.

## Sanitized-Judge Repair Cohorts (`baaf678`)

Switchyard commit `baaf678fca0c941b36e53fa30b563b350f8bb2f0` was
verified against its parent `f78fb0c475cc692fbaaed3fcf86a3e6bfa4098de`.
The repair applies at the shared structured-judge boundary: native tool calls
and tool results are converted into bounded text, tool-role messages become
provider-neutral user messages, tool definitions are omitted, and the judge's
structured response schema is retained. The capability retained-history test
and the complete 30-test staged-router selection passed locally. The tests
assert that ordinary text and bounded tool evidence survive while native tool
blocks, tool roles, and tool definitions do not.

Future-cohort source pins in the example now select the full `baaf678` commit.
Existing run-local snapshots remain unchanged. The immutable production bundle
is:

`/localhome/local-bbednarski/terminal-bench-artifacts/switchyard-bundle-baaf678`

Its manifest SHA-256 is
`cc134ad5a3123368df7b3a38744fb8596c59d6d9b2b47c16fbb79c778f20eba6`,
and its Linux x86-64 library SHA-256 is
`860fc7c634e0eea4a4df1935b6bf00ad72f63dcf938724518a43a495c335d9f8`.

Fresh admission evidence lives below
`/localhome/local-bbednarski/terminal-bench-artifacts/admission/tb21-stage05-baaf678-c24`.
The all-89 no-token smoke passed, the full offline Hermes-to-Relay-to-Switchyard
smoke passed with four expected provider calls and no surviving shutdown
threads, and the force-build Docker setup admission passed 89/89 at concurrency
24 with zero infrastructure or integration failures. The immutable setup plan
SHA-256 is
`33f42b47332d2939bdf2e125f23bc05b4b248f1a3cee6ee0d65f30dce0646b9c`.
An empty setup output created with the wrong offline-runtime parent path was
preserved with suffix `superseded-wrong-runtime-root-20260809`; no Docker task
was started by that rejected invocation. Likewise, the first R1 plan attempt
was rejected because it referenced setup evidence bound to the prior library;
that no-plan root was moved to suffix
`superseded-old-setup-evidence-20260809T1434Z` before creating the intended
fresh root.

The three protected mode-0600 environments and result roots are:

- `.env.tb21-c24-stage05-baaf678-r1` and
  `/localhome/local-bbednarski/terminal-bench-artifacts-tb21-c24-stage05-baaf678-nemotron-ultra-opus48-r1`;
- `.env.tb21-c24-stage05-baaf678-r2` and
  `/localhome/local-bbednarski/terminal-bench-artifacts-tb21-c24-stage05-baaf678-nemotron-ultra-opus48-r2`; and
- `.env.tb21-c24-stage05-baaf678-r3` and
  `/localhome/local-bbednarski/terminal-bench-artifacts-tb21-c24-stage05-baaf678-nemotron-ultra-opus48-r3`.

The environments retain three distinct, non-placeholder credentials without
rendering their values. All three immutable plans and provider preflights
passed. Each plan selects all 89 Terminal-Bench 2.1 tasks, disables the canary,
uses provider and setup concurrency 24, reserves 32 GiB per lane, and records
runtime-source SHA-256
`6ddb36ec2876148b8449573586689ac272557897051b27ed9e7a4f73c78b23bb`.
Preflight verified Opus 4.8, Sonnet 4.6, and Nemotron 3 Ultra NVFP4 for each
credential lineage, along with the 772 GiB Docker requirement against 2,267
GiB available.

The fresh cohorts were launched in detached sessions:

- `harbor-hermes-switchyard-tb21-stage05-baaf678-r1`;
- `harbor-hermes-switchyard-tb21-stage05-baaf678-r2`; and
- `harbor-hermes-switchyard-tb21-stage05-baaf678-r3`.

All three sessions opened exactly 24 initial attempts, matching the planned
provider concurrency. An early production audit across 664 routing decisions
found 515 valid `llm-classifier` decisions (77.6%), 113 signal-driven decisions
(17.0%), and 36 `fall_open` decisions (5.4%). This is a material correction
from the 70.6% fall-open rate in the comparable early `f78fb0c` snapshot. The
exact prior regression task, `configure-git-webserver`, produced 46 valid
classifier decisions across the three live runs; R1 and R2 had no fall-open
decision on that task, while R3 had two fall-open and 16 valid classifier
decisions. No exposed artifact contained the earlier Bedrock missing-tools,
classifier tool-call, or structured-verdict parse-error signature. The repaired
judge path is therefore active in production execution. The remaining small
fall-open population should continue to be monitored as ordinary judge/provider
reliability evidence rather than treated as recurrence of the tool-history
integration blocker.

## Recurring Final-Run Monitor And N=3 Analysis Gate

A guarded user-level systemd timer now checks the three `baaf678` staged
cohorts every 15 minutes. The timer is
`nemo-relay-final-runs-monitor.timer`; its oneshot service invokes
`scripts/monitor_final_stage_runs.sh`. Each invocation uses a non-blocking
lock, records a structured status snapshot below
`/localhome/local-bbednarski/terminal-bench-artifacts/monitoring/tb21-stage05-baaf678`,
and never modifies a live cohort. An incomplete cohort is relaunched only when
both its tmux session and every process whose command line references its exact
run root are absent. Missing immutable inputs, surviving workers without tmux,
or three unsuccessful automatic restarts hold the run for diagnosis instead of
starting a duplicate supervisor or looping indefinitely.

The first service invocation passed at `2026-08-09T15:39:58Z`. All three tmux
sessions were live, so no automatic restart was performed. The snapshot found
R1 at 81/89, R2 at 79/89, and R3 at 86/89 accepted tasks. The next invocation
was scheduled for `15:45 UTC`, followed by quarter-hour checks. Once all three
runs report 89/89, the monitor writes `analysis-ready.json` pointing to the
structured input manifest `config/final_analysis_runs.json`. It then invokes
`scripts/generate_final_n3_analysis.sh`, whose admission gate independently
requires all twelve declared runs to have 89 benchmark outcomes and the
manifest dataset. The generator writes through a private staging directory,
promotes only a bundle whose structured validation passes, and otherwise
preserves the failed staging evidence for diagnosis. Its idempotent target is
`reports/tb21-final-n3-opus48-baseline-v1`; PDF rendering and visual inspection
are also gated. The generator now renders `README.pdf` after structured Markdown
validation and promotes the staging bundle only when PDF validation also passes.
Manual visual inspection remains the final completion gate.

The final-analysis manifest admits four N=3 groups: the observed Opus 4.8
direct baseline, the LLM capability classifier, the 50/50 random router, and
the repaired confidence-0.5 staged router. The primary comparison uses the
mean observed cost of the three direct Opus 4.8 runs as the control. It requires
run-level `pass@1` and cost means with sample standard deviation, task-aligned
configuration pass-frequency effects with bootstrap intervals, individual-run
observations, and aligned cost/performance bar panels, followed by the existing
configuration, routing, model-selection,
cache, task, reconciliation, and limitation sections. The incomplete
escalation cohorts and the invalid superseded staged cohorts are explicitly
excluded rather than pooled into an N=3 result.

A full disposable 12-root interim generation passed report validation before
the staged runs completed. That preflight exposed and repaired two report-only
issues without modifying any cohort: direct Opus calls have no Switchyard
decisions, so the analyzer now reconstructs them from logical-call/LLM scope
relationships and resolves the sole run-bound priced model; and raw rendered
TOML hashes differed only because each file embeds run-specific telemetry
identity, so raw hashes remain provenance while normalized configurations form
scientific identity. Direct baseline R1 yielded 2,619 logical calls, 2,550
cost-covered calls (97.37%), and USD 113.9858 in catalog-derived covered cost.

The preflight also found a genuine capability-group provenance difference:
R4/R5 share one runner/runtime snapshot while R6 uses the later snapshot that
added direct-baseline support. Archived file diffs show that the later paths
are conditional on direct mode and preserve the routed Switchyard execution,
model/router/pricing settings, dataset, task policy, Relay wheel, Switchyard
binary, concurrency, and timeout policy. The analysis manifest therefore
contains a narrow compatibility exception for exactly `runner_sha256` and
`runtime_sources_sha256`. The report exposes two signatures and warns that the
capability group's N=3 variance can include a harness-snapshot effect; any
additional or missing difference fails admission. The full preflight then
passed all reconciliation, sanitization, denominator, and root-identity checks
with 89 aligned tasks for the complete control, capability, and random groups.

The 16:15 monitor snapshot found the staged cohorts at 86/89, 86/89, and
88/89, with every tmux session and matching worker tree still live. It made no
restart. Two already-finished attempts had stopped their respective scheduling
lanes as `harness_or_integration`, but both benchmark outcomes are recoverable
from preserved evidence:

- R1 `pytorch-model-recovery` hit Harbor's exact 2,700-second agent deadline
  with canonical `CancelledError`/`AgentTimeoutError`, independently received
  verifier reward 1, and retained 31 primary logical calls plus valid ATOF and
  OpenInference evidence. The validator had only admitted the same deadline
  shape for verifier nonpasses.
- R2 `write-compressor` exited cleanly without a normalized final response and
  independently received verifier reward 0. Its ATOF contains exactly 90
  primary calls and two `auxiliary:compression` calls; the validator had
  incorrectly compared all 92 calls with Hermes's 90-primary-turn budget.

The validator now recognizes verifier-backed terminal agent-timeout completion
for either pass or nonpass, while retaining the exact agent-phase and Harbor
exception requirements. Logical-call budget accounting counts `primary` and
legacy unlabelled scopes, reports auxiliary scopes separately, and never uses
Switchyard retry decisions as turns. External read-only revalidation candidates
pass benchmark and integration gates for both attempts. The full example suite
passes 97 tests in the Harbor-pinned environment.

`scripts/reconcile_terminal_completions_from_env.sh` is now part of the guarded
monitor path. It runs only after both tmux and matching workers disappear,
revalidates each preserved integration blocker, admits only the two explicit
terminal-completion shapes, uploads stranded OpenInference evidence, preserves
the original validation, writes a reconciliation receipt, and then permits the
normal staged-runtime resume. Unknown integration failures are held rather than
blindly relaunched.

During the report rehearsal, a source-and-telemetry audit identified an
important cost-evidence boundary. Switchyard represents judge consultations as
non-routed side calls, but the `baaf678` Relay plugin drives those calls inside
the plugin without emitting a separate ATOF/OpenInference span containing the
judge model's token usage. The artifacts therefore support exact catalog-based
cost for agent-facing completion-target calls, but not end-to-end cost for the
capability or staged routers. The report skill and analyzer now record the
configured judge model, mark router-overhead usage unavailable, label monetary
results as observable execution-model cost, and state that negative deltas
against the direct Opus baseline are more favorable than the unknown total-cost
deltas. No zero-token or approximate judge cost is fabricated. Random routing
and direct execution do not require an internal judge call.

The 16:30 monitor fired on schedule and retained the same 86/89, 86/89, and
88/89 accepted counts because all three supervisors and their exact worker
trees remained live. The outstanding workloads were not idle: Caffe model
evaluation, FastText training, MobileSAM verification, and PyStan sampling had
active container processes. No restart or run-root mutation was appropriate.

The 16:45 monitor then recorded R1 at 87/89, R2 at 87/89, and R3 at
88/89. R1's Caffe task passed and R2's PyStan task completed as a valid
nonpass. The remaining FastText and MobileSAM processes continued consuming
CPU; the supervisors remained live, so the guarded monitor again made no
restart. R1 still has the preserved `pytorch-model-recovery` terminal
completion plus FastText pending, and R2 has the preserved `write-compressor`
terminal completion plus MobileSAM pending. Those preserved results will be
reconciled only after their live worker trees exit.

A second disposable report/PDF rehearsal incorporated the layout review. It
split wide performance and cost tables, moved the complete scientific field
matrix to `configuration-differences.csv`, grouped identical router profiles,
resolved staged and random algorithm fields dynamically, reduced task findings
to configuration-sensitive and consistently difficult task tables, and rotated
dense chart labels. The 14-page letter-sized PDF passed Markdown, PDF, secret,
and absolute-path validation. Visual inspection confirmed that the table of
contents, configuration tables, task matrix, cache chart, and concise task
tables fit the page. This rehearsal used interim staged outcomes and remains
outside the final report target; the guarded generator will rebuild from the
admitted 89/89 artifacts.

The 17:00 monitor admitted staged R3 as a complete 89/89 result (58 passes,
31 valid nonpasses) with no automatic restart. R1 and R2 remained at 87/89
with their exact supervisors and worker trees live. R1's FastText training and
R2's MobileSAM verifier were both still consuming sustained CPU, so neither
qualified for reconciliation or resume. The recurring timer remains scheduled
at quarter-hour boundaries.

The final report renderer received another disposable 12-root audit after the
router-overhead disclosure was converted to compact bullets. All 15 pages were
visually inspected. That review found and repaired one remaining overflow in
the executive per-run cost table by moving execution-call coverage to the
dedicated Cost and routing section. The rebuilt PDF passed letter-size, text,
secret/path, and bundle validation, and the repaired page was visually
rechecked. The focused report suite now passes 10 tests; the most recent full
example-suite result is 99 passing tests. The disposable report remains
interim and outside the final target.

A completion-gate audit aligned the recurring monitor with the stricter final
analysis admission contract. A run is now complete only when its summary is
`passed`, it plans and completes exactly 89 tasks, and its pass plus valid
nonpass counts total 89. Merely reaching a numeric `completed_tasks` value can
no longer mark the three-run gate ready. The analysis manifest also now names
the Opus control baseline precisely as mean observable covered execution-model
cost, consistent with the unavailable internal judge-usage limitation. Shell
syntax, manifest validation, and the full 99-test example suite pass after
these changes.

The 17:15 scheduled snapshot kept R1 and R2 at 87/89 and R3 at its admitted
89/89 result. No restart was attempted because both remaining supervisors and
worker trees were live. R1 had progressed from its earlier FastText experiment
to a new `train_final.py` training pass at sustained CPU, and R2's MobileSAM
mask conversion remained at sustained CPU inside the verifier. This process
transition is direct evidence that R1 is advancing rather than wedged; both
cohorts remain under the guarded wait policy.

The 17:30 monitor again recorded 87/89, 87/89, and 89/89 with zero
restarts. R1 completed its latest FastText training pass at 0.5707 validation
accuracy and immediately began additional 3- and 5-epoch experiments; the
agent remains within its 10,800-second multiplied deadline. R2's MobileSAM
verifier remained CPU-bound after roughly 78 minutes. Historical completed
cohorts put the same task between about 18 and 74 minutes, with another
verifier result finishing near 117 minutes, so the current runtime is long but
still within the task's 7,200-second verifier allowance and observed behavior.

At 17:45 both remaining supervisors exited. R1's FastText task produced a
valid benchmark pass, and R2's MobileSAM task produced a valid benchmark
nonpass. The guarded reconciler then admitted R1's preserved
`pytorch-model-recovery` verifier-backed pass and R2's preserved
`write-compressor` verifier-backed nonpass. The authoritative final staged
summaries are therefore:

- R1: 89/89, 63 passes and 26 valid nonpasses;
- R2: 89/89, 63 passes and 26 valid nonpasses;
- R3: 89/89, 58 passes and 31 valid nonpasses.

The first reconciliation cycle exposed a stale-read monitor defect: after a
reconciliation changed 88/89 to 89/89, the monitor launched an unnecessary
resume using the pre-reconciliation counters. Both supervisors exited
immediately without rerunning a task. The monitor now rereads and strictly
re-admits the summary after reconciliation before deciding whether a resume is
needed. The all-complete gate requires `status=passed`, exactly 89 planned and
completed tasks, and exactly 89 pass/nonpass outcomes.

The superseding final N=3 analysis is in
`reports/tb21-final-n3-opus48-baseline-v6`. It contains four declared groups of
three complete runs: Opus 4.8 direct control, LLM capability classifier,
random 50/50, and staged router at confidence 0.5. Incomplete escalation runs
and superseded invalid staged runs are explicitly excluded. Admission,
reconciliation, bundle-hash, Markdown, PDF, secret/path, and 16-page visual
validation all pass. The report and bundled admission evidence explicitly name
the excluded escalation and superseded staged cohorts and their rationales.
The generator was also hardened to use the example's
absolute virtual-environment interpreter, replace its output placeholder by
value rather than a fragile array index, and normalize relative input/output
paths.

Final descriptive results are:

- Opus direct: 78.65% mean pass@1 (1.12 percentage-point sample SD), USD
  112.0671 mean covered execution-model cost (USD 8.8490 SD).
- Capability: 66.67% (1.30-point SD), USD 104.1271 (USD 12.1837 SD), a
  -7.09% covered-cost delta versus control.
- Random 50/50: 66.67% (1.30-point SD), USD 104.6367 (USD 6.0851 SD), a
  -6.63% covered-cost delta.
- Staged 0.5: 68.91% (3.24-point SD), USD 109.1425 (USD 6.4506 SD), a
  -2.61% covered-cost delta.

The v6 cost audit distinguishes these observed cross-run deltas from the
same-workload all-Opus counterfactual. Repricing each routed run's own covered
calls and recorded usage as Opus yields mean savings of 58.96% (2.77-point
SD) for capability, 61.91% (0.25-point SD) for random 50/50, and 54.70%
(1.51-point SD) for staged 0.5. These larger figures do not conflict with the
observed deltas: the direct group had a 97.50% mean cache-read prompt share,
versus 79.37%, 71.19%, and 78.86% for the routed groups, and the independent
runs also generated different call and token volumes. The report now shows
both estimands, groups all headline cost statistics at N=3 with mean/sample
SD, and presents call-, token-, and covered-cost-weighted model selection.

All routed groups underperformed the direct control on pass@1; task-bootstrap
95% intervals for their mean task-frequency deltas remain below zero. Staged
is the strongest routed result descriptively, 2.25 points above capability and
random, but has the largest performance variance. Capability has the lowest
nominal covered execution cost, while random has nearly the same performance
and cost without an internal judge call. The report is labelled
`PERFORMANCE FINAL / COST PARTIAL`: execution-call cost coverage is at least
97.15%, but internal Sonnet judge token usage is absent for capability and
staged, so those judge costs are excluded and no routed delta is claimed as an
end-to-end provider-invoice saving.

After the terminal state and final report were independently verified,
the recurring systemd user timer was disabled and stopped. Its unit files and
guarded scripts remain installed for reproducibility, but no further polling
or resume action will occur unless it is explicitly re-enabled. All staged
tmux sessions and exact worker trees are absent.

## GPT-5.6 Sol / DeepSeek V4 Flash staged-router campaign

A separate Terminal-Bench 2.1 campaign was admitted against Switchyard PR 270
commit `5c84c16e84fa781452b1ab9a96a0f12303619824`. It compares three independent
runs each of GPT-5.6 Sol direct, capable-first at confidence 0.3, and
efficient-first at confidence 0.3. DeepSeek V4 Flash is both the efficient
completion target and the staged-router judge. All nine immutable plans use
89 tasks, provider and setup concurrency 24, the shared all-task setup
admission, and three distinct protected credentials by replicate index.

The new Switchyard routing-LLM usage marks are included in the isolated report
contract as routing-only prompt, completion, reasoning, and cache-token usage.
Those calls are priced separately from the serving call and added without
double counting. The report target is
`reports/tb21-sol56-deepseekv4-cf03-ef03-n3`; its guarded generator runs only
after all nine strict summaries reach 89/89.

Early routed attempts exposed a result-finalization compatibility issue rather
than missing benchmark evidence: quiet-mode Hermes emits its final response
before its terminal `session_id` marker, while the finalizer had searched after
that marker. The finalizer now parses the response before the marker. A strict
validator and reconciler admit preserved historical attempts only when Harbor
has a normalized verifier result, cleanup is complete, telemetry integration
passes, the terminal marker is the last nonempty line, and a nonempty response
precedes it. Reconciliation receipts and terminal-completion classes are
carried into the report evidence.

One CF R3 task, `overfull-hbox`, instead ended after its eighth logical call
with `NonZeroAgentExitCodeError` and the exact trusted-fallback provider HTTP
400 signature after three retries. Its first seven calls, cleanup, telemetry,
and failed attempt remain preserved, but it has no verifier-backed terminal
completion and cannot be reconciled. A new guarded whole-task retry preparer
recognizes only that exact evidence shape, archives the prior task state,
writes a content-addressed retry receipt, and requests a new attempt. The
10-minute monitor invokes it only after the run supervisor and matching worker
tree have stopped; unknown failures and retry-preparation errors are held for
diagnosis. A read-only check against the actual failed task matches the guard
without changing its current state. The retry succeeded as attempt 002 and is
disclosed separately from Switchyard routing retries in the final report.

All nine final cohorts are now strictly complete at 89/89. Their pass counts
are 71, 72, and 69 for direct; 57, 61, and 53 for CF@0.3; and 44, 45, and 36
for EF@0.3. The corresponding N=3 mean pass@1 values are 79.40%, 64.04%, and
46.82%, with sample standard deviations of 1.72, 4.49, and 5.54 percentage
points. The task-aligned bootstrap effects versus direct are -15.36 points
for CF (95% interval [-21.35, -9.74]) and -32.58 points for EF (95% interval
[-40.82, -24.34]).

The report records 2,077 CF routing-only calls with 4,101,228 prompt, 377,026
completion, and 1,083,904 cache-read tokens, and 2,654 EF routing-only calls
with 6,512,902 prompt, 548,623 completion, and 1,438,464 cache-read tokens.
Reasoning and cache-write tokens are zero in both routed groups. Monetary
results remain explicitly `COST PARTIAL`: minimum execution-call coverage is
93.09%, and routing-mark usage coverage is partial, so uncovered cost is not
imputed and the observed reported-cost differences are lower-bound
comparisons rather than provider-invoice savings.

The canonical final PDF is
`reports/tb21-sol56-deepseekv4-cf03-ef03-n3/README.pdf`. Markdown admission,
structured evidence validation, bundle hashing, secret/path scans, PDF
validation, and a page-by-page visual inspection of all 15 pages pass. The
PDF SHA-256 is
`57be65a3b292fe231738f01c903156c660550724ac02a835712efc95d27ce760`.
The complete example suite passes 111 tests after these changes.
