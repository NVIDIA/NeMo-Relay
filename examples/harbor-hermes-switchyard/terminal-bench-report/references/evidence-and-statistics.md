<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Evidence and statistical contract

## Contents

1. Evidence hierarchy
2. Scientific configuration identity
3. Performance metrics
4. Analysis modes
5. Cost baselines and reconciliation
6. Routing and cache metrics
7. Admission and sanitization

## Evidence hierarchy

Resolve each supplied root dynamically. Do not assume a parent directory name.

| Evidence | Use |
| --- | --- |
| `plan.json` | Dataset, ordered task manifest, execution settings, source digests, routing inputs. |
| `summary.json` | Planned/completed outcomes, verifier result, integration status, cohort/project identity. |
| Rendered `plugins.toml` | Actual routing targets, algorithm, pricing entries, currency, price date/source. |
| Bundled Switchyard `config.schema.json` | Defaults and the set of configurable judge/router controls. |
| Successful attempt `summary.json` | Benchmark, validation, and Phoenix receipt cross-check. |
| Attempt ATOF JSONL | Logical LLM calls, Switchyard decisions/retries/fallbacks, final route, preferred usage chunks, and dedicated `switchyard.routing.llm_call` routing-only usage marks. |
| Attempt OpenInference NDJSON | Route-scoped usage fallback and recorded-cost/model reconciliation receipts. Caller-stub spans without effective usage are not provider-call evidence. |
| Task reconciliation and provider-retry receipts | Preserved repair provenance and failed-attempt hashes; never additional benchmark observations or costable routing attempts. |

Prefer successful attempts named by the source summary. Never merge failed attempts into a successful run observation. Retain failed-attempt and whole-task provider-retry counts as diagnostics only, distinct from retry decisions inside a successful logical LLM call.

Performance evidence and telemetry evidence have separate gates. A valid verifier result remains a performance observation even if OpenInference or Phoenix evidence is incomplete. Such a task is cost-uncovered, not silently excluded from performance.

## Scientific configuration identity

Build a canonical signature from factors that can affect benchmark results or cost:

- dataset identifier, ordered task manifest, and task-definition digest;
- runtime, runner, Relay wheel, Switchyard library/manifest, and plugin-template digests;
- runtime resource overrides and timeout policy;
- routing algorithm, thresholds, targets, endpoints, and retries;
- complete pricing catalog and provenance; and
- concurrency and memory policy.

Exclude pure run identity:

- source filesystem paths;
- run label, cohort ID, Phoenix project, and supervisor identity;
- generated timestamps; and
- telemetry collector destination.

Hash the normalized, scientifically relevant rendered configuration for configuration identity. Preserve the raw rendered-configuration hash as provenance, but do not place that raw hash inside the scientific signature because run-specific cohort/project and telemetry sink values can change its bytes without changing the experiment.

Aggregate only identical scientific signatures. A report may compare differing signatures, but must list which factors differ and must not present their pooled accuracy as repeated samples of one configuration.

An explicit compatibility exception may group runs only when archived evidence proves that every differing signature field is understood and the user intentionally accepts the residual confounding. The manifest must enumerate the exact differing field paths and a rationale; admission fails if observed and declared differences are not identical. Show the exception beside the group result, and state that its variance can include the harness/provenance effect. This mechanism must never hide model, router, dataset, task, pricing, resource, timeout, or other behaviorally relevant differences.

Do not treat a configuration digest as a sufficient human-readable description. Reports must enumerate exact router and judge fields with provenance. A field absent from both rendered configuration and schema is unresolved. A judge control absent from the classifier schema is not assumed to use a conventional default. A built-in prompt that is compiled into the plugin is identified by the plugin library digest unless its text is explicitly preserved in run evidence.

## Performance metrics

For a complete run:

```text
pass@1 = benchmark passes / planned tasks
```

Each planned task contributes exactly one pass or nonpass. Retries repair infrastructure; they do not create extra benchmark observations.

For an incomplete run, do not emit final `pass@1`. Emit both:

```text
observed accuracy = passes / benchmark-complete tasks
planned-task lower bound = passes / planned tasks
```

Label both interim. Missing tasks are neither verifier nonpasses nor silently discarded observations.

For repeated identical configurations, report:

- each run's `pass@1`;
- arithmetic mean and sample standard deviation across runs;
- minimum and maximum;
- pooled trial-weighted accuracy;
- 95% Wilson interval for each binomial run result; and
- task repeatability: passed in 0 through N runs.

With fewer than five runs, prominently describe variance as descriptive. Do not imply a precise population variance.

## Analysis modes

### Aggregate mode

Use for independent runs with identical scientific signatures. The primary variation statistic is sample standard deviation across run-level `pass@1`. Keep pooled accuracy secondary because it does not express between-run variation.

### Comparison mode

Use for baselines and trials with differing signatures. Compare task-aligned observations when task manifests match:

```text
paired accuracy delta = mean(trial outcome - baseline outcome)
```

Report a deterministic task-bootstrap 95% interval and exact McNemar p-value from discordant task pairs. Also report unmatched or incomplete tasks. These statistics describe association; they do not repair confounding when several configuration factors changed together.

When each configuration is represented by repeated independent runs, first calculate each task's pass frequency within each configuration, then compare the trial frequency with the baseline frequency. Report the mean task-frequency delta, a deterministic bootstrap interval over tasks, and counts of tasks with better, equal, or worse trial frequency. Replicate indices are not natural pairs merely because they share labels such as R1/R2/R3; do not manufacture such a pairing. McNemar's binary paired test does not apply to fractional group means. Preserve individual run-to-run comparisons as machine-readable diagnostics rather than headline inference.

When task manifests differ, compare only explicitly compatible aggregate metrics and state that task-paired inference is unavailable.

## Cost baselines and reconciliation

### Observed control

Use an observed control when the user identifies a run that intentionally represents the baseline configuration. Report observed total cost and total-cost delta. Only call the delta savings when task coverage, workload definition, and cost coverage are comparable and the report explains that it is an observed run-level comparison.

### Expensive-model counterfactual

When no control exists, use each eligible call's recorded usage and reprice it with the run's configured expensive target:

```text
billable_input = prompt_tokens - cache_read_tokens
baseline = billable_input * expensive_input_rate
         + output_tokens * expensive_output_rate
         + cache_read_tokens * expensive_cache_read_rate
         + cache_write_tokens * expensive_cache_write_rate
```

Apply the run-bound cache accounting rule. The default formula above assumes cache reads are included in prompt tokens. Divide per-million prices by one million.

Derive observable execution-model cost from recorded usage and the effective completion target's bound price so every covered call uses one accounting method. Prefer the final ATOF usage chunk. When an explicit retry/fallback path has no final ATOF usage chunk, use a route-scoped OpenInference receipt only when it carries nonzero usage. Compare any recorded `llm.cost.total` value against the derived amount and disclose mismatch counts and deltas; do not silently replace the consistent catalog-derived headline with a partial receipt population. Never use a current web price or a conversation-provided estimate unless the user explicitly supplies it as a new hypothetical scenario, which must be separate from run results.

Audit internal router classifier/judge calls separately from agent-facing logical calls. A routing-decision mark proves a choice, not a separately costable judge call. A `switchyard.routing.llm_call` mark is costable only when it identifies the target/model, carries normalized usage, is explicitly marked as routing overhead, and the run-bound catalog prices every reported token class. These marks exclude the final serving call by contract, so add them to the outer logical-call total without double-counting. Normalize prompt tokens as non-cached input plus cache-read and cache-write detail. Normalize billable completion as output plus separately reported reasoning tokens, while preserving reasoning as an independent statistic. If any expected usage or price is absent, mark router-overhead coverage partial, exclude end-to-end totals, and never assign zero tokens or zero cost to the missing evidence.

At call, task, run, and report levels require:

```text
baseline - actual = savings
```

Disclose floating-point tolerance. A logical execution call missing an unambiguous LLM child, effective model, recorded usage, price, or currency is uncovered. Show covered/eligible execution calls beside every headline and report router-overhead coverage separately. Keep raw route-attempt counts, retry counts, fallback counts, and logical-call counts separate; repeated decisions for one logical call must never double-count its final usage.

## Routing and cache metrics

Count effective provider models once per logical ATOF LLM call, applying an explicit `switchyard.routing.fallback` target after decision attempts. Count raw decision, retry, fallback, target, and reason distributions separately. Reconcile `logical calls + retry decisions = route decisions`; never assume raw decision count equals costed logical-call count.

Report:

```text
cache-read ratio = cache_read_tokens / prompt_tokens
```

Also retain prompt, completion, cache-read, and cache-write token totals. Cache is a diagnostic and a priced usage component. Do not add a separate cache-savings value to model-routing savings.

## Admission and sanitization

A final report requires complete benchmark outcomes for every planned task. Monetary headlines additionally require disclosed cost coverage; use partial status when coverage is below 100%.

An interim report must enumerate missing observations and retain their planned denominator.

Never place these in the bundle:

- credentials, authorization headers, or `.env` contents;
- raw prompts, responses, transcripts, or tool payloads;
- absolute source paths or usernames;
- managed runtime trees, images, wheels, or native libraries; or
- unfiltered telemetry.

Use sanitized labels and content digests for provenance. Scan Markdown, JSON, CSV, SVG, YAML, CSS, and finished PDF metadata before release.
