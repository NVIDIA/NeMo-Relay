---
name: terminal-bench-report
description: Generate reproducible quantitative Terminal-Bench reports from one or more run-root artifact directories. Use when Codex must analyze, aggregate, compare, visualize, or report benchmark performance, pass@1, cross-run variance, task outcomes, model routing, cost versus an observed or synthetic baseline, and cache efficiency; create structured CSV/JSON/SVG evidence; or render a consistent Pandoc PDF without relying on conversation-history estimates.
---

# Terminal-Bench Report

Build reports from source artifacts and run-bound configuration. Never use remembered counts, prices, model names, or directory conventions as evidence.

## Workflow

1. Accept one or more arbitrary run-root paths and the user's analysis question.
2. Treat run roots as read-only. Never stop, resume, clean, or modify a benchmark runtime while reporting.
3. Inspect `summary.json`, `plan.json`, the rendered plugin configuration, successful-attempt summaries, ATOF logical-call/routing/usage events, and OpenInference reconciliation receipts.
4. Resolve the analysis:
   - Use aggregate mode for repeated runs with one scientific configuration signature.
   - Use comparison mode when configuration signatures differ.
   - Keep different configuration groups separate unless the user explicitly requests a justified cross-group statistic.
   - Enumerate the exact router algorithm, classifier/judge target, thresholds, context window, session behavior, retry policy, model roles, transport settings, request-handling flags, and pricing for every configuration group.
   - Distinguish explicitly configured values from bundled-schema defaults. Mark non-exposed judge controls as unavailable; never silently assume `temperature`, `top_p`, `seed`, or a built-in prompt value.
5. Resolve the cost baseline:
   - Prefer an explicitly designated observed control run when one exists.
   - Otherwise derive the counterfactual by repricing each eligible observed call with the expensive model and pricing catalog bound to that run.
   - When both exist, report the observed-control difference and same-workload all-expensive counterfactual as separate estimands. The first compares realized independent runs; the second holds each routed run's recorded usage and cache behavior fixed.
   - Group repeated configurations before headline reporting and show N, mean, and sample standard deviation for observed cost, counterfactual cost, absolute savings, and per-run savings percentage.
   - Never substitute current web pricing for run-bound pricing.
   - Separate agent-facing completion-target calls from internal router classifier/judge calls. Parse `switchyard.routing.llm_call` marks as routing-only usage, verify that they exclude the serving call, and add their catalog-derived cost only at complete mark usage/price coverage. Report prompt, completion, reasoning, cache-read, and cache-write routing tokens independently. Otherwise label monetary results as observable execution-model cost, disclose the missing overhead, and never call the delta end-to-end savings.
6. Run `scripts/analyze.py`. Use `--allow-partial` only when an interim report is appropriate.
7. Inspect `evidence/report-validation.json`, reconciliation evidence, CSV denominators, and every chart before describing results.
8. Render PDF with `scripts/render_pdf.sh` only after Markdown validation passes and the declared Pandoc dependencies are available.

Read [evidence-and-statistics.md](references/evidence-and-statistics.md) when selecting evidence, baselines, comparison statistics, or interpreting incomplete runs. Read [report-format.md](references/report-format.md) before editing report prose, charts, or Pandoc output.

## Quantitative generator

Use explicit labels so reports never expose source paths:

```bash
python terminal-bench-report/scripts/analyze.py \
  --run-root /absolute/run-a \
  --label trial-a \
  --group trial \
  --run-root /absolute/run-b \
  --label trial-b \
  --group trial \
  --output-dir /absolute/new-report-bundle \
  --mode auto \
  --expected-group-size 2 \
  --analysis-request "Aggregate repeated runs and emphasize variance"
```

Options:

- `--mode auto`: aggregate identical scientific configurations; compare differing configurations.
- `--baseline-model MODEL`: override the expensive synthetic model only when the user explicitly requests it and every run-bound catalog contains it.
- `--baseline-run LABEL`: identify an observed control for total-cost comparison. Do not call its total-cost difference call-level routing savings.
- `--group LABEL`: assign each run to a repeated-configuration group; supply it once per run or omit it for every run.
- `--baseline-group LABEL`: use the mean observed cost across one declared group as the control for configuration-level cost comparison.
- `--expected-group-size N`: fail admission unless every declared group contains exactly N independent runs.
- `--group-compatibility-manifest PATH`: admit a repeated group with differing scientific signatures only when the manifest declares the exact differing fields and a nonempty rationale. The report must expose the exception and treat its variance as potentially confounded. Omit this option for normal identical-signature groups.
- `--allow-partial`: emit a clearly marked interim report with missing denominators.
- `--replace`: explicitly replace an existing output bundle; otherwise generation fails closed.

The generator emits:

```text
README.md
aggregate-metrics.json
run-metrics.csv
configuration-group-metrics.csv
configuration-group-comparisons.csv
task-metrics.csv
task-aggregate-metrics.csv
call-metrics.csv
charts/*.svg
evidence/admission.json
evidence/configurations.json
evidence/reconciliation.json
evidence/report-validation.json
evidence/pdf-validation.json  # after PDF rendering
evidence/bundle-manifest.json
report.css
report-header.tex
pandoc-defaults.yaml
README.pdf                   # after PDF rendering
```

## Non-negotiable rules

- Derive every number from structured files. Conversation history may define the question, never the result.
- Never hard-code a dataset version, task count, model, price, run root, cohort, or project.
- Keep benchmark-performance admission independent from integration and cost admission.
- Emit final `pass@1` only for a run with one benchmark-complete outcome for every planned task.
- For incomplete runs, show observed accuracy and pass-count/planned-task lower bound; label both interim.
- Omit the planned-task lower bound from tables and charts when every supplied run is final.
- Preserve nonpasses. Never drop a valid verifier nonpass from the denominator.
- Report cost coverage beside every monetary claim. Never silently impute an uncovered call.
- Derive observable execution-model cost consistently from recorded usage and run-bound pricing; reconcile recorded cost receipts separately, then require baseline minus execution cost to equal the execution-call difference.
- Cost each logical LLM call once. Treat repeated routing decisions as retry attempts and apply an explicit fallback target as the effective route.
- Keep whole-task provider retry receipts separate from routing-decision retries. Analyze only the successful attempt, but count preserved provider retries in reconciliation evidence and disclose them as run provenance.
- Never assume that a routing decision contains the classifier/judge's token usage. Audit `switchyard.routing.llm_call` marks separately; absent usage means absent cost, not zero cost. Treat normalized `input_tokens` as non-cached input, add cache-read and cache-write detail to the aggregate prompt count, and add separately reported reasoning tokens to billable completion tokens while retaining a separate reasoning total.
- Keep cache efficiency diagnostic. Do not add cache reuse as a separate routing-savings term.
- Never infer cost impact from call-count routing share alone. When explaining model selection, show token- and covered-cost-weighted shares beside call share.
- Use sample standard deviation for repeated-run variation and state that small-run estimates are descriptive.
- Require identical scientific signatures within a repeated group by default. A provenance-compatibility exception must be explicit, field-exact, justified from archived evidence, visible in the report, and interpreted as a possible source of variance.
- For one-run configurations, use task-aligned paired binary comparisons. For declared repeated-run groups, compare per-task group pass frequencies with a deterministic task bootstrap; do not manufacture a replicate pairing or apply McNemar to fractional group means. Disclose missing tasks.
- Exclude credentials, transcripts, managed runtimes, absolute source paths, and raw prompt/response content from bundles.
- Refuse final status when required evidence or validation checks fail.

## Report review

Confirm all of the following before handoff:

- The summary accurately identifies aggregate versus comparison mode.
- Configuration signatures and the differing scientific factors are visible.
- Router and judge settings are human-readable, include value provenance, and do not rely on the configuration digest alone.
- Accuracy, cost, routing, and cache figures agree across Markdown, JSON, CSV, and charts.
- Monetary language distinguishes observable completion-target cost from end-to-end total cost whenever router-side model usage is unavailable.
- Variance or paired-effect statistics accompany multi-run claims.
- Synthetic baselines are labelled counterfactual; observed controls are labelled observed.
- The task appendix contains every run/task observation in `task-metrics.csv`.
- `evidence/report-validation.json` passes.
- The PDF, when requested, has been visually inspected for clipped tables, chart labels, page breaks, and working internal structure.
