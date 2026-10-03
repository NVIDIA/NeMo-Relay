<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Report and Pandoc format contract

## Contents

1. Required narrative order
2. Required tables and charts
3. Markdown rules
4. PDF workflow
5. Release inspection

## Required narrative order

1. Title and report status.
2. Executive summary with the comparison question, configurations, performance, cost, baseline, savings, and coverage.
3. Configuration comparison with scientific signatures and meaningful differences.
4. Exact router configuration, including classifier/judge settings and provenance for configured values versus schema defaults.
5. Performance and cross-run variation.
6. Cost baseline, serving-call cost, routing-only prompt/completion/reasoning/cache tokens, router-overhead cost, end-to-end model cost, receipt reconciliation, and coverage.
7. Model selection and routing-reason distributions.
8. Cache efficiency.
9. Task-level findings and link to the full CSV appendix.
10. Methodology, limitations, and provenance.
11. Machine-readable artifact index.

Lead with decisions and results. Keep implementation details in methodology. State denominators beside every percentage.

## Required tables and charts

Always generate:

- executive run table, or an executive repeated-group table when declared N-run groups are present;
- compared-configuration table;
- performance-by-run chart;
- observable execution-model cost versus baseline chart;
- effective-model distribution chart;
- cache-read efficiency chart; and
- task outcome matrix.

For repeated configurations, add run mean, sample standard deviation, range, pooled accuracy, and repeatability distribution. For one-run configuration comparisons, add paired deltas, bootstrap interval, discordant pairs, and McNemar result when task alignment permits. For repeated-configuration comparisons, add task-aligned group pass-frequency deltas, a task-bootstrap interval, and better/equal/worse task counts; do not apply McNemar to fractional group means.

When the comparison contains declared repeated-configuration groups, add configuration-level performance and cost charts that show the mean bar, sample-standard-deviation whisker, and every independent run as a point. Prefer aligned panels with a shared configuration order over a dual-axis overlay, so neither metric's visual scale can imply a spurious relationship.

The router section must identify the algorithm and its complete resolved settings, judge/classifier model, weak and strong models, threshold policy, recent-turn window, session affinity, hash fallback, retry count, default target, target weights, protocols, URLs, caller-body handling, and run-bound prices. Show whether each value was explicitly configured or supplied by the bundled schema. If judge prompt text or sampling controls are internal to the plugin and absent from run artifacts, state that limitation and cite the bound plugin digest rather than inventing values.

The cost section must distinguish completion-target usage from classifier/judge overhead. When `switchyard.routing.llm_call` marks are available, show mark count, prompt tokens, completion tokens, reasoning tokens, cache-read/write tokens, cost, and coverage by configuration group; only then add the overhead to serving-call cost. If router-side model usage is not emitted or any marked token class is unpriced, show that gap by configuration, exclude it explicitly, and avoid the terms total cost, actual end-to-end cost, or end-to-end savings. A configured judge price without judge token usage is provenance, not a cost estimate.

When an observed control and an all-expensive-model counterfactual are both available, report them as separate estimands. Group independent repeats before presenting headline cost: show N, mean, and sample standard deviation for observed covered execution cost, counterfactual cost, absolute savings, and per-run savings percentage. Describe the observed-control comparison as the realized cross-run difference. Describe the counterfactual as same-workload repricing that holds each routed run's recorded usage and cache behavior fixed. Never substitute call-count routing share for token- or cost-weighted share; show those weights when model selection is used to explain savings.

Every chart must have:

- a descriptive title;
- explicit units;
- reconstructable denominator in the chart or adjacent table;
- stable colors across the report;
- readable labels without relying only on color; and
- a structured CSV/JSON source in the same bundle.

Do not use decorative plots, 3-D charts, truncated percentage axes, or unlabeled smoothing.

## Markdown rules

- Use one H1 title. Start major sections at H2 and subsections at H3.
- Keep heading text stable so PDF bookmarks remain predictable.
- Use pipe tables with explicit alignment rows.
- Keep tables narrow. Move high-dimensional data to CSV instead of shrinking it beyond readability.
- Format model IDs, digests, datasets, and field names as inline code.
- Use relative image links below the paragraph that interprets each chart.
- Put interim/failure warnings in blockquotes immediately below the title.
- Write percentages to two decimals, costs to four decimals, and raw counts as integers.
- Never write a percentage without its numerator and denominator in the same table or nearby sentence.
- Use “nonpass” for a valid verifier result of zero and “missing” for no completed verifier observation.
- Use “counterfactual” for an all-expensive-model baseline and “observed control” for a measured control run.
- Omit the planned-task lower-bound column and series when every supplied run is final; it is an interim-only statistic.
- Do not use `pass@N` for repeated independent `pass@1` runs.

## PDF workflow

The generator copies `pandoc-defaults.yaml`, `report-header.tex`, and `report.css` into each bundle. The supported deterministic PDF path uses:

- Pandoc;
- XeLaTeX;
- `rsvg-convert` for SVG figures;
- Poppler `pdfinfo` and `pdftotext` for structural and content validation;
- DejaVu Sans and DejaVu Sans Mono; and
- a letter-sized page with a 0.7-inch margin;
- stable table spacing, link colors, captions, and widow/orphan controls from `report-header.tex`; and
- `report.css` for optional HTML previews only. PDF layout comes from the defaults and TeX header.

Run:

```bash
terminal-bench-report/scripts/render_pdf.sh /absolute/report-bundle
```

Do not silently fall back to another PDF engine. Different engines change pagination, fonts, table wrapping, and SVG handling. If dependencies are absent, preserve the validated Markdown bundle and report the missing dependency.

The renderer requires the PDF to remain inside its bundle. After rendering, it writes `evidence/pdf-validation.json`, scans extracted PDF text for credential and absolute-path patterns, verifies a nonempty letter-sized document, and refreshes `evidence/bundle-manifest.json` so the PDF and its validation receipt are covered.

## Release inspection

Open the finished PDF and inspect every page. Confirm:

- no clipped or overflowing table cells;
- no split headings or orphaned chart captions;
- legible chart legends and axis labels;
- consistent model colors;
- correct page count and bookmarks;
- no absolute path, username, secret, or telemetry payload;
- agreement among PDF, Markdown, JSON, CSV, and chart labels; and
- the bundle manifest was generated after all content was finalized.

If prose is edited after generation, regenerate machine validation and the bundle manifest before release.
