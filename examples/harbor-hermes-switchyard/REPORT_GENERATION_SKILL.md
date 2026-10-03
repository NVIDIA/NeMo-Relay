<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Terminal-Bench report generation skill

The canonical, executable Codex skill is [terminal-bench-report/SKILL.md](terminal-bench-report/SKILL.md).

That skill supersedes the earlier run-specific draft. It accepts arbitrary Terminal-Bench run roots, resolves aggregation versus configuration comparison from artifact-bound scientific configuration signatures, calculates performance and costs from structured evidence, emits CSV/JSON/SVG artifacts, and defines a deterministic Pandoc PDF contract.

Use the skill entrypoint and its linked references rather than duplicating instructions here:

- [Evidence and statistics contract](terminal-bench-report/references/evidence-and-statistics.md)
- [Report and Pandoc format contract](terminal-bench-report/references/report-format.md)
- [Quantitative analyzer](terminal-bench-report/scripts/analyze.py)
- [PDF renderer](terminal-bench-report/scripts/render_pdf.sh)

Example:

```bash
python terminal-bench-report/scripts/analyze.py \
  --run-root /absolute/run-a \
  --label run-a \
  --run-root /absolute/run-b \
  --label run-b \
  --output-dir /absolute/new-report-bundle \
  --mode auto \
  --analysis-request "Aggregate repeated runs and report cross-run variance"
```

Do not add current cohort paths, task counts, model names, prices, or result values to this forwarding document. Those values must always be resolved from the supplied run artifacts.
