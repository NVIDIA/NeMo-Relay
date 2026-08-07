# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a self-contained HTML report for benchmark results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPORT_ROOT = Path(__file__).resolve().parent / "report"
TEMPLATE_PATH = REPORT_ROOT / "template.html"
STYLES_PATH = REPORT_ROOT / "styles.css"
SCRIPT_PATH = REPORT_ROOT / "report.js"


def _embedded_json(results: dict[str, Any]) -> str:
    """Serialize JSON so it cannot terminate its script element."""
    return (
        json.dumps(results, separators=(",", ":"), sort_keys=True)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def render_html_report(results: dict[str, Any]) -> str:
    """Return a portable HTML report with embedded styles, data, and scripts."""
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    replacements = {
        "__BENCHMARK_STYLES__": STYLES_PATH.read_text(encoding="utf-8"),
        "__BENCHMARK_DATA__": _embedded_json(results),
        "__BENCHMARK_SCRIPT__": SCRIPT_PATH.read_text(encoding="utf-8"),
    }
    for marker, value in replacements.items():
        template = template.replace(marker, value)
    return template


def write_html_report(results: dict[str, Any], path: Path) -> None:
    """Write a portable HTML report to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html_report(results), encoding="utf-8")
