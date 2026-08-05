# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the self-contained benchmark report."""

import tempfile
import unittest
from pathlib import Path
from typing import Any

from scripts.latency_benchmark.src.html_report import render_html_report, write_html_report


def _summary(*, interval: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "samples": 5,
        "p50_ms": 1.0,
        "p95_ms": 1.5,
        "p99_ms": 1.6,
        "min_ms": 0.8,
        "max_ms": 1.7,
    }
    if interval:
        result["median_ci95_ms"] = [0.9, 1.1]
    return result


def _sample_results() -> dict[str, Any]:
    absolute = {
        name: {"total": _summary()}
        for name in ("direct", "relay-minimal", "relay-file", "relay-otlp", "relay-guardrails")
    }
    comparisons = {
        name: {"total": _summary(interval=True)}
        for name in (
            "relay-minimal_vs_direct",
            "relay-file_vs_direct",
            "relay-otlp_vs_direct",
            "file_exporter_vs_minimal",
            "otlp_exporter_vs_minimal",
            "relay-guardrails_vs_direct",
            "guardrails_vs_minimal",
        )
    }
    process_absolute = {
        "process_baseline": _summary(),
        "relay-minimal": _summary(),
    }
    return {
        "schema_version": 2,
        "environment": {
            "generated_at": "2026-08-05T12:00:00+00:00",
            "git_commit": "0123456789abcdef",
            "git_dirty": False,
            "relay_version": "nemo-relay 0.1.0",
            "platform": "test",
        },
        "parameters": {
            "tests": ["gateway", "hooks", "startup"],
            "providers": ["openai"],
            "modes": ["buffered"],
            "samples": 5,
            "payload_sizes": [4096],
            "concurrency": [1],
            "middleware": [{"name": "guardrails", "plugin_config": "/tmp/plugins.toml"}],
        },
        "gateway": [
            {
                "provider": "openai",
                "mode": "buffered",
                "payload_bytes": 4096,
                "serialized_request_bytes": 4200,
                "concurrency": 1,
                "absolute": absolute,
                "comparisons": comparisons,
            }
        ],
        "hooks": {
            "absolute": {"process_baseline": _summary(), "codex_minimal": _summary()},
            "comparisons": {"codex_minimal_vs_process_baseline": _summary(interval=True)},
        },
        "startup": {
            "absolute": process_absolute,
            "comparisons": {"relay-minimal_readiness_vs_process_baseline": _summary(interval=True)},
        },
        "exporter_delivery": {"atof_bytes": 8192, "otlp_requests": 4},
    }


class HtmlReportTests(unittest.TestCase):
    def test_report_embeds_results_and_static_assets(self) -> None:
        report = render_html_report(_sample_results())

        self.assertIn("<title>NeMo Relay Latency Report</title>", report)
        self.assertIn("<h1>NeMo Relay Latency Report</h1>", report)
        self.assertNotIn("Coding-Agent Latency Report", report)
        self.assertNotIn('class="eyebrow"', report)
        self.assertNotIn('class="lede"', report)
        self.assertIn('"git_commit":"0123456789abcdef"', report)
        self.assertIn("Gateway Latency by Payload Size", report)
        self.assertIn('"relay-file": "ATOF file exporter"', report)
        self.assertIn('"relay-otlp": "OTLP exporter"', report)
        self.assertIn("function drawLineChart", report)
        self.assertNotIn("__BENCHMARK_DATA__", report)
        self.assertNotIn("__BENCHMARK_STYLES__", report)
        self.assertNotIn("__BENCHMARK_SCRIPT__", report)

    def test_report_escapes_script_terminators_in_embedded_data(self) -> None:
        results = _sample_results()
        results["environment"]["platform"] = "</script><script>alert('unsafe')</script>"

        report = render_html_report(results)

        self.assertNotIn("</script><script>alert", report)
        self.assertIn("\\u003c/script\\u003e", report)

    def test_write_report_creates_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "report.html"

            write_html_report(_sample_results(), path)

            self.assertTrue(path.is_file())
            self.assertIn("<!doctype html>", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
