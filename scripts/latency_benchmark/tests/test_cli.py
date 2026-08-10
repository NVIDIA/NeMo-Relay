# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for latency benchmark orchestration."""

import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.latency_benchmark.src import cli
from scripts.latency_benchmark.src.config import DEFAULT_CONFIG_PATH, load_config


class BenchmarkOrchestrationTests(unittest.TestCase):
    def test_counts_gateway_scenarios_and_process_suites(self) -> None:
        config = load_config(DEFAULT_CONFIG_PATH)

        expected_gateway = (
            len(config.providers) * len(config.modes) * len(config.payload_sizes) * len(config.concurrency)
        )

        self.assertEqual(cli._benchmark_test_count(config), expected_gateway + 2)

    def test_reports_progress_for_completed_suites(self) -> None:
        config = replace(load_config(DEFAULT_CONFIG_PATH), tests=("hooks", "startup"))
        servers = [
            contextlib.nullcontext("http://127.0.0.1:8000"),
            contextlib.nullcontext("http://127.0.0.1:4318"),
        ]

        with (
            mock.patch.object(cli, "environment_record", return_value={}),
            mock.patch.object(cli, "local_server", side_effect=servers),
            mock.patch.object(cli, "write_relay_config"),
            mock.patch.object(cli, "write_plugin_configs", return_value={}),
            mock.patch.object(cli, "benchmark_hooks", return_value={}),
            mock.patch.object(cli, "benchmark_startup", return_value={}),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            cli.run_benchmarks(Path("nemo-relay"), config)

        self.assertIn("[1/2] Completed hooks suite", output.getvalue())
        self.assertIn("[2/2] Completed startup suite", output.getvalue())

    def test_reports_completed_gateway_scenario(self) -> None:
        config = replace(
            load_config(DEFAULT_CONFIG_PATH),
            providers=("openai",),
            modes=("buffered",),
            payload_sizes=(4096,),
            concurrency=(1,),
        )
        report_complete = mock.Mock()

        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(cli, "benchmark_scenario", return_value={"scenario": "complete"}),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            scenarios = cli._benchmark_gateway(
                Path("nemo-relay"),
                Path(temporary),
                "http://127.0.0.1:8000",
                {},
                config,
                report_complete,
            )

        self.assertEqual(scenarios, [{"scenario": "complete"}])
        report_complete.assert_called_once_with("gateway openai buffered, payload=4096, concurrency=1")

    def test_records_exporter_validation_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(cli.OtlpHandler, "request_count", 0):
                delivery, errors = cli._exporter_delivery(Path(temporary))

        self.assertEqual(delivery, {"atof_bytes": 0, "otlp_requests": 0})
        self.assertEqual(
            errors,
            [
                "local ATOF exporter did not write benchmark events",
                "local OTLP receiver did not receive benchmark exports",
            ],
        )

    def test_main_writes_results_before_raising_validation_error(self) -> None:
        results = {
            "schema_version": 2,
            "validation_errors": ["local OTLP receiver did not receive benchmark exports"],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            options = SimpleNamespace(
                relay_bin=root / "nemo-relay",
                config=mock.Mock(),
                output=root / "results.json",
                report=root / "results.html",
            )
            with (
                mock.patch.object(cli, "parse_args", return_value=options),
                mock.patch.object(cli, "run_benchmarks", return_value=results),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                with self.assertRaisesRegex(RuntimeError, "benchmark validation failed"):
                    cli.main([])

            self.assertEqual(json.loads(options.output.read_text(encoding="utf-8")), results)
            self.assertTrue(options.report.is_file())


if __name__ == "__main__":
    unittest.main()
