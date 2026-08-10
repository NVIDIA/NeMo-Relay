# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the latency benchmark fixture."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.latency_benchmark.src import fixtures
from scripts.latency_benchmark.src.config import (
    DEFAULT_CONFIG_PATH,
    MiddlewareVariant,
    build_parser,
    load_config,
    parse_args,
)
from scripts.latency_benchmark.src.fixtures import (
    isolated_environment,
    write_agent_config,
    write_mock_codex,
    write_plugin_configs,
)


class BenchmarkConfigTests(unittest.TestCase):
    def test_default_config_defines_every_suite_and_matrix_axis(self) -> None:
        config = load_config(DEFAULT_CONFIG_PATH)

        self.assertEqual(config.tests, ("gateway", "hooks", "startup"))
        self.assertEqual(config.providers, ("openai", "anthropic"))
        self.assertEqual(config.modes, ("buffered", "streaming"))
        self.assertTrue(config.payload_sizes)
        self.assertTrue(config.concurrency)
        self.assertEqual(config.middleware, ())

    def test_partial_config_and_cli_arguments_override_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            custom_config = root / "quick.toml"
            custom_config.write_text('tests = ["startup"]\nsamples = 7\n', encoding="utf-8")
            relay_bin = root / "nemo-relay"
            relay_bin.touch()

            options = parse_args(
                [
                    "--relay-bin",
                    str(relay_bin),
                    "--output",
                    str(root / "results.json"),
                    "--config",
                    str(custom_config),
                    "--tests",
                    "gateway,hooks",
                    "--samples",
                    "3",
                    "--concurrency",
                    "1",
                    "--providers",
                    "openai",
                ]
            )

        self.assertEqual(options.config.tests, ("gateway", "hooks"))
        self.assertEqual(options.config.samples, 3)
        self.assertEqual(options.config.concurrency, (1,))
        self.assertEqual(options.config.providers, ("openai",))
        self.assertEqual(options.config.modes, ("buffered", "streaming"))
        self.assertEqual(options.report, (root / "results.html").resolve())

    def test_help_uses_just_entrypoint_and_explains_every_option(self) -> None:
        help_text = build_parser().format_help()

        self.assertIn("usage: just latency-benchmark [options]", help_text)
        for option in (
            "-h, --help",
            "--relay-bin",
            "--output",
            "--report",
            "--config",
            "--tests",
            "--providers",
            "--modes",
            "--payload-sizes",
            "--concurrency",
            "--middleware",
            "--samples",
            "--hook-samples",
            "--startup-samples",
            "--warmup",
            "--response-bytes",
            "--stream-chunks",
        ):
            with self.subTest(option=option):
                self.assertIn(option, help_text)

    def test_report_path_can_be_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            relay_bin = root / "nemo-relay"
            relay_bin.touch()

            options = parse_args(
                [
                    "--relay-bin",
                    str(relay_bin),
                    "--output",
                    str(root / "results.json"),
                    "--report",
                    str(root / "site" / "index.html"),
                ]
            )

        self.assertEqual(options.report, (root / "site" / "index.html").resolve())

    def test_middleware_paths_are_relative_to_the_custom_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugin_config = root / "plugins-guardrails.toml"
            plugin_config.write_text("version = 1\ncomponents = []\n", encoding="utf-8")
            config_path = root / "benchmark.toml"
            config_path.write_text(
                '[[middleware]]\nname = "guardrails"\nplugin_config = "plugins-guardrails.toml"\n',
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertEqual(
            config.middleware,
            (MiddlewareVariant(name="guardrails", plugin_config=plugin_config.resolve()),),
        )

    def test_cli_middleware_replaces_configured_middleware(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            relay_bin = root / "nemo-relay"
            relay_bin.touch()
            plugin_config = root / "plugins-redaction.toml"
            plugin_config.write_text("version = 1\ncomponents = []\n", encoding="utf-8")

            options = parse_args(
                [
                    "--relay-bin",
                    str(relay_bin),
                    "--output",
                    str(root / "results.json"),
                    "--middleware",
                    f"redaction={plugin_config}",
                ]
            )

        self.assertEqual(
            options.config.middleware,
            (MiddlewareVariant(name="redaction", plugin_config=plugin_config.resolve()),),
        )

    def test_rejects_reserved_middleware_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugin_config = root / "plugins.toml"
            plugin_config.touch()
            config_path = root / "invalid.toml"
            config_path.write_text(
                '[[middleware]]\nname = "minimal"\nplugin_config = "plugins.toml"\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "reserved by a default variant"):
                load_config(config_path)

    def test_rejects_unknown_config_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "invalid.toml"
            config_path.write_text("sample = 1\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unknown config key"):
                load_config(config_path)

    def test_rejects_gateway_concurrency_greater_than_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "invalid.toml"
            config_path.write_text(
                'tests = ["gateway"]\nsamples = 2\nconcurrency = [4]\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "samples must be greater"):
                load_config(config_path)


class StaticFixtureTests(unittest.TestCase):
    def test_materializes_templates_without_embedded_markers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            custom_config = root / "plugins-guardrails.toml"
            custom_config.write_text("version = 1\ncomponents = []\n", encoding="utf-8")
            configs = write_plugin_configs(
                root,
                "http://127.0.0.1:4318",
                (MiddlewareVariant(name="guardrails", plugin_config=custom_config),),
            )
            mock_codex = write_mock_codex(root)
            agent_config = write_agent_config(root, "test", mock_codex)

            rendered = "\n".join(path.read_text(encoding="utf-8") for path in (*configs.values(), agent_config))

        self.assertEqual(mock_codex.stem, "mock-codex")
        self.assertNotIn("__ATOF_OUTPUT_DIRECTORY__", rendered)
        self.assertNotIn("__OTLP_ENDPOINT__", rendered)
        self.assertNotIn("__CODEX_COMMAND__", rendered)
        self.assertEqual(configs["relay-guardrails"], custom_config)

    def test_materializes_windows_mock_with_crlf_line_endings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(fixtures.os, "name", "nt"):
                mock_codex = write_mock_codex(root)
            contents = mock_codex.read_bytes()

        self.assertIn(b"\r\n", contents)
        self.assertNotIn(b"\n", contents.replace(b"\r\n", b""))

    def test_isolated_environment_removes_values_that_can_skew_results(self) -> None:
        inherited = {
            "ANTHROPIC_API_KEY": "secret",
            "HTTP_PROXY": "http://proxy.example",
            "NEMO_RELAY_CONFIG": "/tmp/developer-config.toml",
            "OPENAI_API_KEY": "secret",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector.example",
            "PATH": "/usr/bin",
            "RUST_LOG": "debug",
            "https_proxy": "http://proxy.example",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.dict(fixtures.os.environ, inherited, clear=True):
                environment = isolated_environment(root)

        self.assertEqual(environment["PATH"], "/usr/bin")
        self.assertEqual(environment["NO_PROXY"], "127.0.0.1,localhost")
        self.assertEqual(environment["no_proxy"], "127.0.0.1,localhost")
        for name in inherited.keys() - {"PATH"}:
            with self.subTest(name=name):
                self.assertNotIn(name, environment)


if __name__ == "__main__":
    unittest.main()
