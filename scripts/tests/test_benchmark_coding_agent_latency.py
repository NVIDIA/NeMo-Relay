# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the coding-agent latency benchmark fixture."""

import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_coding_agent_latency.config import DEFAULT_CONFIG_PATH, load_config, parse_args
from scripts.benchmark_coding_agent_latency.fixtures import write_agent_config, write_fake_codex, write_plugin_configs


class BenchmarkConfigTests(unittest.TestCase):
    def test_default_config_defines_every_suite_and_matrix_axis(self) -> None:
        config = load_config(DEFAULT_CONFIG_PATH)

        self.assertEqual(config.tests, ("gateway", "hooks", "startup"))
        self.assertEqual(config.providers, ("openai", "anthropic"))
        self.assertEqual(config.modes, ("buffered", "streaming"))
        self.assertTrue(config.payload_sizes)
        self.assertTrue(config.concurrency)

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
            configs = write_plugin_configs(root, "http://127.0.0.1:4318")
            fake_codex = write_fake_codex(root)
            agent_config = write_agent_config(root, "test", fake_codex)

            rendered = "\n".join(path.read_text(encoding="utf-8") for path in (*configs.values(), agent_config))

        self.assertNotIn("__ATOF_OUTPUT_DIRECTORY__", rendered)
        self.assertNotIn("__OTLP_ENDPOINT__", rendered)
        self.assertNotIn("__CODEX_COMMAND__", rendered)


if __name__ == "__main__":
    unittest.main()
