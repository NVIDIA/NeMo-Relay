# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for preparation, process cleanup, and routing summaries."""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from run import main, run_task
from summarize import summarize


def mark(name, data, identity):
    return {"kind": "mark", "name": name, "data": data, "uuid": identity}


class SummaryChecks(unittest.TestCase):
    def test_managed_host_is_rejected_before_running_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unused"
            environment = {
                "NVIDIA_API_KEY": "test-only",
                "NVIDIA_BASE_URL": "https://integrate.api.nvidia.com/v1",
                "EFFICIENT_MODEL": "example/efficient",
                "CAPABLE_MODEL": "example/capable",
                "SWITCHYARD_MANIFEST": "not-loaded.toml",
            }
            with (
                mock.patch.dict(os.environ, environment, clear=True),
                mock.patch.object(sys, "argv", ["run.py", "--harness", "codex", "--output", str(output)]),
                mock.patch("run.shutil.which", return_value="installed"),
                mock.patch("run.Path.exists", return_value=True),
                contextlib.redirect_stderr(io.StringIO()) as diagnostic,
                self.assertRaises(SystemExit) as error,
            ):
                main()
            self.assertEqual(error.exception.code, 2)
            self.assertIn("fresh host", diagnostic.getvalue())
            self.assertFalse(output.exists())

    def test_preparation_needs_no_harness_binary_or_credentials(self):
        # Both choices must be inspectable before installing a harness or
        # spending inference tokens. An empty PATH catches accidental launches.
        for harness in ("claude", "codex"):
            with self.subTest(harness=harness), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "prepared"
                environment = {
                    "PATH": "",
                    "NVIDIA_BASE_URL": "https://integrate.api.nvidia.com/v1",
                    "EFFICIENT_MODEL": "example/efficient",
                    "CAPABLE_MODEL": "example/capable",
                    "SWITCHYARD_MANIFEST": str(Path(directory) / "not-installed.toml"),
                }
                result = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("run.py")),
                        "--harness",
                        harness,
                        "--output",
                        str(output),
                        "--prepare-only",
                        "--task",
                        "ticket-id",
                    ],
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(json.loads((output / "comparison.json").read_text())["harness"], harness)
                self.assertTrue((output / "routed/ticket-id-1/service.py").exists())
                self.assertFalse((output / "routed/dependency-plan-1").exists())
                self.assertFalse((output / "results.json").exists())

    def test_acceptance_rejects_unicode_digit_and_prefix_regressions(self):
        # Both defects appeared in real model outputs. Keep the external check
        # capable of rejecting them even when the agent's own tests pass.
        for pattern, flags, expected in (
            (r"INC-\d{4}", "re.IGNORECASE | re.ASCII", 0),
            (r"INC-\d{4}", "re.IGNORECASE", 1),
            (r"INC-[0-9]{4}", "re.IGNORECASE", 1),
        ):
            with self.subTest(pattern=pattern, flags=flags), tempfile.TemporaryDirectory() as directory:
                (Path(directory) / "service.py").write_text(
                    "import re\n"
                    "def normalize_ticket_id(value):\n"
                    "    if not isinstance(value, str): raise ValueError()\n"
                    f"    if not re.fullmatch({pattern!r}, value.strip(), {flags}): raise ValueError()\n"
                    "    return value.strip().upper()\n"
                )
                result = subprocess.run(
                    [sys.executable, str(Path(__file__).with_name("acceptance.py")), directory, "ticket-id"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)

    @unittest.skipUnless(os.name == "posix", "The companion runs on Linux/macOS")
    def test_timeout_stops_child_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "child-survived"
            child = "import pathlib,time; time.sleep(1); pathlib.Path('child-survived').write_text('alive')"
            parent = (
                f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}], "
                "start_new_session=True); time.sleep(60)"
            )
            with (Path(directory) / "agent.log").open("w") as log:
                status = run_task(
                    [sys.executable, "-c", parent], cwd=directory, env=os.environ.copy(), log=log, timeout=0.3
                )
            self.assertEqual(status, "timeout")
            time.sleep(1.1)
            self.assertFalse(marker.exists(), "Harness child outlived the task timeout")

    def test_usage_is_not_double_counted_or_filled_with_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "routes.toml").write_text('[targets.efficient]\nid="small"\n[targets.capable]\nid="large"\n')
            (output / "results.json").write_text(
                json.dumps(
                    [
                        {"arm": "baseline", "accepted": True, "seconds": 2},
                        {"arm": "efficient", "accepted": False, "seconds": 3},
                    ]
                )
            )
            call = mark(
                "switchyard.routing.llm_call",
                {"call_role": "answer", "selected_model": "large", "outcome": "ok"},
                "call",
            )
            decision = mark(
                "switchyard.routing.decision",
                {
                    "selected_model": "large",
                    "served_model": "actual-large",
                    "fallback_used": False,
                    "outcome_id": "classification",
                    "evidence": {"source": "fail_open", "reason_code": "invalid_verdict"},
                },
                "decision",
            )
            metrics = mark(
                "metric",
                {
                    "measurements": [
                        {
                            "name": "switchyard.routing.llm_tokens",
                            "value": value,
                            "attributes": {"call_role": "answer", "token_type": kind, "target_model": "large"},
                        }
                        for kind, value in (("input", 5), ("output", 2), ("total", 7))
                    ]
                },
                "metric",
            )
            metrics["data_schema"] = {"name": "nemo.relay.metric_measurements", "version": "1.0"}
            logs = output / "baseline/logs"
            logs.mkdir(parents=True)
            (logs / "events.jsonl").write_text("\n".join(map(json.dumps, [call, decision, metrics, metrics])))
            with (logs / "events.jsonl").open("a") as stream:
                stream.write(
                    "\n"
                    + json.dumps(
                        {
                            "uuid": "truncated",
                            "kind": "scope",
                            "category": "llm",
                            "scope_category": "end",
                            "category_profile": {"annotated_response": {"finish_reason": "length"}},
                        }
                    )
                )
            with contextlib.redirect_stdout(io.StringIO()):
                rows = summarize(output)
            self.assertEqual(rows[0]["tokens"]["answer"], 7)
            self.assertEqual(rows[0]["tokens_by_model"]["answer"], {"large": {"input": 5, "output": 2, "total": 7}})
            self.assertEqual(rows[0]["truncated_calls"], 1)
            self.assertIsNone(rows[0]["tokens"]["routing"])
            self.assertEqual(rows[0]["served_models"], {"actual-large": 1})
            self.assertEqual(rows[0]["judge_errors"], 0)
            self.assertEqual(rows[0]["fallbacks"], 0)
            self.assertEqual(rows[0]["classifier_recoveries"], 1)
            self.assertEqual(rows[0]["recovery_reasons"], {"invalid_verdict": 1})
            self.assertEqual(rows[1]["accepted"], 0)
            self.assertIsNone(rows[1]["tokens"]["answer"])
            # Claude's Messages response names the same limit stop_reason.
            with (logs / "events.jsonl").open("a") as stream:
                stream.write(
                    "\n"
                    + json.dumps(
                        {
                            "uuid": "claude-truncated",
                            "kind": "scope",
                            "category": "llm",
                            "scope_category": "end",
                            "category_profile": {"annotated_response": {"stop_reason": "max_tokens"}},
                        }
                    )
                )
            with contextlib.redirect_stdout(io.StringIO()):
                rows = summarize(output)
            self.assertEqual(rows[0]["truncated_calls"], 2)
            # One extra observed call without usage makes the aggregate incomplete.
            with (logs / "events.jsonl").open("a") as stream:
                stream.write("\n" + json.dumps({**call, "uuid": "second-call"}))
            with contextlib.redirect_stdout(io.StringIO()):
                rows = summarize(output)
            self.assertIsNone(rows[0]["tokens"]["answer"])
            self.assertEqual(
                rows[0]["tokens_by_model"]["answer"], {"large": {"input": None, "output": None, "total": None}}
            )

    def test_usage_is_separate_for_each_role_model_and_token_type(self):
        calls = [
            {"call_role": role, "selected_model": model}
            for role, model in [("answer", "small"), ("answer", "small"), ("answer", "large"), ("routing", "small")]
        ]
        observations = [
            ("answer", "small", "input", 5),
            ("answer", "small", "input", 8),
            ("answer", "small", "output", 2),
            ("answer", "small", "output", 3),
            ("answer", "small", "total", 7),
            ("answer", "small", "total", 11),
            ("answer", "large", "input", 20),
            ("answer", "large", "output", 4),
            ("answer", "large", "total", 24),
            ("routing", "small", "input", 9),
            ("routing", "small", "output", 1),
            ("routing", "small", "total", 10),
            ("routing", "small", "reasoning", 0),
        ]
        row = self.summarize_usage(calls, observations)
        self.assertEqual(
            row["tokens_by_model"],
            {
                "answer": {
                    "small": {"input": 13, "output": 5, "total": 18},
                    "large": {"input": 20, "output": 4, "total": 24},
                },
                "routing": {"small": {"input": 9, "output": 1, "total": 10, "reasoning": 0}},
            },
        )
        self.assertEqual(row["tokens"], {"answer": 42, "routing": 10})

    def test_missing_or_invalid_usage_affects_only_its_group(self):
        calls = [{"call_role": "answer", "selected_model": model} for model in ("small", "small", "large")]
        for invalid in (None, True, -1, 1.5, "5"):
            with self.subTest(invalid=invalid):
                row = self.summarize_usage(
                    calls,
                    [
                        ("answer", "small", "input", 5),
                        ("answer", "small", "input", invalid),
                        ("answer", "small", "output", 2),
                        ("answer", "small", "output", 3),
                        ("answer", "small", "total", 7),
                        ("answer", "large", "input", 20),
                        ("answer", "large", "output", 4),
                        ("answer", "large", "total", 24),
                    ],
                )
                self.assertEqual(row["tokens_by_model"]["answer"]["small"], {"input": None, "output": 5, "total": None})
                self.assertEqual(row["tokens_by_model"]["answer"]["large"], {"input": 20, "output": 4, "total": 24})
                self.assertIsNone(row["tokens"]["answer"])

    def test_usage_counts_cannot_compensate_across_models(self):
        calls = [{"call_role": "answer", "selected_model": model} for model in ("small", "small", "large")]
        row = self.summarize_usage(
            calls,
            [("answer", "small", "total", 7), ("answer", "large", "total", 24), ("answer", "large", "total", 24)],
        )
        self.assertIsNone(row["tokens_by_model"]["answer"]["small"]["total"])
        self.assertIsNone(row["tokens_by_model"]["answer"]["large"]["total"])
        self.assertIsNone(row["tokens"]["answer"])

    def test_missing_usage_stays_unknown_for_observed_targets(self):
        calls = [{"call_role": "answer", "selected_model": "small"}]
        row = self.summarize_usage(calls, [])
        self.assertEqual(row["tokens_by_model"]["answer"], {"small": {"input": None, "output": None, "total": None}})
        self.assertEqual(row["tokens_by_model"]["routing"], {})

    def summarize_usage(self, calls, observations):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "routes.toml").write_text('[targets.efficient]\nid="small"\n[targets.capable]\nid="large"\n')
            (output / "results.json").write_text("[]")
            events = [mark("switchyard.routing.llm_call", call, f"call-{i}") for i, call in enumerate(calls)]
            metric = mark(
                "metric",
                {
                    "measurements": [
                        {
                            "name": "switchyard.routing.llm_tokens",
                            "value": value,
                            "attributes": {"call_role": role, "target_model": model, "token_type": kind},
                        }
                        for role, model, kind, value in observations
                    ]
                },
                "metric",
            )
            metric["data_schema"] = {"name": "nemo.relay.metric_measurements", "version": "1.0"}
            # Duplicate delivery must not inflate any of the per-model groups.
            events.extend([metric, metric])
            logs = output / "routed/logs"
            logs.mkdir(parents=True)
            (logs / "events.jsonl").write_text("\n".join(map(json.dumps, events)))
            with contextlib.redirect_stdout(io.StringIO()):
                summarize(output)
            return json.loads((output / "summary.json").read_text())[2]


if __name__ == "__main__":
    unittest.main()
