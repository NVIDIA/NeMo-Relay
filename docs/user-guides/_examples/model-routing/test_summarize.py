# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for duplicated events and missing token telemetry."""

import contextlib
import io
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

from run import run_task
from summarize import summarize


def mark(name, data, identity):
    return {"kind": "mark", "name": name, "data": data, "uuid": identity}


class SummaryChecks(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "The companion runs on Linux/macOS")
    def test_timeout_stops_child_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "child-survived"
            child = "import pathlib,time; time.sleep(1); pathlib.Path('child-survived').write_text('alive')"
            parent = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(60)"
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
            self.assertEqual(rows[0]["truncated_calls"], 1)
            self.assertIsNone(rows[0]["tokens"]["routing"])
            self.assertEqual(rows[0]["served_models"], {"actual-large": 1})
            self.assertEqual(rows[0]["judge_errors"], 0)
            self.assertEqual(rows[0]["fallbacks"], 0)
            self.assertEqual(rows[0]["classifier_recoveries"], 1)
            self.assertEqual(rows[0]["recovery_reasons"], {"invalid_verdict": 1})
            self.assertEqual(rows[1]["accepted"], 0)
            self.assertIsNone(rows[1]["tokens"]["answer"])
            # One extra observed call without usage makes the aggregate incomplete.
            with (logs / "events.jsonl").open("a") as stream:
                stream.write("\n" + json.dumps({**call, "uuid": "second-call"}))
            with contextlib.redirect_stdout(io.StringIO()):
                rows = summarize(output)
            self.assertIsNone(rows[0]["tokens"]["answer"])


if __name__ == "__main__":
    unittest.main()
