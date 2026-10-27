# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for unsafe endpoints and misleading local success."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from run import check_events, langsmith_headers, validate_endpoint, verify_phoenix


class CaptureChecks(unittest.TestCase):
    def test_delivery_waits_for_all_local_scopes_and_exported_parents(self):
        events = [
            {"kind": "scope", "scope_category": "end", "category": category, "uuid": category}
            for category in ("agent", "llm", "tool")
        ]
        spans = [
            {
                "context": {"span_id": category},
                "parent_id": None if category == "agent" else "agent",
                "attributes": {"nemo_relay.uuid": category},
            }
            for category in ("agent", "llm", "tool")
        ]
        responses = [
            io.BytesIO(json.dumps({"data": spans[1:]}).encode()),
            io.BytesIO(json.dumps({"data": spans}).encode()),
        ]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("run.urllib.request.urlopen", side_effect=responses) as get,
            patch("run.time.sleep"),
        ):
            verify_phoenix(events, Path(directory))
            self.assertEqual(get.call_count, 2)
            self.assertIn("trace_id=agent", get.call_args.args[0].full_url)
            self.assertEqual(json.loads((Path(directory) / "delivery.json").read_text())["matched_local_scopes"], 3)

    def test_langsmith_addressing_rejects_partial_or_mixed_modes(self):
        for environment in (
            {"LANGSMITH_AGENT_ID": "support-triage"},
            {"LANGSMITH_AGENT_ENVIRONMENT": "development"},
            {
                "LANGSMITH_AGENT_ID": "support-triage",
                "LANGSMITH_AGENT_ENVIRONMENT": "development",
                "LANGSMITH_PROJECT": "legacy",
            },
            {"LANGSMITH_AGENT_ID": "Invalid_ID", "LANGSMITH_AGENT_ENVIRONMENT": "development"},
            {"LANGSMITH_AGENT_ID": "support-triage", "LANGSMITH_AGENT_ENVIRONMENT": "unknown"},
        ):
            with self.subTest(environment=environment), self.assertRaises(ValueError):
                langsmith_headers(environment)
        self.assertEqual(
            langsmith_headers({"LANGSMITH_AGENT_ID": "support-triage", "LANGSMITH_AGENT_ENVIRONMENT": "development"}),
            {"Langsmith-Agent-Id": "support-triage", "Langsmith-Agent-Environment": "development"},
        )

    def test_endpoint_policy(self):
        for endpoint in ("http://localhost:6006/v1/traces", "https://collector.example/v1/traces"):
            validate_endpoint(endpoint)
        for endpoint in (
            "http://localhost.evil.example:6006/v1/traces",
            "https://key:secret@example/v1/traces",
            "https://example/v1/traces?key=secret",
            "file:///tmp/traces",
        ):
            with self.assertRaises(ValueError):
                validate_endpoint(endpoint)
        validate_endpoint("http://otel-collector:4318/v1/traces", allow_insecure=True)

    def test_unrelated_or_unclosed_scopes_do_not_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            events = []
            for category in ("agent", "llm", "tool"):
                for phase in ("start", "end"):
                    events.append(
                        {
                            "kind": "scope",
                            "uuid": category,
                            "category": category,
                            "scope_category": phase,
                            "parent_uuid": None,
                        }
                    )
            path.write_text("\n".join(map(json.dumps, events)))
            with self.assertRaises(ValueError):
                check_events(path)
            for event in events:
                if event["category"] != "agent":
                    event["parent_uuid"] = "agent"
            path.write_text("\n".join(map(json.dumps, events)))
            self.assertEqual(len(check_events(path)), 6)
            path.write_text("\n".join(map(json.dumps, [e for e in events if e["scope_category"] == "start"])))
            with self.assertRaises(ValueError):
                check_events(path)


if __name__ == "__main__":
    unittest.main()
