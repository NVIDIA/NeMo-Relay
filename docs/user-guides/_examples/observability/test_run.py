# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for unsafe endpoints and misleading local success."""

import io
import json
import tempfile
import tomllib
import unittest
import urllib.response
from email.message import Message
from pathlib import Path
from unittest.mock import patch

from run import check_events, langsmith_headers, main, validate_endpoint, verify_phoenix


class CaptureChecks(unittest.TestCase):
    def test_workload_passes_provider_model_and_http_opt_in(self):
        def run_agent(command, directory, environment, timeout):
            (directory / "handoff.md").write_text("INC-1042: checkout\nINC-1044: identity\nINC-1043: reporting\n")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "triage"
            with (
                patch(
                    "sys.argv",
                    [
                        "run.py",
                        "--destination",
                        "phoenix",
                        "--output",
                        str(output),
                        "--provider",
                        "nvidia",
                        "--model",
                        "selected-model",
                        "--allow-insecure-endpoint",
                        "--verify",
                    ],
                ),
                patch.dict("os.environ", {"RELAY_OTLP_ENDPOINT": "http://otel-collector:6006/v1/traces"}, clear=True),
                patch("run.shutil.which", return_value="/example/bin/hermes"),
                patch("run.run_hermes", side_effect=run_agent) as agent,
                patch("run.check_events", return_value=[]),
                patch("run.verify_phoenix") as verify,
            ):
                main()
            command, _, environment, _ = agent.call_args.args
            self.assertEqual(command[:3], ["hermes", "chat", "--oneshot"])
            self.assertEqual(command[3:7], ["--provider", "nvidia", "--model", "selected-model"])
            self.assertEqual(environment["HERMES_NEMO_RELAY_PLUGINS_TOML"], str((output / "plugins.toml").resolve()))
            verify.assert_called_once_with([], output.resolve(), allow_insecure=True)

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
            patch.dict("os.environ", {}, clear=True),
            patch("run.urllib.request.OpenerDirector.open", side_effect=responses) as get,
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

    def test_model_and_tool_need_a_shared_agent_ancestor(self):
        cases = [
            ({"a": None, "b": None, "llm": "a", "tool": "b"}, False),
            ({"a": None, "b": "a", "llm": "b", "tool": "a"}, True),
            ({"a": None, "b": None, "llm": "tool", "tool": "llm"}, False),
        ]
        for parents, accepted in cases:
            with self.subTest(parents=parents), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "events.jsonl"
                events = [
                    {
                        "kind": "scope",
                        "uuid": uid,
                        "category": uid if uid in {"llm", "tool"} else "agent",
                        "parent_uuid": parent,
                        "scope_category": phase,
                    }
                    for uid, parent in parents.items()
                    for phase in ("start", "end")
                ]
                path.write_text("\n".join(map(json.dumps, events)))
                if accepted:
                    self.assertEqual(check_events(path), events)
                else:
                    with self.assertRaises(ValueError):
                        check_events(path)

    def test_generated_config_allows_only_the_selected_private_http_host(self):
        cases = [
            ("https://collector.example/v1/traces", False, []),
            ("http://127.0.0.1:6006/v1/traces", False, []),
            ("http://otel-collector:4318/v1/traces", True, ["otel-collector"]),
            ("http://10.0.0.5:4318/v1/traces", True, ["10.0.0.5"]),
        ]
        for endpoint, allow_insecure, expected in cases:
            with self.subTest(endpoint=endpoint), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "config"
                command = ["run.py", "--destination", "datadog", "--prepare-only", "--output", str(output)]
                if allow_insecure:
                    command.append("--allow-insecure-endpoint")
                with (
                    patch("sys.argv", command),
                    patch.dict("os.environ", {"RELAY_OTLP_ENDPOINT": endpoint}, clear=True),
                    patch("sys.stdout", new_callable=io.StringIO),
                ):
                    main()
                config = tomllib.loads((output / "plugins.toml").read_text())["components"][0]["config"]
                self.assertEqual(config.get("insecure_http_hosts", []), expected)

    def test_phoenix_redirects_never_forward_authorization(self):
        events = [{"kind": "scope", "scope_category": "end", "category": "agent", "uuid": "agent"}]
        # Intercept transport, leaving urllib's real redirect handling in place.
        # Even a regression cannot make an external request with this fixture.
        for target in ("https://other.example/spans", "http://phoenix.example/spans", "https://phoenix.example/new"):
            for code in (301, 302, 303, 307, 308):
                with self.subTest(target=target, code=code), tempfile.TemporaryDirectory() as directory:
                    headers = Message()
                    headers["Location"] = target
                    response = urllib.response.addinfourl(
                        io.BytesIO(b""), headers, "https://phoenix.example/spans", code
                    )
                    response.msg = "Redirect"
                    with (
                        patch.dict(
                            "os.environ",
                            {"PHOENIX_API_URL": "https://phoenix.example", "PHOENIX_AUTHORIZATION": "Bearer synthetic"},
                            clear=True,
                        ),
                        patch("run.urllib.request.OpenerDirector._open", return_value=response) as send,
                        self.assertRaisesRegex(ValueError, "redirect"),
                    ):
                        verify_phoenix(events, Path(directory))
                    self.assertEqual(send.call_count, 1)

    def test_phoenix_private_http_opt_in_does_not_trust_another_host(self):
        events = [{"kind": "scope", "scope_category": "end", "category": "agent", "uuid": "agent"}]
        spans = [{"context": {"span_id": "agent"}, "attributes": {"nemo_relay.uuid": "agent"}}]
        for api_host in ("otel-collector", "other.example"):
            with self.subTest(api_host=api_host), tempfile.TemporaryDirectory() as directory:
                with (
                    patch.dict(
                        "os.environ",
                        {
                            "RELAY_OTLP_ENDPOINT": "http://otel-collector:6006/v1/traces",
                            "PHOENIX_API_URL": f"http://{api_host}:6006",
                        },
                        clear=True,
                    ),
                    patch(
                        "run.urllib.request.OpenerDirector.open",
                        return_value=io.BytesIO(json.dumps({"data": spans}).encode()),
                    ) as send,
                ):
                    if api_host == "otel-collector":
                        verify_phoenix(events, Path(directory), allow_insecure=True)
                        send.assert_called_once()
                    else:
                        with self.assertRaises(ValueError):
                            verify_phoenix(events, Path(directory), allow_insecure=True)
                        send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
