# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise Hermes #77915 and Switchyard #270 against a fake provider."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
import tomllib
from pathlib import Path
from typing import Any


async def exercise(model: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from agent.relay_runtime import RelayRuntime

    import nemo_relay

    host = RelayRuntime(profile_key="phase2-offline")
    downstream_called = False

    async def forbidden_downstream(_request: Any) -> dict[str, Any]:
        nonlocal downstream_called
        downstream_called = True
        raise AssertionError("Switchyard managed request reached Relay downstream callback")

    responses: list[dict[str, Any]] = []
    try:
        cases = (
            ("phase2-offline-weak-session", "reply with the smoke marker"),
            ("phase2-offline-strong-session", "force strong route and reply with the smoke marker"),
        )
        for session_id, prompt in cases:
            session = host.ensure_session({"session_id": session_id})
            if session is None:
                raise RuntimeError("Hermes Relay runtime did not open a session")
            request = nemo_relay.LLMRequest(
                {},
                {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
            )
            response = await host.run_in_session_async(
                session,
                nemo_relay.llm.execute,
                "openai.chat_completions",
                request,
                forbidden_downstream,
                model_name=model,
                response_codec=nemo_relay.codecs.OpenAIChatCodec(),
            )
            responses.append(response)
            host.close_session({"session_id": session_id})
        active_report = nemo_relay.plugin.report()
        if active_report is None:
            raise AssertionError("Relay did not expose an active plugin report")
        report = active_report.to_dict() if hasattr(active_report, "to_dict") else active_report
    finally:
        host.shutdown()
    if downstream_called:
        raise AssertionError("Relay downstream callback was invoked")
    contents = [response["choices"][0]["message"]["content"] for response in responses]
    if contents != ["OFFLINE_SWITCHYARD_OK", "OFFLINE_SWITCHYARD_OK"]:
        raise AssertionError(f"unexpected fake-provider responses: {contents!r}")
    return responses, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugins", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--request-log", type=Path, required=True)
    parser.add_argument("--model", default="ollama-route-stub")
    args = parser.parse_args()
    config = tomllib.loads(args.plugins.read_text(encoding="utf-8"))
    plugin = config["plugins"]["dynamic"][0]["config"]
    algorithm = plugin["algorithm"]
    targets = plugin["targets"]
    classifier_model = targets[algorithm["classifier_target"]]["model"]
    weak_model = targets[algorithm["weak_target"]]["model"]
    strong_model = targets[algorithm["strong_target"]]["model"]
    artifacts = args.artifacts.resolve()
    artifacts.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.environ["HERMES_NEMO_RELAY_PLUGINS_TOML"] = str(args.plugins.resolve())
    responses, report = asyncio.run(exercise(args.model))

    atof = artifacts / "relay" / "trajectory.atof.jsonl"
    atif = sorted((artifacts / "relay" / "atif").glob("trajectory-*.atif.json"))
    if not atof.is_file() or atof.stat().st_size == 0:
        raise AssertionError("ATOF file sink did not emit")
    if not atif:
        raise AssertionError("ATIF file sink did not emit")
    marks = []
    for line in atof.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        name = event.get("name")
        if isinstance(name, str) and name.startswith("switchyard.routing."):
            marks.append(name)
    if not marks:
        raise AssertionError("Switchyard routing marks were not emitted")
    requests = [json.loads(line) for line in args.request_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(requests) != 4 or not all(item.get("authorization_present") for item in requests):
        raise AssertionError("fake provider did not receive exactly four authenticated requests")
    request_kinds = [item.get("request_kind") for item in requests]
    if request_kinds != ["classifier", "completion", "classifier", "completion"]:
        raise AssertionError(f"unexpected provider request sequence: {request_kinds}")
    request_models = [item.get("model") for item in requests]
    expected_models = [
        classifier_model,
        weak_model,
        classifier_model,
        strong_model,
    ]
    if request_models != expected_models:
        raise AssertionError(f"unexpected provider model sequence: {request_models}")
    if args.model in request_models:
        raise AssertionError("Hermes caller stub reached the provider")
    surviving = [
        thread.name for thread in threading.enumerate() if thread.name.startswith("hermes-nemo-relay-shutdown-")
    ]
    if surviving:
        raise AssertionError(f"Hermes shutdown threads survived: {surviving}")

    result = {
        "schema_version": "harbor-hermes-switchyard.offline-smoke.v1",
        "status": "passed",
        "responses": [response["choices"][0]["message"]["content"] for response in responses],
        "provider_requests": len(requests),
        "provider_request_kinds": request_kinds,
        "provider_models": request_models,
        "hermes_caller_model": args.model,
        "relay_downstream_callback_called": False,
        "switchyard_routing_marks": marks,
        "active_plugin_report_before_shutdown": report,
        "atof": str(atof),
        "atif": [str(path) for path in atif],
        "surviving_shutdown_threads": surviving,
    }
    output = artifacts / "offline-smoke.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
