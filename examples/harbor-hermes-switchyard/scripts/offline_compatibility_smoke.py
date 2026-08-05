# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise Hermes #77915 and Switchyard #270 against a fake provider."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
from pathlib import Path
from typing import Any


async def exercise(model: str, session_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    from agent.relay_runtime import RelayRuntime

    import nemo_relay

    host = RelayRuntime(profile_key="phase1-offline")
    session = host.ensure_session({"session_id": session_id})
    if session is None:
        raise RuntimeError("Hermes Relay runtime did not open a session")
    downstream_called = False

    async def forbidden_downstream(_request: Any) -> dict[str, Any]:
        nonlocal downstream_called
        downstream_called = True
        raise AssertionError("Switchyard managed request reached Relay downstream callback")

    request = nemo_relay.LLMRequest(
        {},
        {
            "model": model,
            "messages": [{"role": "user", "content": "reply with the smoke marker"}],
            "stream": False,
        },
    )
    try:
        response = await host.run_in_session_async(
            session,
            nemo_relay.llm.execute,
            "openai.chat_completions",
            request,
            forbidden_downstream,
            model_name=model,
            response_codec=nemo_relay.codecs.OpenAIChatCodec(),
        )
        active_report = nemo_relay.plugin.report()
        if active_report is None:
            raise AssertionError("Relay did not expose an active plugin report")
        report = active_report.to_dict() if hasattr(active_report, "to_dict") else active_report
        host.close_session({"session_id": session_id})
    finally:
        host.shutdown()
    if downstream_called:
        raise AssertionError("Relay downstream callback was invoked")
    content = response["choices"][0]["message"]["content"]
    if content != "OFFLINE_SWITCHYARD_OK":
        raise AssertionError(f"unexpected fake-provider response: {content!r}")
    return response, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugins", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--request-log", type=Path, required=True)
    parser.add_argument("--model", default="phase1/fake-model")
    args = parser.parse_args()
    artifacts = args.artifacts.resolve()
    artifacts.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.environ["HERMES_NEMO_RELAY_PLUGINS_TOML"] = str(args.plugins.resolve())
    session_id = "phase1-offline-session"
    response, report = asyncio.run(exercise(args.model, session_id))

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
    if len(requests) != 1 or not requests[0].get("authorization_present"):
        raise AssertionError("fake provider did not receive exactly one authenticated request")
    surviving = [
        thread.name for thread in threading.enumerate() if thread.name.startswith("hermes-nemo-relay-shutdown-")
    ]
    if surviving:
        raise AssertionError(f"Hermes shutdown threads survived: {surviving}")

    result = {
        "schema_version": "harbor-hermes-switchyard.offline-smoke.v1",
        "status": "passed",
        "response": response["choices"][0]["message"]["content"],
        "provider_requests": len(requests),
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
