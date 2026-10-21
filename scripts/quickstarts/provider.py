# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic provider protocols. Only the actual agent may execute a tool or emit ATOF."""

from __future__ import annotations

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

TOOL_MARKER = "RELAY_QUICKSTART_TOOL_OK"
RESPONSE_MARKER = "RELAY_QUICKSTART_RESPONSE_OK"
SPEC = importlib.util.spec_from_file_location(
    "relay_protocol_fixture", Path(__file__).resolve().parents[1] / "test-support/codex_mock_provider.py"
)
assert SPEC and SPEC.loader
PROTOCOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROTOCOL)


def tool_result(request: dict[str, Any]) -> bool:
    """Inspect protocol result positions; the prompt itself always contains the marker."""
    for message in request.get("messages", []) + (
        request.get("input", []) if isinstance(request.get("input"), list) else []
    ):
        if (message.get("role") == "tool" or message.get("type") == "function_call_output") and (
            message.get("tool_call_id", message.get("call_id")) == "call_quickstart"
        ):
            content = message.get("content", message.get("output", ""))
            try:
                parsed = json.loads(content) if isinstance(content, str) else content
            except ValueError:
                parsed = content
            if isinstance(parsed, dict):
                if not parsed.get("error") and not parsed.get("is_error"):
                    return True
            elif isinstance(parsed, str) and not parsed.lower().startswith("error"):
                return True
        for part in message.get("content", []) if isinstance(message.get("content"), list) else []:
            if (
                part.get("type") == "tool_result"
                and part.get("tool_use_id") == "call_quickstart"
                and not part.get("is_error", False)
            ):
                return True
    return False


def choose_tool(request: dict[str, Any], project: Path) -> tuple[str, dict[str, Any]]:
    tools = {tool.get("name", tool.get("function", {}).get("name")): tool for tool in request.get("tools", [])}
    for name, arguments in (
        ("emit_marker", {}),
        ("Write", {"file_path": str(project / "relay-check.txt"), "content": TOOL_MARKER}),
        ("write_file", {"path": "relay-check.txt", "content": TOOL_MARKER}),
        ("write", {"path": "relay-check.txt", "content": TOOL_MARKER}),
        ("exec_command", {"cmd": "printf '%s' " + TOOL_MARKER + " > relay-check.txt"}),
        ("shell_command", {"command": "printf '%s' " + TOOL_MARKER + " > relay-check.txt"}),
    ):
        if name in tools:
            return name, arguments
    raise ValueError("Provider request did not advertise a supported Quick Start tool")


class Fixture:
    def __init__(self, project: Path, log: Path):
        self.project, self.log = project, log
        self.requests = 0
        self.tool_calls = 0
        self.completions = 0
        self.diagnostics = 0
        self.errors: list[str] = []
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def send_json(self, value: Any, status: int = 200) -> None:
                payload = json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:
                self.send_json(
                    {
                        "object": "list",
                        "data": [
                            {"id": name, "object": "model", "owned_by": "fixture"}
                            for name in (
                                "relay-quickstart",
                                "nvidia/nemotron-3.5-lightning-30b-a3b",
                                "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
                            )
                        ],
                    }
                )

            def do_POST(self) -> None:
                try:
                    request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    fixture.requests += 1
                    if fixture.requests > 16:
                        raise ValueError("Fixture request limit exceeded; agent did not complete the tool cycle")
                    final = tool_result(request)
                    diagnostic = not request.get("tools") and not final
                    text_only = final or diagnostic
                    reply = RESPONSE_MARKER if final else "Quick Start provider is ready."
                    name, arguments = ("", {}) if text_only else choose_tool(request, project)
                    with log.open("a") as stream:
                        # Do not persist request headers, credentials, prompts or unrelated host context.
                        stream.write(
                            json.dumps(
                                {
                                    "path": self.path,
                                    "model": request.get("model"),
                                    "tool": name,
                                    "tool_result": final,
                                    "diagnostic": diagnostic,
                                }
                            )
                            + "\n"
                        )
                    if final:
                        fixture.completions += 1
                    elif diagnostic:
                        # Host health checks and title generation cannot satisfy the
                        # required tool cycle or receive its completion marker.
                        fixture.diagnostics += 1
                    else:
                        fixture.tool_calls += 1
                    if diagnostic and self.path == "/api/show":
                        # Hermes probes loopback providers for Ollama capabilities.
                        # A negative capability response leaves its real OpenAI path
                        # intact; this optional probe is not a failed model call.
                        self.send_json({"error": "This is not an Ollama endpoint"}, 404)
                        return
                    if self.path.endswith("/responses"):
                        events = (
                            PROTOCOL.response_events(request) if text_only else PROTOCOL.response_tool_events(request)
                        )
                        # Preserve the existing tested Responses envelopes, replacing only scenario payloads.
                        encoded = json.dumps(events).replace("pong", reply)
                        events = json.loads(encoded)
                        if not text_only:
                            for event in events:
                                if event["type"] == "response.function_call_arguments.delta":
                                    event["delta"] = json.dumps(arguments)
                                if event["type"] == "response.function_call_arguments.done":
                                    event["arguments"] = json.dumps(arguments)
                                items = event.get("response", {}).get("output", []) + (
                                    [event["item"]] if "item" in event else []
                                )
                                for item in items:
                                    if item.get("type") == "function_call":
                                        item.update(
                                            name=name,
                                            call_id="call_quickstart",
                                            arguments="" if event["type"].endswith("added") else json.dumps(arguments),
                                        )
                        if not request.get("stream"):
                            self.send_json(events[-1]["response"])
                            return
                        frames = [(event["type"], event) for event in events]
                    elif self.path.split("?")[0].endswith("/messages"):
                        block = (
                            {"type": "text", "text": reply}
                            if text_only
                            else {"type": "tool_use", "id": "call_quickstart", "name": name, "input": arguments}
                        )
                        message = {
                            "id": "msg_quickstart",
                            "type": "message",
                            "role": "assistant",
                            "model": request.get("model"),
                            "content": [block],
                            "stop_reason": "end_turn" if text_only else "tool_use",
                            "stop_sequence": None,
                            "usage": {"input_tokens": 10, "output_tokens": 10},
                        }
                        if not request.get("stream"):
                            self.send_json(message)
                            return
                        frames = [
                            (
                                "message_start",
                                {"type": "message_start", "message": {**message, "content": [], "stop_reason": None}},
                            ),
                            (
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": 0,
                                    "content_block": {"type": "text", "text": ""}
                                    if text_only
                                    else {**block, "input": {}},
                                },
                            ),
                            (
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": 0,
                                    "delta": {"type": "text_delta", "text": reply}
                                    if text_only
                                    else {"type": "input_json_delta", "partial_json": json.dumps(arguments)},
                                },
                            ),
                            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                            (
                                "message_delta",
                                {
                                    "type": "message_delta",
                                    "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
                                    "usage": {"output_tokens": 10},
                                },
                            ),
                            ("message_stop", {"type": "message_stop"}),
                        ]
                    elif self.path.endswith("/chat/completions"):
                        call = {
                            "id": "call_quickstart",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                        message: dict[str, Any] = {"role": "assistant", "content": reply if text_only else None}
                        if not text_only:
                            message["tool_calls"] = [call]
                        response = {
                            "id": "chatcmpl_quickstart",
                            "object": "chat.completion",
                            "created": 1,
                            "model": request.get("model"),
                            "choices": [
                                {"index": 0, "message": message, "finish_reason": "stop" if text_only else "tool_calls"}
                            ],
                            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                        }
                        if not request.get("stream"):
                            self.send_json(response)
                            return
                        delta = {**message}
                        if not text_only:
                            delta["tool_calls"] = [{"index": 0, **call}]
                        frames = [
                            (
                                "",
                                {
                                    **response,
                                    "object": "chat.completion.chunk",
                                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                                },
                            ),
                            (
                                "",
                                {
                                    **response,
                                    "object": "chat.completion.chunk",
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {},
                                            "finish_reason": response["choices"][0]["finish_reason"],
                                        }
                                    ],
                                },
                            ),
                        ]
                    else:
                        raise ValueError(f"Unexpected provider route: {self.path}")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    for event_type, event in frames:
                        if event_type:
                            self.wfile.write(f"event: {event_type}\n".encode())
                        self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    self.close_connection = True
                except (ValueError, KeyError) as error:
                    fixture.errors.append(str(error))
                    self.send_json({"error": {"message": str(error)}}, 400)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def verify(self) -> None:
        if self.errors or not self.tool_calls or not self.completions:
            raise ValueError(f"Incomplete provider tool cycle: {self.requests} requests; {self.errors}")
