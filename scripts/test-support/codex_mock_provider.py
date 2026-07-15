# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local OpenAI and Anthropic fixture for coding-agent plugin E2E tests."""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def response_events(request: dict[str, Any]) -> list[dict[str, Any]]:
    response_id = f"resp_{uuid.uuid4().hex}"
    item_id = f"msg_{uuid.uuid4().hex}"
    model = request.get("model", "gpt-5-codex")
    created_at = int(time.time())
    item = {
        "id": item_id,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": "pong",
                "annotations": [],
                "logprobs": [],
            }
        ],
    }
    response = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "completed_at": created_at,
        "status": "completed",
        "background": False,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "max_tool_calls": None,
        "model": model,
        "output": [item],
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "prompt_cache_key": None,
        "reasoning": {"effort": "medium", "summary": None},
        "safety_identifier": None,
        "service_tier": "default",
        "store": False,
        "temperature": None,
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "tool_choice": "auto",
        "tools": [],
        "top_logprobs": 0,
        "top_p": None,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 1,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 1,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 2,
        },
        "user": None,
        "metadata": {},
    }
    in_progress = {**response, "completed_at": None, "status": "in_progress", "output": []}
    return [
        {"type": "response.created", "response": in_progress},
        {
            "type": "response.output_item.added",
            "response_id": response_id,
            "output_index": 0,
            "item": {**item, "status": "in_progress", "content": []},
        },
        {
            "type": "response.content_part.added",
            "response_id": response_id,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []},
        },
        {
            "type": "response.output_text.delta",
            "response_id": response_id,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "delta": "pong",
            "logprobs": [],
        },
        {
            "type": "response.output_text.done",
            "response_id": response_id,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "text": "pong",
            "logprobs": [],
        },
        {
            "type": "response.content_part.done",
            "response_id": response_id,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "part": item["content"][0],
        },
        {
            "type": "response.output_item.done",
            "response_id": response_id,
            "output_index": 0,
            "item": item,
        },
        {"type": "response.completed", "response": response},
    ]


def anthropic_events(request: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    message_id = f"msg_{uuid.uuid4().hex}"
    model = request.get("model", "claude-sonnet-4-5")
    return [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "pong"},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]


def chat_completion_chunks(request: dict[str, Any]) -> list[dict[str, Any]]:
    completion_id = f"chatcmpl_{uuid.uuid4().hex}"
    model = request.get("model", "gpt-4o-mini")
    created = int(time.time())
    base = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
    }
    return [
        {
            **base,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
        },
        {
            **base,
            "choices": [{"index": 0, "delta": {"content": "pong"}, "finish_reason": None}],
        },
        {
            **base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    ]


class Provider(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], log_path: Path, barrier_dir: Path) -> None:
        super().__init__(address, Handler)
        self.log_path = log_path
        self.log_lock = threading.Lock()
        self.barrier_dir = barrier_dir
        self.barrier_lock = threading.Lock()

    def log_request_record(self, record: dict[str, Any]) -> None:
        with self.log_lock, self.log_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")

    def wait_at_barrier_if_enabled(self) -> None:
        if not (self.barrier_dir / "enabled").exists():
            return
        with self.barrier_lock:
            arrivals = self.barrier_dir / "arrivals"
            count = int(arrivals.read_text(encoding="utf-8") or "0") if arrivals.exists() else 0
            temporary = arrivals.with_suffix(".tmp")
            temporary.write_text(str(count + 1), encoding="utf-8")
            temporary.replace(arrivals)
        deadline = time.monotonic() + 30
        release = self.barrier_dir / "release"
        while not release.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("concurrent Codex provider barrier timed out")
            time.sleep(0.02)


class Handler(BaseHTTPRequestHandler):
    server: Provider

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        del format, args

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        self.server.log_request_record(
            {
                "method": "GET",
                "path": self.path,
                "authorization": self.headers.get("authorization"),
                "relay_client_token": self.headers.get("x-nemo-relay-client-token"),
            }
        )
        if not path.endswith("/models"):
            self.send_error(404)
            return
        body = json.dumps(
            {
                "object": "list",
                "data": [
                    {"id": "gpt-5-codex", "object": "model", "owned_by": "openai"},
                    {"id": "gpt-4o-mini", "object": "model", "owned_by": "openai"},
                ],
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length", "0"))
        raw = self.rfile.read(length)
        request = json.loads(raw or b"{}")
        path = urlparse(self.path).path
        response_stream = response_events(request) if path.endswith("/responses") else None
        anthropic_stream = anthropic_events(request) if path.endswith("/messages") else None
        chat_stream = chat_completion_chunks(request) if path.endswith("/chat/completions") else None
        self.server.log_request_record(
            {
                "method": "POST",
                "path": self.path,
                "authorization": self.headers.get("authorization"),
                "x_api_key": self.headers.get("x-api-key"),
                "relay_client_token": self.headers.get("x-nemo-relay-client-token"),
                "model": request.get("model"),
                "response_id": (response_stream[-1]["response"]["id"] if response_stream else None),
            }
        )
        if path.endswith("/messages/count_tokens"):
            body = json.dumps({"input_tokens": 1}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if response_stream is None and anthropic_stream is None and chat_stream is None:
            self.send_error(404)
            return
        self.server.wait_at_barrier_if_enabled()
        if anthropic_stream is not None and not request.get("stream", False):
            body = json.dumps(
                {
                    "id": f"msg_{uuid.uuid4().hex}",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "pong"}],
                    "model": request.get("model", "claude-sonnet-4-5"),
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if chat_stream is not None and not request.get("stream", False):
            body = json.dumps(
                {
                    "id": f"chatcmpl_{uuid.uuid4().hex}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request.get("model", "gpt-4o-mini"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "pong"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        if response_stream is not None:
            for event in response_stream:
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
        elif chat_stream is not None:
            for event in chat_stream:
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            for event_name, event in anthropic_stream or []:
                self.wfile.write(f"event: {event_name}\ndata: {json.dumps(event)}\n\n".encode())
        self.wfile.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--barrier-dir", type=Path, required=True)
    args = parser.parse_args()
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    args.log_file.write_text("", encoding="utf-8")
    args.barrier_dir.mkdir(parents=True, exist_ok=True)
    # This test-only HTTP provider is deliberately restricted to loopback and an ephemeral port;
    # it never accepts remote traffic or production credentials.
    server = Provider(("127.0.0.1", 0), args.log_file, args.barrier_dir)
    temporary = args.ready_file.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"address": f"127.0.0.1:{server.server_port}"}),
        encoding="utf-8",
    )
    temporary.replace(args.ready_file)
    server.serve_forever()


if __name__ == "__main__":
    main()
