# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local OpenAI Responses API fixture for the opt-in Codex plugin E2E test."""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


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
        self.server.log_request_record(
            {
                "method": "GET",
                "path": self.path,
                "authorization": self.headers.get("authorization"),
            }
        )
        if not self.path.endswith("/models"):
            self.send_error(404)
            return
        body = json.dumps(
            {
                "object": "list",
                "data": [{"id": "gpt-5-codex", "object": "model", "owned_by": "openai"}],
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
        events = response_events(request) if self.path.endswith("/responses") else None
        self.server.log_request_record(
            {
                "method": "POST",
                "path": self.path,
                "authorization": self.headers.get("authorization"),
                "model": request.get("model"),
                "response_id": events[-1]["response"]["id"] if events else None,
            }
        )
        if events is None:
            self.send_error(404)
            return
        self.server.wait_at_barrier_if_enabled()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for event in events:
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
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
