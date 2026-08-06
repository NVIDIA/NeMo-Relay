# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small OpenAI Chat-compatible provider used by the offline Phase 1 smoke."""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


class Handler(BaseHTTPRequestHandler):
    token: str
    request_log: Path

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        if self.headers.get("authorization") != f"Bearer {self.token}":
            self.send_error(401)
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            request = json.loads(self.rfile.read(length))
        except (ValueError, json.JSONDecodeError):
            self.send_error(400)
            return
        messages = request.get("messages", [])
        serialized_messages = json.dumps(messages, separators=(",", ":"))
        is_classifier = request.get("response_format") is not None or (
            "p_solve" in serialized_messages and "capability_boundary" in serialized_messages
        )
        log_entry = {
            "path": self.path,
            "model": request.get("model"),
            "message_count": len(messages),
            "request_kind": "classifier" if is_classifier else "completion",
            "authorization_present": True,
        }
        with self.request_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(log_entry, separators=(",", ":")) + "\n")
        content = "OFFLINE_SWITCHYARD_OK"
        if is_classifier:
            force_strong = "force strong route" in serialized_messages
            content = json.dumps(
                {
                    "p_solve": 0.01 if force_strong else 0.99,
                    "capability_boundary": "supported",
                    "primary_rule": "SUP-1",
                    "crux": "deterministic offline smoke task",
                },
                separators=(",", ":"),
            )
        response = {
            "id": "chatcmpl-phase1",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.get("model", "phase1-model"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 3,
                "total_tokens": 7,
            },
        }
        body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--token", required=True)
    parser.add_argument("--request-log", type=Path, required=True)
    args = parser.parse_args()
    args.request_log.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    Handler.token = args.token
    Handler.request_log = args.request_log
    ThreadingHTTPServer((args.bind, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
