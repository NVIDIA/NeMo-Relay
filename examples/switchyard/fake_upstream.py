# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic local OpenAI-compatible upstream for Switchyard plugin smoke tests."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CLASSIFIER_VERDICT = json.dumps(
    {
        "recommended_route": "efficient",
        "p_solve": 0.9,
        "confidence": 0.95,
        "abstain": False,
        "capability_boundary": "supported",
        "primary_rule": "SUP-1",
        "crux": "bounded task",
    },
    separators=(",", ":"),
)


class Handler(BaseHTTPRequestHandler):
    log_path: Path

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        with self.log_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps({"path": self.path, "body": body}) + "\n")
        model = body.get("model", "fake-model")
        content = CLASSIFIER_VERDICT if model == "provider/classifier" else "fake"
        if body.get("stream"):
            chunks = [
                {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion.chunk",
                    "model": model,
                    "system_fingerprint": "fp_switchyard_example",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": content},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion.chunk",
                    "model": model,
                    "system_fingerprint": "fp_switchyard_example",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            ]
            payload = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
            payload += "data: [DONE]\n\n"
            encoded = payload.encode()
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            return
        response = {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "model": model,
            "system_fingerprint": "fp_switchyard_example",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        encoded = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=4101)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    Handler.log_path = args.log
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
