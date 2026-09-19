# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accept OTLP/HTTP protobuf requests for the offline compatibility smoke."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


class Handler(BaseHTTPRequestHandler):
    request_log: Path

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/healthz"}:
            self.send_response(200)
            self.end_headers()
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/traces":
            self.send_error(404)
            return
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length)
        with self.request_log.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "path": self.path,
                        "content_type": self.headers.get("content-type"),
                        "bytes": len(body),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
        response = b"{}"
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=4318)
    parser.add_argument("--request-log", type=Path, required=True)
    args = parser.parse_args()
    args.request_log.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    Handler.request_log = args.request_log
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
