# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loopback provider and OTLP servers used by the benchmark."""

from __future__ import annotations

import contextlib
import http.client
import json
import socket
import threading
import uuid
from collections.abc import Iterable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar
from urllib.parse import urlparse

from .protocol import anthropic_events, openai_events, response_document


class QuietThreadingServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class ProviderHandler(BaseHTTPRequestHandler):
    """Serve deterministic OpenAI Responses and Anthropic Messages payloads."""

    protocol_version = "HTTP/1.1"

    def setup(self) -> None:
        super().setup()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        del format, args

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length", "0"))
        request = json.loads(self.rfile.read(length) or b"{}")
        path = urlparse(self.path).path
        text = getattr(self.server, "response_fill") * getattr(self.server, "response_bytes")
        chunks = getattr(self.server, "stream_chunks")
        if path.endswith("/responses"):
            self._openai(request, text, chunks)
        elif path.endswith("/messages"):
            self._anthropic(request, text, chunks)
        else:
            self.send_error(404)

    def _openai(self, request: dict[str, Any], text: str, chunks: int) -> None:
        model = request.get("model", getattr(self.server, "openai_model"))
        if request.get("stream", False):
            frames = [
                f"data: {json.dumps(event, separators=(',', ':'))}\n\n".encode()
                for event in openai_events(model, text, chunks)
            ]
            frames.append(b"data: [DONE]\n\n")
            self._send_stream(frames)
            return
        self._send_json(response_document(model, text))

    def _anthropic(self, request: dict[str, Any], text: str, chunks: int) -> None:
        model = request.get("model", getattr(self.server, "anthropic_model"))
        if request.get("stream", False):
            frames = []
            for event in anthropic_events(model, text, chunks):
                name = event["type"]
                frames.append(f"event: {name}\ndata: {json.dumps(event, separators=(',', ':'))}\n\n".encode())
            self._send_stream(frames)
            return
        self._send_json(
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
                "model": model,
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )

    def _send_json(self, value: dict[str, Any]) -> None:
        body = json.dumps(value, separators=(",", ":")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        # SONAR_IGNORE_START: the local benchmark server intentionally echoes JSON fixtures.
        self.wfile.write(body)
        # SONAR_IGNORE_END

    def _send_stream(self, frames: Iterable[bytes]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for frame in frames:
            # SONAR_IGNORE_START: the local benchmark server intentionally emits JSON fixture frames.
            self.wfile.write(f"{len(frame):X}\r\n".encode())
            self.wfile.write(frame)
            # SONAR_IGNORE_END
            self.wfile.write(b"\r\n")
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


class OtlpHandler(BaseHTTPRequestHandler):
    """Accept OTLP requests and track whether the exporter delivered data."""

    protocol_version = "HTTP/1.1"
    request_count: ClassVar[int] = 0
    request_count_lock: ClassVar[threading.Lock] = threading.Lock()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        del format, args

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length", "0"))
        self.rfile.read(length)
        with self.request_count_lock:
            type(self).request_count += 1
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    @classmethod
    def reset(cls) -> None:
        with cls.request_count_lock:
            cls.request_count = 0


@contextlib.contextmanager
def local_server(handler: type[BaseHTTPRequestHandler], **attributes: Any) -> Iterator[str]:
    server = QuietThreadingServer(("127.0.0.1", 0), handler)
    for name, value in attributes.items():
        setattr(server, name, value)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def connection_for(url: str) -> http.client.HTTPConnection:
    parsed = urlparse(url)
    if parsed.hostname is None:
        raise ValueError(f"URL does not contain a host: {url}")
    return http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=30)
