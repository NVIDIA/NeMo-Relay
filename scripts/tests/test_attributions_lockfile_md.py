# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import io
import sys
import tarfile
import tempfile
import threading
import unittest
import zipfile
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Self
from unittest import mock

import urllib3
import urllib3.util.retry

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "licensing" / "attributions_lockfile_md.py"
SPEC = importlib.util.spec_from_file_location("attributions_lockfile_md", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ATTRIBUTIONS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ATTRIBUTIONS
SPEC.loader.exec_module(ATTRIBUTIONS)

DATA = b"artifact"
Reply = Callable[[BaseHTTPRequestHandler], None]


def _ok(handler: BaseHTTPRequestHandler) -> None:
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Length", str(len(DATA)))
    handler.end_headers()
    handler.wfile.write(DATA)


def _status(code: HTTPStatus, **headers: str) -> Reply:
    def reply(handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(code)
        handler.send_header("Content-Length", "0")
        for name, value in headers.items():
            handler.send_header(name.replace("_", "-"), value)
        handler.end_headers()

    return reply


def _drop(handler: BaseHTTPRequestHandler) -> None:
    handler.close_connection = True
    handler.connection.close()


class ScriptedServer:
    def __init__(self, replies: list[Reply]) -> None:
        self.replies = replies
        self.requests = 0
        self.lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                with server.lock:
                    index = server.requests
                    server.requests += 1
                server.replies[index](self)

            def log_message(self, format: str, *args: object) -> None:
                del format, args

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)

    def __enter__(self) -> Self:
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/artifact.tar.gz"


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _wheel_bytes(license_name: str) -> bytes:
    return _wheel_from_metadata(f"Name: example\nVersion: 1.0\nLicense: {license_name}\n")


def _wheel_from_metadata(metadata: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("example-1.0.dist-info/METADATA", metadata)
    return buffer.getvalue()


def _sdist_from_pkg_info(pkg_info: str) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tf:
        payload = pkg_info.encode()
        info = tarfile.TarInfo("example-1.0/PKG-INFO")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


class DownloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.expected_hash = _sha256(DATA)
        cache_dir = tempfile.TemporaryDirectory()
        self.addCleanup(cache_dir.cleanup)
        self.cache_dir = Path(cache_dir.name) / "cache"
        self.cached = self.cache_dir / self.expected_hash.removeprefix("sha256:")
        self.enterContext(mock.patch.object(ATTRIBUTIONS, "ARTIFACT_CACHE_DIR", self.cache_dir))
        self.enterContext(mock.patch.object(urllib3.Retry, "sleep"))

    def test_retries_server_errors_then_caches_artifact(self) -> None:
        with ScriptedServer([_status(HTTPStatus.SERVICE_UNAVAILABLE), _status(HTTPStatus.BAD_GATEWAY), _ok]) as server:
            result = ATTRIBUTIONS._download_locked_artifact(server.url, self.expected_hash)
            self.assertEqual(server.requests, 3)
        self.assertEqual(result, DATA)
        self.assertEqual(self.cached.read_bytes(), DATA)

    def test_retries_dropped_connection(self) -> None:
        with ScriptedServer([_drop, _drop, _ok]) as server:
            result = ATTRIBUTIONS._download_locked_artifact(server.url, self.expected_hash)
            self.assertEqual(server.requests, 3)
        self.assertEqual(result, DATA)

    def test_client_error_is_not_retried(self) -> None:
        with ScriptedServer([_status(HTTPStatus.NOT_FOUND)]) as server:
            with self.assertRaises(urllib3.exceptions.HTTPError):
                ATTRIBUTIONS._download_locked_artifact(server.url, self.expected_hash)
            self.assertEqual(server.requests, 1)
        self.assertFalse(self.cache_dir.exists())

    def test_gives_up_after_configured_retries(self) -> None:
        replies = [_status(HTTPStatus.SERVICE_UNAVAILABLE)] * (ATTRIBUTIONS.FETCH_RETRIES + 1)
        with ScriptedServer(replies) as server:
            with self.assertRaises(urllib3.exceptions.MaxRetryError):
                ATTRIBUTIONS._download_locked_artifact(server.url, self.expected_hash)
            self.assertEqual(server.requests, ATTRIBUTIONS.FETCH_RETRIES + 1)

    def test_cache_hit_skips_network(self) -> None:
        self.cache_dir.mkdir(parents=True)
        self.cached.write_bytes(DATA)
        with ScriptedServer([]) as server:
            result = ATTRIBUTIONS._download_locked_artifact(server.url, self.expected_hash)
            self.assertEqual(server.requests, 0)
        self.assertEqual(result, DATA)

    def test_corrupt_cache_entry_is_replaced(self) -> None:
        self.cache_dir.mkdir(parents=True)
        self.cached.write_bytes(b"corrupt")
        with ScriptedServer([_ok]) as server:
            result = ATTRIBUTIONS._download_locked_artifact(server.url, self.expected_hash)
            self.assertEqual(server.requests, 1)
        self.assertEqual(result, DATA)
        self.assertEqual(self.cached.read_bytes(), DATA)

    def test_path_shaped_hash_is_rejected_before_any_file_access(self) -> None:
        victim = Path(self.enterContext(tempfile.TemporaryDirectory())) / "victim.txt"
        victim.write_bytes(b"keep")
        for expected_hash in (f"sha256:{victim}", "sha256:../../victim.txt", "sha256:" + "A" * 64, "md5:" + "0" * 32):
            with self.subTest(expected_hash=expected_hash), ScriptedServer([]) as server:
                with self.assertRaises(ValueError):
                    ATTRIBUTIONS._download_locked_artifact(server.url, expected_hash)
                self.assertEqual(server.requests, 0)
        self.assertEqual(victim.read_bytes(), b"keep")

    def test_hash_mismatch_fails_and_caches_nothing(self) -> None:
        with ScriptedServer([_ok]) as server, self.assertRaises(ValueError):
            ATTRIBUTIONS._download_locked_artifact(server.url, _sha256(b"other"))
        self.assertFalse(self.cache_dir.exists())


class RetryAfterTests(unittest.TestCase):
    def test_retry_after_is_capped_at_fetch_timeout(self) -> None:
        with (
            mock.patch.object(
                ATTRIBUTIONS, "ARTIFACT_CACHE_DIR", Path(self.enterContext(tempfile.TemporaryDirectory()))
            ),
            mock.patch.object(urllib3.util.retry.time, "sleep") as sleep,
            ScriptedServer([_status(HTTPStatus.SERVICE_UNAVAILABLE, Retry_After="1800"), _ok]) as server,
        ):
            ATTRIBUTIONS._download_locked_artifact(server.url, _sha256(DATA))
        sleep.assert_called_once_with(ATTRIBUTIONS.FETCH_TIMEOUT_SECONDS)


class MetadataParsingTests(unittest.TestCase):
    def test_license_expression_is_canonicalized(self) -> None:
        wheel = _wheel_from_metadata(
            "Metadata-Version: 2.4\nName: example\nVersion: 1.0\nLicense-Expression: mit or apache-2.0\n"
        )
        self.assertEqual(
            ATTRIBUTIONS._wheel_metadata_from_bytes(wheel, package_name="example"), ("MIT OR Apache-2.0", [])
        )

    def test_multiline_license_field_is_read_in_full(self) -> None:
        body = "\n        ".join(["MIT License", "", *(["Permission is hereby granted"] * 12)])
        sdist = _sdist_from_pkg_info(f"Metadata-Version: 2.1\nName: example\nVersion: 1.0\nLicense: {body}\n")
        license_name, _ = ATTRIBUTIONS._sdist_metadata_from_bytes(sdist, package_name="example")
        self.assertEqual(license_name, "See package metadata")


class WheelMetadataTests(unittest.TestCase):
    def test_only_first_wheel_is_downloaded(self) -> None:
        first = _wheel_bytes("MIT")
        second = _wheel_bytes("Apache-2.0")
        wheels = [
            {"url": "https://example.invalid/first.whl", "hash": _sha256(first)},
            {"url": "https://example.invalid/second.whl", "hash": _sha256(second)},
        ]
        with mock.patch.object(ATTRIBUTIONS, "_download_locked_artifact", return_value=first) as download:
            metadata = ATTRIBUTIONS._wheel_artifact_metadata(wheels, "example", None)
        self.assertEqual(metadata, ("MIT", []))
        download.assert_called_once_with(wheels[0]["url"], wheels[0]["hash"])

    def test_weak_wheel_license_returns_first_wheel_metadata(self) -> None:
        first = _wheel_bytes("UNKNOWN")
        wheels = [
            {"url": "https://example.invalid/first.whl", "hash": _sha256(first)},
            {"url": "https://example.invalid/second.whl", "hash": _sha256(first)},
        ]
        sdist_metadata = ("UNKNOWN", [("LICENSE", "text\n")])
        with mock.patch.object(ATTRIBUTIONS, "_download_locked_artifact", return_value=first) as download:
            metadata = ATTRIBUTIONS._wheel_artifact_metadata(wheels, "example", sdist_metadata)
        self.assertEqual(metadata, ("UNKNOWN", []))
        self.assertEqual(download.call_count, 1)

    def test_strong_wheel_license_keeps_sdist_texts(self) -> None:
        first = _wheel_bytes("MIT")
        wheels = [{"url": "https://example.invalid/first.whl", "hash": _sha256(first)}]
        sdist_metadata = ("UNKNOWN", [("LICENSE", "text\n")])
        with mock.patch.object(ATTRIBUTIONS, "_download_locked_artifact", return_value=first):
            metadata = ATTRIBUTIONS._wheel_artifact_metadata(wheels, "example", sdist_metadata)
        self.assertEqual(metadata, ("MIT", [("LICENSE", "text\n")]))


if __name__ == "__main__":
    unittest.main()
