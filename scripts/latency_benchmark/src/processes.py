# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Relay subprocess lifecycle helpers for latency measurements."""

from __future__ import annotations

import socket
import subprocess
import time
import uuid
from pathlib import Path
from typing import IO

from .fixtures import isolated_environment, write_agent_config, write_mock_codex
from .servers import connection_for


class RelayProcess:
    """Run a normal Relay gateway and record readiness latency."""

    def __init__(
        self,
        binary: Path,
        root: Path,
        provider_url: str,
        plugin_config: Path,
        name: str,
    ) -> None:
        self.binary = binary
        self.root = root
        self.provider_url = provider_url
        self.plugin_config = plugin_config
        self.name = name
        self.process: subprocess.Popen[bytes] | None = None
        self.url = ""
        self.startup_ns = 0
        self.log_handle: IO[bytes] | None = None

    def start(self) -> None:
        log_path = self.root / f"{self.name}.log"
        self.log_handle = log_path.open("ab")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        command = [
            str(self.binary),
            "--bind",
            f"127.0.0.1:{port}",
            "--openai-base-url",
            f"{self.provider_url}/v1",
            "--anthropic-base-url",
            self.provider_url,
            "--config",
            str(self.root / "config.toml"),
            "--plugin-config-path",
            str(self.plugin_config),
        ]
        started = time.perf_counter_ns()
        self.process = subprocess.Popen(
            command,
            cwd=self.root,
            env=isolated_environment(self.root),
            stdout=subprocess.DEVNULL,
            stderr=self.log_handle,
        )
        deadline = time.monotonic() + 15
        while not self._healthy():
            if self.process.poll() is not None:
                self._raise_start_error(log_path)
            if time.monotonic() >= deadline:
                self.stop()
                raise RuntimeError(f"timed out waiting for {self.name} readiness")
            time.sleep(0.0005)
        self.startup_ns = time.perf_counter_ns() - started

    def _healthy(self) -> bool:
        connection = connection_for(self.url)
        connection.timeout = 0.1
        try:
            connection.request("GET", "/healthz")
            response = connection.getresponse()
            response.read()
            return response.status == 200
        except OSError:
            return False
        finally:
            connection.close()

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None

    def _raise_start_error(self, log_path: Path) -> None:
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None
        details = log_path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(f"{self.name} exited during startup:\n{details}")

    def __enter__(self) -> RelayProcess:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


class TransparentRelayProcess:
    """Run Relay's transparent coding-agent path for hook benchmarks."""

    def __init__(
        self,
        binary: Path,
        root: Path,
        provider_url: str,
        plugin_config: Path,
        name: str,
    ) -> None:
        self.binary = binary
        self.root = root
        self.provider_url = provider_url
        self.plugin_config = plugin_config
        self.name = name
        self.process: subprocess.Popen[bytes] | None = None
        self.url = ""
        self.log_handle: IO[bytes] | None = None
        self.stop_file = root / f"{name}-{uuid.uuid4().hex}.stop"

    def start(self) -> None:
        gateway_file = self.root / f"{self.name}-{uuid.uuid4().hex}.gateway"
        config = write_agent_config(self.root, self.name, write_mock_codex(self.root))
        log_path = self.root / f"{self.name}.log"
        self.log_handle = log_path.open("ab")
        environment = isolated_environment(self.root)
        environment["BENCHMARK_GATEWAY_FILE"] = str(gateway_file)
        environment["BENCHMARK_STOP_FILE"] = str(self.stop_file)
        command = [
            str(self.binary),
            "run",
            "--agent",
            "codex",
            "--config",
            str(config),
            "--openai-base-url",
            f"{self.provider_url}/v1",
            "--anthropic-base-url",
            self.provider_url,
            "--plugin-config-path",
            str(self.plugin_config),
        ]
        self.process = subprocess.Popen(
            command,
            cwd=self.root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=self.log_handle,
        )
        deadline = time.monotonic() + 15
        while True:
            if gateway_file.is_file():
                gateway_url = gateway_file.read_text(encoding="utf-8").strip()
                if gateway_url:
                    self.url = gateway_url
                    break
            if self.process.poll() is not None:
                self._raise_start_error(log_path)
            if time.monotonic() >= deadline:
                self.stop()
                raise RuntimeError(f"timed out waiting for {self.name} transparent gateway")
            time.sleep(0.001)

    def stop(self) -> None:
        self.stop_file.touch()
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None

    def _raise_start_error(self, log_path: Path) -> None:
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None
        details = log_path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(f"{self.name} exited during startup:\n{details}")

    def __enter__(self) -> TransparentRelayProcess:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
