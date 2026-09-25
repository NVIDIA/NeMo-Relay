# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the candidate binary: python crates/cli/tests/prepare_smoke.py BINARY."""

import contextlib
import http.server
import json
import os
import selectors
import subprocess
import sys
import tempfile
import threading
import tomllib
import urllib.error
import urllib.request
from pathlib import Path


def frame(process):
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        assert selector.select(15), "controller did not reply within 15 seconds"
    line = process.stdout.readline()
    assert line, "controller exited before replying"
    return json.loads(line)


def check(binary):
    seen = []

    class Upstream(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"object":"list","data":[]}')

        def log_message(self, *_):
            pass

    with tempfile.TemporaryDirectory() as temporary, contextlib.ExitStack() as stack:
        root = Path(temporary).resolve()
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        stack.callback(server.server_close)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        stack.callback(server.shutdown)
        runtime = root / "config.toml"
        runtime.write_text("")
        plugins = root / "plugins.toml"
        plugins.write_text("version = 1\n")
        executables = root / "bin"
        executables.mkdir()
        for name in ("codex", "claude"):
            executable = executables / name
            executable.write_text(f"#!/bin/sh\ntouch '{root / 'native-spawned'}'\nexit 9\n")
            executable.chmod(0o700)
        controllers = []
        for index, agent in enumerate(("codex", "claude")):
            home = root / str(index)
            home.mkdir(mode=0o700)
            path = home / ("config.toml" if agent == "codex" else "settings.json")
            original = 'model_provider = "gym"\n' if agent == "codex" else '{"env":{"KEEP":"yes"}}'
            path.write_text(original)
            environment = {
                **os.environ,
                "HOME": str(root),
                "XDG_CONFIG_HOME": str(root / "xdg"),
                "XDG_DATA_HOME": str(root / "data"),
                "PATH": str(executables) + os.pathsep + os.environ["PATH"],
                "CODEX_HOME": str(home),
                "CLAUDE_CONFIG_DIR": str(home),
                "OPENAI_API_KEY": "synthetic-local-only",
            }
            for name in list(environment):
                if name.startswith("NEMO_RELAY_"):
                    del environment[name]
            environment["NEMO_RELAY_INVOCATION_STATE_DIR"] = str(home / "bootstrap")
            stderr = stack.enter_context((root / f"stderr-{index}").open("w+"))
            process = subprocess.Popen(
                [str(binary), "--config", str(runtime), "--plugin-config-path", str(plugins), "prepare"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                env=environment,
                cwd=root,
                text=True,
            )
            stack.callback(lambda p=process: p.kill() if p.poll() is None else None)
            argv = (
                [agent, "exec", "--json", "--", "prompt"]
                if agent == "codex"
                else [agent, "-p", "--model", "test", "--", "prompt"]
            )
            request_payload = {
                "version": 1,
                "agent": agent,
                "argv": argv,
                "home": str(home),
                "upstream_url": f"http://127.0.0.1:{server.server_port}/rollout/{index}/v1",
            }
            process.stdin.write(json.dumps(request_payload) + "\n")
            process.stdin.flush()
            try:
                ready = frame(process)
            except AssertionError:
                stderr.seek(0)
                raise AssertionError(stderr.read()) from None
            assert ready["version"] == 1 and "environment" in ready
            patch = ready["environment"]
            assert patch["NEMO_RELAY_INVOCATION_STATE_DIR"] == str(home / "bootstrap")
            route = patch["NEMO_RELAY_GATEWAY_URL"]
            token = patch["NEMO_RELAY_PROXY_CREDENTIAL"]
            if agent == "codex":
                config = tomllib.loads(path.read_text())
                assert config["model_provider"] == "nemo-relay-openai"
                assert config["model_providers"]["nemo-relay-openai"]["base_url"] == route + "/v1"
                assert config["hooks"]["SessionStart"]
                assert any(str(path) in key for key in config["hooks"]["state"])
            else:
                config = json.loads(path.read_text())
                assert config["env"]["ANTHROPIC_BASE_URL"] == route
                assert config["env"]["KEEP"] == "yes" and config["hooks"]["SessionStart"]
            controllers.append((process, path, original, route, token))
        assert controllers[0][3] != controllers[1][3]
        for index, (_, _, _, route, token) in enumerate(controllers):
            request = urllib.request.Request(route + "/v1/models", headers={"x-nemo-relay-proxy-token": token})
            with urllib.request.urlopen(request, timeout=5) as response:
                assert response.status == 200
            assert seen[-1] == f"/rollout/{index}/v1/models", seen
        foreign = urllib.request.Request(
            controllers[0][3] + "/v1/models",
            headers={"x-nemo-relay-proxy-token": controllers[1][4]},
        )
        try:
            urllib.request.urlopen(foreign, timeout=5)
        except urllib.error.HTTPError as error:
            assert error.code == 401
        else:
            raise AssertionError("a sibling credential was accepted")
        # One owner's teardown cannot change its sibling's routing.
        for index, (process, path, original, route, token) in enumerate(controllers):
            process.stdin.close()
            assert frame(process) == {"version": 1, "cleanup": "complete"}
            assert process.wait(timeout=10) == 0
            assert path.read_text() == original
            try:
                urllib.request.urlopen(route + "/healthz", timeout=1)
            except urllib.error.URLError:
                pass
            else:
                raise AssertionError("gateway survived owner EOF")
            if index == 0:
                sibling = controllers[1]
                request = urllib.request.Request(
                    sibling[3] + "/v1/models", headers={"x-nemo-relay-proxy-token": sibling[4]}
                )
                with urllib.request.urlopen(request, timeout=5) as response:
                    assert response.status == 200
                assert seen[-1] == "/rollout/1/v1/models"
        for update in (
            {"version": 2},
            {"argv": ["claude", "-p", "--bare"]},
            {"argv": ["claude", "-p", "--settings", "other.json"]},
            {"home": str(root)},
        ):
            rejected = subprocess.run(
                [str(binary), "--config", str(runtime), "--plugin-config-path", str(plugins), "prepare"],
                input=json.dumps({**request_payload, **update}) + "\n",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                cwd=root,
                text=True,
                timeout=10,
            )
            assert rejected.returncode != 0 and not rejected.stdout, "invalid preparation reached readiness"
            assert path.read_text() == original
        assert not (root / "native-spawned").exists(), "prepare spawned a native executable"
    print("PASS: two private gateways, full upstream paths, zero native launches, EOF cleanup")


if __name__ == "__main__":
    check(Path(sys.argv[1]).resolve())
