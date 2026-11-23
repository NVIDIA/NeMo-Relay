# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify Pi's explicit built-ins with the managed Relay extension and isolated user/project settings."""

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
from scripts.quickstarts import provider  # noqa: E402


def serve_managed_requests(handler_class, credential, posts):
    model_post = handler_class.do_POST

    def post(handler):
        if handler.headers.get("x-nemo-relay-client-token") != credential:
            handler.send_error(401)
        elif handler.path == "/hooks/pi":
            posts.append(json.loads(handler.rfile.read(int(handler.headers["Content-Length"]))))
            handler.send_json({})
        else:
            # Node streams a forwarded Request body using chunked transfer encoding.
            if handler.headers.get("Transfer-Encoding") == "chunked":
                chunks = bytearray()
                while size := int(handler.rfile.readline().strip(), 16):
                    chunks.extend(handler.rfile.read(size))
                    assert handler.rfile.read(2) == b"\r\n"
                handler.rfile = BytesIO(chunks)
                handler.headers["Content-Length"] = str(len(chunks))
            model_post(handler)

    handler_class.do_POST = post


def run(pi: Path, root: Path) -> None:
    credential = base64.urlsafe_b64encode(os.urandom(32)).decode().rstrip("=")
    agent = root / "agent"
    project = root / "project"
    project.mkdir()
    for directory in (agent, project / ".pi"):
        (directory / "extensions").mkdir(parents=True)
        marker = directory / "discovered"
        (directory / "extensions" / "probe.ts").write_text(
            "import {writeFileSync} from 'node:fs'; export default () => writeFileSync("
            + json.dumps(str(marker))
            + ", 'loaded');"
        )
        (directory / "mcp.json").write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "probe": {"command": sys.executable, "args": ["-c", f"open({str(marker)!r},'w').close()"]}
                    }
                }
            )
        )
    managed = root / "managed"
    shutil.copytree(REPO / "crates/cli/src/daemon/managed/pi_extension", managed)
    dispatcher = root / "dispatcher"
    dispatcher.write_text(
        "#!/usr/bin/env python3\nimport json,sys\nfor line in sys.stdin:\n"
        " m=json.loads(line)\n if m.get('method')=='initialize':\n"
        "  print(json.dumps({'jsonrpc':'2.0','id':m['id'],'result':"
        "{'protocolVersion':'2025-11-25','serverInfo':{'name':'nemo-relay'}}}),flush=True)\n"
    )
    dispatcher.chmod(0o755)
    fixture = provider.Fixture(project, root / "provider.jsonl")
    (managed / "managed-config.json").write_text(
        json.dumps(
            {"schema": "nemo-relay-managed-pi-v1", "daemonAddress": fixture.url, "dispatcherCommand": str(dispatcher)}
        )
    )
    (agent / "models.json").write_text(
        json.dumps(
            {
                "providers": {
                    "fixture": {
                        "api": "openai-completions",
                        "apiKey": "fixture",
                        "baseUrl": "http://127.0.0.1:1/v1",
                        "models": [{"id": "model", "contextWindow": 64000, "maxTokens": 4096}],
                    }
                }
            }
        )
    )
    posts = []
    serve_managed_requests(fixture.server.RequestHandlerClass, credential, posts)
    (project / "fixture.txt").write_text("managed-read-ok")
    (agent / "settings.json").write_text(json.dumps({"defaultProvider": "fixture", "defaultModel": "model"}))
    fixture.start()
    try:

        def choose(request, _project):
            names = {entry["function"]["name"] for entry in request["tools"]}
            assert {"read", "codemode", "tool_search"} <= names, names
            return "codemode", {"code": "text(await tools.read({path:'fixture.txt'}))"}

        with (
            patch.object(provider, "choose_tool", choose),
            patch.object(
                provider,
                "tool_result",
                lambda req: any(
                    m.get("role") == "tool" and "managed-read-ok" in json.dumps(m.get("content", ""))
                    for m in req["messages"]
                ),
            ),
        ):
            result = subprocess.run(
                [
                    str(pi),
                    "--no-extensions",
                    "-e",
                    str(managed / "index.ts"),
                    "-e",
                    "builtin:codemode",
                    "-e",
                    "builtin:tool-search",
                    "--tools",
                    "read,codemode,tool_search",
                    "--approve",
                    "--no-session",
                    "--offline",
                    "--model",
                    "fixture/model",
                    "-p",
                    "Run fixture",
                ],
                cwd=project,
                env={
                    "HOME": str(root),
                    "PI_CODING_AGENT_DIR": str(agent),
                    "PATH": os.environ["PATH"],
                    "NEMO_RELAY_CLIENT_TOKEN": credential,
                    "PI_OFFLINE": "1",
                },
                capture_output=True,
                text=True,
                timeout=60,
            )
        assert result.returncode == 0 and provider.RESPONSE_MARKER in result.stdout, result.stderr[-1500:]
        gates = [event for event in posts if event["hook_event_name"] == "tool_call"]
        assert [e["tool_name"] for e in gates] == ["codemode", "read"], gates
        assert gates[1]["parent_tool_call_id"] == gates[0]["tool_call_id"]
        for directory in (agent, project / ".pi"):
            assert not (directory / "discovered").exists()
    finally:
        fixture.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi", required=True, type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="relay-managed-pi-") as directory:
        run(args.pi.absolute(), Path(directory))
