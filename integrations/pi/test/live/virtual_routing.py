# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Route a real Pi virtual model through Relay, including a late provider."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
from scripts.quickstarts import provider  # noqa: E402


def reject_relay_headers(handler_class):
    handle_post = handler_class.do_POST

    def checked(request):
        internal = {
            "x-nemo-relay-session-id",
            "x-nemo-relay-upstream-base-url",
            "x-nemo-relay-proxy-token",
            "x-nemo-relay-client-token",
        }
        # Header names alone identify regressions without recording credentials.
        forbidden = internal.intersection(key.lower() for key in request.headers)
        if forbidden:
            request.send_error(400, "Relay headers reached upstream: " + ", ".join(sorted(forbidden)))
            return
        handle_post(request)

    handler_class.do_POST = checked


def run(pi: Path, relay: Path, root: Path) -> None:
    models = {}
    with ExitStack() as stack:
        for name in ("first", "second"):
            fixture = provider.Fixture(root, root / f"{name}.jsonl")
            reject_relay_headers(fixture.server.RequestHandlerClass)
            fixture.start()
            stack.callback(fixture.close)
            models[name] = {
                "api": "openai-completions",
                "apiKey": "fixture-only",
                "baseUrl": fixture.url + "/v1",
                "models": [
                    {
                        "id": name,
                        "name": name,
                        "input": ["text"],
                        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                        "reasoning": False,
                        "contextWindow": 64000,
                        "maxTokens": 4096,
                    }
                ],
            }

        agent = root / "agent"
        agent.mkdir()
        (agent / "models.json").write_text(json.dumps({"providers": {"first": models["first"]}}))
        router = root / "router.mjs"
        router.write_text(
            "export default pi => pi.registerVirtualModel({provider:'router', id:'auto', name:'Auto', "
            "route(req, ctx) { const id = req.reason === 'user' ? 'first' : 'second'; "
            "if (id === 'second' && !ctx.modelRegistry.find(id, id)) "
            f"pi.registerProvider(id, {json.dumps(models['second'])}); "
            "return {model:ctx.modelRegistry.find(id,id)}; }});"
        )
        config = root / "config.toml"
        config.write_text(
            f"[agents.pi]\ncommand = {json.dumps(str(pi))}\n[upstream]\n"
            f"openai_base_url = {json.dumps(models['first']['baseUrl'])}\n"
        )
        (root / "plugins.toml").write_text((REPO / "docs/getting-started/quick-start/_code/plugins.toml").read_text())
        env = {
            "PATH": os.environ["PATH"],
            "HOME": str(root),
            "PI_CODING_AGENT_DIR": str(agent),
            "PI_OFFLINE": "1",
            "XDG_CONFIG_HOME": str(root / "config"),
            "XDG_STATE_HOME": str(root / "state"),
            "NEMO_RELAY_TEST_SYSTEM_CONFIG_DIR": str(root / "system"),
            "NEMO_RELAY_PI_FAIL": "closed",
            "NEMO_RELAY_PI_EXTENSION": str(REPO / "crates/cli/assets/pi-extension/index.ts"),
        }
        result = subprocess.run(
            [
                str(relay),
                "--config",
                str(config),
                "pi",
                "--",
                "--offline",
                "--no-session",
                "-e",
                str(router),
                "--model",
                "router/auto",
                "-p",
                "Run the fixture tool",
            ],
            cwd=root,
            env=env,
            timeout=90,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert result.returncode == 0, result.stdout[-3000:]
        assert provider.RESPONSE_MARKER in result.stdout, result.stdout[-1200:]
        assert (root / "relay-check.txt").read_text() == provider.TOOL_MARKER
        for name in models:
            requests = [json.loads(line) for line in (root / f"{name}.jsonl").read_text().splitlines()]
            assert [request["model"] for request in requests] == [name], requests
        log = root / "logs" / "events.jsonl"
        starts = [
            event
            for line in log.read_text().splitlines()
            if (event := json.loads(line)).get("category") == "llm" and event.get("scope_category") == "start"
        ]
        traced_models = [e["category_profile"]["annotated_request"]["model"] for e in starts]
        assert traced_models == ["first", "second"], traced_models
        assert len({e["metadata"]["session_id"] for e in starts}) == 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi", required=True, type=Path)
    parser.add_argument("--relay", required=True, type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="relay-pi-virtual-") as directory:
        run(args.pi.absolute(), args.relay.absolute(), Path(directory))
