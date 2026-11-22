# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare real Pi request semantics directly and through Relay for two provider protocols."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from io import BytesIO
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
from scripts.quickstarts import provider  # noqa: E402


def record_messages(handler_class, records):
    model_post = handler_class.do_POST

    def capture(handler):
        body = handler.rfile.read(int(handler.headers["Content-Length"]))
        handler.rfile = BytesIO(body)
        request = json.loads(body)
        records.append(
            {
                "roles": [message["role"] for message in request["messages"]],
                "tools": request.get("tools", []),
                "system": {
                    "system": request.get("system"),
                    "instructions": [
                        message["content"]
                        for message in request["messages"]
                        if message["role"] in {"system", "developer"}
                    ],
                },
                "thread": request.get("thread"),
                "path": handler.path,
            }
        )
        model_post(handler)

    handler_class.do_POST = capture


def run(pi, relay, root, api, domain, upstream_key):
    agent = root / "agent"
    agent.mkdir(parents=True)
    fixture = provider.Fixture(root, root / "requests.jsonl")
    records = []
    record_messages(fixture.server.RequestHandlerClass, records)
    fixture.start()
    try:
        # Pi 1.6.0 detects these services by substring in baseUrl. A path segment
        # exercises the same serializer decision while all traffic stays on loopback.
        url = fixture.url + "/" + domain + ("/v1" if api == "openai-completions" else "")
        (agent / "models.json").write_text(
            json.dumps(
                {
                    "providers": {
                        "fixture": {
                            "api": api,
                            "baseUrl": url,
                            "apiKey": "test-fixture",
                            "models": [{"id": "fixture", "reasoning": True, "contextWindow": 64000, "maxTokens": 4096}],
                        }
                    }
                }
            )
        )
        config = root / "config.toml"
        config.write_text(
            f"[agents.pi]\ncommand = {json.dumps(str(pi))}\n[upstream]\n{upstream_key} = {json.dumps(url)}\n"
        )
        env = {
            "PATH": os.environ["PATH"],
            "HOME": str(root),
            "PI_CODING_AGENT_DIR": str(agent),
            "PI_OFFLINE": "1",
            "NEMO_RELAY_PI_EXTENSION": str(REPO / "crates/cli/assets/pi-extension/index.ts"),
            "NEMO_RELAY_TEST_SYSTEM_CONFIG_DIR": str(root / "system"),
            "XDG_CONFIG_HOME": str(root / "config"),
        }
        (root / "plugins.toml").write_text((REPO / "docs/getting-started/quick-start/_code/plugins.toml").read_text())
        direct = []
        for launch in [[str(pi)], [str(relay), "--config", str(config), "pi", "--"]]:
            records.clear()
            result = subprocess.run(
                launch + ["--no-session", "--model", "fixture/fixture", "-p", "Run the fixture tool"],
                env=env,
                cwd=root,
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert result.returncode == 0 and provider.RESPONSE_MARKER in result.stdout, (
                result.stdout + result.stderr
            )[-1600:]
            assert len(records) == 2, records
            if direct:
                assert records == direct, (records, direct)
            else:
                direct.extend(records)
        if api == "openai-completions":
            assert direct[0]["roles"][0] == "developer", direct
        else:
            assert [r["thread"]["type"] for r in direct] == ["create", "continue"], direct
            assert (
                direct[1]["roles"] == ["user"] and not direct[1]["tools"] and not any(direct[1]["system"].values())
            ), direct
    finally:
        fixture.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi", type=Path, required=True)
    parser.add_argument("--relay", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="relay-pi-wire-") as directory:
        for api, domain, key in [
            ("openai-completions", "api.openai.com", "openai_base_url"),
            ("anthropic-messages", "api.anthropic.com", "anthropic_base_url"),
        ]:
            run(args.pi.absolute(), args.relay.absolute(), Path(directory) / api, api, domain, key)
