# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Instrument an application-owned model request and real file operations."""

import asyncio
import json
import urllib.request

import nemo_relay as relay
import nemo_relay.observability
from nemo_relay.codecs import OpenAIChatCodec


async def run(directory, environment, timeout):
    endpoint = environment.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
    model = environment["NVIDIA_MODEL"]
    key = environment["NVIDIA_API_KEY"]

    def invoke(request):
        # Keep the provider credential in this callback, outside event payloads.
        headers = {"Content-Type": "application/json", "Authorization": "Bearer " + key}
        http_request = urllib.request.Request(
            endpoint + "/chat/completions",
            data=json.dumps(request.content).encode(),
            headers=headers,
        )
        with urllib.request.urlopen(http_request, timeout=timeout) as response:
            return json.load(response)

    async with relay.plugin.activate(relay.plugin.PluginConfig(), str(directory / "plugins.toml")) as activation:
        errors = [d for d in activation.report["config"]["diagnostics"] if d["level"] == "error"]
        if errors:
            raise ValueError("Plugin activation failed: " + str(errors))
        with relay.scope.scope("support-triage", relay.ScopeType.Agent):
            read = await relay.tools.execute(
                "read_file",
                {"path": "requests.csv"},
                lambda _: relay.ToolExecutionResult({"content": (directory / "requests.csv").read_text()}),
            )
            request = relay.LLMRequest(
                {},
                {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Read this synthetic incident queue and return a concise Markdown handoff. "
                                "Include every incident ID, prioritize customer impact, and give an action for each. "
                                "Mention the synthetic contact once. Use no external services.\n"
                                + read.result["content"]
                            ),
                        }
                    ],
                    "max_tokens": 4096,
                },
            )
            response = await relay.llm.execute(
                "openai.chat_completions",
                request,
                lambda req: asyncio.to_thread(invoke, req),
                model_name=model,
                codec=OpenAIChatCodec(),
                response_codec=OpenAIChatCodec(),
            )
            text = response["choices"][0]["message"].get("content")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Model returned no handoff text; inspect provider/output limits")

            def write(_):
                (directory / "handoff.md").write_text(text)
                return relay.ToolExecutionResult({"path": "handoff.md", "content": text})

            await relay.tools.execute("write_file", {"path": "handoff.md", "content": text}, write)
