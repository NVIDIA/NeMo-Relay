# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concrete agent example for the NeMo Guardrails plugin."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import plugin as nemoguardrails_plugin

from nemo_relay import Json, JsonObject, LLMRequest, ScopeType, llm, scope, tools
from nemo_relay import plugin as relay_plugin
from nemo_relay.codecs import OpenAIChatCodec

EXAMPLE_ROOT = Path(__file__).resolve().parent

DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NVIDIA_MODEL = "meta/llama-3.1-8b-instruct"
EXAMPLE_CONFIG_PATH = EXAMPLE_ROOT / "example_config.yml"
DEFAULT_RAILS_PATH = EXAMPLE_ROOT / "rails"
PASSTHROUGH_GUARDRAILS_CONFIG = """
models:
  - type: main
    engine: nvidia_ai_endpoints
    model: meta/llama-3.1-8b-instruct

rails:
  input:
    flows: []
  output:
    flows: []
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the NeMo Guardrails example agent.")
    parser.add_argument(
        "--guardrails-config",
        choices=("passthrough", "inline", "path"),
        default=os.environ.get("NEMO_GUARDRAILS_CONFIG", "passthrough"),
        help=(
            "Use fast passthrough config_yaml, inline self-check config_yaml, or a config_path directory. "
            "Defaults to NEMO_GUARDRAILS_CONFIG or passthrough."
        ),
    )
    parser.add_argument(
        "--config-path",
        default=os.environ.get("NEMO_GUARDRAILS_CONFIG_PATH", str(DEFAULT_RAILS_PATH)),
        help="NeMo Guardrails config directory used when --guardrails-config=path.",
    )
    parser.add_argument(
        "--tool",
        choices=("current_time", "weather"),
        default=os.environ.get("NEMO_GUARDRAILS_TOOL", "current_time"),
        help="Example tool to execute before the LLM call. Defaults to NEMO_GUARDRAILS_TOOL or current_time.",
    )
    return parser.parse_args()


def _require_api_key() -> str:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit("Set NVIDIA_API_KEY before running this example agent.")
    return api_key


def _chat_completions_url() -> str:
    explicit_url = os.environ.get("NVIDIA_CHAT_COMPLETIONS_URL")
    if explicit_url:
        return _validate_http_url(explicit_url)
    base_url = os.environ.get("NVIDIA_BASE_URL", DEFAULT_NVIDIA_BASE_URL).rstrip("/")
    return _validate_http_url(f"{base_url}/chat/completions")


def _validate_http_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("NVIDIA chat completion URL must be an absolute http(s) URL.")
    return url


def _guardrails_component_config(args: argparse.Namespace) -> JsonObject:
    config: dict[str, Json] = {
        "codec": "openai_chat",
        "input": True,
        "output": True,
        "tool_input": True,
        "tool_output": True,
    }
    if args.guardrails_config == "path":
        config["config_path"] = args.config_path
    elif args.guardrails_config == "inline":
        config["config_yaml"] = EXAMPLE_CONFIG_PATH.read_text(encoding="utf-8")
    else:
        config["config_yaml"] = PASSTHROUGH_GUARDRAILS_CONFIG
    return cast(JsonObject, config)


def _plugin_config(args: argparse.Namespace) -> relay_plugin.PluginConfig:
    return relay_plugin.PluginConfig(
        components=[
            relay_plugin.ComponentSpec(
                kind=nemoguardrails_plugin.DEFAULT_KIND,
                config=_guardrails_component_config(args),
            )
        ]
    )


async def _weather_lookup(args: Json) -> JsonObject:
    city = "Phoenix"
    if isinstance(args, dict):
        value = args.get("city")
        if isinstance(value, str) and value:
            city = value
    return {
        "city": city,
        "forecast": "Clear, warm, and dry",
        "source": "local example tool",
    }


async def _current_time(args: Json) -> JsonObject:
    requested_timezone = "UTC"
    if isinstance(args, dict):
        value = args.get("timezone")
        if isinstance(value, str) and value:
            requested_timezone = value
    return {
        "timezone": requested_timezone,
        "iso_time": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "source": "local example tool",
    }


async def _execute_example_tool(tool_name: str) -> Json:
    if tool_name == "weather":
        return await tools.execute("weather_lookup", {"city": "Phoenix"}, _weather_lookup)
    return await tools.execute("current_time", {"timezone": "UTC"}, _current_time)


def _post_chat_completion(request: LLMRequest) -> JsonObject:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    headers.update({key: str(value) for key, value in request.headers.items()})
    http_request = Request(
        _chat_completions_url(),
        data=json.dumps(request.content).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urlopen(http_request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"NVIDIA chat completion failed with HTTP {error.code}: {detail}") from error

    if not isinstance(payload, dict):
        raise RuntimeError("NVIDIA chat completion returned a non-object JSON payload.")
    return cast(JsonObject, payload)


async def _nvidia_chat(request: LLMRequest) -> JsonObject:
    return await asyncio.to_thread(_post_chat_completion, request)


def _assistant_text(response: Json) -> str:
    if not isinstance(response, dict):
        return json.dumps(response, indent=2, sort_keys=True)

    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return json.dumps(response, indent=2, sort_keys=True)

    message = choices[0].get("message")
    if not isinstance(message, dict):
        return json.dumps(response, indent=2, sort_keys=True)

    content = message.get("content")
    return content if isinstance(content, str) else json.dumps(response, indent=2, sort_keys=True)


async def run_agent() -> None:
    args = _parse_args()
    api_key = _require_api_key()
    model = os.environ.get("NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL)

    registered = False
    try:
        nemoguardrails_plugin.register()
        registered = True
        await relay_plugin.initialize(_plugin_config(args))

        with scope.scope("nemoguardrails-example-agent", ScopeType.Agent):
            tool_result = await _execute_example_tool(args.tool)
            prompt = (
                "You are a concise assistant. Use this tool result to answer in one sentence: "
                f"{json.dumps(tool_result, sort_keys=True)}"
            )
            response = await llm.execute(
                "nvidia_chat_completions",
                LLMRequest(
                    {"Authorization": f"Bearer {api_key}"},
                    {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.2,
                        "max_tokens": 120,
                    },
                ),
                _nvidia_chat,
                model_name=model,
                response_codec=OpenAIChatCodec(),
            )

        guardrails_source = "passthrough config_yaml"
        if args.guardrails_config == "inline":
            guardrails_source = f"inline config_yaml {EXAMPLE_CONFIG_PATH}"
        if args.guardrails_config == "path":
            guardrails_source = f"config_path {args.config_path}"
        print(f"Guardrails config: {guardrails_source}")
        print(f"Tool: {args.tool}")
        print("Tool result:")
        print(json.dumps(tool_result, indent=2, sort_keys=True))
        print("\nAssistant:")
        print(_assistant_text(response))
    finally:
        relay_plugin.clear()
        if registered:
            nemoguardrails_plugin.deregister()


if __name__ == "__main__":
    asyncio.run(run_agent())
