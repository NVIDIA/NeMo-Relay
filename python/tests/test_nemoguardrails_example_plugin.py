# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the example NeMo Guardrails plugin.

The tests inject fake ``nemoguardrails`` modules into ``sys.modules`` before
plugin initialization, so CI does not need the optional dependency installed.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
from nemo_flow import JsonObject, LLMRequest, llm, plugin, tools


def _load_example_plugin() -> Any:
    module_path = Path(__file__).resolve().parents[2] / "examples" / "nemoguardrails" / "example" / "plugin.py"
    spec = importlib.util.spec_from_file_location(
        "nemoguardrails_example_plugin",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load NeMo Guardrails example plugin")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ngr = _load_example_plugin()


@dataclass
class FakeGuardrailsResult:
    status: str
    content: str = ""
    rail: str | None = None


class FakeRailType:
    INPUT = "input"
    OUTPUT = "output"


class FakeRailStatus:
    PASSED = "passed"
    MODIFIED = "modified"
    BLOCKED = "blocked"


class FakeRailsConfig:
    loaded: ClassVar[list[dict[str, str | None]]] = []

    @staticmethod
    def from_path(path: str) -> dict[str, str]:
        FakeRailsConfig.loaded.append({"source": "path", "value": path})
        return {"source": "path", "value": path}

    @staticmethod
    def from_content(
        colang_content: str | None = None,
        yaml_content: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, str | None]:
        FakeRailsConfig.loaded.append(
            {
                "source": "content",
                "colang_content": colang_content,
                "yaml_content": yaml_content,
                "config": str(config) if config is not None else None,
            }
        )
        return {"source": "content", "value": yaml_content}


class FakeRails:
    queued_results: ClassVar[list[FakeGuardrailsResult]] = []
    instances: ClassVar[list[FakeRails]] = []

    def __init__(self, config: dict[str, str]) -> None:
        self.config = config
        self.calls: list[tuple[list[dict[str, Any]], list[str] | None]] = []
        FakeRails.instances.append(self)

    async def check_async(self, messages: list[dict[str, Any]], rail_types: list[str] | None = None):
        self.calls.append(([dict(message) for message in messages], rail_types))
        if not FakeRails.queued_results:
            raise AssertionError("No fake NeMo Guardrails result was queued")
        return FakeRails.queued_results.pop(0)


@pytest.fixture(autouse=True)
def reset_fake_guardrails_state() -> Iterator[None]:
    FakeRails.queued_results = []
    FakeRails.instances = []
    FakeRailsConfig.loaded = []
    yield
    FakeRails.queued_results = []
    FakeRails.instances = []
    FakeRailsConfig.loaded = []


@pytest.fixture
def guardrails_kind():
    kind = f"python.test_nemoguardrails.{uuid.uuid4().hex}"
    plugin.clear()
    yield kind
    plugin.clear()
    plugin.deregister(kind)


def _install_fake_guardrails(monkeypatch: pytest.MonkeyPatch, results: list[FakeGuardrailsResult]) -> None:
    FakeRails.queued_results = list(results)
    FakeRails.instances = []
    FakeRailsConfig.loaded = []

    guardrails_mod = types.ModuleType("nemoguardrails")
    rails_pkg = types.ModuleType("nemoguardrails.rails")
    llm_pkg = types.ModuleType("nemoguardrails.rails.llm")
    options_mod = types.ModuleType("nemoguardrails.rails.llm.options")

    setattr(guardrails_mod, "RailsConfig", FakeRailsConfig)
    setattr(guardrails_mod, "LLMRails", FakeRails)
    setattr(guardrails_mod, "rails", rails_pkg)
    setattr(rails_pkg, "llm", llm_pkg)
    setattr(llm_pkg, "options", options_mod)
    setattr(options_mod, "RailType", FakeRailType)
    setattr(options_mod, "RailStatus", FakeRailStatus)

    monkeypatch.setitem(sys.modules, "nemoguardrails", guardrails_mod)
    monkeypatch.setitem(sys.modules, "nemoguardrails.rails", rails_pkg)
    monkeypatch.setitem(sys.modules, "nemoguardrails.rails.llm", llm_pkg)
    monkeypatch.setitem(sys.modules, "nemoguardrails.rails.llm.options", options_mod)


def _plugin_config(kind: str, **overrides: Any) -> plugin.PluginConfig:
    config = {
        "config_yaml": "rails:\n  input:\n    flows: []\n  output:\n    flows: []\n",
        "codec": "openai_chat",
    }
    config.update(overrides)
    return plugin.PluginConfig(components=[plugin.ComponentSpec(kind=kind, config=cast(JsonObject, config))])


def _last_message_content(request: LLMRequest) -> str:
    messages = cast(list[dict[str, Any]], request.content["messages"])
    return cast(str, messages[-1]["content"])


async def _activate(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    results: list[FakeGuardrailsResult],
    **config_overrides: Any,
) -> None:
    _install_fake_guardrails(monkeypatch, results)
    ngr.register(kind)
    report = await plugin.initialize(_plugin_config(kind, **config_overrides))
    assert report["diagnostics"] == []


def _chat_request(content: str = "unsafe input") -> LLMRequest:
    return LLMRequest(
        {"Authorization": "Bearer test"},
        {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.2,
        },
    )


def _chat_response(content: str = "raw answer") -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def _anthropic_request(content: str = "unsafe input") -> LLMRequest:
    return LLMRequest(
        {},
        {
            "model": "claude-sonnet-test",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": content}],
        },
    )


def _anthropic_response(content: str = "raw answer") -> dict[str, Any]:
    return {
        "id": "msg-test",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-test",
        "content": [{"type": "text", "text": content}],
        "stop_reason": "end_turn",
    }


def _openai_responses_request(content: str = "unsafe input") -> LLMRequest:
    return LLMRequest(
        {},
        {
            "model": "gpt-4o",
            "input": [{"role": "user", "content": content}],
        },
    )


def _openai_responses_response(content: str = "raw answer") -> dict[str, Any]:
    return {
        "id": "resp-test",
        "model": "gpt-4o",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content}],
            }
        ],
    }


class TestNeMoGuardrailsPluginValidation:
    def test_validate_does_not_import_nemoguardrails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fail_import(name: str):
            raise AssertionError(f"validate should not import {name}")

        monkeypatch.setattr(ngr.importlib, "import_module", fail_import)
        diagnostics = ngr.NeMoGuardrailsPlugin().validate(
            {
                "config_yaml": "rails: {}\n",
                "codec": "openai_chat",
            }
        )

        assert diagnostics == []

    def test_validate_rejects_invalid_config(self) -> None:
        diagnostics = ngr.NeMoGuardrailsPlugin().validate(
            {
                "config_yaml": "",
                "codec": "not-supported",
                "colang_content": "",
                "input": False,
                "output": False,
            }
        )
        codes = {diagnostic["code"] for diagnostic in diagnostics}

        assert "nemoguardrails.invalid_config_yaml" in codes
        assert "nemoguardrails.unsupported_codec" in codes
        assert "nemoguardrails.invalid_colang_content" in codes
        assert "nemoguardrails.no_rails_enabled" in codes

    def test_validate_accepts_tool_only_config(self) -> None:
        diagnostics = ngr.NeMoGuardrailsPlugin().validate(
            {
                "config_yaml": "rails: {}\n",
                "input": False,
                "output": False,
                "tool_input": True,
            }
        )

        assert diagnostics == []

    async def test_initialize_loads_config_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        guardrails_kind: str,
    ) -> None:
        _install_fake_guardrails(monkeypatch, [])
        ngr.register(guardrails_kind)

        report = await plugin.initialize(
            plugin.PluginConfig(
                components=[
                    plugin.ComponentSpec(
                        kind=guardrails_kind,
                        config=cast(
                            JsonObject,
                            {
                                "config_path": "/tmp/example-rails",
                                "codec": "openai_chat",
                            },
                        ),
                    )
                ]
            )
        )

        assert report["diagnostics"] == []
        assert FakeRailsConfig.loaded == [{"source": "path", "value": "/tmp/example-rails"}]

    async def test_initialize_reports_missing_optional_dependency(
        self,
        monkeypatch: pytest.MonkeyPatch,
        guardrails_kind: str,
    ) -> None:
        def missing_dependency(name: str):
            if name.startswith("nemoguardrails"):
                raise ImportError(name)
            raise AssertionError(f"unexpected import {name}")

        monkeypatch.setattr(ngr.importlib, "import_module", missing_dependency)
        ngr.register(guardrails_kind)

        with pytest.raises(RuntimeError, match="NeMo Guardrails is required"):
            await plugin.initialize(_plugin_config(guardrails_kind))


class TestNeMoGuardrailsPluginRuntime:
    async def test_input_pass_calls_provider(
        self,
        monkeypatch: pytest.MonkeyPatch,
        guardrails_kind: str,
    ) -> None:
        await _activate(
            monkeypatch,
            guardrails_kind,
            [FakeGuardrailsResult(FakeRailStatus.PASSED)],
            output=False,
        )
        seen_requests = []

        async def provider(request: LLMRequest):
            seen_requests.append(request)
            return _chat_response("provider answer")

        result = await llm.execute("gpt-4o", _chat_request("hello"), provider)

        assert result["choices"][0]["message"]["content"] == "provider answer"
        assert _last_message_content(seen_requests[0]) == "hello"
        assert (
            FakeRailsConfig.loaded[0]["yaml_content"] == "rails:\n  input:\n    flows: []\n  output:\n    flows: []\n"
        )
        assert FakeRails.instances[0].calls == [([{"role": "user", "content": "hello"}], [FakeRailType.INPUT])]

    async def test_input_block_stops_before_provider(
        self,
        monkeypatch: pytest.MonkeyPatch,
        guardrails_kind: str,
    ) -> None:
        await _activate(
            monkeypatch,
            guardrails_kind,
            [FakeGuardrailsResult(FakeRailStatus.BLOCKED, rail="jailbreak")],
            output=False,
        )
        provider_called = False

        async def provider(_request: LLMRequest):
            nonlocal provider_called
            provider_called = True
            return _chat_response()

        with pytest.raises(RuntimeError, match="input rail blocked"):
            await llm.execute("gpt-4o", _chat_request("bad"), provider)

        assert provider_called is False

    async def test_input_modified_rewrites_provider_request(
        self,
        monkeypatch: pytest.MonkeyPatch,
        guardrails_kind: str,
    ) -> None:
        await _activate(
            monkeypatch,
            guardrails_kind,
            [FakeGuardrailsResult(FakeRailStatus.MODIFIED, content="safe input")],
            output=False,
        )
        original = _chat_request("unsafe input")
        seen_requests = []

        async def provider(request: LLMRequest):
            seen_requests.append(request)
            return _chat_response("provider answer")

        result = await llm.execute("gpt-4o", original, provider)

        assert result["choices"][0]["message"]["content"] == "provider answer"
        assert _last_message_content(seen_requests[0]) == "safe input"
        assert _last_message_content(original) == "unsafe input"

    async def test_output_pass_returns_provider_response(
        self,
        monkeypatch: pytest.MonkeyPatch,
        guardrails_kind: str,
    ) -> None:
        await _activate(
            monkeypatch,
            guardrails_kind,
            [
                FakeGuardrailsResult(FakeRailStatus.PASSED),
                FakeGuardrailsResult(FakeRailStatus.PASSED),
            ],
        )
        response = _chat_response("raw answer")

        async def provider(_request: LLMRequest):
            return response

        result = await llm.execute("gpt-4o", _chat_request("hello"), provider)

        assert result == response
        assert FakeRails.instances[0].calls[1] == (
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "raw answer"},
            ],
            [FakeRailType.OUTPUT],
        )

    async def test_output_block_raises_after_provider(
        self,
        monkeypatch: pytest.MonkeyPatch,
        guardrails_kind: str,
    ) -> None:
        await _activate(
            monkeypatch,
            guardrails_kind,
            [
                FakeGuardrailsResult(FakeRailStatus.PASSED),
                FakeGuardrailsResult(FakeRailStatus.BLOCKED, rail="toxicity"),
            ],
        )
        provider_called = False

        async def provider(_request: LLMRequest):
            nonlocal provider_called
            provider_called = True
            return _chat_response("bad answer")

        with pytest.raises(RuntimeError, match="output rail blocked"):
            await llm.execute("gpt-4o", _chat_request("hello"), provider)

        assert provider_called is True

    async def test_output_pass_returns_anthropic_messages_response(
        self,
        monkeypatch: pytest.MonkeyPatch,
        guardrails_kind: str,
    ) -> None:
        await _activate(
            monkeypatch,
            guardrails_kind,
            [
                FakeGuardrailsResult(FakeRailStatus.PASSED),
                FakeGuardrailsResult(FakeRailStatus.PASSED),
            ],
            codec="anthropic_messages",
        )

        async def provider(_request: LLMRequest):
            return _anthropic_response("raw answer")

        result = await llm.execute("claude", _anthropic_request("hello"), provider)

        assert result["content"][0]["text"] == "raw answer"
        assert FakeRails.instances[0].calls[1] == (
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "raw answer"},
            ],
            [FakeRailType.OUTPUT],
        )

    async def test_output_pass_returns_openai_responses_response(
        self,
        monkeypatch: pytest.MonkeyPatch,
        guardrails_kind: str,
    ) -> None:
        await _activate(
            monkeypatch,
            guardrails_kind,
            [
                FakeGuardrailsResult(FakeRailStatus.PASSED),
                FakeGuardrailsResult(FakeRailStatus.PASSED),
            ],
            codec="openai_responses",
        )

        async def provider(_request: LLMRequest):
            return _openai_responses_response("raw answer")

        result = await llm.execute("gpt-4o", _openai_responses_request("hello"), provider)

        assert result["output"][0]["content"][0]["text"] == "raw answer"
        assert FakeRails.instances[0].calls[1] == (
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "raw answer"},
            ],
            [FakeRailType.OUTPUT],
        )

    async def test_output_modified_raises_without_rewriting_provider_response(
        self,
        monkeypatch: pytest.MonkeyPatch,
        guardrails_kind: str,
    ) -> None:
        await _activate(
            monkeypatch,
            guardrails_kind,
            [
                FakeGuardrailsResult(FakeRailStatus.PASSED),
                FakeGuardrailsResult(FakeRailStatus.MODIFIED, content="safe answer"),
            ],
        )
        provider_response = _chat_response("raw answer")

        async def provider(_request: LLMRequest):
            return provider_response

        with pytest.raises(RuntimeError, match="does not rewrite provider responses"):
            await llm.execute("gpt-4o", _chat_request("hello"), provider)

        assert provider_response["choices"][0]["message"]["content"] == "raw answer"


class TestNeMoGuardrailsExamplePluginToolRuntime:
    async def test_tool_only_config_does_not_require_codec(
        self,
        monkeypatch: pytest.MonkeyPatch,
        guardrails_kind: str,
    ) -> None:
        _install_fake_guardrails(monkeypatch, [FakeGuardrailsResult(FakeRailStatus.PASSED)])
        ngr.register(guardrails_kind)
        report = await plugin.initialize(
            plugin.PluginConfig(
                components=[
                    plugin.ComponentSpec(
                        kind=guardrails_kind,
                        config=cast(
                            JsonObject,
                            {
                                "config_yaml": "rails: {}\n",
                                "input": False,
                                "output": False,
                                "tool_input": True,
                            },
                        ),
                    )
                ]
            )
        )
        assert report["diagnostics"] == []

        async def tool_impl(args):
            return {"result": args["query"].upper()}

        result = await tools.execute("search", {"query": "hello"}, tool_impl)

        assert result == {"result": "HELLO"}
        assert FakeRails.instances[0].calls == [
            (
                [{"role": "user", "content": '{"arguments":{"query":"hello"},"tool_name":"search"}'}],
                [FakeRailType.INPUT],
            )
        ]

    async def test_tool_input_pass_calls_tool(
        self,
        monkeypatch: pytest.MonkeyPatch,
        guardrails_kind: str,
    ) -> None:
        await _activate(
            monkeypatch,
            guardrails_kind,
            [FakeGuardrailsResult(FakeRailStatus.PASSED)],
            input=False,
            output=False,
            tool_input=True,
        )
        seen_args = []

        async def tool_impl(args):
            seen_args.append(args)
            return {"result": args["query"].upper()}

        result = await tools.execute("search", {"query": "hello"}, tool_impl)

        assert result == {"result": "HELLO"}
        assert seen_args == [{"query": "hello"}]
        assert FakeRails.instances[0].calls == [
            (
                [{"role": "user", "content": '{"arguments":{"query":"hello"},"tool_name":"search"}'}],
                [FakeRailType.INPUT],
            )
        ]

    async def test_tool_input_block_stops_before_tool(
        self,
        monkeypatch: pytest.MonkeyPatch,
        guardrails_kind: str,
    ) -> None:
        await _activate(
            monkeypatch,
            guardrails_kind,
            [FakeGuardrailsResult(FakeRailStatus.BLOCKED, rail="tool policy")],
            input=False,
            output=False,
            tool_input=True,
        )
        tool_called = False

        async def tool_impl(_args):
            nonlocal tool_called
            tool_called = True
            return {"result": "unreachable"}

        with pytest.raises(RuntimeError, match="tool_input rail blocked"):
            await tools.execute("search", {"query": "secret"}, tool_impl)

        assert tool_called is False

    async def test_tool_input_modified_rewrites_tool_args(
        self,
        monkeypatch: pytest.MonkeyPatch,
        guardrails_kind: str,
    ) -> None:
        await _activate(
            monkeypatch,
            guardrails_kind,
            [
                FakeGuardrailsResult(
                    FakeRailStatus.MODIFIED,
                    content='{"tool_name":"search","arguments":{"query":"safe"}}',
                )
            ],
            input=False,
            output=False,
            tool_input=True,
        )
        seen_args = []

        async def tool_impl(args):
            seen_args.append(args)
            return {"query": args["query"]}

        result = await tools.execute("search", {"query": "unsafe"}, tool_impl)

        assert result == {"query": "safe"}
        assert seen_args == [{"query": "safe"}]

    async def test_tool_input_modified_requires_arguments_field(
        self,
        monkeypatch: pytest.MonkeyPatch,
        guardrails_kind: str,
    ) -> None:
        await _activate(
            monkeypatch,
            guardrails_kind,
            [
                FakeGuardrailsResult(
                    FakeRailStatus.MODIFIED,
                    content='{"tool_name":"search","result":{"query":"safe"}}',
                )
            ],
            input=False,
            output=False,
            tool_input=True,
        )

        async def tool_impl(_args):
            return {"result": "unreachable"}

        with pytest.raises(RuntimeError, match="without a 'arguments' field"):
            await tools.execute("search", {"query": "unsafe"}, tool_impl)

    async def test_tool_output_block_raises_after_tool(
        self,
        monkeypatch: pytest.MonkeyPatch,
        guardrails_kind: str,
    ) -> None:
        await _activate(
            monkeypatch,
            guardrails_kind,
            [FakeGuardrailsResult(FakeRailStatus.BLOCKED, rail="tool result policy")],
            input=False,
            output=False,
            tool_output=True,
        )
        tool_called = False

        async def tool_impl(_args):
            nonlocal tool_called
            tool_called = True
            return {"result": "unsafe"}

        with pytest.raises(RuntimeError, match="tool_output rail blocked"):
            await tools.execute("search", {"query": "hello"}, tool_impl)

        assert tool_called is True

    async def test_tool_output_modified_rewrites_tool_result(
        self,
        monkeypatch: pytest.MonkeyPatch,
        guardrails_kind: str,
    ) -> None:
        await _activate(
            monkeypatch,
            guardrails_kind,
            [
                FakeGuardrailsResult(
                    FakeRailStatus.MODIFIED,
                    content='{"tool_name":"search","result":{"result":"safe"}}',
                )
            ],
            input=False,
            output=False,
            tool_output=True,
        )

        async def tool_impl(_args):
            return {"result": "unsafe"}

        result = await tools.execute("search", {"query": "hello"}, tool_impl)

        assert result == {"result": "safe"}
