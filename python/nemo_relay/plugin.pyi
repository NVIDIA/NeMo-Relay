# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import AsyncContextManager, Literal, Protocol, TypedDict

from nemo_relay import (
    Event,
    JsonObject,
    LlmConditionalExecutionGuardrail,
    LlmExecutionIntercept,
    LlmRequestIntercept,
    LlmSanitizeRequestGuardrail,
    LlmSanitizeResponseGuardrail,
    LlmStreamExecutionIntercept,
    ToolConditionalExecutionGuardrail,
    ToolExecutionIntercept,
    ToolRequestIntercept,
    ToolSanitizeGuardrail,
)

UnsupportedBehavior = Literal["ignore", "warn", "error"]

class _ConfigDiagnosticRequired(TypedDict):
    level: Literal["warning", "error"]
    code: str
    message: str

class ConfigDiagnostic(_ConfigDiagnosticRequired, total=False):
    component: str
    field: str

class ConfigReport(TypedDict):
    diagnostics: list[ConfigDiagnostic]

class PluginContext(Protocol):
    def register_subscriber(self, name: str, callback: Callable[[Event], None]) -> None: ...
    def register_tool_sanitize_request_guardrail(
        self, name: str, priority: int, callback: ToolSanitizeGuardrail
    ) -> None: ...
    def register_tool_sanitize_response_guardrail(
        self, name: str, priority: int, callback: ToolSanitizeGuardrail
    ) -> None: ...
    def register_tool_conditional_execution_guardrail(
        self, name: str, priority: int, callback: ToolConditionalExecutionGuardrail
    ) -> None: ...
    def register_llm_sanitize_request_guardrail(
        self, name: str, priority: int, callback: LlmSanitizeRequestGuardrail
    ) -> None: ...
    def register_llm_sanitize_response_guardrail(
        self, name: str, priority: int, callback: LlmSanitizeResponseGuardrail
    ) -> None: ...
    def register_llm_conditional_execution_guardrail(
        self, name: str, priority: int, callback: LlmConditionalExecutionGuardrail
    ) -> None: ...
    def register_llm_request_intercept(
        self, name: str, priority: int, break_chain: bool, callback: LlmRequestIntercept
    ) -> None: ...
    def register_llm_execution_intercept(self, name: str, priority: int, callback: LlmExecutionIntercept) -> None: ...
    def register_llm_stream_execution_intercept(
        self, name: str, priority: int, callback: LlmStreamExecutionIntercept
    ) -> None: ...
    def register_tool_request_intercept(
        self, name: str, priority: int, break_chain: bool, callback: ToolRequestIntercept
    ) -> None: ...
    def register_tool_execution_intercept(self, name: str, priority: int, callback: ToolExecutionIntercept) -> None: ...

class Plugin(Protocol):
    def validate(self, plugin_config: JsonObject) -> list[ConfigDiagnostic] | None: ...
    def register(self, plugin_config: JsonObject, context: PluginContext) -> None: ...

class ConfigPolicy:
    unknown_component: UnsupportedBehavior
    unknown_field: UnsupportedBehavior
    unsupported_value: UnsupportedBehavior

    def __init__(
        self,
        unknown_component: UnsupportedBehavior = "warn",
        unknown_field: UnsupportedBehavior = "warn",
        unsupported_value: UnsupportedBehavior = "error",
    ) -> None: ...
    def to_dict(self) -> JsonObject: ...

class ComponentSpec:
    kind: str
    enabled: bool
    config: JsonObject

    def __init__(
        self,
        kind: str,
        enabled: bool = True,
        config: JsonObject = ...,
    ) -> None: ...
    def to_dict(self) -> JsonObject: ...

class PluginConfig:
    version: int
    components: list[object]
    policy: ConfigPolicy

    def __init__(
        self,
        version: int = 1,
        components: list[object] = ...,
        policy: ConfigPolicy = ...,
    ) -> None: ...
    def to_dict(self) -> JsonObject: ...

def validate(config: PluginConfig | JsonObject) -> ConfigReport: ...
async def initialize(config: PluginConfig | JsonObject) -> ConfigReport: ...
def clear() -> None: ...
def plugin(config: PluginConfig | JsonObject) -> AsyncContextManager[ConfigReport]: ...
def report() -> ConfigReport | None: ...
def list_kinds() -> list[str]: ...
def register(plugin_kind: str, plugin: Plugin) -> None: ...
def deregister(plugin_kind: str) -> bool: ...
