# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import Literal, Protocol, Self, TypedDict

from nemo_relay import (
    Event,
    EventMetadataInjectorCallback,
    EventSanitizeGuardrail,
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
from nemo_relay.runtime_registrations import ConditionalMiddlewareGuardrail, RuntimeRegistrationKind

UnsupportedBehavior = Literal["ignore", "warn", "error"]
DynamicPluginKind = Literal["rust_dynamic", "worker"]

class _ConfigDiagnosticRequired(TypedDict):
    level: Literal["warning", "error"]
    code: str
    message: str

class ConfigDiagnostic(_ConfigDiagnosticRequired, total=False):
    component: str
    field: str

class _RuntimeDiagnosticRequired(TypedDict):
    code: str
    component: str
    message: str
    count: int

class RuntimeDiagnostic(_RuntimeDiagnosticRequired, total=False):
    field: str
    session_id: str

class ConfigReport(TypedDict):
    diagnostics: list[ConfigDiagnostic]
    runtime_diagnostics: list[RuntimeDiagnostic]

DynamicPluginCheckState = Literal["unknown", "valid", "invalid"]

class _DynamicPluginValidationStatusRequired(TypedDict):
    manifest: DynamicPluginCheckState
    compatibility: DynamicPluginCheckState
    integrity: DynamicPluginCheckState
    environment: DynamicPluginCheckState
    authenticity: DynamicPluginCheckState
    policy_satisfied: DynamicPluginCheckState

class DynamicPluginValidationStatus(_DynamicPluginValidationStatusRequired, total=False):
    checked_at: str | None
    message: str | None

class DynamicPluginFailure(TypedDict):
    phase: str
    code: str
    message: str

class _DynamicPluginValidationReportRequired(TypedDict):
    plugin_id: str
    manifest_ref: str
    kind: DynamicPluginKind
    status: DynamicPluginValidationStatus
    selected: bool

class DynamicPluginValidationReport(_DynamicPluginValidationReportRequired, total=False):
    failure: DynamicPluginFailure | None

class PluginHostReport(TypedDict):
    config: ConfigReport
    dynamic_plugins: list[DynamicPluginValidationReport]

class PluginContext(Protocol):
    def register_subscriber(self, name: str, callback: Callable[[Event], None]) -> None: ...
    def register_conditional_middleware_guardrail(
        self,
        name: str,
        kinds: set[RuntimeRegistrationKind],
        registration_name: str,
        guardrail: ConditionalMiddlewareGuardrail,
    ) -> None: ...
    def register_event_metadata_injector(
        self, name: str, priority: int, callback: EventMetadataInjectorCallback
    ) -> None: ...
    def register_mark_sanitize_guardrail(self, name: str, priority: int, callback: EventSanitizeGuardrail) -> None: ...
    def register_scope_sanitize_start_guardrail(
        self, name: str, priority: int, callback: EventSanitizeGuardrail
    ) -> None: ...
    def register_scope_sanitize_end_guardrail(
        self, name: str, priority: int, callback: EventSanitizeGuardrail
    ) -> None: ...
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

class PluginHostActivation:
    @property
    def report(self) -> PluginHostReport: ...
    @property
    def is_active(self) -> bool: ...
    async def close(self) -> None: ...
    async def __aenter__(self) -> Self: ...
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

async def initialize(
    config: PluginConfig | JsonObject,
    additional_plugins_toml: str | os.PathLike[str] | None = None,
) -> PluginHostActivation: ...
def activate(
    config: PluginConfig | JsonObject,
    additional_plugins_toml: str | os.PathLike[str] | None = None,
) -> AbstractAsyncContextManager[PluginHostActivation]: ...
def validate(
    config: PluginConfig | JsonObject,
    additional_plugins_toml: str | os.PathLike[str] | None = None,
) -> PluginHostReport: ...
def validate_exact(config: PluginConfig | JsonObject) -> PluginHostReport: ...
def list_kinds() -> list[str]: ...
def register(plugin_kind: str, plugin: Plugin) -> None: ...
def deregister(plugin_kind: str) -> bool: ...
