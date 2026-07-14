# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stubs for the compiled ``nemo_relay._native`` extension module.

Summary:
    Type and documentation surface for the compiled Rust extension.

Description:
    The native module owns scope-stack state, lifecycle handles, event objects,
    observability exporters, codec bridges, middleware registries, plugin
    activation, and adaptive runtime hooks. Python wrapper modules generally
    provide user-facing names and examples, while this file records the native
    API shape that those wrappers import.

Exceptional flow:
    Native functions may raise exceptions produced by the Rust runtime, by
    Python callbacks invoked from middleware, or by invalid JSON-like values.
    Individual declarations call out the main exceptional paths where the
    behavior is specific.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Generator, Mapping, Sequence
from datetime import datetime
from typing import ClassVar, Literal, Optional, TypeAlias, TypedDict

_JsonPrimitive: TypeAlias = str | int | float | bool | None
_JsonValue: TypeAlias = _JsonPrimitive | list["_JsonValue"] | dict[str, "_JsonValue"]
_JsonObject: TypeAlias = dict[str, _JsonValue]
_Json: TypeAlias = _JsonValue

class _EventSanitizeFields(TypedDict):
    data: _Json | None
    category_profile: _JsonObject | None
    metadata: _Json | None

_ToolSanitizeGuardrail: TypeAlias = Callable[[str, _Json], _Json]
_ToolConditionalExecutionGuardrail: TypeAlias = Callable[[str, _Json], Optional[str]]
_LlmSanitizeRequestGuardrail: TypeAlias = Callable[["LLMRequest"], "LLMRequest"]
_LlmSanitizeResponseGuardrail: TypeAlias = Callable[[_JsonObject], _JsonObject]
_EventSanitizeGuardrail: TypeAlias = Callable[[ScopeEvent | MarkEvent, _EventSanitizeFields], _EventSanitizeFields]
_LlmConditionalExecutionGuardrail: TypeAlias = Callable[["LLMRequest"], Optional[str]]
_ToolRequestIntercept: TypeAlias = Callable[[str, _Json], _Json]
_ToolExecutionIntercept: TypeAlias = Callable[
    [str, _Json, Callable[[_Json], Awaitable[_Json]]],
    "ToolExecutionInterceptOutcome | Awaitable[ToolExecutionInterceptOutcome]",
]
_LlmRequestIntercept: TypeAlias = Callable[
    [str, "LLMRequest", "AnnotatedLLMRequest | None"],
    "LLMRequestInterceptOutcome",
]
_LlmExecutionIntercept: TypeAlias = Callable[
    [str, "LLMRequest", Callable[["LLMRequest"], Awaitable[_Json]]],
    _Json | Awaitable[_Json],
]
_LlmStreamExecutionIntercept: TypeAlias = Callable[
    ["LLMRequest", Callable[["LLMRequest"], Awaitable[AsyncIterator[_Json]]]],
    AsyncIterator[_Json] | Awaitable[AsyncIterator[_Json]],
]

class ScopeAttributes:
    """Bitflags describing scope properties.

    Summary:
        Compact flag container attached to scope handles and scope events.

    Description:
        ``ScopeAttributes`` values can be created from a raw integer bitmask and
        combined with ``|`` or intersected with ``&``. The native runtime uses
        these flags to record semantic execution properties on scopes.

    Flag constants:
        ``PARALLEL`` indicates child work may execute in parallel.
        ``RELOCATABLE`` indicates work may move to another execution context.

    Exceptional flow:
        Construction or bit operations may raise native type errors when values
        cannot be converted to the expected bitmask representation.
    """

    PARALLEL: int
    RELOCATABLE: int

    def __init__(self, value: int = 0) -> None:
        """Create a scope-attribute bitmask.

        Args:
            value: Raw integer bitmask. Defaults to no flags.

        Returns:
            ``None``.
        """
        ...
    @property
    def is_parallel(self) -> bool:
        """Return whether the ``PARALLEL`` flag is set."""
        ...
    @property
    def is_relocatable(self) -> bool:
        """Return whether the ``RELOCATABLE`` flag is set."""
        ...
    @property
    def value(self) -> int:
        """Return the raw integer bitmask."""
        ...
    def __or__(self, other: ScopeAttributes) -> ScopeAttributes:
        """Return the union of two scope-attribute bitmasks."""
        ...
    def __and__(self, other: ScopeAttributes) -> ScopeAttributes:
        """Return the intersection of two scope-attribute bitmasks."""
        ...

class ToolAttributes:
    """Bitflags describing tool call properties.

    Summary:
        Compact flag container attached to tool handles and tool events.

    Description:
        ``ToolAttributes`` records semantic properties of a tool call. Values
        can be created from raw integer bitmasks and combined with bit
        operators.

    Flag constants:
        ``REMOTE`` indicates that the tool call is remote.
    """

    REMOTE: int

    def __init__(self, value: int = 0) -> None:
        """Create a tool-attribute bitmask.

        Args:
            value: Raw integer bitmask. Defaults to no flags.

        Returns:
            ``None``.
        """
        ...
    @property
    def is_remote(self) -> bool:
        """Return whether the ``REMOTE`` flag is set."""
        ...
    @property
    def value(self) -> int:
        """Return the raw integer bitmask."""
        ...
    def __or__(self, other: ToolAttributes) -> ToolAttributes:
        """Return the union of two tool-attribute bitmasks."""
        ...
    def __and__(self, other: ToolAttributes) -> ToolAttributes:
        """Return the intersection of two tool-attribute bitmasks."""
        ...

class LLMAttributes:
    """Bitflags describing LLM call properties.

    Summary:
        Compact flag container attached to LLM handles and LLM events.

    Description:
        ``LLMAttributes`` records semantic properties of an LLM call. Values can
        be created from raw integer bitmasks and combined with bit operators.

    Flag constants:
        ``STATEFUL`` indicates that the LLM call uses stateful context.
        ``STREAMING`` indicates that the LLM call returns a stream.
    """

    STATEFUL: int
    STREAMING: int

    def __init__(self, value: int = 0) -> None:
        """Create an LLM-attribute bitmask.

        Args:
            value: Raw integer bitmask. Defaults to no flags.

        Returns:
            ``None``.
        """
        ...
    @property
    def is_stateful(self) -> bool:
        """Return whether the ``STATEFUL`` flag is set."""
        ...
    @property
    def is_streaming(self) -> bool:
        """Return whether the ``STREAMING`` flag is set."""
        ...
    @property
    def value(self) -> int:
        """Return the raw integer bitmask."""
        ...
    def __or__(self, other: LLMAttributes) -> LLMAttributes:
        """Return the union of two LLM-attribute bitmasks."""
        ...
    def __and__(self, other: LLMAttributes) -> LLMAttributes:
        """Return the intersection of two LLM-attribute bitmasks."""
        ...

class ScopeType:
    """Enum identifying the kind of execution scope.

    Summary:
        Native scope category used when creating scopes.

    Description:
        The selected ``ScopeType`` is recorded on handles and emitted events so
        subscribers can distinguish agents, tools, LLM calls, guardrails, and
        other semantic units of work.
    """

    Agent: ScopeType
    """Autonomous agent scope."""
    Function: ScopeType
    """Generic function-call scope."""
    Tool: ScopeType
    """Tool invocation scope."""
    Llm: ScopeType
    """LLM call scope."""
    Retriever: ScopeType
    """Retriever or RAG lookup scope."""
    Embedder: ScopeType
    """Embedding model scope."""
    Reranker: ScopeType
    """Reranking model scope."""
    Guardrail: ScopeType
    """Guardrail evaluation scope."""
    Evaluator: ScopeType
    """Evaluator or judge scope."""
    Custom: ScopeType
    """User-defined scope type."""
    Unknown: ScopeType
    """Unknown or unspecified scope type."""

class ScopeHandle:
    """An active execution scope in the scope stack.

    Summary:
        Immutable native handle returned by scope creation APIs.

    Description:
        A ``ScopeHandle`` identifies one pushed scope. Pass it back to
        ``pop_scope`` or wrapper APIs to close the scope, attach child work, or
        register scope-local middleware.

    Exceptional flow:
        Accessing properties may propagate native errors if the handle has been
        invalidated by the runtime.
    """

    @property
    def uuid(self) -> str:
        """Return the globally unique scope identifier."""
        ...
    @property
    def name(self) -> str:
        """Return the human-readable scope name."""
        ...
    @property
    def scope_type(self) -> ScopeType:
        """Return the semantic scope type."""
        ...
    @property
    def attributes(self) -> ScopeAttributes:
        """Return the scope attribute bitmask."""
        ...
    @property
    def parent_uuid(self) -> Optional[str]:
        """Return the parent scope UUID, or ``None`` for a root scope."""
        ...
    @property
    def data(self) -> Optional[_Json]:
        """Return application data captured on the scope, if any."""
        ...
    @property
    def metadata(self) -> Optional[_Json]:
        """Return metadata captured on the scope, if any."""
        ...

class ToolHandle:
    """An active tool call.

    Summary:
        Native handle returned by manual tool-call start APIs.

    Description:
        Pass this handle to ``tool_call_end`` or wrapper APIs to close the tool
        lifecycle span and emit the corresponding end event.
    """

    @property
    def uuid(self) -> str:
        """Return the globally unique tool-call identifier."""
        ...
    @property
    def name(self) -> str:
        """Return the tool name."""
        ...
    @property
    def attributes(self) -> ToolAttributes:
        """Return the tool attribute bitmask."""
        ...
    @property
    def parent_uuid(self) -> Optional[str]:
        """Return the parent scope UUID, if any."""
        ...
    @property
    def data(self) -> Optional[_Json]:
        """Return application data captured on the tool call, if any."""
        ...
    @property
    def metadata(self) -> Optional[_Json]:
        """Return metadata captured on the tool call, if any."""
        ...

class LLMHandle:
    """An active LLM call.

    Summary:
        Native handle returned by manual LLM-call start APIs.

    Description:
        Pass this handle to ``llm_call_end`` or wrapper APIs to close the LLM
        lifecycle span and emit the corresponding end event.
    """

    @property
    def uuid(self) -> str:
        """Return the globally unique LLM-call identifier."""
        ...
    @property
    def name(self) -> str:
        """Return the LLM provider or logical call name."""
        ...
    @property
    def attributes(self) -> LLMAttributes:
        """Return the LLM attribute bitmask."""
        ...
    @property
    def parent_uuid(self) -> Optional[str]:
        """Return the parent scope UUID, if any."""
        ...
    @property
    def data(self) -> Optional[_Json]:
        """Return application data captured on the LLM call, if any."""
        ...
    @property
    def metadata(self) -> Optional[_Json]:
        """Return metadata captured on the LLM call, if any."""
        ...

class LLMRequest:
    """An LLM request carrying headers and a content payload.

    Summary:
        Provider request object passed through LLM middleware.

    Description:
        ``headers`` stores provider or transport metadata. ``content`` stores
        the JSON request body. Request intercepts and codecs consume this
        object to normalize, rewrite, and execute LLM calls.
    """

    def __init__(self, headers: Mapping[str, _JsonValue], content: _JsonObject) -> None:
        """Create an LLM request.

        Args:
            headers: Header-like metadata mapping.
            content: JSON object payload sent to the provider.

        Returns:
            ``None``.

        Exceptional flow:
            Raises native conversion errors if ``headers`` or ``content`` are
            not JSON-compatible mappings.
        """
        ...
    @property
    def headers(self) -> _JsonObject:
        """Return the request headers as a JSON object."""
        ...
    @property
    def content(self) -> _JsonObject:
        """Return the request content body as a JSON object."""
        ...

class PendingMarkSpec:
    """A runtime-owned mark specification returned by lifecycle middleware."""
    def __init__(
        self,
        name: str,
        category: Optional[str] = ...,
        category_profile: Optional[_Json] = ...,
        data: Optional[_Json] = ...,
        metadata: Optional[_Json] = ...,
    ) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def category(self) -> Optional[str]: ...
    @property
    def category_profile(self) -> Optional[_Json]: ...
    @property
    def data(self) -> Optional[_Json]: ...
    @property
    def metadata(self) -> Optional[_Json]: ...

class LLMRequestInterceptOutcome:
    """Canonical result returned by an LLM request intercept."""
    def __init__(
        self,
        request: LLMRequest,
        annotated_request: Optional[AnnotatedLLMRequest] = ...,
        pending_marks: list[PendingMarkSpec] = ...,
        optimization_contributions: Optional[Sequence[Mapping[str, _JsonValue]]] = ...,
    ) -> None: ...
    @property
    def request(self) -> LLMRequest: ...
    @property
    def annotated_request(self) -> Optional[AnnotatedLLMRequest]: ...
    @property
    def pending_marks(self) -> list[PendingMarkSpec]: ...
    @property
    def optimization_contributions(self) -> list[_JsonObject]:
        """Return ordered plugin-neutral optimization contribution objects."""
        ...

class ToolExecutionInterceptOutcome:
    """Canonical result returned by a tool execution intercept.

    ``result`` is passed to the remaining middleware and application.
    ``pending_marks`` are Relay-owned lifecycle metadata emitted after the
    tool-end event and are not included in the application-visible result.
    """
    def __init__(
        self,
        result: _Json,
        pending_marks: list[PendingMarkSpec] = ...,
    ) -> None: ...
    @property
    def result(self) -> _Json: ...
    @property
    def pending_marks(self) -> list[PendingMarkSpec]: ...

class AnnotatedLLMRequest:
    """Structured view of an LLM request produced by a codec.

    Summary:
        Provider-neutral request view for annotated LLM middleware.

    Description:
        Codecs decode provider-specific request bodies into this normalized
        shape so request intercepts can operate on messages, model name,
        parameters, tools, and extra provider fields.
    """

    def __init__(
        self,
        messages: Sequence[Mapping[str, _JsonValue]],
        *,
        model: Optional[str] = None,
        params: Optional[Mapping[str, _JsonValue]] = None,
        tools: Optional[Sequence[Mapping[str, _JsonValue]]] = None,
        tool_choice: Optional[str | Mapping[str, _JsonValue]] = None,
        extra: Optional[Mapping[str, _JsonValue]] = None,
    ) -> None:
        """Create a normalized LLM request view.

        Args:
            messages: Provider-normalized message objects.
            model: Optional model name.
            params: Optional provider parameters.
            tools: Optional tool declarations.
            tool_choice: Optional tool-selection directive.
            extra: Optional provider-specific fields.

        Returns:
            ``None``.

        Exceptional flow:
            Raises conversion errors when JSON-like inputs cannot be converted
            to the native normalized representation.
        """
        ...
    @property
    def messages(self) -> list[_JsonObject]:
        """Return normalized message objects."""
        ...
    @messages.setter
    def messages(self, value: Sequence[Mapping[str, _JsonValue]]) -> None:
        """Replace normalized message objects."""
        ...
    @property
    def model(self) -> Optional[str]:
        """Return the normalized model name, if present."""
        ...
    @model.setter
    def model(self, value: Optional[str]) -> None:
        """Set or clear the normalized model name."""
        ...
    @property
    def params(self) -> Optional[_JsonObject]:
        """Return provider parameters, if present."""
        ...
    @params.setter
    def params(self, value: Optional[Mapping[str, _JsonValue]]) -> None:
        """Set or clear provider parameters."""
        ...
    @property
    def tools(self) -> Optional[list[_JsonObject]]:
        """Return normalized tool declarations, if present."""
        ...
    @tools.setter
    def tools(self, value: Optional[Sequence[Mapping[str, _JsonValue]]]) -> None:
        """Set or clear normalized tool declarations."""
        ...
    @property
    def tool_choice(self) -> Optional[str | _JsonObject]:
        """Return the normalized tool-choice directive, if present."""
        ...
    @tool_choice.setter
    def tool_choice(self, value: Optional[str | Mapping[str, _JsonValue]]) -> None:
        """Set or clear the normalized tool-choice directive."""
        ...
    @property
    def extra(self) -> _JsonObject:
        """Return provider-specific request fields."""
        ...
    @extra.setter
    def extra(self, value: Mapping[str, _JsonValue]) -> None:
        """Replace provider-specific request fields."""
        ...
    def system_prompt(self) -> Optional[str]:
        """Return the first normalized system prompt, if one is present."""
        ...
    def last_user_message(self) -> Optional[str]:
        """Return the last normalized user message text, if one is present."""
        ...
    def has_tool_calls(self) -> bool:
        """Return whether the normalized request includes tool declarations."""
        ...

class AnnotatedLLMResponse:
    """Structured view of an LLM response produced by a response codec.

    Summary:
        Provider-neutral response view for LLM end-event annotations.

    Description:
        Response codecs decode provider responses into this normalized shape so
        subscribers can inspect model, text, tool-call, usage, and
        provider-specific fields consistently.
    """

    def __init__(
        self,
        id: Optional[str] = None,
        model: Optional[str] = None,
        message: Optional[_JsonValue] = None,
        tool_calls: Optional[Sequence[Mapping[str, _JsonValue]]] = None,
        finish_reason: Optional[str | Mapping[str, _JsonValue]] = None,
        usage: Optional[Mapping[str, _JsonValue]] = None,
        api_specific: Optional[Mapping[str, _JsonValue]] = None,
        extra: Optional[Mapping[str, _JsonValue]] = None,
    ) -> None:
        """Create a normalized LLM response view.

        Args:
            id: Optional provider response identifier.
            model: Optional model that served the response.
            message: Optional normalized assistant response content.
            tool_calls: Optional normalized response tool-call payloads.
            finish_reason: Optional normalized finish reason.
            usage: Optional normalized usage accounting.
            api_specific: Optional API-specific response fields.
            extra: Optional additional response fields.

        Returns:
            ``None``.

        Exceptional flow:
            Raises conversion errors when JSON-like inputs cannot be converted
            to the native normalized representation.
        """
        ...
    @property
    def id(self) -> Optional[str]:
        """Return the provider response identifier, if present."""
        ...
    @property
    def model(self) -> Optional[str]:
        """Return the provider model name, if present."""
        ...
    @property
    def message(self) -> Optional[_Json]:
        """Return the normalized primary message payload, if present."""
        ...
    @property
    def tool_calls(self) -> Optional[list[_JsonObject]]:
        """Return normalized tool-call payloads, if present."""
        ...
    @property
    def finish_reason(self) -> Optional[str]:
        """Return the provider finish reason, if present."""
        ...
    @property
    def usage(self) -> Optional[_JsonObject]:
        """Return normalized usage accounting, if present."""
        ...
    @property
    def optimization_summary(self) -> Optional[_JsonObject]:
        """Return Relay's close-time optimization accounting, if present."""
        ...
    @property
    def api_specific(self) -> Optional[_JsonObject]:
        """Return provider-specific response fields, if present."""
        ...
    @property
    def extra(self) -> _JsonObject:
        """Return additional normalized response fields."""
        ...
    def response_text(self) -> Optional[str]:
        """Return extracted response text, if present."""
        ...
    def has_tool_calls(self) -> bool:
        """Return whether the response contains tool-call payloads."""
        ...

class ScopeEvent:
    """ATOF scope lifecycle event emitted to subscribers.

    Summary:
        Event emitted when a scope, tool call, or LLM call starts or ends.

    Description:
        Scope events contain hierarchy identifiers, semantic category data,
        user payloads, metadata, and optional normalized LLM request/response
        annotations.
    """

    @property
    def kind(self) -> Literal["scope"]:
        """Return the discriminant value ``"scope"``."""
        ...
    @property
    def scope_category(self) -> Literal["start", "end"]:
        """Return whether this is a start or end lifecycle event."""
        ...
    @property
    def atof_version(self) -> str:
        """Return the ATOF schema version used for this event."""
        ...
    @property
    def parent_uuid(self) -> Optional[str]:
        """Return the parent event UUID, if present."""
        ...
    @property
    def uuid(self) -> str:
        """Return the event UUID."""
        ...
    @property
    def timestamp(self) -> str:
        """Return the event timestamp as an RFC 3339 string."""
        ...
    @property
    def name(self) -> str:
        """Return the event or scope name."""
        ...
    @property
    def data(self) -> Optional[_Json]:
        """Return application data attached to the event, if any."""
        ...
    @property
    def metadata(self) -> Optional[_Json]:
        """Return metadata attached to the event, if any."""
        ...
    @property
    def attributes(self) -> list[str]:
        """Return stringified semantic attributes attached to the event."""
        ...
    @property
    def category(self) -> str:
        """Return the semantic event category."""
        ...
    @property
    def category_profile(self) -> Optional[_JsonObject]:
        """Return category-specific profile data, if any."""
        ...
    @property
    def data_schema(self) -> Optional[_JsonObject]:
        """Return a schema descriptor for ``data``, if one is present."""
        ...
    @property
    def annotated_request(self) -> Optional[AnnotatedLLMRequest]:
        """Return the normalized LLM request annotation, if present."""
        ...
    @property
    def annotated_response(self) -> Optional[AnnotatedLLMResponse]:
        """Return the normalized LLM response annotation, if present."""
        ...
    def to_dict(self) -> _JsonObject:
        """Return this event as the canonical subscriber JSON dictionary."""
        ...
    def to_json(self) -> str:
        """Return this event as canonical subscriber JSON."""
        ...

class MarkEvent:
    """ATOF point-in-time mark event emitted to subscribers.

    Summary:
        Event emitted for standalone marks and guardrail rejections.

    Description:
        Mark events record a named point in time under the current scope
        hierarchy. They do not have a start/end lifecycle pair.
    """

    @property
    def kind(self) -> Literal["mark"]:
        """Return the discriminant value ``"mark"``."""
        ...
    @property
    def atof_version(self) -> str:
        """Return the ATOF schema version used for this event."""
        ...
    @property
    def parent_uuid(self) -> Optional[str]:
        """Return the parent event UUID, if present."""
        ...
    @property
    def uuid(self) -> str:
        """Return the event UUID."""
        ...
    @property
    def timestamp(self) -> str:
        """Return the event timestamp as an RFC 3339 string."""
        ...
    @property
    def name(self) -> str:
        """Return the mark event name."""
        ...
    @property
    def data(self) -> Optional[_Json]:
        """Return application data attached to the event, if any."""
        ...
    @property
    def metadata(self) -> Optional[_Json]:
        """Return metadata attached to the event, if any."""
        ...
    @property
    def category(self) -> Optional[str]:
        """Return the semantic mark category, if present."""
        ...
    @property
    def category_profile(self) -> Optional[_JsonObject]:
        """Return category-specific profile data, if any."""
        ...
    @property
    def data_schema(self) -> Optional[_JsonObject]:
        """Return a schema descriptor for ``data``, if one is present."""
        ...
    def to_dict(self) -> _JsonObject:
        """Return this event as the canonical subscriber JSON dictionary."""
        ...
    def to_json(self) -> str:
        """Return this event as canonical subscriber JSON."""
        ...

class AtifExporter:
    """ATIF trajectory exporter that collects events and exports trajectories.

    Summary:
        Subscriber implementation that accumulates lifecycle events.

    Description:
        Register the exporter under a subscriber name, run application code, and
        export the collected event set as an ATIF trajectory dictionary or JSON
        string.
    """

    def __init__(
        self,
        session_id: str,
        agent_name: str,
        agent_version: str,
        *,
        model_name: Optional[str] = None,
        tool_definitions: Optional[list[_JsonObject]] = None,
        extra: Optional[_Json] = None,
    ) -> None:
        """Create an ATIF exporter.

        Args:
            session_id: Stable session identifier for exported trajectories.
            agent_name: Human-readable agent name.
            agent_version: Agent version string.
            model_name: Optional primary model name.
            tool_definitions: Optional tool schema objects.
            extra: Optional additional JSON-compatible trajectory metadata.

        Returns:
            ``None``.
        """
        ...
    def register(self, name: str) -> None:
        """Register this exporter as a global event subscriber."""
        ...
    def deregister(self, name: str) -> bool:
        """Deregister this exporter and return whether a subscriber was removed."""
        ...
    def export(self) -> _JsonObject:
        """Return collected events as an ATIF trajectory object."""
        ...
    def export_json(self) -> str:
        """Return collected events as an ATIF trajectory JSON string."""
        ...
    def clear(self) -> None:
        """Clear collected events without changing subscriber registration."""
        ...

class AtofExporterMode:
    """File write mode for ``AtofExporter``."""

    Append: ClassVar[AtofExporterMode]
    Overwrite: ClassVar[AtofExporterMode]

class AtofEndpointConfig:
    """Streaming destination for raw ATOF events."""

    url: str
    transport: str
    headers: dict[str, str]
    timeout_millis: int
    field_name_policy: str

    def __init__(
        self,
        url: str,
        *,
        transport: str = "http_post",
        headers: dict[str, str] | None = None,
        timeout_millis: int = 3000,
        field_name_policy: str = "preserve",
    ) -> None:
        """Create an ATOF streaming endpoint config.

        ``headers=None`` is converted to an empty dict; the instance field is
        always non-optional.
        """

class AtofExporterConfig:
    """Mutable configuration for the filesystem-backed ATOF JSONL exporter."""

    output_directory: str
    mode: AtofExporterMode
    filename: str
    endpoints: list[AtofEndpointConfig]

    def __init__(self) -> None:
        """Create an ATOF exporter config with native defaults."""
        ...

class AtofExporter:
    """Filesystem-backed exporter that writes raw ATOF events as JSONL."""

    def __init__(self, config: AtofExporterConfig) -> None:
        """Create an ATOF JSONL exporter from config."""
        ...
    @property
    def path(self) -> str:
        """Return the JSONL output path."""
        ...
    def register(self, name: str) -> None:
        """Register the exporter under ``name``."""
        ...
    def deregister(self, name: str) -> bool:
        """Deregister ``name`` and return whether it existed."""
        ...
    def force_flush(self) -> None:
        """Flush the output file."""
        ...
    def shutdown(self) -> None:
        """Flush the output file before shutdown."""
        ...

class ScopeStack:
    """An isolated scope stack for per-request or per-task isolation.

    Summary:
        Native stack object containing active scope hierarchy state.

    Description:
        Scope stacks are installed into Python context variables or native
        thread-local storage so nested scope, tool, and LLM events share the
        correct hierarchy.
    """

    def __repr__(self) -> str:
        """Return a debug representation of the native scope stack."""
        ...

class LlmStream:
    """An async iterator of JSON chunks from a streaming LLM response.

    Summary:
        Native stream returned by managed streaming LLM execution.

    Description:
        The object can be awaited by compatibility wrappers and iterated with
        ``async for`` to receive post-intercept JSON chunks.
    """

    def __await__(self) -> Generator[object, None, LlmStream]:
        """Return an awaitable that resolves to this stream."""
        ...
    def __aiter__(self) -> AsyncIterator[_Json]:
        """Return the async iterator for stream chunks."""
        ...
    async def __anext__(self) -> _Json:
        """Return the next JSON chunk or raise ``StopAsyncIteration``."""
        ...

class OpenTelemetryConfig:
    """Mutable configuration for ``OpenTelemetrySubscriber``.

    Summary:
        Native OpenTelemetry exporter configuration object.

    Description:
        Configure transport, endpoint, service identity, exporter timeout,
        headers, and resource attributes before constructing a subscriber.
    """

    transport: str
    endpoint: Optional[str]
    service_name: str
    service_namespace: Optional[str]
    service_version: Optional[str]
    instrumentation_scope: str
    timeout_millis: int

    def __init__(self) -> None:
        """Create an OpenTelemetry config with native defaults."""
        ...
    @property
    def headers(self) -> dict[str, str]:
        """Return additional exporter headers."""
        ...
    @headers.setter
    def headers(self, value: dict[str, str]) -> None:
        """Replace additional exporter headers."""
        ...
    @property
    def resource_attributes(self) -> dict[str, str]:
        """Return additional OpenTelemetry resource attributes."""
        ...
    @resource_attributes.setter
    def resource_attributes(self, value: dict[str, str]) -> None:
        """Replace additional OpenTelemetry resource attributes."""
        ...
    def set_header(self, key: str, value: str) -> None:
        """Set one exporter header key/value pair."""
        ...
    def set_resource_attribute(self, key: str, value: str) -> None:
        """Set one OpenTelemetry resource attribute key/value pair."""
        ...

class OpenTelemetrySubscriber:
    """OpenTelemetry-backed NeMo Relay event subscriber.

    Summary:
        Native subscriber that exports lifecycle events as OpenTelemetry spans.

    Description:
        Register the subscriber under a name to receive runtime events. Flush or
        shut it down before process exit when deterministic export is required.
    """

    def __init__(self, config: OpenTelemetryConfig) -> None:
        """Create a subscriber from an OpenTelemetry config."""
        ...
    def register(self, name: str) -> None:
        """Register the subscriber under ``name``."""
        ...
    def deregister(self, name: str) -> bool:
        """Deregister ``name`` and return whether it existed."""
        ...
    def force_flush(self) -> None:
        """Flush pending telemetry through the configured exporter."""
        ...
    def shutdown(self) -> None:
        """Shut down native OpenTelemetry resources."""
        ...

class OpenInferenceConfig:
    """Mutable configuration for ``OpenInferenceSubscriber``.

    Summary:
        Native OpenInference exporter configuration object.

    Description:
        Configure transport, endpoint, service identity, exporter timeout,
        headers, and resource attributes before constructing a subscriber.
    """

    transport: str
    endpoint: Optional[str]
    service_name: str
    service_namespace: Optional[str]
    service_version: Optional[str]
    instrumentation_scope: str
    timeout_millis: int

    def __init__(self) -> None:
        """Create an OpenInference config with native defaults."""
        ...
    @property
    def headers(self) -> dict[str, str]:
        """Return additional exporter headers."""
        ...
    @headers.setter
    def headers(self, value: dict[str, str]) -> None:
        """Replace additional exporter headers."""
        ...
    @property
    def resource_attributes(self) -> dict[str, str]:
        """Return additional OpenInference resource attributes."""
        ...
    @resource_attributes.setter
    def resource_attributes(self, value: dict[str, str]) -> None:
        """Replace additional OpenInference resource attributes."""
        ...
    def set_header(self, key: str, value: str) -> None:
        """Set one exporter header key/value pair."""
        ...
    def set_resource_attribute(self, key: str, value: str) -> None:
        """Set one OpenInference resource attribute key/value pair."""
        ...

class OpenInferenceSubscriber:
    """OpenInference-backed NeMo Relay event subscriber.

    Summary:
        Native subscriber that exports lifecycle events as OpenInference spans.

    Description:
        Register the subscriber under a name to receive runtime events. Flush or
        shut it down before process exit when deterministic export is required.
    """

    def __init__(self, config: OpenInferenceConfig) -> None:
        """Create a subscriber from an OpenInference config."""
        ...
    def register(self, name: str) -> None:
        """Register the subscriber under ``name``."""
        ...
    def deregister(self, name: str) -> bool:
        """Deregister ``name`` and return whether it existed."""
        ...
    def force_flush(self) -> None:
        """Flush pending telemetry through the configured exporter."""
        ...
    def shutdown(self) -> None:
        """Shut down native OpenInference resources."""
        ...

class OpenAIChatCodec:
    """Built-in codec for OpenAI Chat Completions requests and responses.

    Summary:
        Native codec bridge for Chat Completions payloads.
    """

    def __init__(self) -> None:
        """Create an OpenAI Chat codec."""
        ...
    def decode(self, request: LLMRequest) -> AnnotatedLLMRequest:
        """Decode a Chat Completions request into a normalized request view."""
        ...
    def encode(self, annotated: AnnotatedLLMRequest, original: LLMRequest) -> LLMRequest:
        """Encode a normalized request back into Chat Completions shape."""
        ...
    def decode_response(self, response: _Json) -> AnnotatedLLMResponse:
        """Decode a Chat Completions response into a normalized response view."""
        ...

class OpenAIResponsesCodec:
    """Built-in codec for OpenAI Responses requests and responses.

    Summary:
        Native codec bridge for OpenAI Responses payloads.
    """

    def __init__(self) -> None:
        """Create an OpenAI Responses codec."""
        ...
    def decode(self, request: LLMRequest) -> AnnotatedLLMRequest:
        """Decode a Responses request into a normalized request view."""
        ...
    def encode(self, annotated: AnnotatedLLMRequest, original: LLMRequest) -> LLMRequest:
        """Encode a normalized request back into Responses shape."""
        ...
    def decode_response(self, response: _Json) -> AnnotatedLLMResponse:
        """Decode a Responses response into a normalized response view."""
        ...

class AnthropicMessagesCodec:
    """Built-in codec for Anthropic Messages requests and responses.

    Summary:
        Native codec bridge for Anthropic Messages payloads.
    """

    def __init__(self) -> None:
        """Create an Anthropic Messages codec."""
        ...
    def decode(self, request: LLMRequest) -> AnnotatedLLMRequest:
        """Decode an Anthropic Messages request into a normalized request view."""
        ...
    def encode(self, annotated: AnnotatedLLMRequest, original: LLMRequest) -> LLMRequest:
        """Encode a normalized request back into Anthropic Messages shape."""
        ...
    def decode_response(self, response: _Json) -> AnnotatedLLMResponse:
        """Decode an Anthropic response into a normalized response view."""
        ...

class AdaptiveRuntime:
    """Hosted adaptive runtime bridge implemented by the native extension.

    Summary:
        Native runtime for configured adaptive components.

    Description:
        ``AdaptiveRuntime`` validates and owns adaptive state for external
        integrations that need lifecycle-managed adaptive features.
    """

    def __init__(self, config: object) -> None:
        """Create a runtime from an adaptive config object or mapping."""
        ...
    async def register(self) -> None:
        """Validate and register configured adaptive components."""
        ...
    def deregister(self) -> None:
        """Deregister adaptive components owned by this runtime."""
        ...
    async def shutdown(self) -> None:
        """Deregister components and release owned native resources."""
        ...
    def wait_for_idle(self) -> None:
        """Block until the adaptive telemetry drain has processed pending work."""
        ...
    def report(self) -> _JsonObject:
        """Return the runtime validation report."""
        ...
    def bind_scope(self, scope_handle: ScopeHandle) -> None:
        """Bind runtime ACG behavior to an active scope handle."""
        ...
    def build_cache_request_facts(
        self,
        *,
        provider: str,
        request_id: str,
        annotated_request: object,
        agent_id: str,
        timestamp: Optional[str] = None,
    ) -> Optional[_JsonObject]:
        """Build cache-diagnostics facts for a provider request.

        Args:
            provider: Logical provider name.
            request_id: Stable request identifier.
            annotated_request: Normalized request or equivalent mapping.
            agent_id: Agent identifier associated with the request.
            timestamp: Optional RFC 3339 timestamp override.

        Returns:
            Derived cache facts, or ``None`` when there is not enough state to
            produce a valid event.
        """
        ...

class PluginContext:
    """Native plugin registration context passed to plugin implementations.

    Summary:
        Internal bridge used by the native plugin system.

    Description:
        Python plugin protocols expose the public shape. The native class exists
        for runtime registration callbacks.
    """
    def register_mark_sanitize_guardrail(self, name: str, priority: int, callback: _EventSanitizeGuardrail) -> None: ...
    def register_scope_sanitize_start_guardrail(
        self, name: str, priority: int, callback: _EventSanitizeGuardrail
    ) -> None: ...
    def register_scope_sanitize_end_guardrail(
        self, name: str, priority: int, callback: _EventSanitizeGuardrail
    ) -> None: ...

def register_mark_sanitize_guardrail(name: str, priority: int, guardrail: _EventSanitizeGuardrail) -> None: ...
def deregister_mark_sanitize_guardrail(name: str) -> bool: ...
def register_scope_sanitize_start_guardrail(name: str, priority: int, guardrail: _EventSanitizeGuardrail) -> None: ...
def deregister_scope_sanitize_start_guardrail(name: str) -> bool: ...
def register_scope_sanitize_end_guardrail(name: str, priority: int, guardrail: _EventSanitizeGuardrail) -> None: ...
def deregister_scope_sanitize_end_guardrail(name: str) -> bool: ...
def scope_register_mark_sanitize_guardrail(
    scope_uuid: str, name: str, priority: int, guardrail: _EventSanitizeGuardrail
) -> None: ...
def scope_deregister_mark_sanitize_guardrail(scope_uuid: str, name: str) -> bool: ...
def scope_register_scope_sanitize_start_guardrail(
    scope_uuid: str, name: str, priority: int, guardrail: _EventSanitizeGuardrail
) -> None: ...
def scope_deregister_scope_sanitize_start_guardrail(scope_uuid: str, name: str) -> bool: ...
def scope_register_scope_sanitize_end_guardrail(
    scope_uuid: str, name: str, priority: int, guardrail: _EventSanitizeGuardrail
) -> None: ...
def scope_deregister_scope_sanitize_end_guardrail(scope_uuid: str, name: str) -> bool: ...
def create_scope_stack() -> ScopeStack:
    """Create a fresh native scope stack.

    Returns:
        A new ``ScopeStack`` that is not installed into any Python context or
        native thread-local slot.

    Exceptional flow:
        Native allocation errors propagate unchanged.
    """
    ...

def set_thread_scope_stack(stack: ScopeStack) -> None:
    """Install a scope stack into native thread-local storage.

    Args:
        stack: Scope stack to use for subsequent native calls on this thread.

    Returns:
        ``None``.
    """
    ...

def sync_thread_scope_stack(stack: ScopeStack) -> None:
    """Synchronize thread-local storage without marking it explicitly active.

    Args:
        stack: Scope stack to synchronize into the native thread-local slot.

    Returns:
        ``None``.
    """
    ...

def scope_stack_active() -> bool:
    """Return whether native thread-local storage has an explicitly active stack."""
    ...

def get_handle() -> ScopeHandle:
    """Return the current top-of-stack scope handle.

    Returns:
        The active ``ScopeHandle``.

    Exceptional flow:
        Raises a native runtime error if the active stack has no current scope.
    """
    ...

def push_scope(
    name: str,
    scope_type: ScopeType,
    *,
    handle: ScopeHandle | None = None,
    attributes: ScopeAttributes | None = None,
    data: _Json | None = None,
    metadata: _Json | None = None,
    input: _Json | None = None,
    timestamp: datetime | None = None,
) -> ScopeHandle:
    """Push a new scope onto the active native stack.

    Args:
        name: Human-readable scope name.
        scope_type: Semantic scope type.
        handle: Optional parent scope handle. When omitted, the current
            top-of-stack scope becomes the parent.
        attributes: Optional scope attribute bitflags.
        data: Optional JSON application payload stored on the scope handle.
        metadata: Optional JSON metadata recorded on the start event.
        input: Optional JSON payload exported as semantic scope input on the
            start event.
        timestamp: Optional timezone-aware datetime recorded as the handle
            start time and on the start event. When omitted, the current runtime
            time is used.

    Returns:
        Handle for the newly pushed scope.

    Exceptional flow:
        Raises native runtime errors for invalid parent handles, invalid JSON
        payloads, invalid timestamp types, naive datetimes, or inactive stack
        state.
    """
    ...

def pop_scope(
    handle: ScopeHandle,
    output: Optional[_Json] = None,
    metadata: Optional[_Json] = None,
    timestamp: datetime | None = None,
) -> None:
    """Pop a scope and emit its end event.

    Args:
        handle: Handle returned by ``push_scope``.
        output: Optional semantic output payload recorded on the end event.
        metadata: Optional JSON metadata recorded on the end event.
        timestamp: Optional timezone-aware datetime recorded on the end event.
            When omitted, the runtime default end timestamp is used.

    Returns:
        ``None``.

    Exceptional flow:
        Raises native runtime errors if ``handle`` is not the current scope or
        if ``output`` or ``metadata`` cannot be converted to JSON-compatible data.
        Raises for invalid timestamp types or naive datetimes.
    """
    ...

def event(
    name: str,
    *,
    handle: ScopeHandle | None = None,
    data: _Json | None = None,
    metadata: _Json | None = None,
    timestamp: datetime | None = None,
) -> None:
    """Emit a point-in-time mark event.

    Args:
        name: Mark event name.
        handle: Optional parent scope handle. When omitted, the current
            top-of-stack scope becomes the parent.
        data: Optional JSON data payload recorded on the mark event.
        metadata: Optional JSON metadata payload recorded on the mark event.
        timestamp: Optional timezone-aware datetime recorded on the mark event.
            When omitted, the current runtime time is used.

    Returns:
        ``None``.

    Exceptional flow:
        Raises for invalid JSON payloads, invalid timestamp types, or naive
        datetimes.
    """
    ...

def tool_call(
    name: str,
    args: _Json,
    *,
    handle: ScopeHandle | None = None,
    attributes: ToolAttributes | None = None,
    data: _Json | None = None,
    metadata: _Json | None = None,
    tool_call_id: str | None = None,
    timestamp: datetime | None = None,
) -> ToolHandle:
    """Begin a manual tool lifecycle span.

    Args:
        name: Tool name.
        args: JSON-compatible tool arguments recorded on the start event after
            sanitize-request guardrails.
        handle: Optional parent scope handle. When omitted, the current
            top-of-stack scope becomes the parent.
        attributes: Optional tool attribute bitflags.
        data: Optional JSON application payload stored on the tool handle.
        metadata: Optional JSON metadata recorded on the start event.
        tool_call_id: Optional provider-specific tool-call correlation ID.
        timestamp: Optional timezone-aware datetime recorded on the start event.
            When omitted, the current runtime time is used.

    Returns:
        Tool handle that must be passed to ``tool_call_end``.

    Exceptional flow:
        Raises for invalid JSON payloads, invalid timestamp types, or naive
        datetimes.
    """
    ...

def tool_call_end(
    handle: ToolHandle,
    result: _Json,
    *,
    data: _Json | None = None,
    metadata: _Json | None = None,
    timestamp: datetime | None = None,
) -> None:
    """End a manual tool lifecycle span.

    Args:
        handle: Tool handle returned by ``tool_call``.
        result: JSON-compatible tool result recorded on the end event after
            sanitize-response guardrails unless it sanitizes to JSON null.
        data: Optional JSON payload used when the sanitized result is JSON null.
        metadata: Optional JSON metadata recorded on the end event.
        timestamp: Optional timezone-aware datetime recorded on the end event.
            When omitted, the runtime default end timestamp is used.

    Returns:
        ``None``.

    Exceptional flow:
        Raises for invalid JSON payloads, invalid timestamp types, or naive
        datetimes.
    """
    ...

def tool_call_execute(
    name: str,
    args: _Json,
    func: Callable[[_Json], Awaitable[_Json]],
    **kwargs: object,
) -> Awaitable[_Json]:
    """Execute a tool through the managed native middleware pipeline.

    Args:
        name: Tool name.
        args: Initial JSON-compatible tool arguments.
        func: Awaitable tool implementation called with final arguments.
        **kwargs: Optional parent handle, attributes, data, and metadata.

    Returns:
        Awaitable that resolves to the JSON-compatible tool result.

    Exceptional flow:
        Conditional guardrails may reject execution. Callback and native errors
        propagate through the returned awaitable.
    """
    ...

def llm_call(
    name: str,
    request: LLMRequest,
    *,
    handle: ScopeHandle | None = None,
    attributes: LLMAttributes | None = None,
    data: _Json | None = None,
    metadata: _Json | None = None,
    model_name: str | None = None,
    timestamp: datetime | None = None,
) -> LLMHandle:
    """Begin a manual LLM lifecycle span.

    Args:
        name: Provider or logical call name.
        request: LLM request recorded on the start event after
            sanitize-request guardrails.
        handle: Optional parent scope handle. When omitted, the current
            top-of-stack scope becomes the parent.
        attributes: Optional LLM attribute bitflags.
        data: Optional JSON application payload stored on the LLM handle.
        metadata: Optional JSON metadata recorded on the start event.
        model_name: Optional normalized model name recorded in the LLM event
            category profile.
        timestamp: Optional timezone-aware datetime recorded on the start event.
            When omitted, the current runtime time is used.

    Returns:
        LLM handle that must be passed to ``llm_call_end``.

    Exceptional flow:
        Raises for invalid timestamp types or naive datetimes.
    """
    ...

def llm_call_end(
    handle: LLMHandle,
    response: _Json,
    *,
    data: _Json | None = None,
    metadata: _Json | None = None,
    annotated_response: AnnotatedLLMResponse | Mapping[str, _JsonValue] | None = None,
    response_codec: object | None = None,
    timestamp: datetime | None = None,
) -> None:
    """End a manual LLM lifecycle span.

    Args:
        handle: LLM handle returned by ``llm_call``.
        response: JSON-compatible LLM response recorded on the end event after
            sanitize-response guardrails unless it sanitizes to JSON null.
        data: Optional JSON payload used when the sanitized response is JSON null.
        metadata: Optional JSON metadata recorded on the end event.
        annotated_response: Optional normalized response annotation attached to
            the end event. Accepts an ``AnnotatedLLMResponse`` instance or a
            JSON-compatible mapping matching that schema.
        response_codec: Optional object implementing ``decode_response`` used
            to derive ``annotated_response`` from ``response`` for observability
            when ``annotated_response`` is omitted.
        timestamp: Optional timezone-aware datetime recorded on the end event.
            When omitted, the runtime default end timestamp is used.

    Returns:
        ``None``.

    Exceptional flow:
        Raises for invalid JSON payloads, invalid timestamp types, or naive
        datetimes.
    """
    ...

def llm_call_execute(
    name: str,
    request: LLMRequest,
    func: Callable[[LLMRequest], Awaitable[_Json]],
    **kwargs: object,
) -> Awaitable[_Json]:
    """Execute a non-streaming LLM call through native middleware.

    Args:
        name: Provider or logical call name.
        request: Initial LLM request.
        func: Awaitable provider implementation called with the final request.
        **kwargs: Optional parent handle, attributes, data, metadata, model
            name, request codec, and response codec.

    Returns:
        Awaitable that resolves to a JSON-compatible LLM response.

    Exceptional flow:
        Conditional guardrails may reject execution. Request intercept,
        execution intercept, codec, callback, and native errors propagate
        through the returned awaitable.
    """
    ...

def llm_stream_call_execute(
    name: str,
    request: LLMRequest,
    func: Callable[[LLMRequest], AsyncIterator[_Json]],
    collector: Callable[[_Json], None],
    finalizer: Callable[[], _Json],
    **kwargs: object,
) -> LlmStream:
    """Execute a streaming LLM call through native middleware.

    Args:
        name: Provider or logical call name.
        request: Initial LLM request.
        func: Provider implementation returning an async iterator of chunks.
        collector: Callback invoked for each post-intercept chunk.
        finalizer: Callback that returns the aggregate final response.
        **kwargs: Optional parent handle, attributes, data, metadata, model
            name, request codec, and response codec.

    Returns:
        ``LlmStream`` async iterator of JSON-compatible chunks.

    Exceptional flow:
        Conditional guardrails may reject execution. Streaming callback,
        collector, finalizer, codec, and native errors propagate while awaiting
        or iterating the stream.
    """
    ...

def tool_request_intercepts(name: str, args: _Json) -> _Json:
    """Run the registered tool request-intercept chain.

    Args:
        name: Tool name used to select and invoke intercept callbacks.
        args: Current JSON-compatible tool arguments.

    Returns:
        Transformed tool arguments after all applicable request intercepts.

    Exceptional flow:
        Callback exceptions and native middleware errors propagate unchanged.
    """
    ...

def tool_conditional_execution(name: str, args: _Json) -> None:
    """Run tool conditional-execution guardrails.

    Args:
        name: Tool name used to select and invoke guardrail callbacks.
        args: Current JSON-compatible tool arguments.

    Returns:
        ``None`` when all guardrails allow execution.

    Exceptional flow:
        Raises a native rejection error when a guardrail returns a rejection
        message. Callback exceptions also propagate.
    """
    ...

def llm_request_intercepts(name: str, request: LLMRequest) -> LLMRequestInterceptOutcome:
    """Run the registered LLM request-intercept chain.

    Args:
        name: Provider or logical LLM call name.
        request: Current LLM request.

    Returns:
        Transformed request after all applicable request intercepts.

    Exceptional flow:
        Callback exceptions and native middleware errors propagate unchanged.
    """
    ...

def llm_conditional_execution(request: LLMRequest) -> None:
    """Run LLM conditional-execution guardrails.

    Args:
        request: LLM request to evaluate.

    Returns:
        ``None`` when all guardrails allow execution.

    Exceptional flow:
        Raises a native rejection error when a guardrail returns a rejection
        message. Callback exceptions also propagate.
    """
    ...

def register_tool_sanitize_request_guardrail(name: str, priority: int, guardrail: _ToolSanitizeGuardrail) -> None:
    """Register a global tool sanitize-request guardrail.

    Args:
        name: Unique guardrail name.
        priority: Execution order; lower values run first.
        guardrail: Callback that returns the request payload recorded on start
            events.

    Returns:
        ``None``.
    """
    ...

def deregister_tool_sanitize_request_guardrail(name: str) -> bool:
    """Remove a global tool sanitize-request guardrail by name.

    Args:
        name: Guardrail name to remove.

    Returns:
        ``True`` if a guardrail was removed, otherwise ``False``.
    """
    ...

def register_tool_sanitize_response_guardrail(name: str, priority: int, guardrail: _ToolSanitizeGuardrail) -> None:
    """Register a global tool sanitize-response guardrail.

    Args:
        name: Unique guardrail name.
        priority: Execution order; lower values run first.
        guardrail: Callback that returns the response payload recorded on end
            events.

    Returns:
        ``None``.
    """
    ...

def deregister_tool_sanitize_response_guardrail(name: str) -> bool:
    """Remove a global tool sanitize-response guardrail by name.

    Returns:
        ``True`` if a guardrail was removed, otherwise ``False``.
    """
    ...

def register_tool_conditional_execution_guardrail(
    name: str, priority: int, guardrail: _ToolConditionalExecutionGuardrail
) -> None:
    """Register a global tool conditional-execution guardrail.

    Args:
        name: Unique guardrail name.
        priority: Execution order; lower values run first.
        guardrail: Callback that returns ``None`` to allow execution or a
            rejection message to block execution.

    Returns:
        ``None``.
    """
    ...

def deregister_tool_conditional_execution_guardrail(name: str) -> bool:
    """Remove a global tool conditional-execution guardrail by name.

    Returns:
        ``True`` if a guardrail was removed, otherwise ``False``.
    """
    ...

def register_llm_sanitize_request_guardrail(name: str, priority: int, guardrail: _LlmSanitizeRequestGuardrail) -> None:
    """Register a global LLM sanitize-request guardrail.

    Args:
        name: Unique guardrail name.
        priority: Execution order; lower values run first.
        guardrail: Callback that returns the request recorded on start events.

    Returns:
        ``None``.
    """
    ...

def deregister_llm_sanitize_request_guardrail(name: str) -> bool:
    """Remove a global LLM sanitize-request guardrail by name.

    Returns:
        ``True`` if a guardrail was removed, otherwise ``False``.
    """
    ...

def register_llm_sanitize_response_guardrail(
    name: str, priority: int, guardrail: _LlmSanitizeResponseGuardrail
) -> None:
    """Register a global LLM sanitize-response guardrail.

    Args:
        name: Unique guardrail name.
        priority: Execution order; lower values run first.
        guardrail: Callback that returns the response object recorded on end
            events.

    Returns:
        ``None``.
    """
    ...

def deregister_llm_sanitize_response_guardrail(name: str) -> bool:
    """Remove a global LLM sanitize-response guardrail by name.

    Returns:
        ``True`` if a guardrail was removed, otherwise ``False``.
    """
    ...

def register_llm_conditional_execution_guardrail(
    name: str, priority: int, guardrail: _LlmConditionalExecutionGuardrail
) -> None:
    """Register a global LLM conditional-execution guardrail.

    Args:
        name: Unique guardrail name.
        priority: Execution order; lower values run first.
        guardrail: Callback that returns ``None`` to allow execution or a
            rejection message to block execution.

    Returns:
        ``None``.
    """
    ...

def deregister_llm_conditional_execution_guardrail(name: str) -> bool:
    """Remove a global LLM conditional-execution guardrail by name.

    Returns:
        ``True`` if a guardrail was removed, otherwise ``False``.
    """
    ...

def register_tool_request_intercept(
    name: str,
    priority: int,
    break_chain: bool,
    callable: _ToolRequestIntercept,
) -> None:
    """Register a global tool request intercept.

    Args:
        name: Unique intercept name.
        priority: Execution order; lower values run first.
        break_chain: Whether later request intercepts should be skipped after
            this one runs.
        callable: Callback that rewrites tool arguments.

    Returns:
        ``None``.
    """
    ...

def deregister_tool_request_intercept(name: str) -> bool:
    """Remove a global tool request intercept and return whether it existed."""
    ...

def register_tool_execution_intercept(name: str, priority: int, callable: _ToolExecutionIntercept) -> None:
    """Register a global tool execution intercept.

    Args:
        name: Unique intercept name.
        priority: Execution order; lower values run first.
        callable: Middleware callback returning
            ``ToolExecutionInterceptOutcome``. It may call or short-circuit
            ``next``; ``next`` resolves to the raw downstream result while
            Relay retains downstream pending marks.

    Returns:
        ``None``.
    """
    ...

def deregister_tool_execution_intercept(name: str) -> bool:
    """Remove a global tool execution intercept.

    Returns:
        ``True`` if an intercept was removed, otherwise ``False``.
    """
    ...

def register_llm_request_intercept(
    name: str,
    priority: int,
    break_chain: bool,
    callable: _LlmRequestIntercept,
) -> None:
    """Register a global LLM request intercept.

    Args:
        name: Unique intercept name.
        priority: Execution order; lower values run first.
        break_chain: Whether lower-priority request intercepts are skipped
            after this callback.
        callable: Callback that rewrites the raw and optional annotated request.

    Returns:
        ``None``.
    """
    ...

def deregister_llm_request_intercept(name: str) -> bool:
    """Remove a global LLM request intercept.

    Returns:
        ``True`` if an intercept was removed, otherwise ``False``.
    """
    ...

def register_llm_execution_intercept(name: str, priority: int, callable: _LlmExecutionIntercept) -> None:
    """Register a global LLM execution intercept.

    Args:
        name: Unique intercept name.
        priority: Execution order; lower values run first.
        callable: Middleware callback that may call or short-circuit ``next``.

    Returns:
        ``None``.
    """
    ...

def deregister_llm_execution_intercept(name: str) -> bool:
    """Remove a global LLM execution intercept.

    Returns:
        ``True`` if an intercept was removed, otherwise ``False``.
    """
    ...

def register_llm_stream_execution_intercept(
    name: str,
    priority: int,
    callable: _LlmStreamExecutionIntercept,
) -> None:
    """Register a global LLM stream-execution intercept.

    Args:
        name: Unique intercept name.
        priority: Execution order; lower values run first.
        callable: Streaming middleware callback.

    Returns:
        ``None``.
    """
    ...

def deregister_llm_stream_execution_intercept(name: str) -> bool:
    """Remove a global LLM stream-execution intercept.

    Returns:
        ``True`` if an intercept was removed, otherwise ``False``.
    """
    ...

def register_subscriber(name: str, callback: Callable[[ScopeEvent | MarkEvent], None]) -> None:
    """Register a global event subscriber callback.

    Args:
        name: Unique subscriber name.
        callback: Function invoked for each emitted scope or mark event.

    Returns:
        ``None``.

    Notes:
        Native event emission queues subscriber callbacks and does not wait for
        observer work before returning.
    """
    ...

def deregister_subscriber(name: str) -> bool:
    """Remove a global event subscriber.

    Returns:
        ``True`` if a subscriber was removed, otherwise ``False``.
    """
    ...

def flush_subscribers() -> None:
    """Wait for subscriber callbacks already queued by native event emission."""
    ...

def scope_register_tool_sanitize_request_guardrail(
    scope_uuid: str, name: str, priority: int, guardrail: _ToolSanitizeGuardrail
) -> None:
    """Register a scope-local tool sanitize-request guardrail.

    Args:
        scope_uuid: UUID of the owning scope.
        name: Unique guardrail name within that scope.
        priority: Execution order; lower values run first.
        guardrail: Callback used while the owning scope is active.

    Returns:
        ``None``.
    """
    ...

def scope_deregister_tool_sanitize_request_guardrail(scope_uuid: str, name: str) -> bool:
    """Remove a scope-local tool sanitize-request guardrail.

    Returns:
        ``True`` if a guardrail was removed, otherwise ``False``.
    """
    ...

def scope_register_tool_sanitize_response_guardrail(
    scope_uuid: str, name: str, priority: int, guardrail: _ToolSanitizeGuardrail
) -> None:
    """Register a scope-local tool sanitize-response guardrail.

    Args:
        scope_uuid: UUID of the owning scope.
        name: Unique guardrail name within that scope.
        priority: Execution order; lower values run first.
        guardrail: Callback used while the owning scope is active.

    Returns:
        ``None``.
    """
    ...

def scope_deregister_tool_sanitize_response_guardrail(scope_uuid: str, name: str) -> bool:
    """Remove a scope-local tool sanitize-response guardrail.

    Returns:
        ``True`` if a guardrail was removed, otherwise ``False``.
    """
    ...

def scope_register_tool_conditional_execution_guardrail(
    scope_uuid: str, name: str, priority: int, guardrail: _ToolConditionalExecutionGuardrail
) -> None:
    """Register a scope-local tool conditional-execution guardrail.

    Args:
        scope_uuid: UUID of the owning scope.
        name: Unique guardrail name within that scope.
        priority: Execution order; lower values run first.
        guardrail: Callback used while the owning scope is active.

    Returns:
        ``None``.
    """
    ...

def scope_deregister_tool_conditional_execution_guardrail(scope_uuid: str, name: str) -> bool:
    """Remove a scope-local tool conditional-execution guardrail.

    Returns:
        ``True`` if a guardrail was removed, otherwise ``False``.
    """
    ...

def scope_register_tool_request_intercept(
    scope_uuid: str,
    name: str,
    priority: int,
    break_chain: bool,
    callable: _ToolRequestIntercept,
) -> None:
    """Register a scope-local tool request intercept.

    Args:
        scope_uuid: UUID of the owning scope.
        name: Unique intercept name within that scope.
        priority: Execution order; lower values run first.
        break_chain: Whether lower-priority request intercepts are skipped.
        callable: Callback used while the owning scope is active.

    Returns:
        ``None``.
    """
    ...

def scope_deregister_tool_request_intercept(scope_uuid: str, name: str) -> bool:
    """Remove a scope-local tool request intercept.

    Returns:
        ``True`` if an intercept was removed, otherwise ``False``.
    """
    ...

def scope_register_tool_execution_intercept(
    scope_uuid: str,
    name: str,
    priority: int,
    callable: _ToolExecutionIntercept,
) -> None:
    """Register a scope-local tool execution intercept.

    Args:
        scope_uuid: UUID of the owning scope.
        name: Unique intercept name within that scope.
        priority: Execution order; lower values run first.
        callable: Middleware callback returning
            ``ToolExecutionInterceptOutcome`` while the owning scope is active.
            Its ``next`` continuation resolves to the raw downstream result
            while Relay retains downstream pending marks.

    Returns:
        ``None``.
    """
    ...

def scope_deregister_tool_execution_intercept(scope_uuid: str, name: str) -> bool:
    """Remove a scope-local tool execution intercept.

    Returns:
        ``True`` if an intercept was removed, otherwise ``False``.
    """
    ...

def scope_register_llm_sanitize_request_guardrail(
    scope_uuid: str, name: str, priority: int, guardrail: _LlmSanitizeRequestGuardrail
) -> None:
    """Register a scope-local LLM sanitize-request guardrail.

    Args:
        scope_uuid: UUID of the owning scope.
        name: Unique guardrail name within that scope.
        priority: Execution order; lower values run first.
        guardrail: Callback used while the owning scope is active.

    Returns:
        ``None``.
    """
    ...

def scope_deregister_llm_sanitize_request_guardrail(scope_uuid: str, name: str) -> bool:
    """Remove a scope-local LLM sanitize-request guardrail.

    Returns:
        ``True`` if a guardrail was removed, otherwise ``False``.
    """
    ...

def scope_register_llm_sanitize_response_guardrail(
    scope_uuid: str, name: str, priority: int, guardrail: _LlmSanitizeResponseGuardrail
) -> None:
    """Register a scope-local LLM sanitize-response guardrail.

    Args:
        scope_uuid: UUID of the owning scope.
        name: Unique guardrail name within that scope.
        priority: Execution order; lower values run first.
        guardrail: Callback used while the owning scope is active.

    Returns:
        ``None``.
    """
    ...

def scope_deregister_llm_sanitize_response_guardrail(scope_uuid: str, name: str) -> bool:
    """Remove a scope-local LLM sanitize-response guardrail.

    Returns:
        ``True`` if a guardrail was removed, otherwise ``False``.
    """
    ...

def scope_register_llm_conditional_execution_guardrail(
    scope_uuid: str, name: str, priority: int, guardrail: _LlmConditionalExecutionGuardrail
) -> None:
    """Register a scope-local LLM conditional-execution guardrail.

    Args:
        scope_uuid: UUID of the owning scope.
        name: Unique guardrail name within that scope.
        priority: Execution order; lower values run first.
        guardrail: Callback used while the owning scope is active.

    Returns:
        ``None``.
    """
    ...

def scope_deregister_llm_conditional_execution_guardrail(scope_uuid: str, name: str) -> bool:
    """Remove a scope-local LLM conditional-execution guardrail.

    Returns:
        ``True`` if a guardrail was removed, otherwise ``False``.
    """
    ...

def scope_register_llm_request_intercept(
    scope_uuid: str,
    name: str,
    priority: int,
    break_chain: bool,
    callable: _LlmRequestIntercept,
) -> None:
    """Register a scope-local LLM request intercept.

    Args:
        scope_uuid: UUID of the owning scope.
        name: Unique intercept name within that scope.
        priority: Execution order; lower values run first.
        break_chain: Whether lower-priority request intercepts are skipped.
        callable: Callback used while the owning scope is active.

    Returns:
        ``None``.
    """
    ...

def scope_deregister_llm_request_intercept(scope_uuid: str, name: str) -> bool:
    """Remove a scope-local LLM request intercept.

    Returns:
        ``True`` if an intercept was removed, otherwise ``False``.
    """
    ...

def scope_register_llm_execution_intercept(
    scope_uuid: str,
    name: str,
    priority: int,
    callable: _LlmExecutionIntercept,
) -> None:
    """Register a scope-local LLM execution intercept.

    Args:
        scope_uuid: UUID of the owning scope.
        name: Unique intercept name within that scope.
        priority: Execution order; lower values run first.
        callable: Middleware callback used while the owning scope is active.

    Returns:
        ``None``.
    """
    ...

def scope_deregister_llm_execution_intercept(scope_uuid: str, name: str) -> bool:
    """Remove a scope-local LLM execution intercept.

    Returns:
        ``True`` if an intercept was removed, otherwise ``False``.
    """
    ...

def scope_register_llm_stream_execution_intercept(
    scope_uuid: str,
    name: str,
    priority: int,
    callable: _LlmStreamExecutionIntercept,
) -> None:
    """Register a scope-local LLM stream-execution intercept.

    Args:
        scope_uuid: UUID of the owning scope.
        name: Unique intercept name within that scope.
        priority: Execution order; lower values run first.
        callable: Streaming middleware callback used while the owning scope is
            active.

    Returns:
        ``None``.
    """
    ...

def scope_deregister_llm_stream_execution_intercept(scope_uuid: str, name: str) -> bool:
    """Remove a scope-local LLM stream-execution intercept.

    Returns:
        ``True`` if an intercept was removed, otherwise ``False``.
    """
    ...

def scope_register_subscriber(
    scope_uuid: str,
    name: str,
    callback: Callable[[ScopeEvent | MarkEvent], None],
) -> None:
    """Register a scope-local event subscriber callback.

    Args:
        scope_uuid: UUID of the owning scope.
        name: Unique subscriber name within that scope.
        callback: Event callback used while the owning scope is active.

    Returns:
        ``None``.
    """
    ...

def scope_deregister_subscriber(scope_uuid: str, name: str) -> bool:
    """Remove a scope-local event subscriber.

    Returns:
        ``True`` if a subscriber was removed, otherwise ``False``.
    """
    ...

def validate_plugin_config(config: object) -> _JsonObject:
    """Validate a plugin configuration without changing active runtime state.

    Args:
        config: Plugin configuration object or equivalent mapping.

    Returns:
        Validation report as a JSON object.

    Exceptional flow:
        Raises native conversion or validation errors for malformed config.
    """
    ...

class _PluginHostActivation:
    """Native owner for one process-wide dynamic plugin host."""

    @property
    def report(self) -> _JsonObject:
        """Return the validation report captured during activation."""
        ...

    @property
    def is_active(self) -> bool:
        """Return whether this activation handle has not begun teardown.

        ``False`` does not guarantee another process-wide activation can start;
        failed teardown may intentionally retain the activation owner.
        """
        ...

    def close(self) -> Awaitable[None]:
        """Clear and unload this activation; repeated calls are safe."""
        ...

def activate_dynamic_plugins(config: object, dynamic_plugins: object) -> Awaitable[_PluginHostActivation]:
    """Load dynamic plugins and activate their components as one owned host.

    Args:
        config: Base plugin configuration object.
        dynamic_plugins: Sequence of dynamic plugin activation specifications.

    Returns:
        Awaitable resolving to the native activation owner.

    Exceptional flow:
        Invalid configuration, load, ownership, and registration errors are
        raised through the awaitable.
    """
    ...

def initialize_plugins(config: object) -> Awaitable[_JsonObject]:
    """Validate and activate plugin configuration.

    Args:
        config: Plugin configuration object or equivalent mapping.

    Returns:
        Awaitable resolving to the activation report.

    Exceptional flow:
        Activation errors propagate through the awaitable. The native runtime
        rolls back partial registration when possible.
    """
    ...

def clear_plugin_configuration() -> None:
    """Clear active plugin configuration while preserving registered kinds.

    Returns:
        ``None``.

    Exceptional flow:
        Native cleanup errors propagate unchanged.
    """
    ...

def active_plugin_report() -> Optional[_JsonObject]:
    """Return the active plugin report.

    Returns:
        Report JSON object for the last active configuration, or ``None`` if no
        plugin configuration is active.
    """
    ...

def list_plugin_kinds() -> list[str]:
    """Return registered custom plugin kind names.

    Returns:
        Sorted plugin kind names known to the native registry.
    """
    ...

def register_plugin(plugin_kind: str, plugin: object) -> None:
    """Register a custom plugin implementation under a kind string.

    Args:
        plugin_kind: Unique top-level component kind string.
        plugin: Plugin implementation object.

    Returns:
        ``None``.

    Exceptional flow:
        Raises native registry errors for duplicate or invalid registrations.
    """
    ...

def deregister_plugin(plugin_kind: str) -> bool:
    """Deregister a custom plugin kind.

    Args:
        plugin_kind: Kind string to remove.

    Returns:
        ``True`` if a plugin kind was removed, otherwise ``False``.
    """
    ...

def build_cache_telemetry_event(*args: object, **kwargs: object) -> _JsonObject:
    """Build a normalized adaptive cache telemetry event.

    Args:
        *args: Positional values accepted by the native adaptive helper.
        **kwargs: Provider, request, usage, template, model, tenant, and
            timestamp values accepted by the native adaptive helper.

    Returns:
        Cache telemetry event as a JSON object.
    """
    ...

def validate_adaptive_config(config: object) -> _JsonObject:
    """Validate adaptive configuration.

    Args:
        config: Adaptive configuration object or equivalent mapping.

    Returns:
        Validation report as a JSON object.

    Exceptional flow:
        Native conversion and validation errors propagate unchanged.
    """
    ...

def set_latency_sensitivity(level: int) -> None:
    """Set the process-local manual latency-sensitivity override.

    Args:
        level: Positive integer sensitivity value for the current execution
            context.

    Returns:
        ``None``.

    Exceptional flow:
        Native validation errors propagate when ``level`` is unsupported.
    """
    ...

def __getattr__(name: str) -> object:
    """Resolve dynamic native attributes not represented by static stubs.

    Args:
        name: Attribute name to resolve.

    Returns:
        Native attribute object.

    Exceptional flow:
        Raises ``AttributeError`` when no native attribute exists.
    """
    ...
