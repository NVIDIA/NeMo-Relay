# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Author out-of-process NeMo Relay worker plugins in Python.

Use :class:`WorkerPlugin` and :class:`PluginContext` to define plugin behavior,
then pass the implementation to :func:`serve_plugin`. The SDK owns the
``grpc-v1`` transport, host authentication, JSON envelopes, continuation calls,
and task-local scope-stack propagation.

The worker can keep multiple RPCs in flight, but callback execution is
cooperative. Asynchronous callbacks overlap only when they yield control at an
``await``. Synchronous callbacks run on the worker event-loop thread and must
not perform blocking I/O or long-running CPU work. Wrap blocking work in an
asynchronous callback and offload it with :func:`asyncio.to_thread` or another
appropriate executor.

Public data types:
    Json: Any JSON-serializable Python value.
    DataSchema: Schema identity for a mark's opaque data payload.
    EventCategory: Well-known semantic category for a Relay event.
    LogSeverity: Telemetry severity assigned to an exported mark log.
    MetricKind: OpenTelemetry instrument kind for a metric measurement.
    MetricValueType: Explicit numeric representation for a metric measurement.
    MetricMeasurement: One recording operation in a Relay metric mark.
    Event: A Relay event represented as a JSON object.
    EventSanitizeFields: Mutable event observability fields.
    LlmRequest: A Relay LLM request represented as a JSON object.
    LlmCodecIdentity: Typed discriminator for the active LLM codec.
    LlmSanitizeRequestContext: Per-call context supplied to an LLM request sanitizer.
    LlmSanitizeResponseContext: Per-call context supplied to an LLM response sanitizer.
    WorkerRequestCodec: Invocation-scoped async proxy for an active request codec.
    WorkerResponseCodec: Invocation-scoped async proxy for an active response codec.
    AnnotatedLlmRequest: An annotated Relay LLM request represented as a JSON
        object.
    PendingMarkSpec: A mark Relay emits under its managed lifecycle scope.
    LlmOptimizationContribution: Plugin-neutral LLM optimization evidence.
    LlmOptimizationDataSchema: Schema tag for custom optimization evidence.
    LlmOptimizationEvidenceQuality: Whether token evidence was observed or estimated.
    LlmOptimizationModel: Model identity used by optimization accounting.
    LlmOptimizationModelTransition: Baseline and effective model identities.
    LlmOptimizationTokens: Explicit token evidence by category.
    LlmOptimizationTokenImpact: Baseline, effective, and saved token evidence.
    LlmRequestInterceptOutcome: Canonical LLM request-intercept result.
    ToolExecutionResult: Canonical application-visible tool result.
    ToolExecutionInterceptOutcome: Canonical tool execution-intercept result.
    DiagnosticLevel: Severity of a configuration diagnostic.
    ConfigDiagnostic: Structured configuration warning or error.
    RuntimeDiagnostic: One aggregated host runtime diagnostic entry.
    RuntimeDiagnostics: Immutable host runtime diagnostics snapshot.
    RuntimeRegistrationKind: Kind of globally registered runtime behavior.
    RuntimeRegistrationOwnerKind: Kind of owner that installed a registration.
    RuntimeRegistrationOwner: Structured identity of a registration owner.
    RuntimeRegistrationIdentity: Structured identity of a runtime registration.
    ConditionalMiddlewareGuardrailHandle: Opaque handle for a registered guardrail.
    ScopeType: Semantic category for a Relay execution scope.
    WorkerSdkError: SDK, host-call, or worker protocol error.

Public callback aliases:
    ConditionalMiddlewareCallback: Runtime registration eligibility callback.
    SubscriberCallback: Event subscriber callback.
    EventMetadataInjectorCallback: Event metadata injector callback.
    EventSanitizeCallback: Mark or scope event sanitizer callback.
    ToolSanitizeCallback: Tool request or response sanitizer callback.
    ToolConditionalCallback: Tool execution guardrail callback.
    ToolRequestCallback: Tool request intercept callback.
    ToolExecutionCallback: Tool execution intercept callback.
    LlmSanitizeRequestCallback: LLM request sanitizer callback.
    LlmSanitizeResponseCallback: LLM response sanitizer callback.
    LlmConditionalCallback: LLM execution guardrail callback.
    LlmRequestCallback: LLM request intercept callback.
    LlmExecutionCallback: Unary LLM execution intercept callback.
    LlmStreamExecutionCallback: Streaming LLM execution intercept callback.

Public authoring types:
    WorkerPlugin: Base validation and registration contract for a plugin.
    PluginContext: Component-scoped callback registration context.
    PluginRuntime: Host runtime handle for event and scope operations.
    ToolNext: Continuation for a tool execution intercept.
    LlmNext: Continuation for a unary LLM execution intercept.
    LlmStreamNext: Continuation for a streaming LLM execution intercept.

Public functions:
    serve_plugin: Run a local ``grpc-v1`` worker until host shutdown.
"""

from ._api import (
    AnnotatedLlmRequest,
    ConditionalMiddlewareCallback,
    ConditionalMiddlewareGuardrailHandle,
    ConfigDiagnostic,
    DataSchema,
    DiagnosticLevel,
    Event,
    EventCategory,
    EventMetadataInjectorCallback,
    EventSanitizeCallback,
    EventSanitizeFields,
    Json,
    LlmCodecIdentity,
    LlmConditionalCallback,
    LlmExecutionCallback,
    LlmNext,
    LlmOptimizationContribution,
    LlmOptimizationDataSchema,
    LlmOptimizationEvidenceQuality,
    LlmOptimizationModel,
    LlmOptimizationModelTransition,
    LlmOptimizationTokenImpact,
    LlmOptimizationTokens,
    LlmRequest,
    LlmRequestCallback,
    LlmRequestInterceptOutcome,
    LlmSanitizeRequestCallback,
    LlmSanitizeRequestContext,
    LlmSanitizeResponseCallback,
    LlmSanitizeResponseContext,
    LlmStreamExecutionCallback,
    LlmStreamNext,
    LogSeverity,
    MetricKind,
    MetricMeasurement,
    MetricValueType,
    PendingMarkSpec,
    PluginContext,
    PluginRuntime,
    RuntimeDiagnostic,
    RuntimeDiagnostics,
    RuntimeRegistrationIdentity,
    RuntimeRegistrationKind,
    RuntimeRegistrationOwner,
    RuntimeRegistrationOwnerKind,
    ScopeType,
    SubscriberCallback,
    ToolConditionalCallback,
    ToolExecutionCallback,
    ToolExecutionInterceptOutcome,
    ToolExecutionResult,
    ToolNext,
    ToolRequestCallback,
    ToolSanitizeCallback,
    WorkerPlugin,
    WorkerRequestCodec,
    WorkerResponseCodec,
    WorkerSdkError,
    serve_plugin,
)

__all__ = [
    "AnnotatedLlmRequest",
    "ConditionalMiddlewareCallback",
    "ConditionalMiddlewareGuardrailHandle",
    "ConfigDiagnostic",
    "DataSchema",
    "DiagnosticLevel",
    "Event",
    "EventCategory",
    "EventMetadataInjectorCallback",
    "EventSanitizeCallback",
    "EventSanitizeFields",
    "Json",
    "LlmConditionalCallback",
    "LlmCodecIdentity",
    "LlmExecutionCallback",
    "LogSeverity",
    "MetricKind",
    "MetricMeasurement",
    "MetricValueType",
    "LlmOptimizationContribution",
    "LlmOptimizationDataSchema",
    "LlmOptimizationEvidenceQuality",
    "LlmOptimizationModel",
    "LlmOptimizationModelTransition",
    "LlmOptimizationTokenImpact",
    "LlmOptimizationTokens",
    "LlmNext",
    "LlmRequest",
    "LlmSanitizeRequestContext",
    "LlmSanitizeResponseContext",
    "LlmRequestCallback",
    "LlmRequestInterceptOutcome",
    "LlmSanitizeRequestCallback",
    "LlmSanitizeResponseCallback",
    "LlmStreamNext",
    "LlmStreamExecutionCallback",
    "PluginContext",
    "PluginRuntime",
    "RuntimeDiagnostic",
    "RuntimeDiagnostics",
    "RuntimeRegistrationIdentity",
    "RuntimeRegistrationKind",
    "RuntimeRegistrationOwner",
    "RuntimeRegistrationOwnerKind",
    "PendingMarkSpec",
    "ScopeType",
    "SubscriberCallback",
    "ToolConditionalCallback",
    "ToolExecutionCallback",
    "ToolExecutionInterceptOutcome",
    "ToolExecutionResult",
    "ToolNext",
    "ToolRequestCallback",
    "ToolSanitizeCallback",
    "WorkerRequestCodec",
    "WorkerResponseCodec",
    "WorkerPlugin",
    "WorkerSdkError",
    "serve_plugin",
]
