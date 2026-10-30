// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The public Rust SDK for NeMo Relay.
//!
//! Instrument through [`scope`], [`tool`], and [`llm`], register middleware through
//! [`registry`], and observe events through [`subscriber`] or [`exporter`].
//! Shared request, response, error, and codec contracts are re-exported directly.
//! Bindings and component implementations use their owning crates and the execution
//! engine in `nemo-relay-runtime` instead of depending on this SDK.

pub use nemo_relay_builtins::initialize_runtime;

/// Runtime-bound exporter constructors and their configuration contracts.
///
/// Exporter pipeline state is private; use lifecycle methods and diagnostic snapshots.
///
/// ```compile_fail,E0616
/// use nemo_relay::exporter::otel::OpenTelemetrySubscriber;
/// fn pipeline(subscriber: &OpenTelemetrySubscriber) { let _ = &subscriber.inner; }
/// ```
///
/// ```compile_fail,E0616
/// use nemo_relay::exporter::otel_logs::OpenTelemetryLogSubscriber;
/// fn pipeline(subscriber: &OpenTelemetryLogSubscriber) { let _ = &subscriber.inner; }
/// ```
///
/// ```compile_fail,E0616
/// use nemo_relay::exporter::otel_metrics::OpenTelemetryMetricSubscriber;
/// fn pipeline(subscriber: &OpenTelemetryMetricSubscriber) { let _ = &subscriber.inner; }
/// ```
pub mod exporter {
    pub use nemo_relay_builtins::exporter::*;
    pub use nemo_relay_observability::{
        MarkProjection, OpenTelemetryTlsConfig, OpenTelemetryType, OtlpAttributeMapping,
    };
    pub use nemo_relay_observability::{atif, atof, otel, otel_file, otel_logs, otel_metrics};
}
pub use nemo_relay_types::api::event::Event;
pub use nemo_relay_types::api::llm::LlmRequest;
pub use nemo_relay_types::codec::{
    normalization::LlmNormalizer,
    request::AnnotatedLlmRequest,
    response::{AnnotatedLlmResponse, Usage},
    streaming::{NormalizedStreamEvent, StreamingCodec},
    traits::{LlmCodec, LlmResponseCodec},
};
pub use nemo_relay_types::error::{FlowError, Result};
pub use nemo_relay_types::{Json, codec, editor_config, error};

/// Canonical lifecycle events and normalized LLM accessors.
pub mod event {
    pub use nemo_relay_runtime::api::event::EventNormalizationExt;
    pub use nemo_relay_types::api::event::*;
}
/// Task, thread, and distributed request context propagation.
pub mod context {
    pub use nemo_relay_runtime::api::runtime::scope_stack::snapshot_scope_stack;
    pub use nemo_relay_runtime::api::runtime::{
        PropagationContext, ScopeStackHandle, ThreadScopeStackBinding, capture_propagation_context,
        capture_propagation_context_with_root, capture_rootless_propagation_context,
        capture_thread_scope_stack, capture_traceparent, create_scope_stack,
        create_scope_stack_from_propagation, current_scope_stack_snapshot, fork_scope_stack,
        propagate_scope_to_thread, restore_thread_scope_stack, set_thread_scope_stack,
        with_scope_stack, with_task_scope_stack,
    };
}
/// Scope lifecycle and event emission.
pub mod scope {
    pub use nemo_relay_runtime::api::scope::{
        COMPACTION_EVENT_NAME, EmitMarkEventParams, EmitMetricEventParams, PopScopeParams,
        PushScopeParams, ScopeHandle, event, get_handle, metric, pop_scope, push_scope,
    };
    pub use nemo_relay_types::api::scope::*;
}
/// Managed and manual LLM calls.
pub mod llm {
    pub use nemo_relay_runtime::api::llm::{
        LlmCallEndParams, LlmCallExecuteParams, LlmCallParams, LlmHandle,
        LlmStreamCallExecuteParams, llm_call, llm_call_end, llm_call_execute,
        llm_conditional_execution, llm_request_intercepts, llm_stream_call_execute,
    };
    pub use nemo_relay_types::api::llm::*;
}
/// Managed and manual tool calls.
pub mod tool {
    pub use nemo_relay_runtime::api::tool::{
        ToolCallEndParams, ToolCallExecuteParams, ToolCallParams, ToolHandle, tool_call,
        tool_call_end, tool_call_execute, tool_conditional_execution, tool_request_intercepts,
    };
    pub use nemo_relay_types::api::tool::*;
}
/// Shared middleware callback, stream, and invocation-context contracts.
pub mod middleware {
    pub use nemo_relay_types::api::runtime::*;
}
/// Global and scope-local middleware registration.
pub mod registry {
    pub use nemo_relay_runtime::api::registry::{
        deregister_conditional_middleware_guardrail, deregister_event_metadata_injector,
        deregister_llm_conditional_execution_guardrail, deregister_llm_execution_intercept,
        deregister_llm_request_intercept, deregister_llm_sanitize_request_guardrail,
        deregister_llm_sanitize_response_guardrail, deregister_llm_stream_execution_intercept,
        deregister_mark_sanitize_guardrail, deregister_scope_sanitize_end_guardrail,
        deregister_scope_sanitize_start_guardrail, deregister_tool_conditional_execution_guardrail,
        deregister_tool_execution_intercept, deregister_tool_request_intercept,
        deregister_tool_sanitize_request_guardrail, deregister_tool_sanitize_response_guardrail,
        list_runtime_registrations, register_conditional_middleware_guardrail,
        register_event_metadata_injector, register_llm_conditional_execution_guardrail,
        register_llm_execution_intercept, register_llm_request_intercept,
        register_llm_sanitize_request_guardrail, register_llm_sanitize_response_guardrail,
        register_llm_stream_execution_intercept, register_mark_sanitize_guardrail,
        register_scope_sanitize_end_guardrail, register_scope_sanitize_start_guardrail,
        register_tool_conditional_execution_guardrail, register_tool_execution_intercept,
        register_tool_request_intercept, register_tool_sanitize_request_guardrail,
        register_tool_sanitize_response_guardrail, scope_deregister_event_metadata_injector,
        scope_deregister_llm_conditional_execution_guardrail,
        scope_deregister_llm_execution_intercept, scope_deregister_llm_request_intercept,
        scope_deregister_llm_sanitize_request_guardrail,
        scope_deregister_llm_sanitize_response_guardrail,
        scope_deregister_llm_stream_execution_intercept, scope_deregister_mark_sanitize_guardrail,
        scope_deregister_scope_sanitize_end_guardrail,
        scope_deregister_scope_sanitize_start_guardrail,
        scope_deregister_tool_conditional_execution_guardrail,
        scope_deregister_tool_execution_intercept, scope_deregister_tool_request_intercept,
        scope_deregister_tool_sanitize_request_guardrail,
        scope_deregister_tool_sanitize_response_guardrail, scope_register_event_metadata_injector,
        scope_register_llm_conditional_execution_guardrail, scope_register_llm_execution_intercept,
        scope_register_llm_request_intercept, scope_register_llm_sanitize_request_guardrail,
        scope_register_llm_sanitize_response_guardrail,
        scope_register_llm_stream_execution_intercept, scope_register_mark_sanitize_guardrail,
        scope_register_scope_sanitize_end_guardrail, scope_register_scope_sanitize_start_guardrail,
        scope_register_tool_conditional_execution_guardrail,
        scope_register_tool_execution_intercept, scope_register_tool_request_intercept,
        scope_register_tool_sanitize_request_guardrail,
        scope_register_tool_sanitize_response_guardrail,
    };
    pub use nemo_relay_types::api::registry::*;
}
/// Lifecycle event subscriptions and delivery barriers.
pub mod subscriber {
    pub use nemo_relay_runtime::api::subscriber::{
        deregister_subscriber, flush_subscribers, register_subscriber, scope_deregister_subscriber,
        scope_register_subscriber,
    };
    pub use nemo_relay_types::api::runtime::EventSubscriberFn;
}
/// Adaptive contribution recording for Rust integrations.
pub use nemo_relay_runtime::api::optimization;
/// Plugin contracts, configuration, and activation.
///
/// Initialize through the functions in this module so the SDK composes its
/// built-in plugins before validation and activation. Host-only constructors
/// are not available on the returned handle:
///
/// ```compile_fail,E0624
/// use nemo_relay::plugin::{PluginConfig, PluginHostActivation};
/// async fn bypass_composition() {
///     let _ = PluginHostActivation::initialize_exact(PluginConfig::default()).await;
/// }
/// ```
pub mod plugin {
    pub use nemo_relay_builtins::plugin::{
        ConfigReport, PluginConfig, PluginError, PluginHostActivation, PluginHostConfig,
        PluginHostReport, deregister_plugin, ensure_builtin_plugins_registered, initialize,
        initialize_exact, initialize_with_diagnostics, initialize_with_verified_specs,
        list_plugin_kinds, lookup_plugin, plugin_config_schema, register_plugin, validate,
        validate_exact, validate_static_plugin_config,
    };
    pub use nemo_relay_plugin::component::{
        ConfigDiagnostic, ConfigPolicy, DiagnosticLevel, Plugin, PluginComponentSpec,
        PluginRegistrar, PluginRegistration, PluginRegistrationCleanupOutcome, Result,
        RuntimeDiagnostic, UnsupportedBehavior, apply_global_config_policy,
    };
}
/// Structured resource measurements and on-demand collection.
pub mod resource_metrics {
    pub use nemo_relay_builtins::resource_metrics::{RESOURCE_METRICS_EVENT_NAME, collect};
    pub use nemo_relay_resource_metrics::config::*;
    pub use nemo_relay_types::api::resource_metrics::*;
}
