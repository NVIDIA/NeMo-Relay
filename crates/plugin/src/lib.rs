// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared in-process plugin contracts, stable native ABI, and Rust authoring helpers.
//!
//! This crate intentionally does not depend on the `nemo-relay` runtime crate.
//! Native plugins built with it communicate with a host through versioned
//! checked ABI-safe values and foreign futures.

/// Execution support for native component adapters.
#[doc(hidden)]
pub mod abi_executor;
/// Independently versioned checked interface declarations.
pub use nemo_relay_plugin_abi as abi;
/// Lossless SDK JSON conversions.
pub mod abi_value;
/// Native authoring adapters.
#[doc(hidden)]
pub mod checked;
/// Diagnostic submission and authorized reads.
pub mod diagnostics;
mod executor_config;
mod logging;
mod runtime_report;

pub use executor_config::NativeExecutorConfig;
pub use logging::PluginLogLevel;
pub use runtime_report::{RuntimeDiagnostic, RuntimeDiagnostics};

/// Shared built-in component contracts.
pub mod component;
pub use checked::{
    ConditionalMiddlewareGuardrailHandle, EmitMarkOptions, LlmExecutionContext, LlmJsonAsyncStream,
    LlmNext, LlmRequestContext, LlmResponseContext, LlmSanitizeRequestContext,
    LlmSanitizeResponseContext, LlmStreamNext, NativePlugin, Plugin, PluginContext, PluginRuntime,
    PluginTaskSpawner, ScopeGuard, ScopeHandle, ScopeOptions, ScopeStack, ScopeStackBinding,
    ThreadScopeStackGuard, ToolExecutionContext, ToolNext,
};
pub use nemo_relay_types::Json;
pub use nemo_relay_types::api::event::{
    CategoryProfile, DataSchema, Event, EventCategory, EventSanitizeFields, LogSeverity,
    METRIC_DATA_SCHEMA_NAME, METRIC_DATA_SCHEMA_VERSION, MetricEnvelope, MetricKind,
    MetricMeasurement, MetricValueType, PendingMarkSpec, ScopeCategory,
};
pub use nemo_relay_types::api::llm::{LlmAttributes, LlmRequest, LlmRequestInterceptOutcome};
pub use nemo_relay_types::api::registry::{
    RuntimeRegistrationIdentity, RuntimeRegistrationKind, RuntimeRegistrationOwner,
    RuntimeRegistrationOwnerKind,
};
pub use nemo_relay_types::api::scope::{HandleAttributes, ScopeAttributes, ScopeType};
pub use nemo_relay_types::api::scope::{ScopeStackEntry, ScopeStackSnapshot};
pub use nemo_relay_types::api::tool::{
    TOOL_EXECUTION_INTERCEPT_OUTCOME_SCHEMA, TOOL_EXECUTION_RESULT_SCHEMA, ToolAttributes,
    ToolExecutionInterceptOutcome, ToolExecutionResult,
};
pub use nemo_relay_types::codec::identity::{BuiltinLlmCodec, LlmCodecIdentity};
pub use nemo_relay_types::codec::optimization::{
    LlmOptimizationContribution, LlmOptimizationEvidenceQuality, LlmOptimizationKind,
    LlmOptimizationModel, LlmOptimizationModelTransition, LlmOptimizationPayload,
    LlmOptimizationSummary, LlmOptimizationSummaryStatus, LlmOptimizationTokenImpact,
    LlmOptimizationTokens,
};
pub use nemo_relay_types::codec::request::AnnotatedLlmRequest;
pub use nemo_relay_types::codec::response::AnnotatedLlmResponse;
pub use nemo_relay_types::plugin::http_error_message;
pub use nemo_relay_types::plugin::{ConfigDiagnostic, DiagnosticLevel, PluginError, PluginFuture};
pub use serde_json::Map;
/// Result returned by native SDK helpers.
pub type Result<T> = checked::Result<T>;

/// Export a checked native plugin with its SDK-required contract version.
#[macro_export]
macro_rules! nemo_relay_plugin {
    (kind = $kind:literal, $constructor:expr) => {
        $crate::nemo_relay_checked_plugin!(kind = $kind, $constructor);
    };
}
