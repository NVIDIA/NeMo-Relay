// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust SDK for NeMo Relay out-of-process gRPC worker plugins.
//!
//! # Invocation cancellation
//!
//! The `nvidia.nemo.relay.worker.v1` service tracks active unary and streaming callbacks by the
//! host-provided invocation ID. Relay sends `CancelInvocation` when a managed
//! caller is cancelled, an invocation times out, or a host stream is abandoned.
//! The SDK aborts the matching async callback task, waits for callback-owned
//! cleanup, and reports `worker.cancelled`; cancellation of an unknown, completed, or already
//! cancelled ID returns a negative acknowledgment.
//!
//! Cancellation is cooperative. Dropping an async callback future releases its
//! Rust-owned resources, but an accepted acknowledgment does not prove that
//! arbitrary blocking work started by the callback has stopped.

/// Negotiated, bounded plugin diagnostic submission and delivery controls.
pub mod diagnostics;
mod invocation;
use invocation::{CallbackInvocation, CallbackPayload};

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::future::Future;
use std::net::{SocketAddr, ToSocketAddrs};
#[cfg(unix)]
use std::os::unix::fs::FileTypeExt;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::{SystemTime, UNIX_EPOCH};

use futures_util::{Stream, StreamExt};
#[cfg(unix)]
use hyper_util::rt::TokioIo;
pub use nemo_relay_types::Json;
pub use nemo_relay_types::api::event::{
    DataSchema, Event, EventCategory, EventSanitizeFields, LogSeverity, METRIC_DATA_SCHEMA_NAME,
    METRIC_DATA_SCHEMA_VERSION, MetricEnvelope, MetricKind, MetricMeasurement, MetricValueType,
    PendingMarkSpec,
};
pub use nemo_relay_types::api::llm::{LlmRequest, LlmRequestInterceptOutcome};
pub use nemo_relay_types::api::registry::{
    RuntimeRegistrationIdentity, RuntimeRegistrationKind, RuntimeRegistrationOwner,
    RuntimeRegistrationOwnerKind,
};
pub use nemo_relay_types::api::scope::{ScopeStackEntry, ScopeStackSnapshot, ScopeType};
pub use nemo_relay_types::api::tool::{
    ToolExecutionContext, ToolExecutionInterceptOutcome, ToolExecutionResult,
};
pub use nemo_relay_types::codec::identity::{BuiltinLlmCodec, LlmCodecIdentity};
pub use nemo_relay_types::codec::optimization::{
    LlmOptimizationContribution, LlmOptimizationEvidenceQuality, LlmOptimizationKind,
    LlmOptimizationModel, LlmOptimizationModelTransition, LlmOptimizationPayload,
    LlmOptimizationSummary, LlmOptimizationSummaryStatus, LlmOptimizationTokenImpact,
    LlmOptimizationTokens,
};
pub use nemo_relay_types::codec::request::{ANNOTATED_LLM_REQUEST_SCHEMA, AnnotatedLlmRequest};
pub use nemo_relay_types::codec::response::AnnotatedLlmResponse;
pub use nemo_relay_types::plugin::http_error_message;
pub use nemo_relay_types::plugin::{ConfigDiagnostic, DiagnosticLevel, PluginError, PluginFuture};
use nemo_relay_worker_proto::conversions::{self, WireValue};
use nemo_relay_worker_proto::nvidia_worker_v1 as wire;
use nemo_relay_worker_proto::nvidia_worker_v1::WORKER_PROTOCOL;
use nemo_relay_worker_proto::nvidia_worker_v1::plugin_worker_service_server::{
    PluginWorkerService as PluginWorker, PluginWorkerServiceServer as PluginWorkerServer,
};
use nemo_relay_worker_proto::nvidia_worker_v1::relay_host_runtime_service_client::RelayHostRuntimeServiceClient as RelayHostRuntimeClient;
use nemo_relay_worker_proto::nvidia_worker_v1::*;
use nemo_relay_worker_proto::{
    CallbackMessageLimitService, DOWNSTREAM_CONTINUATION_METADATA, WorkerCallbackMessageLimits,
};
type JsonEnvelope = wire::JsonValue;
type StreamChunk = wire::InvokeStreamResponse;
type LogLevel = wire::LogSeverity;
type ProtoRuntimeDiagnostic = wire::RuntimeDiagnostic;
pub use serde_json::Map;
use tokio::net::TcpListener;
#[cfg(unix)]
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{OnceCell, mpsc, watch};
use tokio_stream::wrappers::TcpListenerStream;
#[cfg(unix)]
use tokio_stream::wrappers::UnixListenerStream;
use tonic::transport::{Channel, Endpoint, Server};
use tonic::{Request, Response, Status};
#[cfg(unix)]
use tower::service_fn;

/// SDK result type.
pub type Result<T> = std::result::Result<T, WorkerSdkError>;

/// One bounded runtime diagnostic reported by the Relay host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeDiagnostic {
    /// Stable identifier for the diagnostic condition.
    pub code: String,
    /// Most recently recorded message for this condition.
    pub message: String,
    /// Total number of occurrences recorded for this condition.
    pub count: u64,
}

/// Opaque activation-owned key for removing a worker-created gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConditionalMiddlewareGuardrailHandle {
    handle: String,
    callback_name: String,
}

/// Bounded snapshot of active host runtime diagnostics.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimeDiagnostics {
    entries: Vec<RuntimeDiagnostic>,
}

impl RuntimeDiagnostics {
    /// Return diagnostics in stable code order.
    pub fn entries(&self) -> &[RuntimeDiagnostic] {
        &self.entries
    }

    /// Return a diagnostic by its stable code.
    pub fn get(&self, code: &str) -> Option<&RuntimeDiagnostic> {
        self.entries
            .iter()
            .find(|diagnostic| diagnostic.code == code)
    }
}

/// Boxed future returned by async worker callbacks.
pub type BoxFutureResult<T> = Pin<Box<dyn Future<Output = Result<T>> + Send>>;

/// Boxed JSON stream returned by streaming worker callbacks.
pub type JsonStream = Pin<Box<dyn tokio_stream::Stream<Item = Result<Json>> + Send>>;

tokio::task_local! {
    static TASK_SCOPE_CONTEXT: Option<ScopeContext>;
    static TASK_CALLBACK_NAME: Option<String>;
}

thread_local! {
    static THREAD_SCOPE_CONTEXT: RefCell<Option<ScopeContext>> = const { RefCell::new(None) };
    static THREAD_CALLBACK_NAME: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Error returned by worker SDK callbacks and runtime helpers.
#[derive(Debug, thiserror::Error)]
pub enum WorkerSdkError {
    /// Invalid host-provided input.
    #[error("invalid input: {0}")]
    InvalidInput(String),
    /// Worker callback failed.
    #[error("callback failed: {0}")]
    Callback(String),
    /// Worker transport failed.
    #[error("transport failed: {0}")]
    Transport(String),
    /// An unchanged downstream continuation failure, preserving its original display.
    /// Propagate this error to retain provenance; constructing a new error represents
    /// a distinct plugin failure. The host checks the invocation's continuation ID.
    #[error("{source}")]
    Downstream {
        /// Original SDK error, with unchanged execution/reporting semantics.
        source: Box<WorkerSdkError>,
        /// Host-issued continuation that returned the failure.
        continuation_id: String,
    },
    /// JSON serialization failed.
    #[error("serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
}

impl From<WorkerSdkError> for PluginError {
    fn from(error: WorkerSdkError) -> Self {
        match error {
            WorkerSdkError::InvalidInput(message) => Self::InvalidConfig(message),
            WorkerSdkError::Serialization(error) => Self::Serialization(error),
            other => Self::RegistrationFailed(other.to_string()),
        }
    }
}

/// Optional fields supported by the additive `nvidia.nemo.relay.worker.v1` mark-emission request.
#[derive(Debug, Clone, Default)]
pub struct EmitMarkOptions {
    /// Schema identifier for the mark's opaque data payload.
    pub data_schema: Option<DataSchema>,
    /// Telemetry severity for OTLP log projection.
    pub severity: Option<LogSeverity>,
}

/// Process-log severity for records emitted through the Relay host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginLogLevel {
    /// An error prevented the plugin from completing expected work.
    Error,
    /// The plugin encountered a recoverable condition.
    Warn,
    /// Normal plugin lifecycle information.
    Info,
    /// Diagnostic information useful while debugging a plugin.
    Debug,
    /// Fine-grained diagnostic information.
    Trace,
}

/// Emit a formatted trace-level gRPC worker record; await the returned future.
#[macro_export]
macro_rules! relay_trace { ($runtime:expr, target: $target:expr, fields: $fields:expr, $($arg:tt)+) => {{ async { let message = format!($($arg)+); $runtime.log($crate::PluginLogLevel::Trace, $target, &message, Some(&$fields)).await } }}; ($runtime:expr, target: $target:expr, $($arg:tt)+) => {{ async { let message = format!($($arg)+); $runtime.log($crate::PluginLogLevel::Trace, $target, &message, None).await } }}; }
/// Emit a formatted debug-level gRPC worker record; await the returned future.
#[macro_export]
macro_rules! relay_debug { ($runtime:expr, target: $target:expr, fields: $fields:expr, $($arg:tt)+) => {{ async { let message = format!($($arg)+); $runtime.log($crate::PluginLogLevel::Debug, $target, &message, Some(&$fields)).await } }}; ($runtime:expr, target: $target:expr, $($arg:tt)+) => {{ async { let message = format!($($arg)+); $runtime.log($crate::PluginLogLevel::Debug, $target, &message, None).await } }}; }
/// Emit a formatted info-level gRPC worker record; await the returned future.
#[macro_export]
macro_rules! relay_info { ($runtime:expr, target: $target:expr, fields: $fields:expr, $($arg:tt)+) => {{ async { let message = format!($($arg)+); $runtime.log($crate::PluginLogLevel::Info, $target, &message, Some(&$fields)).await } }}; ($runtime:expr, target: $target:expr, $($arg:tt)+) => {{ async { let message = format!($($arg)+); $runtime.log($crate::PluginLogLevel::Info, $target, &message, None).await } }}; }
/// Emit a formatted warn-level gRPC worker record; await the returned future.
#[macro_export]
macro_rules! relay_warn { ($runtime:expr, target: $target:expr, fields: $fields:expr, $($arg:tt)+) => {{ async { let message = format!($($arg)+); $runtime.log($crate::PluginLogLevel::Warn, $target, &message, Some(&$fields)).await } }}; ($runtime:expr, target: $target:expr, $($arg:tt)+) => {{ async { let message = format!($($arg)+); $runtime.log($crate::PluginLogLevel::Warn, $target, &message, None).await } }}; }
/// Emit a formatted error-level gRPC worker record; await the returned future.
#[macro_export]
macro_rules! relay_error { ($runtime:expr, target: $target:expr, fields: $fields:expr, $($arg:tt)+) => {{ async { let message = format!($($arg)+); $runtime.log($crate::PluginLogLevel::Error, $target, &message, Some(&$fields)).await } }}; ($runtime:expr, target: $target:expr, $($arg:tt)+) => {{ async { let message = format!($($arg)+); $runtime.log($crate::PluginLogLevel::Error, $target, &message, None).await } }}; }

impl PluginLogLevel {
    fn wire_value(self) -> i32 {
        match self {
            Self::Error => LogLevel::Error as i32,
            Self::Warn => LogLevel::Warn as i32,
            Self::Info => LogLevel::Info as i32,
            Self::Debug => LogLevel::Debug as i32,
            Self::Trace => LogLevel::Trace as i32,
        }
    }
}

/// Trait implemented by Rust out-of-process worker plugins.
pub trait Plugin: Send + Sync + 'static {
    /// Returns the unique plugin kind.
    fn plugin_kind(&self) -> &str;

    /// Whether multiple configured components may use this plugin kind.
    fn allows_multiple_components(&self) -> bool {
        true
    }

    /// Validate one component before runtime registration.
    fn validate(&self, _config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        Vec::new()
    }

    /// Registers callbacks into the worker context.
    fn register<'a>(
        &'a self,
        config: &'a Map<String, Json>,
        ctx: &'a mut PluginContext,
    ) -> PluginFuture<'a, ()>;
}

/// Alternate export of the worker [`Plugin`] authoring trait.
pub use Plugin as WorkerPlugin;

type SubscriberFn = Arc<dyn Fn(&Event) + Send + Sync>;
type EventSanitizeFn =
    Arc<dyn Fn(&Event, EventSanitizeFields) -> BoxFutureResult<EventSanitizeFields> + Send + Sync>;
type EventMetadataInjectorFn =
    Arc<dyn Fn(Arc<Event>) -> BoxFutureResult<BTreeMap<String, Json>> + Send + Sync>;
type ToolSanitizeFn = Arc<dyn Fn(&str, Json) -> BoxFutureResult<Json> + Send + Sync>;
type ToolConditionalFn = Arc<dyn Fn(String, Json) -> BoxFutureResult<Option<String>> + Send + Sync>;
type ToolRequestFn = Arc<dyn Fn(String, Json) -> BoxFutureResult<Json> + Send + Sync>;
type ToolExecutionFn = Arc<
    dyn Fn(ToolExecutionContext, ToolNext) -> BoxFutureResult<ToolExecutionInterceptOutcome>
        + Send
        + Sync,
>;
type LlmSanitizeRequestFn = Arc<
    dyn Fn(LlmRequest, LlmSanitizeRequestContext) -> BoxFutureResult<Option<LlmRequest>>
        + Send
        + Sync,
>;
type LlmSanitizeResponseFn =
    Arc<dyn Fn(Json, LlmSanitizeResponseContext) -> BoxFutureResult<Option<Json>> + Send + Sync>;

/// Active codec identity and capability for an LLM request.
#[derive(Clone)]
pub struct LlmSanitizeRequestContext {
    /// Identity of the active codec.
    pub codec: LlmCodecIdentity,
    runtime: Option<PluginRuntime>,
    codec_capability_id: Option<String>,
    invocation_id: Option<String>,
}

/// Active codec identity and capability for an LLM response.
#[derive(Clone)]
pub struct LlmSanitizeResponseContext {
    /// Identity of the active codec.
    pub codec: LlmCodecIdentity,
    runtime: Option<PluginRuntime>,
    codec_capability_id: Option<String>,
    invocation_id: Option<String>,
}

/// Request codec context supplied to an LLM execution intercept.
pub type LlmRequestContext = LlmSanitizeRequestContext;

/// Response codec context supplied to an LLM execution intercept.
pub type LlmResponseContext = LlmSanitizeResponseContext;

impl LlmSanitizeRequestContext {
    /// Resolves the active request codec for this callback.
    #[must_use]
    pub fn resolve_codec(&self) -> Option<WorkerRequestCodec> {
        Some(WorkerRequestCodec {
            runtime: self.runtime.clone()?,
            capability_id: self.codec_capability_id.clone()?,
            invocation_id: self.invocation_id.clone()?,
        })
    }
}

impl LlmSanitizeResponseContext {
    /// Resolves the active response codec for this callback.
    #[must_use]
    pub fn resolve_codec(&self) -> Option<WorkerResponseCodec> {
        Some(WorkerResponseCodec {
            runtime: self.runtime.clone()?,
            capability_id: self.codec_capability_id.clone()?,
            invocation_id: self.invocation_id.clone()?,
        })
    }
}

/// Invocation-scoped proxy for the active LLM request codec.
#[derive(Clone)]
pub struct WorkerRequestCodec {
    runtime: PluginRuntime,
    capability_id: String,
    invocation_id: String,
}

impl WorkerRequestCodec {
    /// Decodes an opaque request into its normalized representation.
    pub async fn decode(&self, request: &LlmRequest) -> Result<AnnotatedLlmRequest> {
        self.runtime
            .decode_llm_codec_request(&self.capability_id, &self.invocation_id, request)
            .await
    }
    /// Encodes normalized request changes onto the original opaque request.
    pub async fn encode(
        &self,
        annotated: &AnnotatedLlmRequest,
        original: &LlmRequest,
    ) -> Result<LlmRequest> {
        self.runtime
            .encode_llm_codec_request(
                &self.capability_id,
                &self.invocation_id,
                annotated,
                original,
            )
            .await
    }
}

/// Invocation-scoped proxy for the active LLM response codec.
#[derive(Clone)]
pub struct WorkerResponseCodec {
    runtime: PluginRuntime,
    capability_id: String,
    invocation_id: String,
}

impl WorkerResponseCodec {
    /// Decodes an opaque response into its normalized representation.
    pub async fn decode(&self, response: &Json) -> Result<AnnotatedLlmResponse> {
        self.runtime
            .decode_llm_codec_response(&self.capability_id, &self.invocation_id, response)
            .await
    }
}

/// Invocation-scoped codec identities and operations for an LLM execution interceptor.
///
/// This context identifies the codecs selected when Relay created the managed
/// invocation. Rewriting a request into another provider's wire format does
/// not change these identities; codec operations reject incompatible payloads.
#[derive(Clone)]
pub struct LlmExecutionContext {
    request_codec: LlmRequestContext,
    response_codec: Option<LlmResponseContext>,
}

impl std::fmt::Debug for LlmExecutionContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LlmExecutionContext")
            .field("request_codec", &self.request_codec.codec)
            .field(
                "response_codec",
                &self.response_codec.as_ref().map(|context| &context.codec),
            )
            .finish_non_exhaustive()
    }
}

impl LlmExecutionContext {
    /// Request codec identity and invocation-scoped operations.
    #[must_use]
    pub fn request_codec(&self) -> &LlmRequestContext {
        &self.request_codec
    }

    /// Unary response codec identity and invocation-scoped decode operation.
    ///
    /// Streaming execution returns `None` because Relay response codecs decode
    /// completed provider responses, not individual stream chunks.
    #[must_use]
    pub fn response_codec(&self) -> Option<&LlmResponseContext> {
        self.response_codec.as_ref()
    }
}

type LlmConditionalFn = Arc<dyn Fn(LlmRequest) -> BoxFutureResult<Option<String>> + Send + Sync>;
type ConditionalMiddlewareFn = Arc<
    dyn Fn(BTreeSet<RuntimeRegistrationKind>, String) -> BoxFutureResult<Option<String>>
        + Send
        + Sync,
>;
type LlmRequestFn = Arc<
    dyn Fn(
            String,
            LlmRequest,
            Option<AnnotatedLlmRequest>,
        ) -> BoxFutureResult<LlmRequestInterceptOutcome>
        + Send
        + Sync,
>;
type LlmExecutionFn = Arc<
    dyn Fn(&str, LlmRequest, LlmExecutionContext, LlmNext) -> BoxFutureResult<Json> + Send + Sync,
>;
type LlmStreamExecutionFn = Arc<
    dyn Fn(&str, LlmRequest, LlmExecutionContext, LlmStreamNext) -> BoxFutureResult<JsonStream>
        + Send
        + Sync,
>;

#[derive(Default)]
struct WorkerHandlers {
    registrations: Vec<Registration>,
    conditional_middleware_guardrails: Vec<ConditionalMiddlewareGuardrailRegistration>,
    conditional_middleware_callbacks: HashMap<String, ConditionalMiddlewareFn>,
    subscribers: HashMap<String, SubscriberFn>,
    event_metadata_injectors: HashMap<String, EventMetadataInjectorFn>,
    mark_sanitizers: HashMap<String, EventSanitizeFn>,
    scope_start_sanitizers: HashMap<String, EventSanitizeFn>,
    scope_end_sanitizers: HashMap<String, EventSanitizeFn>,
    tool_sanitize_requests: HashMap<String, ToolSanitizeFn>,
    tool_sanitize_responses: HashMap<String, ToolSanitizeFn>,
    tool_conditionals: HashMap<String, ToolConditionalFn>,
    tool_requests: HashMap<String, ToolRequestFn>,
    tool_executions: HashMap<String, ToolExecutionFn>,
    llm_sanitize_requests: HashMap<String, LlmSanitizeRequestFn>,
    llm_sanitize_responses: HashMap<String, LlmSanitizeResponseFn>,
    llm_conditionals: HashMap<String, LlmConditionalFn>,
    llm_requests: HashMap<String, LlmRequestFn>,
    llm_executions: HashMap<String, LlmExecutionFn>,
    llm_stream_executions: HashMap<String, LlmStreamExecutionFn>,
}

/// Registration context passed to [`WorkerPlugin::register`].
pub struct PluginContext {
    handlers: WorkerHandlers,
    runtime: Option<PluginRuntime>,
}

impl PluginContext {
    /// Creates an empty worker registration context.
    pub fn new() -> Self {
        Self {
            handlers: WorkerHandlers::default(),
            runtime: None,
        }
    }

    /// Creates an empty worker registration context with a host runtime handle.
    pub fn with_runtime(runtime: PluginRuntime) -> Self {
        Self {
            handlers: WorkerHandlers::default(),
            runtime: Some(runtime),
        }
    }

    /// Returns the host runtime handle for event and scope operations.
    pub fn runtime(&self) -> Option<PluginRuntime> {
        self.runtime.clone()
    }

    /// Declares a callback-based gate installed with this component activation.
    pub fn register_conditional_middleware_guardrail<F, Fut>(
        &mut self,
        name: &str,
        kinds: BTreeSet<RuntimeRegistrationKind>,
        registration_name: &str,
        callback: F,
    ) where
        F: Fn(BTreeSet<RuntimeRegistrationKind>, String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Option<String>>> + Send + 'static,
    {
        self.handlers.conditional_middleware_guardrails.push(
            ConditionalMiddlewareGuardrailRegistration {
                name: name.into(),
                kinds: kinds
                    .into_iter()
                    .map(registration_surface)
                    .map(|surface| surface as i32)
                    .collect(),
                registration_name: registration_name.into(),
                reason: String::new(),
                callback: true,
            },
        );
        self.handlers.conditional_middleware_callbacks.insert(
            name.into(),
            Arc::new(move |kinds, registration_name| Box::pin(callback(kinds, registration_name))),
        );
    }

    /// Registers an event subscriber.
    pub fn register_subscriber<F>(&mut self, name: &str, callback: F)
    where
        F: Fn(&Event) + Send + Sync + 'static,
    {
        self.push_registration(name, RegistrationSurface::Subscriber, 0, false);
        self.handlers
            .subscribers
            .insert(name.into(), Arc::new(callback));
    }

    /// Registers an Event metadata injector.
    pub fn register_event_metadata_injector<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) where
        F: Fn(Arc<Event>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<BTreeMap<String, Json>>> + Send + 'static,
    {
        self.push_registration(
            name,
            RegistrationSurface::EventMetadataInjector,
            priority,
            false,
        );
        self.handlers.event_metadata_injectors.insert(
            name.into(),
            Arc::new(move |event| Box::pin(callback(event))),
        );
    }

    fn register_event_sanitizer<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        surface: RegistrationSurface,
        callback: F,
    ) where
        F: Fn(&Event, EventSanitizeFields) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<EventSanitizeFields>> + Send + 'static,
    {
        self.push_registration(name, surface, priority, false);
        let sanitizers = match surface {
            RegistrationSurface::MarkSanitizeGuardrail => &mut self.handlers.mark_sanitizers,
            RegistrationSurface::ScopeSanitizeStartGuardrail => {
                &mut self.handlers.scope_start_sanitizers
            }
            RegistrationSurface::ScopeSanitizeEndGuardrail => {
                &mut self.handlers.scope_end_sanitizers
            }
            _ => unreachable!("event sanitizer registration requires an event sanitizer surface"),
        };
        sanitizers.insert(
            name.into(),
            Arc::new(move |event, fields| Box::pin(callback(event, fields))),
        );
    }

    /// Registers a mark event sanitizer.
    pub fn register_mark_sanitize_guardrail<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) where
        F: Fn(&Event, EventSanitizeFields) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<EventSanitizeFields>> + Send + 'static,
    {
        self.register_event_sanitizer(
            name,
            priority,
            RegistrationSurface::MarkSanitizeGuardrail,
            callback,
        );
    }

    /// Registers a scope-start event sanitizer.
    pub fn register_scope_sanitize_start_guardrail<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) where
        F: Fn(&Event, EventSanitizeFields) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<EventSanitizeFields>> + Send + 'static,
    {
        self.register_event_sanitizer(
            name,
            priority,
            RegistrationSurface::ScopeSanitizeStartGuardrail,
            callback,
        );
    }

    /// Registers a scope-end event sanitizer.
    pub fn register_scope_sanitize_end_guardrail<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) where
        F: Fn(&Event, EventSanitizeFields) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<EventSanitizeFields>> + Send + 'static,
    {
        self.register_event_sanitizer(
            name,
            priority,
            RegistrationSurface::ScopeSanitizeEndGuardrail,
            callback,
        );
    }

    /// Registers a tool sanitize-request guardrail.
    pub fn register_tool_sanitize_request_guardrail<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) where
        F: Fn(&str, Json) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Json>> + Send + 'static,
    {
        self.push_registration(
            name,
            RegistrationSurface::ToolSanitizeRequestGuardrail,
            priority,
            false,
        );
        self.handlers.tool_sanitize_requests.insert(
            name.into(),
            Arc::new(move |tool_name, value| Box::pin(callback(tool_name, value))),
        );
    }

    /// Registers a tool sanitize-response guardrail.
    pub fn register_tool_sanitize_response_guardrail<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) where
        F: Fn(&str, Json) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Json>> + Send + 'static,
    {
        self.push_registration(
            name,
            RegistrationSurface::ToolSanitizeResponseGuardrail,
            priority,
            false,
        );
        self.handlers.tool_sanitize_responses.insert(
            name.into(),
            Arc::new(move |tool_name, value| Box::pin(callback(tool_name, value))),
        );
    }

    /// Registers a tool conditional-execution guardrail.
    pub fn register_tool_conditional_execution_guardrail<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) where
        F: Fn(String, Json) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Option<String>>> + Send + 'static,
    {
        self.push_registration(
            name,
            RegistrationSurface::ToolConditionalExecutionGuardrail,
            priority,
            false,
        );
        self.handlers.tool_conditionals.insert(
            name.into(),
            Arc::new(move |tool_name, value| Box::pin(callback(tool_name, value))),
        );
    }

    /// Registers a tool request intercept.
    pub fn register_tool_request_intercept<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        break_chain: bool,
        callback: F,
    ) where
        F: Fn(String, Json) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Json>> + Send + 'static,
    {
        self.push_registration(
            name,
            RegistrationSurface::ToolRequestIntercept,
            priority,
            break_chain,
        );
        self.handlers.tool_requests.insert(
            name.into(),
            Arc::new(move |tool_name, value| Box::pin(callback(tool_name, value))),
        );
    }

    /// Registers a tool execution intercept.
    ///
    /// The callback returns a [`ToolExecutionInterceptOutcome`]. Calling
    /// [`ToolNext::call`] continues the chain and returns the downstream
    /// [`ToolExecutionResult`]; Relay retains downstream pending marks.
    /// `ToolNext` may be called repeatedly or concurrently while the callback
    /// is active. Each call snapshots its visible worker scope stack, and Relay
    /// rejects late calls or cancels unfinished calls when the callback settles.
    pub fn register_tool_execution_intercept<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) where
        F: Fn(ToolExecutionContext, ToolNext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<ToolExecutionInterceptOutcome>> + Send + 'static,
    {
        self.push_registration(
            name,
            RegistrationSurface::ToolExecutionIntercept,
            priority,
            false,
        );
        self.handlers.tool_executions.insert(
            name.into(),
            Arc::new(move |context, next| Box::pin(callback(context, next))),
        );
    }

    /// Registers an LLM sanitize-request guardrail.
    pub fn register_llm_sanitize_request_guardrail<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) where
        F: Fn(LlmRequest, LlmSanitizeRequestContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Option<LlmRequest>>> + Send + 'static,
    {
        self.push_registration(
            name,
            RegistrationSurface::LlmSanitizeRequestGuardrail,
            priority,
            false,
        );
        self.handlers.llm_sanitize_requests.insert(
            name.into(),
            Arc::new(move |request, context| Box::pin(callback(request, context))),
        );
    }

    /// Registers an LLM sanitize-response guardrail.
    pub fn register_llm_sanitize_response_guardrail<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) where
        F: Fn(Json, LlmSanitizeResponseContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Option<Json>>> + Send + 'static,
    {
        self.push_registration(
            name,
            RegistrationSurface::LlmSanitizeResponseGuardrail,
            priority,
            false,
        );
        self.handlers.llm_sanitize_responses.insert(
            name.into(),
            Arc::new(move |response, context| Box::pin(callback(response, context))),
        );
    }

    /// Registers an LLM conditional-execution guardrail.
    pub fn register_llm_conditional_execution_guardrail<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) where
        F: Fn(LlmRequest) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Option<String>>> + Send + 'static,
    {
        self.push_registration(
            name,
            RegistrationSurface::LlmConditionalExecutionGuardrail,
            priority,
            false,
        );
        self.handlers.llm_conditionals.insert(
            name.into(),
            Arc::new(move |request| Box::pin(callback(request))),
        );
    }

    /// Registers an LLM request intercept.
    pub fn register_llm_request_intercept<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        break_chain: bool,
        callback: F,
    ) where
        F: Fn(String, LlmRequest, Option<AnnotatedLlmRequest>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<LlmRequestInterceptOutcome>> + Send + 'static,
    {
        self.push_registration(
            name,
            RegistrationSurface::LlmRequestIntercept,
            priority,
            break_chain,
        );
        self.handlers.llm_requests.insert(
            name.into(),
            Arc::new(move |model_name, request, annotated| {
                Box::pin(callback(model_name, request, annotated))
            }),
        );
    }

    /// Registers an LLM execution intercept.
    ///
    /// [`LlmNext::call`] may run repeatedly or concurrently while the callback
    /// is active. Each call snapshots its visible worker scope stack, and Relay
    /// rejects late calls or cancels unfinished calls when the callback settles.
    pub fn register_llm_execution_intercept<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) where
        F: Fn(&str, LlmRequest, LlmExecutionContext, LlmNext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Json>> + Send + 'static,
    {
        self.push_registration(
            name,
            RegistrationSurface::LlmExecutionIntercept,
            priority,
            false,
        );
        self.handlers.llm_executions.insert(
            name.into(),
            Arc::new(move |model, request, context, next| {
                Box::pin(callback(model, request, context, next))
            }),
        );
    }

    /// Registers an LLM stream execution intercept.
    ///
    /// [`LlmStreamNext::call`] may run repeatedly or concurrently. Each call
    /// snapshots its visible worker scope stack. The interceptor's returned
    /// stream keeps the callback active until it closes; Relay then rejects
    /// late calls and cancels unfinished calls.
    pub fn register_llm_stream_execution_intercept<F, Fut>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) where
        F: Fn(&str, LlmRequest, LlmExecutionContext, LlmStreamNext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<JsonStream>> + Send + 'static,
    {
        self.push_registration(
            name,
            RegistrationSurface::LlmStreamExecutionIntercept,
            priority,
            false,
        );
        self.handlers.llm_stream_executions.insert(
            name.into(),
            Arc::new(move |model, request, context, next| {
                Box::pin(callback(model, request, context, next))
            }),
        );
    }

    fn push_registration(
        &mut self,
        name: &str,
        surface: RegistrationSurface,
        priority: i32,
        break_chain: bool,
    ) {
        self.handlers.registrations.push(Registration {
            local_name: name.into(),
            surface: surface as i32,
            priority,
            break_chain,
        });
    }
}

impl Default for PluginContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Cloneable handle for calling the Relay host runtime from worker callbacks.
#[derive(Clone)]
pub struct PluginRuntime {
    diagnostics: Arc<OnceCell<diagnostics::WorkerDiagnostics>>,
    activation_id: String,
    auth_token: String,
    host_endpoint: String,
    host_channel: Arc<OnceCell<Channel>>,
    max_message_bytes: WorkerCallbackMessageLimits,
    callback_name: Option<String>,
    conditional_middleware_callbacks: Arc<Mutex<HashMap<String, ConditionalMiddlewareFn>>>,
}

impl PluginRuntime {
    fn credentials(&self) -> wire::Credentials {
        wire::Credentials {
            activation_id: self.activation_id.clone(),
            auth_token: self.auth_token.clone(),
        }
    }
    /// Negotiate once per activation. Disabled/unsupported facades skip lazy
    /// diagnostic factories and leave execution and operational logging intact.
    pub async fn diagnostics(&self) -> &diagnostics::WorkerDiagnostics {
        self.diagnostics
            .get_or_init(|| async {
                let channel = self
                    .host_channel
                    .get_or_try_init(|| connect_host_endpoint(&self.host_endpoint))
                    .await
                    .cloned();
                diagnostics::WorkerDiagnostics::connect(
                    channel,
                    self.activation_id.clone(),
                    self.auth_token.clone(),
                )
                .await
            })
            .await
    }
    /// Emits one operational log record through the Relay host's configured sinks.
    ///
    /// The host authenticates the activation and applies its own log-level filter.
    /// `target` identifies the plugin subsystem; pass an empty string to use the
    /// plugin's default target.
    pub async fn log(
        &self,
        level: PluginLogLevel,
        target: &str,
        message: &str,
        fields: Option<&Map<String, Json>>,
    ) -> Result<()> {
        let mut client = self.host_client().await?;
        let _response = client
            .log(Request::new(LogRequest {
                credentials: Some(self.credentials()),
                level: level.wire_value(),
                target: target.into(),
                message: message.into(),
                fields: fields.map(encode_wire).transpose()?,
            }))
            .await
            .map_err(|err| WorkerSdkError::Transport(err.to_string()))?
            .into_inner();
        Ok(())
    }

    /// Emits an info-level operational log record through the Relay host.
    pub async fn trace(
        &self,
        target: &str,
        message: &str,
        fields: Option<&Map<String, Json>>,
    ) -> Result<()> {
        self.log(PluginLogLevel::Trace, target, message, fields)
            .await
    }
    /// Emits a debug-level operational log record through the Relay host.
    pub async fn debug(
        &self,
        target: &str,
        message: &str,
        fields: Option<&Map<String, Json>>,
    ) -> Result<()> {
        self.log(PluginLogLevel::Debug, target, message, fields)
            .await
    }
    /// Emits an info-level operational log record through the Relay host.
    pub async fn info(
        &self,
        target: &str,
        message: &str,
        fields: Option<&Map<String, Json>>,
    ) -> Result<()> {
        self.log(PluginLogLevel::Info, target, message, fields)
            .await
    }

    /// Emits a warn-level operational log record through the Relay host.
    pub async fn warn(
        &self,
        target: &str,
        message: &str,
        fields: Option<&Map<String, Json>>,
    ) -> Result<()> {
        self.log(PluginLogLevel::Warn, target, message, fields)
            .await
    }

    /// Emits an error-level operational log record through the Relay host.
    pub async fn error(
        &self,
        target: &str,
        message: &str,
        fields: Option<&Map<String, Json>>,
    ) -> Result<()> {
        self.log(PluginLogLevel::Error, target, message, fields)
            .await
    }

    /// List global runtime registrations, optionally filtered by kind.
    pub async fn list_runtime_registrations(
        &self,
        kinds: Option<BTreeSet<RuntimeRegistrationKind>>,
    ) -> Result<Vec<RuntimeRegistrationIdentity>> {
        let mut client = self.host_client().await?;
        let response = client
            .list_runtime_registrations(Request::new(ListRuntimeRegistrationsRequest {
                credentials: Some(self.credentials()),
                kinds: kinds
                    .unwrap_or_default()
                    .into_iter()
                    .map(|kind| registration_surface(kind) as i32)
                    .collect(),
            }))
            .await
            .map_err(|error| WorkerSdkError::Transport(error.to_string()))?
            .into_inner();
        response
            .registrations
            .into_iter()
            .map(|v| conversions::decode(v).map_err(Into::into))
            .collect()
    }

    /// Register a callback-based gate owned by this worker activation.
    pub async fn register_conditional_middleware_guardrail<F, Fut>(
        &self,
        name: &str,
        kinds: BTreeSet<RuntimeRegistrationKind>,
        registration_name: &str,
        callback: F,
    ) -> Result<ConditionalMiddlewareGuardrailHandle>
    where
        F: Fn(BTreeSet<RuntimeRegistrationKind>, String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Option<String>>> + Send + 'static,
    {
        self.try_insert_conditional_middleware_callback(name, callback)?;
        let mut client = match self.host_client().await {
            Ok(client) => client,
            Err(error) => {
                self.remove_conditional_middleware_callback(name);
                return Err(error);
            }
        };
        let response = match client
            .register_conditional_middleware_guardrail(Request::new(
                RegisterConditionalMiddlewareGuardrailRequest {
                    credentials: Some(self.credentials()),
                    registration: Some(wire::ConditionalMiddlewareGuardrailRegistration {
                        name: name.into(),
                        kinds: kinds
                            .into_iter()
                            .map(|kind| registration_surface(kind) as i32)
                            .collect(),
                        registration_name: registration_name.into(),
                        reason: String::new(),
                        callback: true,
                    }),
                },
            ))
            .await
        {
            Ok(response) => response.into_inner(),
            Err(error) => {
                self.remove_conditional_middleware_callback(name);
                return Err(WorkerSdkError::Transport(error.to_string()));
            }
        };
        if let Err(error) = nemo_relay_worker_proto::validation::validate(&response) {
            self.remove_conditional_middleware_callback(name);
            return Err(WorkerSdkError::InvalidInput(error));
        }
        Ok(ConditionalMiddlewareGuardrailHandle {
            handle: response.handle,
            callback_name: name.into(),
        })
    }

    /// Deregister a callback-based gate owned by this worker activation.
    pub async fn deregister_conditional_middleware_guardrail(
        &self,
        handle: &ConditionalMiddlewareGuardrailHandle,
    ) -> Result<bool> {
        let mut client = self.host_client().await?;
        let response = client
            .deregister_conditional_middleware_guardrail(Request::new(
                DeregisterConditionalMiddlewareGuardrailRequest {
                    credentials: Some(self.credentials()),
                    handle: handle.handle.clone(),
                },
            ))
            .await
            .map_err(|error| WorkerSdkError::Transport(error.to_string()))?
            .into_inner();
        if response.removed {
            self.remove_conditional_middleware_callback(&handle.callback_name);
        }
        Ok(response.removed)
    }

    fn try_insert_conditional_middleware_callback<F, Fut>(
        &self,
        name: &str,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(BTreeSet<RuntimeRegistrationKind>, String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Option<String>>> + Send + 'static,
    {
        let mut callbacks = self
            .conditional_middleware_callbacks
            .lock()
            .map_err(|error| WorkerSdkError::Callback(error.to_string()))?;
        if callbacks.contains_key(name) {
            return Err(WorkerSdkError::InvalidInput(format!(
                "conditional middleware callback '{name}' is already registered"
            )));
        }
        callbacks.insert(
            name.into(),
            Arc::new(move |kinds, registration_name| Box::pin(callback(kinds, registration_name))),
        );
        Ok(())
    }

    fn remove_conditional_middleware_callback(&self, name: &str) {
        self.conditional_middleware_callbacks
            .lock()
            .expect("conditional middleware callback lock")
            .remove(name);
    }

    async fn decode_llm_codec_request(
        &self,
        capability_id: &str,
        invocation_id: &str,
        request: &LlmRequest,
    ) -> Result<AnnotatedLlmRequest> {
        let mut client = self.host_client().await?;
        let response = client
            .decode_llm_codec_request(Request::new(DecodeLlmCodecRequestRequest {
                credentials: Some(self.credentials()),
                codec_capability_id: capability_id.into(),
                invocation_id: invocation_id.into(),
                request: Some(encode_wire(request)?),
            }))
            .await
            .map_err(|err| WorkerSdkError::Transport(err.to_string()))?
            .into_inner();
        decode_rpc_result(response)
    }

    async fn encode_llm_codec_request(
        &self,
        capability_id: &str,
        invocation_id: &str,
        annotated: &AnnotatedLlmRequest,
        original: &LlmRequest,
    ) -> Result<LlmRequest> {
        let mut client = self.host_client().await?;
        let response = client
            .encode_llm_codec_request(Request::new(EncodeLlmCodecRequestRequest {
                credentials: Some(self.credentials()),
                codec_capability_id: capability_id.into(),
                invocation_id: invocation_id.into(),
                annotated_request: Some(encode_wire(annotated)?),
                original_request: Some(encode_wire(original)?),
            }))
            .await
            .map_err(|err| WorkerSdkError::Transport(err.to_string()))?
            .into_inner();
        decode_rpc_result(response)
    }

    async fn decode_llm_codec_response(
        &self,
        capability_id: &str,
        invocation_id: &str,
        response: &Json,
    ) -> Result<AnnotatedLlmResponse> {
        let mut client = self.host_client().await?;
        let response = client
            .decode_llm_codec_response(Request::new(DecodeLlmCodecResponseRequest {
                credentials: Some(self.credentials()),
                codec_capability_id: capability_id.into(),
                invocation_id: invocation_id.into(),
                response: Some(encode_wire(response)?),
            }))
            .await
            .map_err(|err| WorkerSdkError::Transport(err.to_string()))?
            .into_inner();
        decode_rpc_result(response)
    }
    /// Emits a mark event through the host runtime.
    pub async fn emit_mark(
        &self,
        name: &str,
        data: Option<Json>,
        metadata: Option<Json>,
    ) -> Result<()> {
        self.emit_mark_with_options(name, data, metadata, EmitMarkOptions::default())
            .await
    }

    /// Emits a mark event through the host runtime with optional schema and severity fields.
    pub async fn emit_mark_with_options(
        &self,
        name: &str,
        data: Option<Json>,
        metadata: Option<Json>,
        options: EmitMarkOptions,
    ) -> Result<()> {
        self.emit_mark_impl(name, data, metadata, options, None)
            .await
    }

    /// Emits a mark event through the host runtime with optional schema, severity, and category.
    pub async fn emit_mark_with_options_and_category(
        &self,
        name: &str,
        data: Option<Json>,
        metadata: Option<Json>,
        options: EmitMarkOptions,
        category: EventCategory,
    ) -> Result<()> {
        self.emit_mark_impl(name, data, metadata, options, Some(category))
            .await
    }

    async fn emit_mark_impl(
        &self,
        name: &str,
        data: Option<Json>,
        metadata: Option<Json>,
        options: EmitMarkOptions,
        category: Option<EventCategory>,
    ) -> Result<()> {
        let scope = self.current_scope_context();
        let mut client = self.host_client().await?;
        let _response = client
            .emit_mark(Request::new(EmitMarkRequest {
                credentials: Some(self.credentials()),
                scope,
                mark: Some(conversions::encode(PendingMarkSpec {
                    name: name.into(),
                    data,
                    metadata,
                    data_schema: options.data_schema,
                    severity: options.severity,
                    category,
                    category_profile: None,
                })?),
            }))
            .await
            .map_err(|err| WorkerSdkError::Transport(err.to_string()))?
            .into_inner();
        Ok(())
    }

    /// Read owned root-to-top scope identities from the exact current binding.
    /// Returns `None` without a binding and an error if the host cannot read
    /// the binding or it has expired. Reads the live scope stack on each call.
    pub async fn current_scope_stack_snapshot(&self) -> Result<Option<ScopeStackSnapshot>> {
        let scope = self.current_scope_context();
        let mut client = self.host_client().await?;
        let response = client
            .get_current_scope_stack_snapshot(Request::new(
                nemo_relay_worker_proto::nvidia_worker_v1::GetCurrentScopeStackSnapshotRequest {
                    credentials: Some(self.credentials()),
                    scope,
                },
            ))
            .await
            .map_err(|error| {
                if error.code() == tonic::Code::Unimplemented {
                    WorkerSdkError::Callback(
                        "scope stack snapshots are unsupported by this host; \
                         require the worker v1 snapshot contract"
                            .into(),
                    )
                } else {
                    WorkerSdkError::Transport(error.to_string())
                }
            })?
            .into_inner();
        match response.result {
            Some(wire::get_current_scope_stack_snapshot_response::Result::Snapshot(v)) => {
                Ok(Some(conversions::decode(v)?))
            }
            Some(wire::get_current_scope_stack_snapshot_response::Result::NoContext(_)) => Ok(None),
            None => Err(WorkerSdkError::InvalidInput(
                "missing scope snapshot result".into(),
            )),
        }
    }

    /// Return a bounded snapshot of active host runtime diagnostics.
    pub async fn runtime_diagnostics(&self) -> Result<RuntimeDiagnostics> {
        let mut client = self.host_client().await?;
        let response = client
            .get_runtime_diagnostics(Request::new(GetRuntimeDiagnosticsRequest {
                credentials: Some(self.credentials()),
            }))
            .await
            .map_err(|error| {
                if error.code() == tonic::Code::Unimplemented {
                    WorkerSdkError::Callback(
                        "runtime diagnostics require a Relay host with the worker v1 diagnostics contract"
                            .into(),
                    )
                } else {
                    WorkerSdkError::Transport(error.to_string())
                }
            })?
            .into_inner();
        Ok(RuntimeDiagnostics {
            entries: response
                .entries
                .into_iter()
                .map(|diagnostic: ProtoRuntimeDiagnostic| RuntimeDiagnostic {
                    code: diagnostic.code,
                    message: diagnostic.message,
                    count: diagnostic.count,
                })
                .collect(),
        })
    }

    /// Emits a validated Relay metric-measurement mark through the host runtime.
    pub async fn emit_metric(
        &self,
        name: &str,
        measurements: Vec<MetricMeasurement>,
        metadata: Option<Json>,
    ) -> Result<()> {
        self.emit_metric_impl(name, measurements, metadata, None)
            .await
    }

    /// Emits a validated Relay metric-measurement mark with an event category.
    pub async fn emit_metric_with_category(
        &self,
        name: &str,
        measurements: Vec<MetricMeasurement>,
        metadata: Option<Json>,
        category: EventCategory,
    ) -> Result<()> {
        self.emit_metric_impl(name, measurements, metadata, Some(category))
            .await
    }

    async fn emit_metric_impl(
        &self,
        name: &str,
        measurements: Vec<MetricMeasurement>,
        metadata: Option<Json>,
        category: Option<EventCategory>,
    ) -> Result<()> {
        let envelope = MetricEnvelope { measurements };
        envelope
            .validate()
            .map_err(|err| WorkerSdkError::InvalidInput(err.to_string()))?;
        self.emit_mark_impl(
            name,
            Some(serde_json::to_value(envelope)?),
            metadata,
            EmitMarkOptions {
                data_schema: Some(
                    DataSchema::builder()
                        .name(METRIC_DATA_SCHEMA_NAME)
                        .version(METRIC_DATA_SCHEMA_VERSION)
                        .build(),
                ),
                severity: None,
            },
            category,
        )
        .await
    }

    /// Creates an isolated host-owned scope stack.
    pub async fn create_scope_stack(&self) -> Result<String> {
        let mut client = self.host_client().await?;
        let response = client
            .create_scope_stack(Request::new(CreateScopeStackRequest {
                credentials: Some(self.credentials()),
            }))
            .await
            .map_err(|err| WorkerSdkError::Transport(err.to_string()))?
            .into_inner();
        nemo_relay_worker_proto::validation::validate(&response)
            .map_err(WorkerSdkError::InvalidInput)?;
        Ok(response.scope_stack_id)
    }

    /// Drops an isolated host-owned scope stack.
    pub async fn drop_scope_stack(&self, scope_stack_id: &str) -> Result<()> {
        let request = DropScopeStackRequest {
            credentials: Some(self.credentials()),
            scope_stack_id: scope_stack_id.into(),
        };
        nemo_relay_worker_proto::validation::validate(&request)
            .map_err(WorkerSdkError::InvalidInput)?;
        let mut client = self.host_client().await?;
        let _response = client
            .drop_scope_stack(Request::new(request))
            .await
            .map_err(|err| WorkerSdkError::Transport(err.to_string()))?
            .into_inner();
        Ok(())
    }

    /// Runs an async operation with runtime calls bound to a specific host-owned scope stack.
    ///
    /// This is useful for isolated stacks created with [`Self::create_scope_stack`]. The previous
    /// worker invocation scope is restored after the future completes.
    pub async fn with_scope_stack<F, Fut, T>(&self, scope_stack_id: &str, f: F) -> T
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = T>,
    {
        let scope = Some(scope_context(scope_stack_id));
        TASK_SCOPE_CONTEXT
            .scope(scope.clone(), async move {
                let future = with_thread_scope(&scope, f);
                future.await
            })
            .await
    }

    /// Pushes a scope through the host runtime.
    pub async fn push_scope(
        &self,
        scope_stack_id: Option<&str>,
        name: &str,
        scope_type: ScopeType,
        data: Option<Json>,
        metadata: Option<Json>,
        input: Option<Json>,
    ) -> Result<String> {
        self.push_scope_with_timestamp(
            scope_stack_id,
            name,
            scope_type,
            data,
            metadata,
            input,
            None,
        )
        .await
    }

    /// Pushes a scope through the host runtime using an explicit start time.
    #[allow(clippy::too_many_arguments)] // Mirrors `push_scope` with one additive timestamp.
    pub async fn push_scope_at(
        &self,
        scope_stack_id: Option<&str>,
        name: &str,
        scope_type: ScopeType,
        data: Option<Json>,
        metadata: Option<Json>,
        input: Option<Json>,
        started_at: SystemTime,
    ) -> Result<String> {
        let timestamp = unix_micros(started_at)?;
        self.push_scope_with_timestamp(
            scope_stack_id,
            name,
            scope_type,
            data,
            metadata,
            input,
            Some(timestamp),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)] // Shared implementation for the parallel public methods.
    async fn push_scope_with_timestamp(
        &self,
        scope_stack_id: Option<&str>,
        name: &str,
        scope_type: ScopeType,
        data: Option<Json>,
        metadata: Option<Json>,
        input: Option<Json>,
        timestamp_unix_micros: Option<i64>,
    ) -> Result<String> {
        let scope = scope_stack_id
            .map(scope_context)
            .or_else(|| self.current_scope_context());
        let mut client = self.host_client().await?;
        let response = client
            .push_scope(Request::new(PushScopeRequest {
                credentials: Some(self.credentials()),
                scope,
                name: name.into(),
                scope_type: proto_scope_type(scope_type),
                data: optional_encode_wire(data)?,
                metadata: optional_encode_wire(metadata)?,
                input: optional_encode_wire(input)?,
                time: micros_to_timestamp(timestamp_unix_micros)?,
            }))
            .await
            .map_err(|err| WorkerSdkError::Transport(err.to_string()))?
            .into_inner();
        nemo_relay_worker_proto::validation::validate(&response)
            .map_err(WorkerSdkError::InvalidInput)?;
        Ok(response.scope_handle_id)
    }

    /// Pops a scope through the host runtime.
    pub async fn pop_scope(
        &self,
        scope_handle_id: &str,
        output: Option<Json>,
        metadata: Option<Json>,
    ) -> Result<()> {
        self.pop_scope_with_timestamp(scope_handle_id, output, metadata, None)
            .await
    }

    /// Pops a scope through the host runtime using an explicit end time.
    pub async fn pop_scope_at(
        &self,
        scope_handle_id: &str,
        output: Option<Json>,
        metadata: Option<Json>,
        ended_at: SystemTime,
    ) -> Result<()> {
        let timestamp = unix_micros(ended_at)?;
        self.pop_scope_with_timestamp(scope_handle_id, output, metadata, Some(timestamp))
            .await
    }

    async fn pop_scope_with_timestamp(
        &self,
        scope_handle_id: &str,
        output: Option<Json>,
        metadata: Option<Json>,
        timestamp_unix_micros: Option<i64>,
    ) -> Result<()> {
        let mut client = self.host_client().await?;
        let _response = client
            .pop_scope(Request::new(PopScopeRequest {
                credentials: Some(self.credentials()),
                scope_handle_id: scope_handle_id.into(),
                output: optional_encode_wire(output)?,
                metadata: optional_encode_wire(metadata)?,
                time: micros_to_timestamp(timestamp_unix_micros)?,
            }))
            .await
            .map_err(|err| WorkerSdkError::Transport(err.to_string()))?
            .into_inner();
        Ok(())
    }

    /// Reuses the host connection while applying this activation's execution caps.
    async fn host_client(
        &self,
    ) -> Result<
        RelayHostRuntimeClient<
            tonic::service::interceptor::InterceptedService<
                Channel,
                nemo_relay_worker_proto::WorkerCallbackMetadata,
            >,
        >,
    > {
        let callback = self.callback_name.clone().or_else(current_callback_name);
        let limit = self
            .max_message_bytes
            .for_callback(callback.as_deref().unwrap_or_default());
        self.host_channel
            .get_or_try_init(|| connect_host_endpoint(&self.host_endpoint))
            .await
            .cloned()
            .map(|channel| {
                RelayHostRuntimeClient::with_interceptor(
                    channel,
                    nemo_relay_worker_proto::WorkerCallbackMetadata(callback),
                )
                .max_decoding_message_size(limit)
                .max_encoding_message_size(limit)
            })
    }

    fn current_scope_context(&self) -> Option<ScopeContext> {
        current_scope_context()
    }
}

fn unix_micros(timestamp: SystemTime) -> Result<i64> {
    let micros = match timestamp.duration_since(UNIX_EPOCH) {
        Ok(duration) => i128::try_from(duration.as_micros()),
        Err(error) => {
            let duration = error.duration();
            i128::try_from(duration.as_micros()).map(|micros| {
                // `Duration::as_micros` truncates toward zero, but a signed
                // Unix timestamp must floor pre-epoch sub-microsecond values.
                -micros - i128::from(duration.subsec_nanos() % 1_000 != 0)
            })
        }
    }
    .map_err(|_| {
        WorkerSdkError::InvalidInput("scope timestamp exceeds the supported range".into())
    })?;
    i64::try_from(micros).map_err(|_| {
        WorkerSdkError::InvalidInput("scope timestamp exceeds the supported range".into())
    })
}

/// Explicit worker server configuration for tests and custom launchers.
#[derive(Debug, Clone)]
pub struct WorkerServerConfig {
    /// Endpoint the worker listens on, such as `unix:///tmp/worker.sock` or `http://127.0.0.1:50051`.
    pub worker_endpoint: String,
    /// Relay host runtime endpoint used for callbacks and continuations.
    pub host_endpoint: String,
    /// Host-issued activation identifier accepted by this worker.
    pub activation_id: String,
    /// Host-issued bearer token accepted by this worker.
    pub auth_token: String,
}

/// Continuation handle for tool execution intercepts.
#[derive(Clone)]
pub struct ToolNext {
    runtime: PluginRuntime,
    continuation_id: String,
}

impl ToolNext {
    /// Calls the remaining tool execution chain.
    ///
    /// Calls may be repeated or concurrent while the owning interceptor is
    /// active. Each call receives an isolated snapshot of the scope stack
    /// visible here. Calls still unfinished when the interceptor settles fail.
    pub async fn call(&self, value: Json) -> Result<ToolExecutionResult> {
        let mut client = self.runtime.host_client().await?;
        let response = client
            .tool_next(Request::new(ToolNextRequest {
                credentials: Some(self.runtime.credentials()),
                continuation_id: self.continuation_id.clone(),
                value: Some(encode_wire(&value)?),
                scope: self.runtime.current_scope_context(),
            }))
            .await
            .map_err(continuation_status_to_sdk)?
            .into_inner();
        tool_execution_result_to_sdk(response)
    }
}

/// Continuation handle for LLM execution intercepts.
#[derive(Clone)]
pub struct LlmNext {
    runtime: PluginRuntime,
    continuation_id: String,
}

impl LlmNext {
    /// Calls the remaining LLM execution chain.
    ///
    /// Calls may be repeated or concurrent while the owning interceptor is
    /// active. Each call receives an isolated snapshot of the scope stack
    /// visible here. Calls still unfinished when the interceptor settles fail.
    pub async fn call(&self, request: LlmRequest) -> Result<Json> {
        let mut client = self.runtime.host_client().await?;
        let response = client
            .llm_next(Request::new(LlmNextRequest {
                credentials: Some(self.runtime.credentials()),
                continuation_id: self.continuation_id.clone(),
                request: Some(encode_wire(&request)?),
                scope: self.runtime.current_scope_context(),
            }))
            .await
            .map_err(continuation_status_to_sdk)?
            .into_inner();
        json_result_to_sdk(response)
    }
}

/// Continuation handle for LLM stream execution intercepts.
#[derive(Clone)]
pub struct LlmStreamNext {
    runtime: PluginRuntime,
    continuation_id: String,
}

impl LlmStreamNext {
    /// Calls the remaining LLM streaming execution chain.
    ///
    /// Calls may be repeated or concurrent while the owning interceptor stream
    /// is active. Each call receives an isolated snapshot of the scope stack
    /// visible here. A returned downstream stream has its ordinary lifetime,
    /// but unfinished calls are cancelled when the interceptor stream closes.
    pub async fn call(&self, request: LlmRequest) -> Result<JsonStream> {
        let scope = self.runtime.current_scope_context();
        let mut client = self.runtime.host_client().await?;
        let response = client
            .llm_stream_next(Request::new(LlmStreamNextRequest {
                credentials: Some(self.runtime.credentials()),
                continuation_id: self.continuation_id.clone(),
                request: Some(encode_wire(&request)?),
                scope: scope.clone(),
            }))
            .await
            .map_err(continuation_status_to_sdk)?;
        let stream = response.into_inner().map(|chunk| match chunk {
            Ok(chunk) => stream_chunk_to_json(chunk),
            Err(err) => Err(continuation_status_to_sdk(err)),
        });
        Ok(Box::pin(ScopedJsonStream::new(Box::pin(stream), scope)))
    }
}

/// Serves a worker plugin using environment variables supplied by the Relay host.
///
/// # Errors
/// Returns an error when required worker environment variables are missing,
/// `NEMO_RELAY_WORKER_MAX_MESSAGE_BYTES` is invalid, or the gRPC server fails.
pub async fn serve_plugin(plugin: impl WorkerPlugin) -> Result<()> {
    serve_plugin_arc(Arc::new(plugin)).await
}

/// Serves a shared worker plugin using environment variables supplied by the Relay host.
///
/// # Errors
/// Returns an error when required worker environment variables are missing,
/// `NEMO_RELAY_WORKER_MAX_MESSAGE_BYTES` is invalid, or the gRPC server fails.
pub async fn serve_plugin_arc(plugin: Arc<dyn WorkerPlugin>) -> Result<()> {
    let config = WorkerServerConfig {
        worker_endpoint: required_env("NEMO_RELAY_WORKER_SOCKET")?,
        host_endpoint: required_env("NEMO_RELAY_HOST_SOCKET")?,
        activation_id: required_env("NEMO_RELAY_WORKER_ID")?,
        auth_token: required_env("NEMO_RELAY_WORKER_TOKEN")?,
    };
    serve_plugin_arc_with_endpoint_file(
        plugin,
        config,
        optional_env("NEMO_RELAY_WORKER_ENDPOINT_FILE").map(PathBuf::from),
    )
    .await
}

/// Serves a shared worker plugin using explicit endpoint and authentication configuration.
///
/// This is primarily useful for tests and custom worker launchers. Relay-spawned
/// workers should normally use [`serve_plugin`] or [`serve_plugin_arc`].
///
/// # Errors
/// Returns an error when the endpoint configuration or
/// `NEMO_RELAY_WORKER_MAX_MESSAGE_BYTES` is invalid, or the gRPC server fails.
pub async fn serve_plugin_arc_with_config(
    plugin: Arc<dyn WorkerPlugin>,
    config: WorkerServerConfig,
) -> Result<()> {
    serve_plugin_arc_with_endpoint_file(plugin, config, None).await
}

/// Validates the transport limit before constructing and serving the worker runtime.
async fn serve_plugin_arc_with_endpoint_file(
    plugin: Arc<dyn WorkerPlugin>,
    config: WorkerServerConfig,
    endpoint_file: Option<PathBuf>,
) -> Result<()> {
    let max_message_bytes = WorkerCallbackMessageLimits::default()
        .from_env_with_defaults()
        .map_err(WorkerSdkError::InvalidInput)?;
    let runtime = PluginRuntime {
        callback_name: None,
        diagnostics: Arc::new(OnceCell::new()),
        activation_id: config.activation_id,
        auth_token: config.auth_token,
        host_endpoint: config.host_endpoint,
        host_channel: Arc::new(OnceCell::new()),
        max_message_bytes,
        conditional_middleware_callbacks: Arc::new(Mutex::new(HashMap::new())),
    };
    let service = WorkerService {
        health_reporter: tonic_health::server::HealthReporter::new(),
        shutdown: Arc::new(tokio::sync::Notify::new()),
        draining: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        registration: Arc::new(tokio::sync::Mutex::new(None)),
        registration_cancel: watch::channel(false).0,
        plugin,
        runtime,
        handlers: Arc::new(Mutex::new(WorkerHandlers::default())),
        active_invocations: Arc::new(Mutex::new(HashMap::new())),
        next_invocation_generation: Arc::new(AtomicU64::new(1)),
    };
    serve_worker_service(service, &config.worker_endpoint, endpoint_file.as_deref()).await
}

#[cfg(unix)]
/// Binds a bounded worker service to a Unix socket or delegates to loopback TCP.
async fn serve_worker_service(
    service: WorkerService,
    endpoint: &str,
    endpoint_file: Option<&Path>,
) -> Result<()> {
    if endpoint.starts_with("unix://") {
        let path = parse_unix_endpoint(endpoint)?;
        remove_stale_socket(&path)?;
        let listener = UnixListener::bind(&path).map_err(|err| {
            WorkerSdkError::Transport(format!("failed to bind worker socket: {err}"))
        })?;
        let shutdown = service.shutdown.clone();
        service
            .health_reporter
            .set_serving::<PluginWorkerServer<WorkerService>>()
            .await;
        let health_service = tonic_health::pb::health_server::HealthServer::new(
            tonic_health::server::HealthService::from_health_reporter(
                service.health_reporter.clone(),
            ),
        );
        return Server::builder()
            .add_service(health_service)
            .add_service(worker_grpc_service(service))
            .serve_with_incoming_shutdown(UnixListenerStream::new(listener), async move {
                shutdown.notified().await
            })
            .await
            .map_err(|err| WorkerSdkError::Transport(err.to_string()));
    }
    serve_tcp_worker_service(service, endpoint, endpoint_file).await
}

#[cfg(not(unix))]
async fn serve_worker_service(
    service: WorkerService,
    endpoint: &str,
    endpoint_file: Option<&Path>,
) -> Result<()> {
    if endpoint.starts_with("unix://") {
        return Err(WorkerSdkError::InvalidInput(
            "unix endpoints are not supported on this platform".into(),
        ));
    }
    serve_tcp_worker_service(service, endpoint, endpoint_file).await
}

/// Publishes the resolved TCP endpoint and serves execution RPCs with finite caps.
async fn serve_tcp_worker_service(
    service: WorkerService,
    endpoint: &str,
    endpoint_file: Option<&Path>,
) -> Result<()> {
    let addr = parse_tcp_endpoint(endpoint)?;
    let listener = TcpListener::bind(addr)
        .await
        .map_err(|err| WorkerSdkError::Transport(format!("failed to bind worker socket: {err}")))?;
    if let Some(path) = endpoint_file {
        let local_addr = listener.local_addr().map_err(|err| {
            WorkerSdkError::Transport(format!("failed to inspect worker socket: {err}"))
        })?;
        write_endpoint_file(path, &format!("http://{local_addr}"))?;
    }
    let shutdown = service.shutdown.clone();
    service
        .health_reporter
        .set_serving::<PluginWorkerServer<WorkerService>>()
        .await;
    let health_service = tonic_health::pb::health_server::HealthServer::new(
        tonic_health::server::HealthService::from_health_reporter(service.health_reporter.clone()),
    );
    Server::builder()
        .add_service(health_service)
        .add_service(worker_grpc_service(service))
        .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async move {
            shutdown.notified().await
        })
        .await
        .map_err(|err| WorkerSdkError::Transport(err.to_string()))
}

/// Applies the runtime's execution limit to both sides of the worker service.
fn worker_grpc_service(
    service: WorkerService,
) -> CallbackMessageLimitService<PluginWorkerServer<WorkerService>> {
    let limits = service.runtime.max_message_bytes.clone();
    CallbackMessageLimitService::new(PluginWorkerServer::new(service), limits, |server, limit| {
        server
            .max_decoding_message_size(limit)
            .max_encoding_message_size(limit)
    })
}

type InstalledRegistration = (Map<String, Json>, wire::RegistrationPlan);

#[derive(Clone)]
struct WorkerService {
    health_reporter: tonic_health::server::HealthReporter,
    shutdown: Arc<tokio::sync::Notify>,
    draining: Arc<std::sync::atomic::AtomicBool>,
    registration: Arc<tokio::sync::Mutex<Option<InstalledRegistration>>>,
    registration_cancel: watch::Sender<bool>,
    plugin: Arc<dyn WorkerPlugin>,
    runtime: PluginRuntime,
    handlers: Arc<Mutex<WorkerHandlers>>,
    active_invocations: Arc<Mutex<HashMap<String, ActiveInvocation>>>,
    next_invocation_generation: Arc<AtomicU64>,
}

struct ActiveInvocation {
    generation: u64,
    abort_handle: tokio::task::AbortHandle,
    stream_cancel: Option<watch::Sender<bool>>,
    completion: watch::Receiver<bool>,
    cancel_requested: bool,
}

// Shared across stream setup and production so cancellation also waits for
// cleanup during the handoff between those tasks.
struct InvocationCompletion {
    completed: watch::Sender<bool>,
    registration: Mutex<Option<ActiveInvocationGuard>>,
}

impl InvocationCompletion {
    fn new(completed: watch::Sender<bool>) -> Arc<Self> {
        Arc::new(Self {
            completed,
            registration: Mutex::new(None),
        })
    }

    fn track(&self, service: &WorkerService, invocation_id: &str, generation: u64) {
        *self
            .registration
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some(ActiveInvocationGuard {
            active_invocations: service.active_invocations.clone(),
            invocation_id: invocation_id.to_owned(),
            generation,
        });
    }
}

impl Drop for InvocationCompletion {
    fn drop(&mut self) {
        self.registration
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take();
        self.completed.send_replace(true);
    }
}

struct InvocationFuture<F> {
    // Field order keeps completion pending until the task's captured state drops,
    // including when cancellation happens before the task is first polled.
    future: Pin<Box<F>>,
    _completion: Arc<InvocationCompletion>,
}

impl<F: Future> Future for InvocationFuture<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.get_mut().future.as_mut().poll(cx)
    }
}

struct ActiveInvocationGuard {
    active_invocations: Arc<Mutex<HashMap<String, ActiveInvocation>>>,
    invocation_id: String,
    generation: u64,
}

impl Drop for ActiveInvocationGuard {
    fn drop(&mut self) {
        if let Ok(mut active) = self.active_invocations.lock()
            && active
                .get(&self.invocation_id)
                .is_some_and(|entry| entry.generation == self.generation)
        {
            active.remove(&self.invocation_id);
        }
    }
}

struct AbortTaskOnDrop {
    abort_handle: tokio::task::AbortHandle,
    armed: bool,
}

impl AbortTaskOnDrop {
    fn new(abort_handle: tokio::task::AbortHandle) -> Self {
        Self {
            abort_handle,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for AbortTaskOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.abort_handle.abort();
        }
    }
}

#[tonic::async_trait]
impl PluginWorker for WorkerService {
    async fn handshake(
        &self,
        request: Request<HandshakeRequest>,
    ) -> std::result::Result<Response<HandshakeResponse>, Status> {
        let request = request.into_inner();
        let credentials = request
            .credentials
            .as_ref()
            .ok_or_else(|| Status::unauthenticated("missing credentials"))?;
        self.authorize(&credentials.activation_id, &credentials.auth_token)?;
        nemo_relay_worker_proto::validation::validate(&request)
            .map_err(Status::invalid_argument)?;
        let protocol = request
            .protocol
            .ok_or_else(|| Status::invalid_argument("missing protocol"))?;
        if protocol.name != WORKER_PROTOCOL
            || protocol.version != wire::CONTRACT_VERSION
            || request.plugin_id != self.plugin.plugin_kind()
        {
            return Err(Status::failed_precondition(format!(
                "incompatible worker identity: expected plugin_id '{}', received plugin_id '{}', protocol '{}@{}'",
                self.plugin.plugin_kind(),
                request.plugin_id,
                protocol.name,
                protocol.version
            )));
        }
        Ok(Response::new(HandshakeResponse {
            identity: Some(self.identity()),
            allows_multiple_components: self.plugin.allows_multiple_components(),
            supported_surfaces: all_surfaces().into_iter().map(|v| v as i32).collect(),
        }))
    }

    async fn health(
        &self,
        request: Request<HealthRequest>,
    ) -> std::result::Result<Response<HealthResponse>, Status> {
        let request = request.into_inner();
        let credentials = request
            .credentials
            .as_ref()
            .ok_or_else(|| Status::unauthenticated("missing credentials"))?;
        self.authorize(&credentials.activation_id, &credentials.auth_token)?;
        nemo_relay_worker_proto::validation::validate(&request)
            .map_err(Status::invalid_argument)?;
        let draining = self.draining.load(Ordering::Acquire);
        Ok(Response::new(HealthResponse {
            status: if draining {
                WorkerHealthStatus::Unhealthy as i32
            } else {
                WorkerHealthStatus::Ready as i32
            },
            message: if draining { "draining" } else { "ready" }.into(),
            identity: Some(self.identity()),
        }))
    }

    async fn validate(
        &self,
        request: Request<ValidateRequest>,
    ) -> std::result::Result<Response<ValidateResponse>, Status> {
        let request = request.into_inner();
        let credentials = request
            .credentials
            .as_ref()
            .ok_or_else(|| Status::unauthenticated("missing credentials"))?;
        self.authorize(&credentials.activation_id, &credentials.auth_token)?;
        nemo_relay_worker_proto::validation::validate(&request)
            .map_err(Status::invalid_argument)?;
        if request.plugin_id != self.plugin.plugin_kind() {
            return Err(Status::failed_precondition("plugin identity mismatch"));
        }
        if self.draining.load(Ordering::Acquire) {
            return Err(Status::failed_precondition("worker is draining"));
        }
        let config = configuration_object(request.config.as_ref())?;
        let diagnostics = self.plugin.validate(&config);
        Ok(Response::new(ValidateResponse {
            result: Some(wire::validate_response::Result::Diagnostics(
                wire::ConfigDiagnostics {
                    entries: diagnostics
                        .into_iter()
                        .map(conversions::encode)
                        .collect::<std::result::Result<_, _>>()
                        .map_err(|e| Status::internal(e.to_string()))?,
                },
            )),
        }))
    }

    /// Reject new registrations while draining and cancel registrations already in progress.
    async fn register(
        &self,
        request: Request<RegisterRequest>,
    ) -> std::result::Result<Response<RegisterResponse>, Status> {
        let request = request.into_inner();
        let credentials = request
            .credentials
            .as_ref()
            .ok_or_else(|| Status::unauthenticated("missing credentials"))?;
        self.authorize(&credentials.activation_id, &credentials.auth_token)?;
        nemo_relay_worker_proto::validation::validate(&request)
            .map_err(Status::invalid_argument)?;
        let config = configuration_object(request.config.as_ref())?;
        if request.plugin_id != self.plugin.plugin_kind() {
            return Err(Status::failed_precondition("plugin identity mismatch"));
        }
        if self.draining.load(Ordering::Acquire) {
            return Err(Status::failed_precondition("worker is draining"));
        }
        let mut cancelled = self.registration_cancel.subscribe();
        let mut installed = tokio::select! {
            biased;
            _ = cancelled.wait_for(|cancelled| *cancelled) => return Err(Status::cancelled("worker registration cancelled by shutdown")),
            installed = self.registration.lock() => installed,
        };
        if self.draining.load(Ordering::Acquire) {
            return Err(Status::failed_precondition("worker is draining"));
        }
        if let Some((previous, plan)) = installed.as_ref() {
            if previous != &config {
                return Err(Status::failed_precondition(
                    "worker is already registered with a different config",
                ));
            }
            return Ok(Response::new(RegisterResponse {
                result: Some(wire::register_response::Result::Plan(plan.clone())),
            }));
        }
        let mut ctx = PluginContext::with_runtime(self.runtime.clone());
        let registered = tokio::select! {
            biased;
            _ = cancelled.wait_for(|cancelled| *cancelled) => return Err(Status::cancelled("worker registration cancelled by shutdown")),
            result = self.plugin.register(&config, &mut ctx) => result,
        };
        if let Err(err) = registered {
            return Ok(Response::new(RegisterResponse {
                result: Some(wire::register_response::Result::Error(sdk_error_to_worker(
                    WorkerSdkError::InvalidInput(err.to_string()),
                ))),
            }));
        }
        if let Err(err) = validate_unique_registrations(&ctx.handlers.registrations) {
            return Ok(Response::new(RegisterResponse {
                result: Some(wire::register_response::Result::Error(sdk_error_to_worker(
                    err,
                ))),
            }));
        }
        if let Err(err) = validate_initial_gates(&ctx.handlers.conditional_middleware_guardrails) {
            return Ok(Response::new(RegisterResponse {
                result: Some(wire::register_response::Result::Error(sdk_error_to_worker(
                    err,
                ))),
            }));
        }
        let registrations = ctx.handlers.registrations.clone();
        let conditional_middleware_guardrails =
            ctx.handlers.conditional_middleware_guardrails.clone();
        {
            // Serialize publication with shutdown's admission fence.
            let _admission = self
                .active_invocations
                .lock()
                .map_err(|err| Status::internal(err.to_string()))?;
            if self.draining.load(Ordering::Acquire) {
                return Err(Status::failed_precondition("worker is draining"));
            }
            let mut callbacks = self
                .runtime
                .conditional_middleware_callbacks
                .lock()
                .map_err(|err| {
                    Status::internal(format!(
                        "conditional middleware callback lock poisoned: {err}"
                    ))
                })?;
            let mut handlers = self
                .handlers
                .lock()
                .map_err(|err| Status::internal(format!("handler lock poisoned: {err}")))?;
            if let Some(name) = ctx
                .handlers
                .conditional_middleware_callbacks
                .keys()
                .find(|name| {
                    callbacks.contains_key(*name)
                        && !handlers
                            .conditional_middleware_callbacks
                            .contains_key(*name)
                })
            {
                return Ok(Response::new(RegisterResponse {
                    result: Some(wire::register_response::Result::Error(sdk_error_to_worker(
                        WorkerSdkError::InvalidInput(format!(
                            "conditional middleware callback '{name}' is already registered"
                        )),
                    ))),
                }));
            }
            for name in handlers.conditional_middleware_callbacks.keys() {
                callbacks.remove(name);
            }
            callbacks.extend(ctx.handlers.conditional_middleware_callbacks.clone());
            *handlers = ctx.handlers;
        }
        let plan = wire::RegistrationPlan {
            registrations,
            conditional_middleware_guardrails,
        };
        *installed = Some((config, plan.clone()));
        Ok(Response::new(RegisterResponse {
            result: Some(wire::register_response::Result::Plan(plan)),
        }))
    }

    async fn invoke(
        &self,
        request: Request<InvokeRequest>,
    ) -> std::result::Result<Response<InvokeResponse>, Status> {
        nemo_relay_worker_proto::validate_callback_request(
            &request,
            &request.get_ref().registration_name,
            &self.runtime.max_message_bytes,
        )?;
        let request = request.into_inner();
        let credentials = request
            .credentials
            .as_ref()
            .ok_or_else(|| Status::unauthenticated("missing credentials"))?;
        self.authorize(&credentials.activation_id, &credentials.auth_token)?;
        nemo_relay_worker_proto::validation::validate(&request)
            .map_err(Status::invalid_argument)?;
        let request = CallbackInvocation::try_from(request).map_err(status_from_sdk)?;
        let invocation_id = request.invocation_id.clone();
        let mut service = self.clone();
        service.runtime.callback_name = Some(request.registration_name.clone());
        let (completion_tx, completion_rx) = watch::channel(false);
        let completion = InvocationCompletion::new(completion_tx);
        let task = tokio::spawn(InvocationFuture {
            future: Box::pin(async move { service.invoke_inner(request).await }),
            _completion: Arc::clone(&completion),
        });
        let abort_handle = task.abort_handle();
        let generation = match self.track_invocation(
            &invocation_id,
            abort_handle.clone(),
            None,
            completion_rx,
        ) {
            Ok(generation) => generation,
            Err(err) => {
                task.abort();
                return Err(err);
            }
        };
        completion.track(self, &invocation_id, generation);
        let mut abort_on_drop = AbortTaskOnDrop::new(abort_handle);
        let response = match task.await {
            Ok(response) => response,
            Err(err) if err.is_cancelled() => cancelled_invoke_response(),
            Err(err) => InvokeResponse {
                result: Some(
                    nemo_relay_worker_proto::nvidia_worker_v1::invoke_response::Result::Error(
                        WorkerError {
                            code: "worker.error".into(),
                            message: format!("worker invocation task failed: {err}"),
                            retryable: false,
                            downstream_continuation_id: None,
                            http_status: None,
                        },
                    ),
                ),
            },
        };
        abort_on_drop.disarm();
        Ok(Response::new(response))
    }

    type InvokeStreamStream =
        Pin<Box<dyn tokio_stream::Stream<Item = std::result::Result<StreamChunk, Status>> + Send>>;

    async fn invoke_stream(
        &self,
        request: Request<InvokeStreamRequest>,
    ) -> std::result::Result<Response<Self::InvokeStreamStream>, Status> {
        nemo_relay_worker_proto::validate_callback_request(
            &request,
            &request.get_ref().registration_name,
            &self.runtime.max_message_bytes,
        )?;
        let request = request.into_inner();
        let callback_name = request.registration_name.clone();
        let mut runtime = self.runtime.clone();
        runtime.callback_name = Some(callback_name.clone());
        let credentials = request
            .credentials
            .as_ref()
            .ok_or_else(|| Status::unauthenticated("missing credentials"))?;
        self.authorize(&credentials.activation_id, &credentials.auth_token)?;
        nemo_relay_worker_proto::validation::validate(&request)
            .map_err(Status::invalid_argument)?;
        let request = CallbackInvocation::from(request);
        let invocation_id = request.invocation_id.clone();
        let scope = invocation_scope_context(request.scope.as_ref());
        let surface = RegistrationSurface::try_from(request.surface)
            .map_err(|_| Status::invalid_argument("unknown registration surface"))?;
        if surface != RegistrationSurface::LlmStreamExecutionIntercept {
            return Err(Status::invalid_argument(
                "InvokeStream only supports LLM stream execution",
            ));
        }
        let handler = self
            .handlers
            .lock()
            .map_err(|err| Status::internal(format!("handler lock poisoned: {err}")))?
            .llm_stream_executions
            .get(&request.registration_name)
            .cloned()
            .ok_or_else(|| Status::not_found("stream execution handler not registered"))?;
        let payload = llm_payload(request.payload).map_err(status_from_sdk)?;
        let execution_context = payload
            .execution_context(&runtime, &invocation_id, false)
            .map_err(status_from_sdk)?;
        let request_value =
            required_json::<LlmRequest>(payload.request, "llm request").map_err(status_from_sdk)?;
        let next = LlmStreamNext {
            runtime,
            continuation_id: request.continuation_id,
        };
        let model_name = payload.model_name;
        let (tx, rx) = mpsc::channel(16);
        let (stream_cancel_tx, stream_cancel_rx) = watch::channel(false);
        let (completion_tx, completion_rx) = watch::channel(false);
        let completion = InvocationCompletion::new(completion_tx);
        let open_completion = Arc::clone(&completion);
        let open_scope = scope.clone();
        let open_callback = callback_name.clone();
        let open_task = tokio::spawn(InvocationFuture {
            future: Box::pin(async move {
                TASK_CALLBACK_NAME
                    .scope(
                        Some(open_callback),
                        TASK_SCOPE_CONTEXT.scope(open_scope.clone(), async {
                            let future = with_thread_scope(&open_scope, || {
                                handler(&model_name, request_value, execution_context, next)
                            });
                            future.await
                        }),
                    )
                    .await
            }),
            _completion: open_completion,
        });
        let open_abort_handle = open_task.abort_handle();
        let generation = match self.track_invocation(
            &invocation_id,
            open_abort_handle.clone(),
            Some(stream_cancel_tx.clone()),
            completion_rx.clone(),
        ) {
            Ok(generation) => generation,
            Err(err) => {
                open_task.abort();
                return Err(err);
            }
        };
        completion.track(self, &invocation_id, generation);
        let mut abort_on_drop = AbortTaskOnDrop::new(open_abort_handle);
        let stream = match open_task.await {
            Ok(Ok(stream)) => stream,
            Ok(Err(err)) => {
                abort_on_drop.disarm();
                return Err(status_from_sdk(err));
            }
            Err(err) if err.is_cancelled() => {
                abort_on_drop.disarm();
                let explicitly_cancelled = *stream_cancel_rx.borrow();
                if explicitly_cancelled {
                    return Ok(Response::new(cancellation_aware_worker_stream(
                        rx,
                        stream_cancel_rx,
                    )));
                }
                return Err(Status::cancelled("worker invocation was cancelled"));
            }
            Err(err) => {
                abort_on_drop.disarm();
                return Err(Status::internal(format!(
                    "worker stream invocation task failed: {err}"
                )));
            }
        };
        abort_on_drop.disarm();
        let task_tx = tx;
        let mut stream = ScopedJsonStream::new(stream, scope);
        stream.callback_name = Some(callback_name);
        let (start_tx, start_rx) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(InvocationFuture {
            future: Box::pin(async move {
                let mut stream = stream;
                if start_rx.await.is_err() {
                    return;
                }
                loop {
                    let item = tokio::select! {
                        item = stream.next() => item,
                        _ = task_tx.closed() => return,
                    };
                    let Some(item) = item else {
                        return;
                    };
                    let (chunk, terminal) = match item {
                        Ok(value) => (
                            StreamChunk {
                                item: Some(wire::invoke_stream_response::Item::Value(
                                    match encode_wire(&value) {
                                        Ok(value) => value,
                                        Err(err) => {
                                            let _ = task_tx
                                                .send(Err(Status::internal(err.to_string())))
                                                .await;
                                            return;
                                        }
                                    },
                                )),
                            },
                            false,
                        ),
                        Err(err) => (
                            StreamChunk {
                                item: Some(wire::invoke_stream_response::Item::Error(
                                    sdk_error_to_worker(err),
                                )),
                            },
                            true,
                        ),
                    };
                    if task_tx.send(Ok(chunk)).await.is_err() {
                        return;
                    }
                    if terminal {
                        return;
                    }
                }
            }),
            _completion: completion,
        });
        let replaced = self.replace_invocation(
            &invocation_id,
            generation,
            ActiveInvocation {
                generation,
                abort_handle: task.abort_handle(),
                stream_cancel: Some(stream_cancel_tx),
                completion: completion_rx,
                cancel_requested: false,
            },
        );
        match replaced {
            Ok(true) => {
                let _ = start_tx.send(());
            }
            Ok(false) => {
                task.abort();
                return Ok(Response::new(cancellation_aware_worker_stream(
                    rx,
                    stream_cancel_rx,
                )));
            }
            Err(err) => {
                task.abort();
                return Err(err);
            }
        }
        Ok(Response::new(cancellation_aware_worker_stream(
            rx,
            stream_cancel_rx,
        )))
    }

    async fn cancel_invocation(
        &self,
        request: Request<CancelInvocationRequest>,
    ) -> std::result::Result<Response<CancelInvocationResponse>, Status> {
        let request = request.into_inner();
        let credentials = request
            .credentials
            .as_ref()
            .ok_or_else(|| Status::unauthenticated("missing credentials"))?;
        self.authorize(&credentials.activation_id, &credentials.auth_token)?;
        nemo_relay_worker_proto::validation::validate(&request)
            .map_err(Status::invalid_argument)?;
        let active = {
            let mut invocations = self
                .active_invocations
                .lock()
                .map_err(|err| Status::internal(format!("invocation lock poisoned: {err}")))?;
            invocations.get_mut(&request.invocation_id).map(|active| {
                let accepted = !active.cancel_requested;
                active.cancel_requested = true;
                (
                    active.abort_handle.clone(),
                    active.stream_cancel.clone(),
                    active.completion.clone(),
                    accepted,
                )
            })
        };
        let Some((abort_handle, stream_cancel, mut completion, accepted)) = active else {
            return Ok(Response::new(CancelInvocationResponse {
                accepted: false,
                message: "invocation is not active".into(),
            }));
        };
        if let Some(cancel) = stream_cancel {
            let _ = cancel.send(true);
        }
        abort_handle.abort();
        while !*completion.borrow_and_update() {
            if completion.changed().await.is_err() {
                break;
            }
        }
        Ok(Response::new(CancelInvocationResponse {
            accepted,
            message: if !accepted {
                "invocation is not active".into()
            } else if request.reason.is_empty() {
                "cancellation accepted".into()
            } else {
                format!("cancellation accepted: {}", request.reason)
            },
        }))
    }

    async fn shutdown(
        &self,
        request: Request<ShutdownRequest>,
    ) -> std::result::Result<Response<ShutdownResponse>, Status> {
        let request = request.into_inner();
        let credentials = request
            .credentials
            .as_ref()
            .ok_or_else(|| Status::unauthenticated("missing credentials"))?;
        self.authorize(&credentials.activation_id, &credentials.auth_token)?;
        nemo_relay_worker_proto::validation::validate(&request)
            .map_err(Status::invalid_argument)?;
        let (first_shutdown, active) = {
            let mut active = self
                .active_invocations
                .lock()
                .map_err(|e| Status::internal(e.to_string()))?;
            let first_shutdown = !self.draining.swap(true, Ordering::AcqRel);
            let active = active
                .values_mut()
                .map(|v| {
                    // Fence opener-to-pump handoffs before releasing the admission lock.
                    v.cancel_requested = true;
                    (
                        v.abort_handle.clone(),
                        v.completion.clone(),
                        v.stream_cancel.clone(),
                    )
                })
                .collect::<Vec<_>>();
            (first_shutdown, active)
        };
        self.registration_cancel.send_replace(true);
        for (abort, _, cancel) in &active {
            if let Some(cancel) = cancel {
                let _ = cancel.send(true);
            }
            abort.abort();
        }
        if first_shutdown {
            self.health_reporter
                .set_not_serving::<PluginWorkerServer<WorkerService>>()
                .await;
            self.health_reporter
                .set_service_status("", tonic_health::ServingStatus::NotServing)
                .await;
            // Only the first shutdown updates and clears health entries, so repeated
            // calls cannot recreate watch senders and prevent graceful exit.
            let mut health_reporter = self.health_reporter.clone();
            health_reporter
                .clear_service_status(
                    <PluginWorkerServer<WorkerService> as tonic::server::NamedService>::NAME,
                )
                .await;
            health_reporter.clear_service_status("").await;
        }
        // The lock is released only after the author future and its context
        // have dropped. Do not close diagnostics while registration owns them.
        drop(
            tokio::time::timeout(std::time::Duration::from_secs(2), self.registration.lock())
                .await
                .map_err(|_| {
                    Status::deadline_exceeded("worker registration cleanup is still in progress")
                })?,
        );
        let _ = tokio::time::timeout(std::time::Duration::from_secs(2), async {
            for (_, mut completion, _) in active {
                while !*completion.borrow_and_update() {
                    if completion.changed().await.is_err() {
                        break;
                    }
                }
            }
        })
        .await;
        if let Some(diagnostics) = self.runtime.diagnostics.get() {
            let _ = diagnostics.close(std::time::Duration::from_secs(2)).await;
        }
        self.shutdown.notify_one();
        Ok(Response::new(ShutdownResponse {}))
    }
}

fn configuration_object(
    config: Option<&wire::JsonValue>,
) -> std::result::Result<Map<String, Json>, Status> {
    let config = config
        .map(decode_wire::<Json>)
        .transpose()
        .map_err(|error| Status::invalid_argument(format!("invalid config JSON: {error}")))?
        .unwrap_or(Json::Null);
    match config {
        Json::Null => Ok(Map::new()),
        Json::Object(config) => Ok(config),
        _ => Err(Status::invalid_argument(
            "plugin configuration must be an object",
        )),
    }
}

fn validate_unique_registrations(registrations: &[Registration]) -> Result<()> {
    let mut seen = HashSet::new();
    for registration in registrations {
        if !seen.insert((registration.surface, registration.local_name.as_str())) {
            let surface = RegistrationSurface::try_from(registration.surface)
                .map_or("UNKNOWN", |surface| surface.as_str_name());
            return Err(WorkerSdkError::InvalidInput(format!(
                "duplicate registration '{}' for surface {}",
                registration.local_name, surface
            )));
        }
    }
    Ok(())
}

fn validate_initial_gates(gates: &[ConditionalMiddlewareGuardrailRegistration]) -> Result<()> {
    let mut names = HashSet::new();
    for gate in gates {
        if gate.name.trim().is_empty() || gate.registration_name.trim().is_empty() {
            return Err(WorkerSdkError::InvalidInput(
                "conditional middleware guardrail names must not be empty".into(),
            ));
        }
        if gate.kinds.is_empty() {
            return Err(WorkerSdkError::InvalidInput(format!(
                "conditional middleware guardrail '{}' has no registration kinds",
                gate.name
            )));
        }
        if !names.insert(gate.name.as_str()) {
            return Err(WorkerSdkError::InvalidInput(format!(
                "duplicate conditional middleware guardrail '{}'",
                gate.name
            )));
        }
    }
    Ok(())
}

impl WorkerService {
    fn authorize(&self, activation_id: &str, auth_token: &str) -> std::result::Result<(), Status> {
        if activation_id != self.runtime.activation_id {
            return Err(Status::unauthenticated("invalid worker activation"));
        }
        if auth_token != self.runtime.auth_token {
            return Err(Status::unauthenticated("invalid worker token"));
        }
        Ok(())
    }

    fn track_invocation(
        &self,
        invocation_id: &str,
        abort_handle: tokio::task::AbortHandle,
        stream_cancel: Option<watch::Sender<bool>>,
        completion: watch::Receiver<bool>,
    ) -> std::result::Result<u64, Status> {
        let generation = self
            .next_invocation_generation
            .fetch_add(1, Ordering::Relaxed);
        self.insert_invocation(
            invocation_id,
            ActiveInvocation {
                generation,
                abort_handle,
                stream_cancel,
                completion,
                cancel_requested: false,
            },
        )?;
        Ok(generation)
    }

    fn insert_invocation(
        &self,
        invocation_id: &str,
        invocation: ActiveInvocation,
    ) -> std::result::Result<(), Status> {
        if invocation_id.is_empty() {
            return Err(Status::invalid_argument("invocation_id must not be empty"));
        }
        let mut active = self
            .active_invocations
            .lock()
            .map_err(|err| Status::internal(format!("invocation lock poisoned: {err}")))?;
        if self.draining.load(Ordering::Acquire) {
            return Err(Status::failed_precondition("worker is draining"));
        }
        if active.contains_key(invocation_id) {
            return Err(Status::already_exists(format!(
                "invocation '{invocation_id}' is already active"
            )));
        }
        active.insert(invocation_id.to_string(), invocation);
        Ok(())
    }

    fn replace_invocation(
        &self,
        invocation_id: &str,
        generation: u64,
        invocation: ActiveInvocation,
    ) -> std::result::Result<bool, Status> {
        let mut active = self
            .active_invocations
            .lock()
            .map_err(|err| Status::internal(format!("invocation lock poisoned: {err}")))?;
        if active
            .get(invocation_id)
            .is_none_or(|entry| entry.generation != generation || entry.cancel_requested)
        {
            return Ok(false);
        }
        active.insert(invocation_id.to_string(), invocation);
        Ok(true)
    }

    async fn invoke_inner(&self, request: CallbackInvocation) -> InvokeResponse {
        match self.invoke_result(request).await {
            Ok(response) => response,
            Err(err) => InvokeResponse {
                result: Some(
                    nemo_relay_worker_proto::nvidia_worker_v1::invoke_response::Result::Error(
                        sdk_error_to_worker(err),
                    ),
                ),
            },
        }
    }

    async fn invoke_result(&self, request: CallbackInvocation) -> Result<InvokeResponse> {
        let scope = invocation_scope_context(request.scope.as_ref());
        TASK_CALLBACK_NAME
            .scope(
                Some(request.registration_name.clone()),
                TASK_SCOPE_CONTEXT.scope(scope.clone(), self.invoke_result_scoped(request, scope)),
            )
            .await
    }

    async fn invoke_result_scoped(
        &self,
        request: CallbackInvocation,
        scope: Option<ScopeContext>,
    ) -> Result<InvokeResponse> {
        let surface = RegistrationSurface::try_from(request.surface)
            .map_err(|_| WorkerSdkError::InvalidInput("unknown registration surface".into()))?;
        match surface {
            RegistrationSurface::ConditionalMiddlewareGuardrail => {
                self.invoke_conditional_middleware_response(request, &scope)
                    .await
            }
            RegistrationSurface::Subscriber => self.invoke_subscriber_response(request, &scope),
            RegistrationSurface::EventMetadataInjector => {
                self.invoke_event_metadata_injector_response(request, &scope)
                    .await
            }
            RegistrationSurface::MarkSanitizeGuardrail
            | RegistrationSurface::ScopeSanitizeStartGuardrail
            | RegistrationSurface::ScopeSanitizeEndGuardrail => {
                self.invoke_event_sanitize_response(request, &scope, surface)
                    .await
            }
            RegistrationSurface::ToolSanitizeRequestGuardrail
            | RegistrationSurface::ToolSanitizeResponseGuardrail
            | RegistrationSurface::ToolConditionalExecutionGuardrail
            | RegistrationSurface::ToolRequestIntercept
            | RegistrationSurface::ToolExecutionIntercept => {
                self.invoke_tool_response(request, &scope, surface).await
            }
            RegistrationSurface::LlmSanitizeRequestGuardrail
            | RegistrationSurface::LlmSanitizeResponseGuardrail
            | RegistrationSurface::LlmConditionalExecutionGuardrail
            | RegistrationSurface::LlmRequestIntercept
            | RegistrationSurface::LlmExecutionIntercept => {
                self.invoke_llm_response(request, &scope, surface).await
            }
            RegistrationSurface::LlmStreamExecutionIntercept | RegistrationSurface::Unspecified => {
                Err(WorkerSdkError::InvalidInput(
                    "surface must use InvokeStream or is unspecified".into(),
                ))
            }
        }
    }

    async fn invoke_conditional_middleware_response(
        &self,
        request: CallbackInvocation,
        scope: &Option<ScopeContext>,
    ) -> Result<InvokeResponse> {
        let payload = match request.payload {
            Some(CallbackPayload::ConditionalMiddleware(payload)) => payload,
            _ => {
                return Err(WorkerSdkError::InvalidInput(
                    "conditional middleware payload is missing".into(),
                ));
            }
        };
        let kinds = payload
            .kinds
            .into_iter()
            .map(|kind| {
                RegistrationSurface::try_from(kind)
                    .map_err(|_| {
                        WorkerSdkError::InvalidInput("unknown runtime registration kind".into())
                    })
                    .and_then(runtime_registration_kind)
            })
            .collect::<Result<BTreeSet<_>>>()?;
        let handler = self
            .runtime
            .conditional_middleware_callbacks
            .lock()
            .map_err(|error| WorkerSdkError::Callback(error.to_string()))?
            .get(&request.registration_name)
            .cloned()
            .ok_or_else(|| {
                WorkerSdkError::InvalidInput(format!(
                    "conditional middleware callback '{}' is not registered",
                    request.registration_name
                ))
            })?;
        let future = with_thread_scope(scope, || handler(kinds, payload.registration_name));
        Ok(guardrail_response(future.await?))
    }

    fn invoke_subscriber_response(
        &self,
        request: CallbackInvocation,
        scope: &Option<ScopeContext>,
    ) -> Result<InvokeResponse> {
        let event = event_payload(request.payload)?;
        let handler = self.subscriber(&request.registration_name)?;
        with_thread_scope(scope, || handler(&event));
        Ok(empty_response())
    }

    async fn invoke_event_metadata_injector_response(
        &self,
        request: CallbackInvocation,
        scope: &Option<ScopeContext>,
    ) -> Result<InvokeResponse> {
        let event = event_payload(request.payload)?;
        let handler = self.event_metadata_injector(&request.registration_name)?;
        let future = with_thread_scope(scope, || handler(Arc::new(event)));
        Ok(InvokeResponse {
            result: Some(wire::invoke_response::Result::MetadataAdditions(
                conversions::encode(future.await?)?,
            )),
        })
    }

    async fn invoke_event_sanitize_response(
        &self,
        request: CallbackInvocation,
        scope: &Option<ScopeContext>,
        surface: RegistrationSurface,
    ) -> Result<InvokeResponse> {
        let event = event_payload(request.payload)?;
        let fields = event.sanitize_fields();
        let handler = self.event_sanitizer(surface, &request.registration_name)?;
        let fields = with_thread_scope(scope, || handler(&event, fields)).await?;
        Ok(InvokeResponse {
            result: Some(wire::invoke_response::Result::EventSanitize(
                conversions::encode(fields)?,
            )),
        })
    }

    async fn invoke_tool_response(
        &self,
        request: CallbackInvocation,
        scope: &Option<ScopeContext>,
        surface: RegistrationSurface,
    ) -> Result<InvokeResponse> {
        match surface {
            RegistrationSurface::ToolSanitizeRequestGuardrail => {
                self.invoke_tool_sanitize_request_response(request, scope)
                    .await
            }
            RegistrationSurface::ToolSanitizeResponseGuardrail => {
                self.invoke_tool_sanitize_response_response(request, scope)
                    .await
            }
            RegistrationSurface::ToolConditionalExecutionGuardrail => {
                self.invoke_tool_conditional_response(request, scope).await
            }
            RegistrationSurface::ToolRequestIntercept => {
                self.invoke_tool_request_response(request, scope).await
            }
            RegistrationSurface::ToolExecutionIntercept => {
                self.invoke_tool_execution_response(request, scope).await
            }
            _ => unreachable!("tool surface was pre-filtered"),
        }
    }

    async fn invoke_tool_sanitize_request_response(
        &self,
        request: CallbackInvocation,
        scope: &Option<ScopeContext>,
    ) -> Result<InvokeResponse> {
        let payload = tool_payload(request.payload)?;
        let handler = self.tool_sanitize_request(&request.registration_name)?;
        Ok(json_response(
            with_thread_scope(scope, || handler(&payload.tool_name, payload.value)).await?,
        ))
    }

    async fn invoke_tool_sanitize_response_response(
        &self,
        request: CallbackInvocation,
        scope: &Option<ScopeContext>,
    ) -> Result<InvokeResponse> {
        let payload = tool_payload(request.payload)?;
        let handler = self.tool_sanitize_response(&request.registration_name)?;
        Ok(json_response(
            with_thread_scope(scope, || handler(&payload.tool_name, payload.value)).await?,
        ))
    }

    async fn invoke_tool_conditional_response(
        &self,
        request: CallbackInvocation,
        scope: &Option<ScopeContext>,
    ) -> Result<InvokeResponse> {
        let payload = tool_payload(request.payload)?;
        let handler = self.tool_conditional(&request.registration_name)?;
        let future = with_thread_scope(scope, || handler(payload.tool_name, payload.value));
        Ok(guardrail_response(future.await?))
    }

    async fn invoke_tool_request_response(
        &self,
        request: CallbackInvocation,
        scope: &Option<ScopeContext>,
    ) -> Result<InvokeResponse> {
        let payload = tool_payload(request.payload)?;
        let handler = self.tool_request(&request.registration_name)?;
        let future = with_thread_scope(scope, || handler(payload.tool_name, payload.value));
        Ok(json_response(future.await?))
    }

    async fn invoke_tool_execution_response(
        &self,
        request: CallbackInvocation,
        scope: &Option<ScopeContext>,
    ) -> Result<InvokeResponse> {
        let payload = tool_payload(request.payload)?;
        let next = ToolNext {
            runtime: self.runtime.clone(),
            continuation_id: request.continuation_id,
        };
        let handler = self.tool_execution(&request.registration_name)?;
        let context = ToolExecutionContext {
            tool_name: payload.tool_name,
            args: payload.value,
            tool_call_id: payload.tool_call_id,
        };
        let future = with_thread_scope(scope, || handler(context, next));
        tool_execution_response(future.await?)
    }

    async fn invoke_llm_response(
        &self,
        request: CallbackInvocation,
        scope: &Option<ScopeContext>,
        surface: RegistrationSurface,
    ) -> Result<InvokeResponse> {
        match surface {
            RegistrationSurface::LlmSanitizeRequestGuardrail => {
                self.invoke_llm_sanitize_request_response(request, scope)
                    .await
            }
            RegistrationSurface::LlmSanitizeResponseGuardrail => {
                self.invoke_llm_sanitize_response_response(request, scope)
                    .await
            }
            RegistrationSurface::LlmConditionalExecutionGuardrail => {
                self.invoke_llm_conditional_response(request, scope).await
            }
            RegistrationSurface::LlmRequestIntercept => {
                self.invoke_llm_request_response(request, scope).await
            }
            RegistrationSurface::LlmExecutionIntercept => {
                self.invoke_llm_execution_response(request, scope).await
            }
            _ => unreachable!("LLM surface was pre-filtered"),
        }
    }

    async fn invoke_llm_sanitize_request_response(
        &self,
        request: CallbackInvocation,
        scope: &Option<ScopeContext>,
    ) -> Result<InvokeResponse> {
        let payload = llm_payload(request.payload)?;
        let mut context = payload.sanitize_request_context(&request.invocation_id)?;
        context.runtime = Some(self.runtime.clone());
        let request_value = required_json::<LlmRequest>(payload.request, "llm request")?;
        let handler = self.llm_sanitize_request(&request.registration_name)?;
        match with_thread_scope(scope, || handler(request_value, context)).await? {
            Some(request) => Ok(InvokeResponse {
                result: Some(wire::invoke_response::Result::LlmRequest(
                    conversions::encode(request)?,
                )),
            }),
            None => Ok(unchanged_response()),
        }
    }

    async fn invoke_llm_sanitize_response_response(
        &self,
        request: CallbackInvocation,
        scope: &Option<ScopeContext>,
    ) -> Result<InvokeResponse> {
        let payload = llm_payload(request.payload)?;
        let mut context = payload.sanitize_response_context(&request.invocation_id)?;
        context.runtime = Some(self.runtime.clone());
        let response = required_json::<Json>(payload.response, "llm response")?;
        let handler = self.llm_sanitize_response(&request.registration_name)?;
        match with_thread_scope(scope, || handler(response, context)).await? {
            Some(response) => Ok(llm_response(response)),
            None => Ok(unchanged_response()),
        }
    }

    async fn invoke_llm_conditional_response(
        &self,
        request: CallbackInvocation,
        scope: &Option<ScopeContext>,
    ) -> Result<InvokeResponse> {
        let payload = llm_payload(request.payload)?;
        let request_value = required_json::<LlmRequest>(payload.request, "llm request")?;
        let handler = self.llm_conditional(&request.registration_name)?;
        let future = with_thread_scope(scope, || handler(request_value));
        Ok(guardrail_response(future.await?))
    }

    async fn invoke_llm_request_response(
        &self,
        request: CallbackInvocation,
        scope: &Option<ScopeContext>,
    ) -> Result<InvokeResponse> {
        let payload = llm_payload(request.payload)?;
        let request_value = required_json::<LlmRequest>(payload.request, "llm request")?;
        let annotated = payload
            .annotated_request
            .map(|value| decode_wire::<AnnotatedLlmRequest>(&value))
            .transpose()?;
        let handler = self.llm_request(&request.registration_name)?;
        let outcome = with_thread_scope(scope, || {
            handler(payload.model_name, request_value, annotated)
        })
        .await?;
        llm_request_response(outcome)
    }

    async fn invoke_llm_execution_response(
        &self,
        request: CallbackInvocation,
        scope: &Option<ScopeContext>,
    ) -> Result<InvokeResponse> {
        let payload = llm_payload(request.payload)?;
        let execution_context =
            payload.execution_context(&self.runtime, &request.invocation_id, true)?;
        let request_value = required_json::<LlmRequest>(payload.request, "llm request")?;
        let handler = self.llm_execution(&request.registration_name)?;
        let next = LlmNext {
            runtime: self.runtime.clone(),
            continuation_id: request.continuation_id,
        };
        let future = with_thread_scope(scope, || {
            handler(&payload.model_name, request_value, execution_context, next)
        });
        Ok(llm_response(future.await?))
    }

    fn subscriber(&self, name: &str) -> Result<SubscriberFn> {
        self.handlers
            .lock()
            .map_err(|err| WorkerSdkError::Callback(format!("handler lock poisoned: {err}")))?
            .subscribers
            .get(name)
            .cloned()
            .ok_or_else(|| {
                WorkerSdkError::InvalidInput(format!("subscriber '{name}' not registered"))
            })
    }

    fn event_metadata_injector(&self, name: &str) -> Result<EventMetadataInjectorFn> {
        self.handlers
            .lock()
            .map_err(|err| WorkerSdkError::Callback(format!("handler lock poisoned: {err}")))?
            .event_metadata_injectors
            .get(name)
            .cloned()
            .ok_or_else(|| {
                WorkerSdkError::InvalidInput(format!(
                    "Event metadata injector '{name}' not registered"
                ))
            })
    }

    fn event_sanitizer(&self, surface: RegistrationSurface, name: &str) -> Result<EventSanitizeFn> {
        let handlers = self
            .handlers
            .lock()
            .map_err(|err| WorkerSdkError::Callback(format!("handler lock poisoned: {err}")))?;
        let sanitizers = match surface {
            RegistrationSurface::MarkSanitizeGuardrail => &handlers.mark_sanitizers,
            RegistrationSurface::ScopeSanitizeStartGuardrail => &handlers.scope_start_sanitizers,
            RegistrationSurface::ScopeSanitizeEndGuardrail => &handlers.scope_end_sanitizers,
            _ => unreachable!("event sanitizer lookup requires an event sanitizer surface"),
        };
        sanitizers.get(name).cloned().ok_or_else(|| {
            WorkerSdkError::InvalidInput(format!("event sanitizer '{name}' not registered"))
        })
    }

    fn tool_sanitize_request(&self, name: &str) -> Result<ToolSanitizeFn> {
        self.handlers
            .lock()
            .map_err(|err| WorkerSdkError::Callback(format!("handler lock poisoned: {err}")))?
            .tool_sanitize_requests
            .get(name)
            .cloned()
            .ok_or_else(|| {
                WorkerSdkError::InvalidInput(format!(
                    "tool request sanitizer '{name}' not registered"
                ))
            })
    }

    fn tool_sanitize_response(&self, name: &str) -> Result<ToolSanitizeFn> {
        self.handlers
            .lock()
            .map_err(|err| WorkerSdkError::Callback(format!("handler lock poisoned: {err}")))?
            .tool_sanitize_responses
            .get(name)
            .cloned()
            .ok_or_else(|| {
                WorkerSdkError::InvalidInput(format!(
                    "tool response sanitizer '{name}' not registered"
                ))
            })
    }

    fn tool_conditional(&self, name: &str) -> Result<ToolConditionalFn> {
        self.handlers
            .lock()
            .map_err(|err| WorkerSdkError::Callback(format!("handler lock poisoned: {err}")))?
            .tool_conditionals
            .get(name)
            .cloned()
            .ok_or_else(|| {
                WorkerSdkError::InvalidInput(format!("tool conditional '{name}' not registered"))
            })
    }

    fn tool_request(&self, name: &str) -> Result<ToolRequestFn> {
        self.handlers
            .lock()
            .map_err(|err| WorkerSdkError::Callback(format!("handler lock poisoned: {err}")))?
            .tool_requests
            .get(name)
            .cloned()
            .ok_or_else(|| {
                WorkerSdkError::InvalidInput(format!("tool request '{name}' not registered"))
            })
    }

    fn tool_execution(&self, name: &str) -> Result<ToolExecutionFn> {
        self.handlers
            .lock()
            .map_err(|err| WorkerSdkError::Callback(format!("handler lock poisoned: {err}")))?
            .tool_executions
            .get(name)
            .cloned()
            .ok_or_else(|| {
                WorkerSdkError::InvalidInput(format!("tool execution '{name}' not registered"))
            })
    }

    fn llm_sanitize_request(&self, name: &str) -> Result<LlmSanitizeRequestFn> {
        self.handlers
            .lock()
            .map_err(|err| WorkerSdkError::Callback(format!("handler lock poisoned: {err}")))?
            .llm_sanitize_requests
            .get(name)
            .cloned()
            .ok_or_else(|| {
                WorkerSdkError::InvalidInput(format!(
                    "llm request sanitizer '{name}' not registered"
                ))
            })
    }

    fn llm_sanitize_response(&self, name: &str) -> Result<LlmSanitizeResponseFn> {
        self.handlers
            .lock()
            .map_err(|err| WorkerSdkError::Callback(format!("handler lock poisoned: {err}")))?
            .llm_sanitize_responses
            .get(name)
            .cloned()
            .ok_or_else(|| {
                WorkerSdkError::InvalidInput(format!(
                    "llm response sanitizer '{name}' not registered"
                ))
            })
    }

    fn llm_conditional(&self, name: &str) -> Result<LlmConditionalFn> {
        self.handlers
            .lock()
            .map_err(|err| WorkerSdkError::Callback(format!("handler lock poisoned: {err}")))?
            .llm_conditionals
            .get(name)
            .cloned()
            .ok_or_else(|| {
                WorkerSdkError::InvalidInput(format!("llm conditional '{name}' not registered"))
            })
    }

    fn llm_request(&self, name: &str) -> Result<LlmRequestFn> {
        self.handlers
            .lock()
            .map_err(|err| WorkerSdkError::Callback(format!("handler lock poisoned: {err}")))?
            .llm_requests
            .get(name)
            .cloned()
            .ok_or_else(|| {
                WorkerSdkError::InvalidInput(format!("llm request '{name}' not registered"))
            })
    }

    fn llm_execution(&self, name: &str) -> Result<LlmExecutionFn> {
        self.handlers
            .lock()
            .map_err(|err| WorkerSdkError::Callback(format!("handler lock poisoned: {err}")))?
            .llm_executions
            .get(name)
            .cloned()
            .ok_or_else(|| {
                WorkerSdkError::InvalidInput(format!("llm execution '{name}' not registered"))
            })
    }
}

struct ToolPayload {
    tool_name: String,
    value: Json,
    tool_call_id: Option<String>,
}

#[derive(Default)]
struct LlmPayload {
    model_name: String,
    request: Option<wire::LlmRequest>,
    annotated_request: Option<wire::AnnotatedLlmRequest>,
    response: Option<wire::JsonValue>,
    request_codec: Option<wire::LlmCodecContext>,
    response_codec: Option<wire::LlmCodecContext>,
}

impl LlmPayload {
    fn execution_context(
        &self,
        runtime: &PluginRuntime,
        invocation_id: &str,
        response_required: bool,
    ) -> Result<LlmExecutionContext> {
        let request = self
            .request_codec
            .as_ref()
            .ok_or_else(|| WorkerSdkError::InvalidInput("missing request codec context".into()))?;
        let response_codec = self
            .response_codec
            .as_ref()
            .map(|context| -> Result<_> {
                Ok(LlmResponseContext {
                    codec: codec_identity_from_proto(context.codec.as_ref())?,
                    runtime: Some(runtime.clone()),
                    codec_capability_id: context.capability_id.clone(),
                    invocation_id: Some(invocation_id.to_owned()),
                })
            })
            .transpose()?;
        if response_required && response_codec.is_none() {
            return Err(WorkerSdkError::InvalidInput(
                "missing response codec context".into(),
            ));
        }
        Ok(LlmExecutionContext {
            request_codec: LlmRequestContext {
                codec: codec_identity_from_proto(request.codec.as_ref())?,
                runtime: Some(runtime.clone()),
                codec_capability_id: request.capability_id.clone(),
                invocation_id: Some(invocation_id.to_owned()),
            },
            response_codec,
        })
    }
    fn sanitize_request_context(&self, invocation_id: &str) -> Result<LlmSanitizeRequestContext> {
        let context = self
            .request_codec
            .as_ref()
            .ok_or_else(|| WorkerSdkError::InvalidInput("missing request codec context".into()))?;
        Ok(LlmSanitizeRequestContext {
            codec: codec_identity_from_proto(context.codec.as_ref())?,
            runtime: None,
            codec_capability_id: context.capability_id.clone(),
            invocation_id: Some(invocation_id.to_owned()),
        })
    }
    fn sanitize_response_context(&self, invocation_id: &str) -> Result<LlmSanitizeResponseContext> {
        let context = self
            .response_codec
            .as_ref()
            .ok_or_else(|| WorkerSdkError::InvalidInput("missing response codec context".into()))?;
        Ok(LlmSanitizeResponseContext {
            codec: codec_identity_from_proto(context.codec.as_ref())?,
            runtime: None,
            codec_capability_id: context.capability_id.clone(),
            invocation_id: Some(invocation_id.to_owned()),
        })
    }
}

fn codec_identity_from_proto(value: Option<&wire::LlmCodecReference>) -> Result<LlmCodecIdentity> {
    let value =
        value.ok_or_else(|| WorkerSdkError::InvalidInput("missing codec identity".into()))?;
    conversions::decode(value.clone()).map_err(Into::into)
}

fn event_payload(payload: Option<CallbackPayload>) -> Result<Event> {
    match payload {
        Some(CallbackPayload::Event(value)) => decode_wire(value.as_ref()),
        _ => Err(WorkerSdkError::InvalidInput(
            "expected event payload".into(),
        )),
    }
}
fn tool_payload(payload: Option<CallbackPayload>) -> Result<ToolPayload> {
    match payload {
        Some(CallbackPayload::Tool(value)) => Ok(value),
        _ => Err(WorkerSdkError::InvalidInput("expected tool payload".into())),
    }
}
fn llm_payload(payload: Option<CallbackPayload>) -> Result<LlmPayload> {
    match payload {
        Some(CallbackPayload::Llm(value)) => Ok(*value),
        _ => Err(WorkerSdkError::InvalidInput("expected LLM payload".into())),
    }
}
fn required_json<D>(value: Option<impl DecodeTo<D>>, field: &str) -> Result<D> {
    value
        .ok_or_else(|| WorkerSdkError::InvalidInput(format!("{field} is missing")))?
        .decode_to()
}

fn empty_response() -> InvokeResponse {
    InvokeResponse {
        result: Some(wire::invoke_response::Result::Completed(())),
    }
}

fn json_response(value: Json) -> InvokeResponse {
    InvokeResponse {
        result: Some(wire::invoke_response::Result::ToolValue(infallible_wire(
            &value,
        ))),
    }
}

fn guardrail_response(reason: Option<String>) -> InvokeResponse {
    InvokeResponse {
        result: Some(wire::invoke_response::Result::Guardrail(
            wire::GuardrailDecision {
                decision: Some(match reason {
                    None => wire::guardrail_decision::Decision::Allow(()),
                    Some(reason) => wire::guardrail_decision::Decision::BlockReason(reason),
                }),
            },
        )),
    }
}

fn llm_request_response(outcome: LlmRequestInterceptOutcome) -> Result<InvokeResponse> {
    Ok(InvokeResponse {
        result: Some(wire::invoke_response::Result::LlmRequestOutcome(
            conversions::encode(outcome)?,
        )),
    })
}

fn tool_execution_response(outcome: ToolExecutionInterceptOutcome) -> Result<InvokeResponse> {
    Ok(InvokeResponse {
        result: Some(wire::invoke_response::Result::ToolExecutionOutcome(
            conversions::encode(outcome)?,
        )),
    })
}

fn tool_execution_result_to_sdk(result: wire::ToolNextResponse) -> Result<ToolExecutionResult> {
    decode_rpc_result(result)
}

fn stream_chunk_to_json(chunk: wire::LlmStreamNextResponse) -> Result<Json> {
    match chunk.item {
        Some(wire::llm_stream_next_response::Item::Value(value)) => {
            Ok(decode_wire::<Json>(&value)?)
        }
        Some(wire::llm_stream_next_response::Item::Error(error)) => Err(worker_error_to_sdk(error)),
        None => Err(WorkerSdkError::InvalidInput("empty stream chunk".into())),
    }
}

fn json_result_to_sdk(result: wire::LlmNextResponse) -> Result<Json> {
    decode_rpc_result(result)
}

fn optional_encode_wire(value: Option<Json>) -> Result<Option<JsonEnvelope>> {
    value.as_ref().map(encode_wire).transpose()
}

fn infallible_wire<D: Clone + WireValue<W>, W>(value: &D) -> W {
    conversions::encode(value.clone()).expect("validated SDK value must encode")
}

fn sdk_error_to_worker(error: WorkerSdkError) -> WorkerError {
    if let WorkerSdkError::Downstream {
        source,
        continuation_id,
    } = error
    {
        let mut wire = sdk_error_to_worker(*source);
        wire.downstream_continuation_id = Some(continuation_id);
        return wire;
    }
    WorkerError {
        code: "worker.error".into(),
        message: match &error {
            WorkerSdkError::Callback(message)
                if nemo_relay_types::plugin::parse_http_error_message(message).is_some() =>
            {
                message.clone()
            }
            _ => error.to_string(),
        },
        retryable: false,
        downstream_continuation_id: None,
        http_status: nemo_relay_types::plugin::parse_http_error_message(match &error {
            WorkerSdkError::Callback(message) => message,
            _ => "",
        })
        .map(|(status, _)| u32::from(status)),
    }
}

fn cancelled_worker_error() -> WorkerError {
    WorkerError {
        code: "worker.cancelled".into(),
        message: "worker invocation was cancelled".into(),
        retryable: false,
        downstream_continuation_id: None,
        http_status: None,
    }
}

fn cancelled_invoke_response() -> InvokeResponse {
    InvokeResponse {
        result: Some(
            nemo_relay_worker_proto::nvidia_worker_v1::invoke_response::Result::Error(
                cancelled_worker_error(),
            ),
        ),
    }
}

fn cancelled_stream_chunk() -> StreamChunk {
    StreamChunk {
        item: Some(wire::invoke_stream_response::Item::Error(
            cancelled_worker_error(),
        )),
    }
}

fn cancellation_aware_worker_stream(
    rx: mpsc::Receiver<std::result::Result<StreamChunk, Status>>,
    cancel_rx: watch::Receiver<bool>,
) -> Pin<Box<dyn Stream<Item = std::result::Result<StreamChunk, Status>> + Send>> {
    #[derive(Clone, Copy)]
    enum CancellationState {
        Watching,
        Closed,
        Done,
    }

    Box::pin(futures_util::stream::unfold(
        (rx, cancel_rx, CancellationState::Watching),
        |(mut rx, mut cancel_rx, mut state)| async move {
            if matches!(state, CancellationState::Done) {
                return None;
            }
            loop {
                if matches!(state, CancellationState::Closed) {
                    return rx
                        .recv()
                        .await
                        .map(|item| (item, (rx, cancel_rx, CancellationState::Closed)));
                }
                tokio::select! {
                    biased;
                    changed = cancel_rx.changed() => match changed {
                        Ok(()) if *cancel_rx.borrow() => {
                            return Some((
                                Ok(cancelled_stream_chunk()),
                                (rx, cancel_rx, CancellationState::Done),
                            ));
                        }
                        Ok(()) => {}
                        Err(_) => state = CancellationState::Closed,
                    },
                    item = rx.recv() => {
                        return item.map(|item| {
                            (item, (rx, cancel_rx, CancellationState::Watching))
                        });
                    }
                }
            }
        },
    ))
}

fn worker_error_to_sdk(error: WorkerError) -> WorkerSdkError {
    let mut error = error;
    if error.code != "worker.cancelled"
        && let Some(status) = error.http_status.and_then(|v| u16::try_from(v).ok())
        && nemo_relay_types::plugin::parse_http_error_message(&error.message).is_none()
        && let Ok(message) = http_error_message(status, &error.message)
    {
        error.message = message;
    }
    let source = if error.code != "worker.cancelled"
        && nemo_relay_types::plugin::parse_http_error_message(&error.message).is_some()
    {
        WorkerSdkError::Callback(error.message)
    } else {
        WorkerSdkError::Callback(format!("{}: {}", error.code, error.message))
    };
    with_downstream_provenance(source, error.downstream_continuation_id)
}

fn with_downstream_provenance(
    source: WorkerSdkError,
    continuation_id: Option<String>,
) -> WorkerSdkError {
    match continuation_id.filter(|id| !id.is_empty()) {
        Some(continuation_id) => WorkerSdkError::Downstream {
            source: Box::new(source),
            continuation_id,
        },
        None => source,
    }
}

fn continuation_status_to_sdk(error: Status) -> WorkerSdkError {
    let continuation_id = error
        .metadata()
        .get(DOWNSTREAM_CONTINUATION_METADATA)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    with_downstream_provenance(
        WorkerSdkError::Transport(error.to_string()),
        continuation_id,
    )
}

fn status_from_sdk(error: WorkerSdkError) -> Status {
    let mut status = Status::internal(error.to_string());
    if let WorkerSdkError::Downstream {
        continuation_id, ..
    } = error
        && let Ok(value) = continuation_id.parse()
    {
        status
            .metadata_mut()
            .insert(DOWNSTREAM_CONTINUATION_METADATA, value);
    }
    status
}

struct ScopedJsonStream {
    inner: JsonStream,
    scope: Option<ScopeContext>,
    callback_name: Option<String>,
}

impl ScopedJsonStream {
    fn new(inner: JsonStream, scope: Option<ScopeContext>) -> Self {
        Self {
            inner,
            scope,
            callback_name: current_callback_name(),
        }
    }
}

impl Stream for ScopedJsonStream {
    type Item = Result<Json>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let scope = this.scope.clone();
        TASK_CALLBACK_NAME.sync_scope(this.callback_name.clone(), || {
            TASK_SCOPE_CONTEXT.sync_scope(scope.clone(), || {
                with_thread_scope(&scope, || this.inner.as_mut().poll_next(cx))
            })
        })
    }
}

fn invocation_scope_context(scope: Option<&ScopeContext>) -> Option<ScopeContext> {
    scope
        .filter(|scope| !scope.scope_stack_id.trim().is_empty())
        .cloned()
}

fn current_scope_context() -> Option<ScopeContext> {
    TASK_SCOPE_CONTEXT
        .try_with(Clone::clone)
        .ok()
        .flatten()
        .or_else(|| THREAD_SCOPE_CONTEXT.with(|scope| scope.borrow().clone()))
}

fn with_thread_scope<T>(scope: &Option<ScopeContext>, f: impl FnOnce() -> T) -> T {
    let _guard = ThreadScopeBinding::new(scope.clone());
    let previous = THREAD_CALLBACK_NAME.with(|current| current.replace(current_callback_name()));
    let _callback = ThreadCallbackBinding(previous);
    f()
}

fn current_callback_name() -> Option<String> {
    TASK_CALLBACK_NAME
        .try_with(Clone::clone)
        .ok()
        .flatten()
        .or_else(|| THREAD_CALLBACK_NAME.with(|name| name.borrow().clone()))
}

struct ThreadCallbackBinding(Option<String>);

impl Drop for ThreadCallbackBinding {
    fn drop(&mut self) {
        THREAD_CALLBACK_NAME.with(|name| name.replace(self.0.take()));
    }
}

struct ThreadScopeBinding {
    previous: Option<ScopeContext>,
}

impl ThreadScopeBinding {
    fn new(scope: Option<ScopeContext>) -> Self {
        let previous = THREAD_SCOPE_CONTEXT.with(|current| current.replace(scope));
        Self { previous }
    }
}

impl Drop for ThreadScopeBinding {
    fn drop(&mut self) {
        let previous = self.previous.take();
        THREAD_SCOPE_CONTEXT.with(|scope| {
            scope.replace(previous);
        });
    }
}

fn scope_context(scope_stack_id: &str) -> ScopeContext {
    ScopeContext {
        scope_stack_id: scope_stack_id.into(),
        parent_scope_id: String::new(),
    }
}

fn registration_surface(kind: RuntimeRegistrationKind) -> RegistrationSurface {
    match kind {
        RuntimeRegistrationKind::Subscriber => RegistrationSurface::Subscriber,
        RuntimeRegistrationKind::EventMetadataInjector => {
            RegistrationSurface::EventMetadataInjector
        }
        RuntimeRegistrationKind::MarkSanitizeGuardrail => {
            RegistrationSurface::MarkSanitizeGuardrail
        }
        RuntimeRegistrationKind::ScopeSanitizeStartGuardrail => {
            RegistrationSurface::ScopeSanitizeStartGuardrail
        }
        RuntimeRegistrationKind::ScopeSanitizeEndGuardrail => {
            RegistrationSurface::ScopeSanitizeEndGuardrail
        }
        RuntimeRegistrationKind::ToolSanitizeRequestGuardrail => {
            RegistrationSurface::ToolSanitizeRequestGuardrail
        }
        RuntimeRegistrationKind::ToolSanitizeResponseGuardrail => {
            RegistrationSurface::ToolSanitizeResponseGuardrail
        }
        RuntimeRegistrationKind::ToolConditionalExecutionGuardrail => {
            RegistrationSurface::ToolConditionalExecutionGuardrail
        }
        RuntimeRegistrationKind::ToolRequestIntercept => RegistrationSurface::ToolRequestIntercept,
        RuntimeRegistrationKind::ToolExecutionIntercept => {
            RegistrationSurface::ToolExecutionIntercept
        }
        RuntimeRegistrationKind::LlmSanitizeRequestGuardrail => {
            RegistrationSurface::LlmSanitizeRequestGuardrail
        }
        RuntimeRegistrationKind::LlmSanitizeResponseGuardrail => {
            RegistrationSurface::LlmSanitizeResponseGuardrail
        }
        RuntimeRegistrationKind::LlmConditionalExecutionGuardrail => {
            RegistrationSurface::LlmConditionalExecutionGuardrail
        }
        RuntimeRegistrationKind::LlmRequestIntercept => RegistrationSurface::LlmRequestIntercept,
        RuntimeRegistrationKind::LlmExecutionIntercept => {
            RegistrationSurface::LlmExecutionIntercept
        }
        RuntimeRegistrationKind::LlmStreamExecutionIntercept => {
            RegistrationSurface::LlmStreamExecutionIntercept
        }
    }
}

fn runtime_registration_kind(surface: RegistrationSurface) -> Result<RuntimeRegistrationKind> {
    match surface {
        RegistrationSurface::Subscriber => Ok(RuntimeRegistrationKind::Subscriber),
        RegistrationSurface::EventMetadataInjector => {
            Ok(RuntimeRegistrationKind::EventMetadataInjector)
        }
        RegistrationSurface::MarkSanitizeGuardrail => {
            Ok(RuntimeRegistrationKind::MarkSanitizeGuardrail)
        }
        RegistrationSurface::ScopeSanitizeStartGuardrail => {
            Ok(RuntimeRegistrationKind::ScopeSanitizeStartGuardrail)
        }
        RegistrationSurface::ScopeSanitizeEndGuardrail => {
            Ok(RuntimeRegistrationKind::ScopeSanitizeEndGuardrail)
        }
        RegistrationSurface::ToolSanitizeRequestGuardrail => {
            Ok(RuntimeRegistrationKind::ToolSanitizeRequestGuardrail)
        }
        RegistrationSurface::ToolSanitizeResponseGuardrail => {
            Ok(RuntimeRegistrationKind::ToolSanitizeResponseGuardrail)
        }
        RegistrationSurface::ToolConditionalExecutionGuardrail => {
            Ok(RuntimeRegistrationKind::ToolConditionalExecutionGuardrail)
        }
        RegistrationSurface::ToolRequestIntercept => {
            Ok(RuntimeRegistrationKind::ToolRequestIntercept)
        }
        RegistrationSurface::ToolExecutionIntercept => {
            Ok(RuntimeRegistrationKind::ToolExecutionIntercept)
        }
        RegistrationSurface::LlmSanitizeRequestGuardrail => {
            Ok(RuntimeRegistrationKind::LlmSanitizeRequestGuardrail)
        }
        RegistrationSurface::LlmSanitizeResponseGuardrail => {
            Ok(RuntimeRegistrationKind::LlmSanitizeResponseGuardrail)
        }
        RegistrationSurface::LlmConditionalExecutionGuardrail => {
            Ok(RuntimeRegistrationKind::LlmConditionalExecutionGuardrail)
        }
        RegistrationSurface::LlmRequestIntercept => {
            Ok(RuntimeRegistrationKind::LlmRequestIntercept)
        }
        RegistrationSurface::LlmExecutionIntercept => {
            Ok(RuntimeRegistrationKind::LlmExecutionIntercept)
        }
        RegistrationSurface::LlmStreamExecutionIntercept => {
            Ok(RuntimeRegistrationKind::LlmStreamExecutionIntercept)
        }
        RegistrationSurface::ConditionalMiddlewareGuardrail | RegistrationSurface::Unspecified => {
            Err(WorkerSdkError::Callback(
                "surface is not a runtime registration kind".into(),
            ))
        }
    }
}

fn proto_scope_type(scope_type: ScopeType) -> i32 {
    (match scope_type {
        ScopeType::Agent => nemo_relay_worker_proto::nvidia_worker_v1::ScopeType::Agent,
        ScopeType::Function => nemo_relay_worker_proto::nvidia_worker_v1::ScopeType::Function,
        ScopeType::Tool => nemo_relay_worker_proto::nvidia_worker_v1::ScopeType::Tool,
        ScopeType::Llm => nemo_relay_worker_proto::nvidia_worker_v1::ScopeType::Llm,
        ScopeType::Retriever => nemo_relay_worker_proto::nvidia_worker_v1::ScopeType::Retriever,
        ScopeType::Embedder => nemo_relay_worker_proto::nvidia_worker_v1::ScopeType::Embedder,
        ScopeType::Reranker => nemo_relay_worker_proto::nvidia_worker_v1::ScopeType::Reranker,
        ScopeType::Guardrail => nemo_relay_worker_proto::nvidia_worker_v1::ScopeType::Guardrail,
        ScopeType::Evaluator => nemo_relay_worker_proto::nvidia_worker_v1::ScopeType::Evaluator,
        ScopeType::Custom => nemo_relay_worker_proto::nvidia_worker_v1::ScopeType::Custom,
        ScopeType::Unknown => nemo_relay_worker_proto::nvidia_worker_v1::ScopeType::Unknown,
    }) as i32
}

fn all_surfaces() -> Vec<RegistrationSurface> {
    vec![
        RegistrationSurface::Subscriber,
        RegistrationSurface::EventMetadataInjector,
        RegistrationSurface::ToolSanitizeRequestGuardrail,
        RegistrationSurface::ToolSanitizeResponseGuardrail,
        RegistrationSurface::ToolConditionalExecutionGuardrail,
        RegistrationSurface::ToolRequestIntercept,
        RegistrationSurface::ToolExecutionIntercept,
        RegistrationSurface::LlmSanitizeRequestGuardrail,
        RegistrationSurface::LlmSanitizeResponseGuardrail,
        RegistrationSurface::LlmConditionalExecutionGuardrail,
        RegistrationSurface::LlmRequestIntercept,
        RegistrationSurface::LlmExecutionIntercept,
        RegistrationSurface::LlmStreamExecutionIntercept,
        RegistrationSurface::MarkSanitizeGuardrail,
        RegistrationSurface::ScopeSanitizeStartGuardrail,
        RegistrationSurface::ScopeSanitizeEndGuardrail,
        RegistrationSurface::ConditionalMiddlewareGuardrail,
    ]
}

async fn connect_host_endpoint(endpoint: &str) -> Result<Channel> {
    if endpoint.starts_with("unix://") {
        return connect_uds(endpoint).await;
    }
    let endpoint = normalize_tcp_endpoint(endpoint)?;
    Endpoint::from_shared(endpoint)
        .map_err(|err| WorkerSdkError::InvalidInput(err.to_string()))?
        .connect()
        .await
        .map_err(|err| WorkerSdkError::Transport(err.to_string()))
}

#[cfg(unix)]
async fn connect_uds(endpoint: &str) -> Result<Channel> {
    let path = Arc::new(parse_unix_endpoint(endpoint)?);
    let endpoint = Endpoint::try_from("http://[::]:50051")
        .map_err(|err| WorkerSdkError::Transport(err.to_string()))?;
    endpoint
        .connect_with_connector(service_fn(move |_| {
            let path = path.clone();
            async move {
                let stream = UnixStream::connect(&*path).await?;
                Ok::<_, std::io::Error>(TokioIo::new(stream))
            }
        }))
        .await
        .map_err(|err| WorkerSdkError::Transport(err.to_string()))
}

#[cfg(not(unix))]
async fn connect_uds(_endpoint: &str) -> Result<Channel> {
    Err(WorkerSdkError::InvalidInput(
        "unix endpoints are not supported on this platform".into(),
    ))
}

fn parse_tcp_endpoint(endpoint: &str) -> Result<SocketAddr> {
    let endpoint = normalize_tcp_endpoint(endpoint)?;
    let authority = endpoint
        .strip_prefix("http://")
        .expect("normalized TCP endpoints always use http scheme");
    if authority.contains('/') {
        return Err(WorkerSdkError::InvalidInput(format!(
            "unsupported TCP endpoint '{endpoint}'"
        )));
    }
    authority
        .to_socket_addrs()
        .map_err(|err| {
            WorkerSdkError::InvalidInput(format!("invalid TCP endpoint '{endpoint}': {err}"))
        })?
        .next()
        .ok_or_else(|| WorkerSdkError::InvalidInput(format!("invalid TCP endpoint '{endpoint}'")))
}

fn normalize_tcp_endpoint(endpoint: &str) -> Result<String> {
    if let Some(authority) = endpoint.strip_prefix("tcp://") {
        if authority.is_empty() {
            return Err(WorkerSdkError::InvalidInput(format!(
                "unsupported endpoint '{endpoint}'"
            )));
        }
        return Ok(format!("http://{authority}"));
    }
    if endpoint.starts_with("http://") {
        return Ok(endpoint.to_owned());
    }
    Err(WorkerSdkError::InvalidInput(format!(
        "unsupported endpoint '{endpoint}'"
    )))
}

#[cfg(unix)]
fn parse_unix_endpoint(endpoint: &str) -> Result<PathBuf> {
    endpoint
        .strip_prefix("unix://")
        .map(PathBuf::from)
        .ok_or_else(|| WorkerSdkError::InvalidInput(format!("unsupported endpoint '{endpoint}'")))
}

#[cfg(unix)]
fn remove_stale_socket(path: &Path) -> Result<()> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(err) => {
            return Err(WorkerSdkError::Transport(format!(
                "failed to inspect worker socket path '{}': {err}",
                path.display()
            )));
        }
    };
    if !metadata.file_type().is_socket() {
        return Err(WorkerSdkError::InvalidInput(format!(
            "worker socket path '{}' exists and is not a socket",
            path.display()
        )));
    }
    std::fs::remove_file(path).map_err(|err| {
        WorkerSdkError::Transport(format!(
            "failed to remove stale worker socket '{}': {err}",
            path.display()
        ))
    })
}

fn required_env(name: &str) -> Result<String> {
    std::env::var(name).map_err(|_| {
        WorkerSdkError::InvalidInput(format!("environment variable {name} is required"))
    })
}

fn optional_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
}

fn write_endpoint_file(path: &Path, endpoint: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|err| {
            WorkerSdkError::Transport(format!(
                "failed to create worker endpoint file directory '{}': {err}",
                parent.display()
            ))
        })?;
    }
    std::fs::write(path, endpoint).map_err(|err| {
        WorkerSdkError::Transport(format!(
            "failed to write worker endpoint file '{}': {err}",
            path.display()
        ))
    })
}

fn rustc_version_runtime() -> String {
    option_env!("RUSTC_VERSION")
        .unwrap_or("unknown")
        .to_string()
}

#[cfg(test)]
#[path = "../tests/unit/codec_identity_tests.rs"]
mod codec_identity_tests;
#[cfg(test)]
#[path = "../tests/unit/error_provenance_tests.rs"]
mod error_provenance_tests;
#[cfg(test)]
#[path = "../tests/unit/execution_context_tests.rs"]
mod execution_context_tests;
#[cfg(test)]
#[path = "../tests/unit/timestamp_tests.rs"]
mod timestamp_tests;

trait DecodeTo<D> {
    fn decode_to(&self) -> Result<D>;
}
impl<D: WireValue<W>, W: Clone> DecodeTo<D> for W {
    fn decode_to(&self) -> Result<D> {
        conversions::decode(self.clone()).map_err(Into::into)
    }
}
fn decode_wire<D>(value: &impl DecodeTo<D>) -> Result<D> {
    value.decode_to()
}
fn encode_wire<D: Clone + WireValue<W>, W>(value: &D) -> Result<W> {
    conversions::encode(value.clone()).map_err(Into::into)
}
impl From<conversions::ConversionError> for WorkerSdkError {
    fn from(error: conversions::ConversionError) -> Self {
        Self::InvalidInput(error.to_string())
    }
}

fn unchanged_response() -> InvokeResponse {
    InvokeResponse {
        result: Some(wire::invoke_response::Result::Unchanged(())),
    }
}
fn llm_response(value: Json) -> InvokeResponse {
    InvokeResponse {
        result: Some(wire::invoke_response::Result::LlmResponse(infallible_wire(
            &value,
        ))),
    }
}
trait RpcResult<D> {
    fn read_result(self) -> Result<D>;
}
fn decode_rpc_result<D>(value: impl RpcResult<D>) -> Result<D> {
    value.read_result()
}
macro_rules! rpc_result {
    ($response:ty, $module:ident, $domain:ty) => {
        impl RpcResult<$domain> for $response {
            fn read_result(self) -> Result<$domain> {
                nemo_relay_worker_proto::validation::validate(&self)
                    .map_err(WorkerSdkError::InvalidInput)?;
                match self.result {
                    Some(wire::$module::Result::Value(v)) => {
                        conversions::decode(v).map_err(Into::into)
                    }
                    Some(wire::$module::Result::Error(e)) => Err(worker_error_to_sdk(e)),
                    None => Err(WorkerSdkError::InvalidInput("missing RPC result".into())),
                }
            }
        }
    };
}
rpc_result!(
    wire::ToolNextResponse,
    tool_next_response,
    ToolExecutionResult
);
rpc_result!(wire::LlmNextResponse, llm_next_response, Json);
rpc_result!(
    wire::DecodeLlmCodecRequestResponse,
    decode_llm_codec_request_response,
    AnnotatedLlmRequest
);
rpc_result!(
    wire::EncodeLlmCodecRequestResponse,
    encode_llm_codec_request_response,
    LlmRequest
);
rpc_result!(
    wire::DecodeLlmCodecResponseResponse,
    decode_llm_codec_response_response,
    AnnotatedLlmResponse
);

impl WorkerService {
    fn identity(&self) -> wire::WorkerIdentity {
        wire::WorkerIdentity {
            plugin_id: self.plugin.plugin_kind().into(),
            plugin_kind: self.plugin.plugin_kind().into(),
            protocol: Some(wire::ProtocolIdentity {
                name: WORKER_PROTOCOL.into(),
                version: wire::CONTRACT_VERSION.into(),
            }),
            sdk_name: "nemo-relay-worker".into(),
            sdk_version: env!("CARGO_PKG_VERSION").into(),
            runtime_name: "rust".into(),
            runtime_version: rustc_version_runtime(),
        }
    }
}

fn micros_to_timestamp(value: Option<i64>) -> Result<Option<prost_types::Timestamp>> {
    value
        .map(|micros| {
            Ok(prost_types::Timestamp {
                seconds: micros.div_euclid(1_000_000),
                nanos: (micros.rem_euclid(1_000_000) * 1000) as i32,
            })
        })
        .transpose()
}
