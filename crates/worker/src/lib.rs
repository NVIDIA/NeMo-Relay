// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![deny(rustdoc::broken_intra_doc_links, rustdoc::private_intra_doc_links)]

//! Rust SDK for NeMo Relay out-of-process gRPC worker plugins.
//!
//! # Invocation cancellation
//!
//! The `grpc-v1` service tracks active unary and streaming callbacks by the
//! host-provided invocation ID. Relay sends `CancelInvocation` when a managed
//! caller is cancelled, an invocation times out, or a host stream is abandoned.
//! The SDK aborts the matching async callback task and reports
//! `worker.cancelled`; cancellation of an unknown, completed, or already
//! cancelled ID returns a negative acknowledgment.
//!
//! Cancellation is cooperative. Dropping an async callback future releases its
//! Rust-owned resources, but an accepted acknowledgment does not prove that
//! arbitrary blocking work started by the callback has stopped.

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
pub use nemo_relay_types::api::scope::ScopeType;
pub use nemo_relay_types::api::tool::{ToolExecutionInterceptOutcome, ToolExecutionResult};
pub use nemo_relay_types::codec::identity::{BuiltinLlmCodec, LlmCodecIdentity};
pub use nemo_relay_types::codec::optimization::{
    LlmOptimizationContribution, LlmOptimizationEvidenceQuality, LlmOptimizationKind,
    LlmOptimizationModel, LlmOptimizationModelTransition, LlmOptimizationPayload,
    LlmOptimizationSummary, LlmOptimizationSummaryStatus, LlmOptimizationTokenImpact,
    LlmOptimizationTokens,
};
pub use nemo_relay_types::codec::request::{ANNOTATED_LLM_REQUEST_SCHEMA, AnnotatedLlmRequest};
pub use nemo_relay_types::codec::response::AnnotatedLlmResponse;
pub use nemo_relay_types::plugin::{ConfigDiagnostic, DiagnosticLevel};
use nemo_relay_worker_proto::v1::plugin_worker_server::{PluginWorker, PluginWorkerServer};
use nemo_relay_worker_proto::v1::relay_host_runtime_client::RelayHostRuntimeClient;
use nemo_relay_worker_proto::v1::{
    CancelInvocationRequest, ConditionalMiddlewareGuardrailRegistration, CreateScopeStackRequest,
    DeregisterConditionalMiddlewareGuardrailRequest, DropScopeStackRequest, EmitMarkRequest,
    EmptyResult, GetRuntimeDiagnosticsRequest, GuardrailResult, HandshakeRequest,
    HandshakeResponse, HealthRequest, HealthResponse, InvokeRequest, InvokeResponse, JsonEnvelope,
    JsonResult, ListRuntimeRegistrationsRequest, LlmCodecDecodeRequest, LlmCodecDecodeResponse,
    LlmCodecEncodeRequest, LlmCodecKind, LlmNextRequest, LlmRequestInterceptResult,
    LlmStreamNextRequest, PopScopeRequest, PushScopeRequest,
    RegisterConditionalMiddlewareGuardrailRequest, RegisterRequest, RegisterResponse, Registration,
    RegistrationSurface, RuntimeDiagnostic as ProtoRuntimeDiagnostic, ScopeContext,
    ShutdownRequest, StreamChunk,
    ToolExecutionInterceptOutcome as ProtoToolExecutionInterceptOutcome,
    ToolExecutionInterceptResult, ToolExecutionResult as ProtoToolExecutionResult,
    ToolExecutionResultResponse, ToolNextRequest, ValidateRequest, ValidateResponse, WorkerAck,
    WorkerError,
};
use nemo_relay_worker_proto::{
    WORKER_PROTOCOL_GRPC_V1, decode_json_envelope, decode_json_value, json_envelope, json_value,
};
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

const JSON_SCHEMA: &str = "nemo.relay.Json@1";
const DATA_SCHEMA_SCHEMA: &str = "nemo.relay.DataSchema@1";
const LLM_REQUEST_SCHEMA: &str = "nemo.relay.LlmRequest@1";

tokio::task_local! {
    static TASK_SCOPE_CONTEXT: Option<ScopeContext>;
}

thread_local! {
    static THREAD_SCOPE_CONTEXT: RefCell<Option<ScopeContext>> = const { RefCell::new(None) };
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
    /// JSON serialization failed.
    #[error("serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Optional fields supported by the additive `grpc-v1` mark-emission request.
#[derive(Debug, Clone, Default)]
pub struct EmitMarkOptions {
    /// Schema identifier for the mark's opaque data payload.
    pub data_schema: Option<DataSchema>,
    /// Telemetry severity for OTLP log projection.
    pub severity: Option<LogSeverity>,
}

/// Trait implemented by Rust out-of-process worker plugins.
pub trait WorkerPlugin: Send + Sync + 'static {
    /// Stable plugin id/kind returned to the Relay host.
    fn plugin_id(&self) -> &str;

    /// Whether multiple configured components of this plugin kind are allowed.
    fn allows_multiple_components(&self) -> bool {
        false
    }

    /// Validates component config.
    fn validate(&self, _config: &Json) -> Vec<ConfigDiagnostic> {
        Vec::new()
    }

    /// Registers callbacks into the worker context.
    fn register(&self, ctx: &mut PluginContext, config: &Json) -> Result<()>;
}

type SubscriberFn = Arc<dyn Fn(&Event) + Send + Sync>;
type EventSanitizeFn =
    Arc<dyn Fn(&Event, EventSanitizeFields) -> BoxFutureResult<EventSanitizeFields> + Send + Sync>;
type EventMetadataInjectorFn =
    Arc<dyn Fn(Arc<Event>) -> BoxFutureResult<BTreeMap<String, Json>> + Send + Sync>;
type ToolSanitizeFn = Arc<dyn Fn(&str, Json) -> BoxFutureResult<Json> + Send + Sync>;
type ToolConditionalFn = Arc<dyn Fn(String, Json) -> BoxFutureResult<Option<String>> + Send + Sync>;
type ToolRequestFn = Arc<dyn Fn(String, Json) -> BoxFutureResult<Json> + Send + Sync>;
type ToolExecutionFn = Arc<
    dyn Fn(&str, Json, ToolNext) -> BoxFutureResult<ToolExecutionInterceptOutcome> + Send + Sync,
>;
type LlmSanitizeRequestFn = Arc<
    dyn Fn(LlmRequest, LlmSanitizeRequestContext) -> BoxFutureResult<Option<LlmRequest>>
        + Send
        + Sync,
>;
type LlmSanitizeResponseFn =
    Arc<dyn Fn(Json, LlmSanitizeResponseContext) -> BoxFutureResult<Option<Json>> + Send + Sync>;

/// Active codec context supplied to an LLM request sanitizer.
#[derive(Clone)]
pub struct LlmSanitizeRequestContext {
    /// Identity of the active codec.
    pub codec: LlmCodecIdentity,
    runtime: Option<PluginRuntime>,
    codec_capability_id: Option<String>,
    invocation_id: Option<String>,
}

/// Active codec context supplied to an LLM response sanitizer.
#[derive(Clone)]
pub struct LlmSanitizeResponseContext {
    /// Identity of the active codec.
    pub codec: LlmCodecIdentity,
    runtime: Option<PluginRuntime>,
    codec_capability_id: Option<String>,
    invocation_id: Option<String>,
}

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
type LlmExecutionFn = Arc<dyn Fn(&str, LlmRequest, LlmNext) -> BoxFutureResult<Json> + Send + Sync>;
type LlmStreamExecutionFn =
    Arc<dyn Fn(&str, LlmRequest, LlmStreamNext) -> BoxFutureResult<JsonStream> + Send + Sync>;

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
        F: Fn(&str, Json, ToolNext) -> Fut + Send + Sync + 'static,
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
            Arc::new(move |tool, value, next| Box::pin(callback(tool, value, next))),
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
        F: Fn(&str, LlmRequest, LlmNext) -> Fut + Send + Sync + 'static,
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
            Arc::new(move |model, request, next| Box::pin(callback(model, request, next))),
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
        F: Fn(&str, LlmRequest, LlmStreamNext) -> Fut + Send + Sync + 'static,
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
            Arc::new(move |model, request, next| Box::pin(callback(model, request, next))),
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
    activation_id: String,
    auth_token: String,
    host_endpoint: String,
    host_channel: Arc<OnceCell<Channel>>,
    conditional_middleware_callbacks: Arc<Mutex<HashMap<String, ConditionalMiddlewareFn>>>,
}

impl PluginRuntime {
    /// List global runtime registrations, optionally filtered by kind.
    pub async fn list_runtime_registrations(
        &self,
        kinds: Option<BTreeSet<RuntimeRegistrationKind>>,
    ) -> Result<Vec<RuntimeRegistrationIdentity>> {
        let mut client = self.host_client().await?;
        let response = client
            .list_runtime_registrations(Request::new(ListRuntimeRegistrationsRequest {
                activation_id: self.activation_id.clone(),
                auth_token: self.auth_token.clone(),
                kinds: kinds
                    .unwrap_or_default()
                    .into_iter()
                    .map(|kind| registration_surface(kind) as i32)
                    .collect(),
            }))
            .await
            .map_err(|error| WorkerSdkError::Transport(error.to_string()))?
            .into_inner();
        if let Some(error) = response.error {
            return Err(worker_error_to_sdk(error));
        }
        response
            .registrations
            .into_iter()
            .map(|registration| {
                let kind = RegistrationSurface::try_from(registration.kind)
                    .map_err(|_| WorkerSdkError::Callback("unknown registration kind".into()))?;
                let owner = registration.owner.ok_or_else(|| {
                    WorkerSdkError::Callback("runtime registration owner is missing".into())
                })?;
                let owner_kind =
                    match nemo_relay_worker_proto::v1::RuntimeRegistrationOwnerKind::try_from(
                        owner.kind,
                    )
                    .map_err(|_| {
                        WorkerSdkError::Callback("unknown registration owner kind".into())
                    })? {
                        nemo_relay_worker_proto::v1::RuntimeRegistrationOwnerKind::Core => {
                            RuntimeRegistrationOwnerKind::Core
                        }
                        nemo_relay_worker_proto::v1::RuntimeRegistrationOwnerKind::GlobalApi => {
                            RuntimeRegistrationOwnerKind::GlobalApi
                        }
                        nemo_relay_worker_proto::v1::RuntimeRegistrationOwnerKind::Plugin => {
                            RuntimeRegistrationOwnerKind::Plugin
                        }
                        nemo_relay_worker_proto::v1::RuntimeRegistrationOwnerKind::Unspecified => {
                            return Err(WorkerSdkError::Callback(
                                "registration owner kind is unspecified".into(),
                            ));
                        }
                    };
                Ok(RuntimeRegistrationIdentity {
                    kind: runtime_registration_kind(kind)?,
                    local_name: registration.local_name,
                    effective_name: registration.effective_name,
                    owner: RuntimeRegistrationOwner {
                        kind: owner_kind,
                        plugin_kind: owner.plugin_kind,
                        component_ordinal: owner.component_ordinal,
                    },
                })
            })
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
                    activation_id: self.activation_id.clone(),
                    auth_token: self.auth_token.clone(),
                    name: name.into(),
                    kinds: kinds
                        .into_iter()
                        .map(|kind| registration_surface(kind) as i32)
                        .collect(),
                    registration_name: registration_name.into(),
                    reason: String::new(),
                    callback: true,
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
        if let Some(error) = response.error {
            self.remove_conditional_middleware_callback(name);
            return Err(worker_error_to_sdk(error));
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
                    activation_id: self.activation_id.clone(),
                    auth_token: self.auth_token.clone(),
                    handle: handle.handle.clone(),
                },
            ))
            .await
            .map_err(|error| WorkerSdkError::Transport(error.to_string()))?
            .into_inner();
        if let Some(error) = response.error {
            return Err(worker_error_to_sdk(error));
        }
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
            .decode_llm_codec_request(Request::new(LlmCodecDecodeRequest {
                activation_id: self.activation_id.clone(),
                auth_token: self.auth_token.clone(),
                codec_capability_id: capability_id.into(),
                invocation_id: invocation_id.into(),
                request: Some(json_envelope(LLM_REQUEST_SCHEMA, request)?),
            }))
            .await
            .map_err(|err| WorkerSdkError::Transport(err.to_string()))?
            .into_inner();
        decode_typed_json_result(response, ANNOTATED_LLM_REQUEST_SCHEMA)
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
            .encode_llm_codec_request(Request::new(LlmCodecEncodeRequest {
                activation_id: self.activation_id.clone(),
                auth_token: self.auth_token.clone(),
                codec_capability_id: capability_id.into(),
                invocation_id: invocation_id.into(),
                annotated_request: Some(json_envelope(ANNOTATED_LLM_REQUEST_SCHEMA, annotated)?),
                original_request: Some(json_envelope(LLM_REQUEST_SCHEMA, original)?),
            }))
            .await
            .map_err(|err| WorkerSdkError::Transport(err.to_string()))?
            .into_inner();
        decode_typed_json_result(response, LLM_REQUEST_SCHEMA)
    }

    async fn decode_llm_codec_response(
        &self,
        capability_id: &str,
        invocation_id: &str,
        response: &Json,
    ) -> Result<AnnotatedLlmResponse> {
        let mut client = self.host_client().await?;
        let response = client
            .decode_llm_codec_response(Request::new(LlmCodecDecodeResponse {
                activation_id: self.activation_id.clone(),
                auth_token: self.auth_token.clone(),
                codec_capability_id: capability_id.into(),
                invocation_id: invocation_id.into(),
                response: Some(json_envelope(JSON_SCHEMA, response)?),
            }))
            .await
            .map_err(|err| WorkerSdkError::Transport(err.to_string()))?
            .into_inner();
        decode_typed_json_result(response, JSON_SCHEMA)
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
        let response = client
            .emit_mark(Request::new(EmitMarkRequest {
                activation_id: self.activation_id.clone(),
                auth_token: self.auth_token.clone(),
                scope,
                name: name.into(),
                data: optional_json_envelope(data)?,
                metadata: optional_json_envelope(metadata)?,
                data_schema: optional_typed_json_envelope(
                    DATA_SCHEMA_SCHEMA,
                    options.data_schema.as_ref(),
                )?,
                severity: options
                    .severity
                    .as_ref()
                    .map(severity_wire_value)
                    .transpose()?
                    .unwrap_or_default(),
                category: category
                    .as_ref()
                    .map(EventCategory::as_str)
                    .unwrap_or_default()
                    .into(),
            }))
            .await
            .map_err(|err| WorkerSdkError::Transport(err.to_string()))?
            .into_inner();
        ack_to_result(response.ok, response.error)
    }

    /// Return a bounded snapshot of active host runtime diagnostics.
    pub async fn runtime_diagnostics(&self) -> Result<RuntimeDiagnostics> {
        let mut client = self.host_client().await?;
        let response = client
            .get_runtime_diagnostics(Request::new(GetRuntimeDiagnosticsRequest {
                activation_id: self.activation_id.clone(),
                auth_token: self.auth_token.clone(),
            }))
            .await
            .map_err(|error| {
                if error.code() == tonic::Code::Unimplemented {
                    WorkerSdkError::Callback(
                        "runtime diagnostics require a Relay host with the grpc-v1 diagnostics extension"
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
                activation_id: self.activation_id.clone(),
                auth_token: self.auth_token.clone(),
            }))
            .await
            .map_err(|err| WorkerSdkError::Transport(err.to_string()))?
            .into_inner();
        if let Some(error) = response.error {
            return Err(worker_error_to_sdk(error));
        }
        Ok(response.scope_stack_id)
    }

    /// Drops an isolated host-owned scope stack.
    pub async fn drop_scope_stack(&self, scope_stack_id: &str) -> Result<()> {
        let mut client = self.host_client().await?;
        let response = client
            .drop_scope_stack(Request::new(DropScopeStackRequest {
                activation_id: self.activation_id.clone(),
                auth_token: self.auth_token.clone(),
                scope_stack_id: scope_stack_id.into(),
            }))
            .await
            .map_err(|err| WorkerSdkError::Transport(err.to_string()))?
            .into_inner();
        ack_to_result(response.ok, response.error)
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
        let scope = scope_stack_id
            .map(scope_context)
            .or_else(|| self.current_scope_context());
        let mut client = self.host_client().await?;
        let response = client
            .push_scope(Request::new(PushScopeRequest {
                activation_id: self.activation_id.clone(),
                auth_token: self.auth_token.clone(),
                scope,
                name: name.into(),
                scope_type: proto_scope_type(scope_type),
                data: optional_json_envelope(data)?,
                metadata: optional_json_envelope(metadata)?,
                input: optional_json_envelope(input)?,
            }))
            .await
            .map_err(|err| WorkerSdkError::Transport(err.to_string()))?
            .into_inner();
        if let Some(error) = response.error {
            return Err(worker_error_to_sdk(error));
        }
        Ok(response.scope_handle_id)
    }

    /// Pops a scope through the host runtime.
    pub async fn pop_scope(
        &self,
        scope_handle_id: &str,
        output: Option<Json>,
        metadata: Option<Json>,
    ) -> Result<()> {
        let mut client = self.host_client().await?;
        let response = client
            .pop_scope(Request::new(PopScopeRequest {
                activation_id: self.activation_id.clone(),
                auth_token: self.auth_token.clone(),
                scope_handle_id: scope_handle_id.into(),
                output: optional_json_envelope(output)?,
                metadata: optional_json_envelope(metadata)?,
            }))
            .await
            .map_err(|err| WorkerSdkError::Transport(err.to_string()))?
            .into_inner();
        ack_to_result(response.ok, response.error)
    }

    async fn host_client(&self) -> Result<RelayHostRuntimeClient<Channel>> {
        self.host_channel
            .get_or_try_init(|| connect_host_endpoint(&self.host_endpoint))
            .await
            .cloned()
            .map(RelayHostRuntimeClient::new)
    }

    fn current_scope_context(&self) -> Option<ScopeContext> {
        current_scope_context()
    }
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
                activation_id: self.runtime.activation_id.clone(),
                auth_token: self.runtime.auth_token.clone(),
                continuation_id: self.continuation_id.clone(),
                value: Some(json_envelope(JSON_SCHEMA, &value)?),
                scope: self.runtime.current_scope_context(),
            }))
            .await
            .map_err(|err| WorkerSdkError::Transport(err.to_string()))?
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
                activation_id: self.runtime.activation_id.clone(),
                auth_token: self.runtime.auth_token.clone(),
                continuation_id: self.continuation_id.clone(),
                request: Some(json_envelope(LLM_REQUEST_SCHEMA, &request)?),
                scope: self.runtime.current_scope_context(),
            }))
            .await
            .map_err(|err| WorkerSdkError::Transport(err.to_string()))?
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
                activation_id: self.runtime.activation_id.clone(),
                auth_token: self.runtime.auth_token.clone(),
                continuation_id: self.continuation_id.clone(),
                request: Some(json_envelope(LLM_REQUEST_SCHEMA, &request)?),
                scope: scope.clone(),
            }))
            .await
            .map_err(|err| WorkerSdkError::Transport(err.to_string()))?;
        let stream = response.into_inner().map(|chunk| match chunk {
            Ok(chunk) => stream_chunk_to_json(chunk),
            Err(err) => Err(WorkerSdkError::Transport(err.to_string())),
        });
        Ok(Box::pin(ScopedJsonStream::new(Box::pin(stream), scope)))
    }
}

/// Serves a worker plugin using environment variables supplied by the Relay host.
///
/// # Errors
/// Returns an error when required worker environment variables are missing or
/// the gRPC server fails.
pub async fn serve_plugin(plugin: impl WorkerPlugin) -> Result<()> {
    serve_plugin_arc(Arc::new(plugin)).await
}

/// Serves a shared worker plugin using environment variables supplied by the Relay host.
///
/// # Errors
/// Returns an error when required worker environment variables are missing or
/// the gRPC server fails.
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
/// Returns an error when the endpoint configuration is invalid or the gRPC
/// server fails.
pub async fn serve_plugin_arc_with_config(
    plugin: Arc<dyn WorkerPlugin>,
    config: WorkerServerConfig,
) -> Result<()> {
    serve_plugin_arc_with_endpoint_file(plugin, config, None).await
}

async fn serve_plugin_arc_with_endpoint_file(
    plugin: Arc<dyn WorkerPlugin>,
    config: WorkerServerConfig,
    endpoint_file: Option<PathBuf>,
) -> Result<()> {
    let runtime = PluginRuntime {
        activation_id: config.activation_id,
        auth_token: config.auth_token,
        host_endpoint: config.host_endpoint,
        host_channel: Arc::new(OnceCell::new()),
        conditional_middleware_callbacks: Arc::new(Mutex::new(HashMap::new())),
    };
    let service = WorkerService {
        plugin,
        runtime,
        handlers: Arc::new(Mutex::new(WorkerHandlers::default())),
        active_invocations: Arc::new(Mutex::new(HashMap::new())),
        next_invocation_generation: Arc::new(AtomicU64::new(1)),
    };
    serve_worker_service(service, &config.worker_endpoint, endpoint_file.as_deref()).await
}

#[cfg(unix)]
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
        return Server::builder()
            .add_service(PluginWorkerServer::new(service))
            .serve_with_incoming(UnixListenerStream::new(listener))
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
    Server::builder()
        .add_service(PluginWorkerServer::new(service))
        .serve_with_incoming(TcpListenerStream::new(listener))
        .await
        .map_err(|err| WorkerSdkError::Transport(err.to_string()))
}

#[derive(Clone)]
struct WorkerService {
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
}

struct ActiveInvocationGuard {
    active_invocations: Arc<Mutex<HashMap<String, ActiveInvocation>>>,
    invocation_id: String,
    generation: u64,
    armed: bool,
}

impl ActiveInvocationGuard {
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for ActiveInvocationGuard {
    fn drop(&mut self) {
        if self.armed
            && let Ok(mut active) = self.active_invocations.lock()
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
        self.authorize(&request.activation_id, &request.auth_token)?;
        Ok(Response::new(HandshakeResponse {
            plugin_id: self.plugin.plugin_id().into(),
            plugin_kind: self.plugin.plugin_id().into(),
            allows_multiple_components: self.plugin.allows_multiple_components(),
            worker_protocol: WORKER_PROTOCOL_GRPC_V1.into(),
            sdk_name: "nemo-relay-worker".into(),
            sdk_version: env!("CARGO_PKG_VERSION").into(),
            runtime_name: "rust".into(),
            runtime_version: rustc_version_runtime(),
            supported_surfaces: all_surfaces()
                .into_iter()
                .map(|surface| surface as i32)
                .collect(),
        }))
    }

    async fn health(
        &self,
        request: Request<HealthRequest>,
    ) -> std::result::Result<Response<HealthResponse>, Status> {
        let request = request.into_inner();
        self.authorize(&request.activation_id, &request.auth_token)?;
        Ok(Response::new(HealthResponse {
            ok: true,
            message: "ready".into(),
            plugin_id: self.plugin.plugin_id().into(),
            worker_protocol: WORKER_PROTOCOL_GRPC_V1.into(),
            sdk_name: "nemo-relay-worker".into(),
            sdk_version: env!("CARGO_PKG_VERSION").into(),
            runtime_name: "rust".into(),
            runtime_version: rustc_version_runtime(),
        }))
    }

    async fn validate(
        &self,
        request: Request<ValidateRequest>,
    ) -> std::result::Result<Response<ValidateResponse>, Status> {
        let request = request.into_inner();
        self.authorize(&request.activation_id, &request.auth_token)?;
        let config = request
            .config
            .as_ref()
            .map(decode_json_envelope::<Json>)
            .transpose()
            .map_err(|err| Status::invalid_argument(format!("invalid config JSON: {err}")))?
            .unwrap_or(Json::Null);
        let diagnostics = self.plugin.validate(&config);
        Ok(Response::new(ValidateResponse {
            diagnostics: Some(infallible_json_envelope(
                "nemo.relay.PluginDiagnostics@1",
                &diagnostics,
            )),
            error: None,
        }))
    }

    async fn register(
        &self,
        request: Request<RegisterRequest>,
    ) -> std::result::Result<Response<RegisterResponse>, Status> {
        let request = request.into_inner();
        self.authorize(&request.activation_id, &request.auth_token)?;
        let config = request
            .config
            .as_ref()
            .map(decode_json_envelope::<Json>)
            .transpose()
            .map_err(|err| Status::invalid_argument(format!("invalid config JSON: {err}")))?
            .unwrap_or(Json::Null);
        let mut ctx = PluginContext::with_runtime(self.runtime.clone());
        if let Err(err) = self.plugin.register(&mut ctx, &config) {
            return Ok(Response::new(RegisterResponse {
                registrations: Vec::new(),
                error: Some(sdk_error_to_worker(err)),
                conditional_middleware_guardrails: Vec::new(),
            }));
        }
        if let Err(err) = validate_unique_registrations(&ctx.handlers.registrations) {
            return Ok(Response::new(RegisterResponse {
                registrations: Vec::new(),
                error: Some(sdk_error_to_worker(err)),
                conditional_middleware_guardrails: Vec::new(),
            }));
        }
        if let Err(err) = validate_initial_gates(&ctx.handlers.conditional_middleware_guardrails) {
            return Ok(Response::new(RegisterResponse {
                registrations: Vec::new(),
                error: Some(sdk_error_to_worker(err)),
                conditional_middleware_guardrails: Vec::new(),
            }));
        }
        let registrations = ctx.handlers.registrations.clone();
        let conditional_middleware_guardrails =
            ctx.handlers.conditional_middleware_guardrails.clone();
        {
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
                    registrations: Vec::new(),
                    error: Some(sdk_error_to_worker(WorkerSdkError::InvalidInput(format!(
                        "conditional middleware callback '{name}' is already registered"
                    )))),
                    conditional_middleware_guardrails: Vec::new(),
                }));
            }
            for name in handlers.conditional_middleware_callbacks.keys() {
                callbacks.remove(name);
            }
            callbacks.extend(ctx.handlers.conditional_middleware_callbacks.clone());
            *handlers = ctx.handlers;
        }
        Ok(Response::new(RegisterResponse {
            registrations,
            error: None,
            conditional_middleware_guardrails,
        }))
    }

    async fn invoke(
        &self,
        request: Request<InvokeRequest>,
    ) -> std::result::Result<Response<InvokeResponse>, Status> {
        let request = request.into_inner();
        self.authorize(&request.activation_id, &request.auth_token)?;
        let invocation_id = request.invocation_id.clone();
        let service = self.clone();
        let task = tokio::spawn(async move { service.invoke_inner(request).await });
        let abort_handle = task.abort_handle();
        let generation = match self.track_invocation(&invocation_id, abort_handle.clone(), None) {
            Ok(generation) => generation,
            Err(err) => {
                task.abort();
                return Err(err);
            }
        };
        let _active_guard = ActiveInvocationGuard {
            active_invocations: self.active_invocations.clone(),
            invocation_id,
            generation,
            armed: true,
        };
        let mut abort_on_drop = AbortTaskOnDrop::new(abort_handle);
        let response = match task.await {
            Ok(response) => response,
            Err(err) if err.is_cancelled() => cancelled_invoke_response(),
            Err(err) => InvokeResponse {
                result: Some(nemo_relay_worker_proto::v1::invoke_response::Result::Error(
                    WorkerError {
                        code: "worker.error".into(),
                        message: format!("worker invocation task failed: {err}"),
                        retryable: false,
                    },
                )),
            },
        };
        abort_on_drop.disarm();
        Ok(Response::new(response))
    }

    type InvokeStreamStream =
        Pin<Box<dyn tokio_stream::Stream<Item = std::result::Result<StreamChunk, Status>> + Send>>;

    async fn invoke_stream(
        &self,
        request: Request<InvokeRequest>,
    ) -> std::result::Result<Response<Self::InvokeStreamStream>, Status> {
        let request = request.into_inner();
        self.authorize(&request.activation_id, &request.auth_token)?;
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
        let request_value =
            required_json::<LlmRequest>(payload.request, "llm request").map_err(status_from_sdk)?;
        let next = LlmStreamNext {
            runtime: self.runtime.clone(),
            continuation_id: request.continuation_id,
        };
        let model_name = payload.model_name;
        let (tx, rx) = mpsc::channel(16);
        let (stream_cancel_tx, stream_cancel_rx) = watch::channel(false);
        let open_scope = scope.clone();
        let open_task = tokio::spawn(async move {
            TASK_SCOPE_CONTEXT
                .scope(open_scope.clone(), async {
                    let future = with_thread_scope(&open_scope, || {
                        handler(&model_name, request_value, next)
                    });
                    future.await
                })
                .await
        });
        let open_abort_handle = open_task.abort_handle();
        let generation = match self.track_invocation(
            &invocation_id,
            open_abort_handle.clone(),
            Some(stream_cancel_tx.clone()),
        ) {
            Ok(generation) => generation,
            Err(err) => {
                open_task.abort();
                return Err(err);
            }
        };
        let mut active_guard = ActiveInvocationGuard {
            active_invocations: self.active_invocations.clone(),
            invocation_id: invocation_id.clone(),
            generation,
            armed: true,
        };
        let mut abort_on_drop = AbortTaskOnDrop::new(open_abort_handle);
        let stream = match open_task.await {
            Ok(Ok(stream)) => stream,
            Ok(Err(err)) => {
                abort_on_drop.disarm();
                return Err(Status::internal(err.to_string()));
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
        let active_invocations = self.active_invocations.clone();
        let task_invocation_id = invocation_id.clone();
        let task_tx = tx;
        let task = tokio::spawn(async move {
            let _active_guard = ActiveInvocationGuard {
                active_invocations,
                invocation_id: task_invocation_id,
                generation,
                armed: true,
            };
            let mut stream = ScopedJsonStream::new(stream, scope);
            loop {
                let item = tokio::select! {
                    item = stream.next() => item,
                    _ = task_tx.closed() => return,
                };
                let Some(item) = item else {
                    return;
                };
                let chunk = match item {
                    Ok(value) => StreamChunk {
                        item: Some(nemo_relay_worker_proto::v1::stream_chunk::Item::Value(
                            match json_envelope(JSON_SCHEMA, &value) {
                                Ok(value) => value,
                                Err(err) => {
                                    let _ =
                                        task_tx.send(Err(Status::internal(err.to_string()))).await;
                                    return;
                                }
                            },
                        )),
                    },
                    Err(err) => StreamChunk {
                        item: Some(nemo_relay_worker_proto::v1::stream_chunk::Item::Error(
                            sdk_error_to_worker(err),
                        )),
                    },
                };
                if task_tx.send(Ok(chunk)).await.is_err() {
                    return;
                }
            }
        });
        let replaced = self.replace_invocation(
            &invocation_id,
            generation,
            ActiveInvocation {
                generation,
                abort_handle: task.abort_handle(),
                stream_cancel: Some(stream_cancel_tx),
            },
        );
        match replaced {
            Ok(true) => active_guard.disarm(),
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
    ) -> std::result::Result<Response<WorkerAck>, Status> {
        let request = request.into_inner();
        self.authorize(&request.activation_id, &request.auth_token)?;
        let active = {
            let mut invocations = self
                .active_invocations
                .lock()
                .map_err(|err| Status::internal(format!("invocation lock poisoned: {err}")))?;
            invocations.remove(&request.invocation_id)
        };
        let Some(active) = active else {
            return Ok(Response::new(WorkerAck {
                accepted: false,
                message: "invocation is not active".into(),
            }));
        };
        if let Some(cancel) = active.stream_cancel {
            let _ = cancel.send(true);
        }
        active.abort_handle.abort();
        Ok(Response::new(WorkerAck {
            accepted: true,
            message: if request.reason.is_empty() {
                "cancellation accepted".into()
            } else {
                format!("cancellation accepted: {}", request.reason)
            },
        }))
    }

    async fn shutdown(
        &self,
        request: Request<ShutdownRequest>,
    ) -> std::result::Result<Response<WorkerAck>, Status> {
        let request = request.into_inner();
        self.authorize(&request.activation_id, &request.auth_token)?;
        Ok(Response::new(WorkerAck {
            accepted: false,
            message: "shutdown is not implemented by the Rust worker SDK yet".into(),
        }))
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
            return Err(Status::permission_denied("invalid worker activation"));
        }
        if auth_token != self.runtime.auth_token {
            return Err(Status::permission_denied("invalid worker token"));
        }
        Ok(())
    }

    fn track_invocation(
        &self,
        invocation_id: &str,
        abort_handle: tokio::task::AbortHandle,
        stream_cancel: Option<watch::Sender<bool>>,
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
            .is_none_or(|entry| entry.generation != generation)
        {
            return Ok(false);
        }
        active.insert(invocation_id.to_string(), invocation);
        Ok(true)
    }

    async fn invoke_inner(&self, request: InvokeRequest) -> InvokeResponse {
        match self.invoke_result(request).await {
            Ok(response) => response,
            Err(err) => InvokeResponse {
                result: Some(nemo_relay_worker_proto::v1::invoke_response::Result::Error(
                    sdk_error_to_worker(err),
                )),
            },
        }
    }

    async fn invoke_result(&self, request: InvokeRequest) -> Result<InvokeResponse> {
        let scope = invocation_scope_context(request.scope.as_ref());
        TASK_SCOPE_CONTEXT
            .scope(scope.clone(), self.invoke_result_scoped(request, scope))
            .await
    }

    async fn invoke_result_scoped(
        &self,
        request: InvokeRequest,
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
        request: InvokeRequest,
        scope: &Option<ScopeContext>,
    ) -> Result<InvokeResponse> {
        let payload = match request.payload {
            Some(nemo_relay_worker_proto::v1::invoke_request::Payload::ConditionalMiddleware(
                payload,
            )) => payload,
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
        request: InvokeRequest,
        scope: &Option<ScopeContext>,
    ) -> Result<InvokeResponse> {
        let event = event_payload(request.payload)?;
        let handler = self.subscriber(&request.registration_name)?;
        with_thread_scope(scope, || handler(&event));
        Ok(empty_response())
    }

    async fn invoke_event_metadata_injector_response(
        &self,
        request: InvokeRequest,
        scope: &Option<ScopeContext>,
    ) -> Result<InvokeResponse> {
        let event = event_payload(request.payload)?;
        let handler = self.event_metadata_injector(&request.registration_name)?;
        let future = with_thread_scope(scope, || handler(Arc::new(event)));
        Ok(json_response(serde_json::to_value(future.await?).map_err(
            |err| WorkerSdkError::Callback(format!("serialize Event metadata additions: {err}")),
        )?))
    }

    async fn invoke_event_sanitize_response(
        &self,
        request: InvokeRequest,
        scope: &Option<ScopeContext>,
        surface: RegistrationSurface,
    ) -> Result<InvokeResponse> {
        let event = event_payload(request.payload)?;
        let fields = event.sanitize_fields();
        let handler = self.event_sanitizer(surface, &request.registration_name)?;
        let fields = with_thread_scope(scope, || handler(&event, fields)).await?;
        Ok(json_response(
            serde_json::to_value(fields).expect("event sanitize fields are JSON serializable"),
        ))
    }

    async fn invoke_tool_response(
        &self,
        request: InvokeRequest,
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
        request: InvokeRequest,
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
        request: InvokeRequest,
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
        request: InvokeRequest,
        scope: &Option<ScopeContext>,
    ) -> Result<InvokeResponse> {
        let payload = tool_payload(request.payload)?;
        let handler = self.tool_conditional(&request.registration_name)?;
        let future = with_thread_scope(scope, || handler(payload.tool_name, payload.value));
        Ok(guardrail_response(future.await?))
    }

    async fn invoke_tool_request_response(
        &self,
        request: InvokeRequest,
        scope: &Option<ScopeContext>,
    ) -> Result<InvokeResponse> {
        let payload = tool_payload(request.payload)?;
        let handler = self.tool_request(&request.registration_name)?;
        let future = with_thread_scope(scope, || handler(payload.tool_name, payload.value));
        Ok(json_response(future.await?))
    }

    async fn invoke_tool_execution_response(
        &self,
        request: InvokeRequest,
        scope: &Option<ScopeContext>,
    ) -> Result<InvokeResponse> {
        let payload = tool_payload(request.payload)?;
        let handler = self.tool_execution(&request.registration_name)?;
        let next = ToolNext {
            runtime: self.runtime.clone(),
            continuation_id: request.continuation_id,
        };
        let future = with_thread_scope(scope, || handler(&payload.tool_name, payload.value, next));
        tool_execution_response(future.await?)
    }

    async fn invoke_llm_response(
        &self,
        request: InvokeRequest,
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
        request: InvokeRequest,
        scope: &Option<ScopeContext>,
    ) -> Result<InvokeResponse> {
        let payload = llm_payload(request.payload)?;
        let mut context = payload.sanitize_request_context(&request.invocation_id);
        context.runtime = Some(self.runtime.clone());
        let request_value = required_json::<LlmRequest>(payload.request, "llm request")?;
        let handler = self.llm_sanitize_request(&request.registration_name)?;
        match with_thread_scope(scope, || handler(request_value, context)).await? {
            Some(request) => Ok(json_response(
                serde_json::to_value(request).expect("LLM request is JSON serializable"),
            )),
            None => Ok(empty_response()),
        }
    }

    async fn invoke_llm_sanitize_response_response(
        &self,
        request: InvokeRequest,
        scope: &Option<ScopeContext>,
    ) -> Result<InvokeResponse> {
        let payload = llm_payload(request.payload)?;
        let mut context = payload.sanitize_response_context(&request.invocation_id);
        context.runtime = Some(self.runtime.clone());
        let response = required_json::<Json>(payload.response, "llm response")?;
        let handler = self.llm_sanitize_response(&request.registration_name)?;
        match with_thread_scope(scope, || handler(response, context)).await? {
            Some(response) => Ok(json_response(response)),
            None => Ok(empty_response()),
        }
    }

    async fn invoke_llm_conditional_response(
        &self,
        request: InvokeRequest,
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
        request: InvokeRequest,
        scope: &Option<ScopeContext>,
    ) -> Result<InvokeResponse> {
        let payload = llm_payload(request.payload)?;
        let request_value = required_json::<LlmRequest>(payload.request, "llm request")?;
        let annotated = payload
            .annotated_request
            .map(|value| {
                decode_expected_json_envelope::<AnnotatedLlmRequest>(
                    &value,
                    "annotated llm request",
                    ANNOTATED_LLM_REQUEST_SCHEMA,
                )
            })
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
        request: InvokeRequest,
        scope: &Option<ScopeContext>,
    ) -> Result<InvokeResponse> {
        let payload = llm_payload(request.payload)?;
        let request_value = required_json::<LlmRequest>(payload.request, "llm request")?;
        let handler = self.llm_execution(&request.registration_name)?;
        let next = LlmNext {
            runtime: self.runtime.clone(),
            continuation_id: request.continuation_id,
        };
        let future = with_thread_scope(scope, || handler(&payload.model_name, request_value, next));
        Ok(json_response(future.await?))
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
}

struct LlmPayload {
    model_name: String,
    request: Option<JsonEnvelope>,
    annotated_request: Option<JsonEnvelope>,
    response: Option<JsonEnvelope>,
    sanitize_context: Option<nemo_relay_worker_proto::v1::llm_invocation::SanitizeContext>,
}

impl LlmPayload {
    fn sanitize_request_context(&self, invocation_id: &str) -> LlmSanitizeRequestContext {
        let codec = match self.sanitize_context.as_ref() {
            Some(nemo_relay_worker_proto::v1::llm_invocation::SanitizeContext::RequestSanitizeContext(context)) => context.codec.as_ref(),
            _ => None,
        };
        LlmSanitizeRequestContext {
            codec: codec_identity_from_proto(codec),
            runtime: None,
            codec_capability_id: match self.sanitize_context.as_ref() {
                Some(nemo_relay_worker_proto::v1::llm_invocation::SanitizeContext::RequestSanitizeContext(context)) => context.codec_capability_id.clone(),
                _ => None,
            },
            invocation_id: Some(invocation_id.to_owned()),
        }
    }

    fn sanitize_response_context(&self, invocation_id: &str) -> LlmSanitizeResponseContext {
        let codec = match self.sanitize_context.as_ref() {
            Some(nemo_relay_worker_proto::v1::llm_invocation::SanitizeContext::ResponseSanitizeContext(context)) => context.codec.as_ref(),
            _ => None,
        };
        LlmSanitizeResponseContext {
            codec: codec_identity_from_proto(codec),
            runtime: None,
            codec_capability_id: match self.sanitize_context.as_ref() {
                Some(nemo_relay_worker_proto::v1::llm_invocation::SanitizeContext::ResponseSanitizeContext(context)) => context.codec_capability_id.clone(),
                _ => None,
            },
            invocation_id: Some(invocation_id.to_owned()),
        }
    }
}

fn codec_identity_from_proto(
    codec: Option<&nemo_relay_worker_proto::v1::LlmCodecIdentity>,
) -> LlmCodecIdentity {
    let codec_kind = codec
        .map(|codec| codec.kind)
        .unwrap_or(LlmCodecKind::Unspecified as i32);
    let codec_id = codec.and_then(|codec| codec.id.clone());
    match LlmCodecKind::try_from(codec_kind).ok() {
        Some(LlmCodecKind::Unspecified) => LlmCodecIdentity::None,
        Some(LlmCodecKind::Builtin) => codec_id
            .as_deref()
            .and_then(BuiltinLlmCodec::from_id)
            .map_or(LlmCodecIdentity::Opaque, LlmCodecIdentity::BuiltIn),
        Some(LlmCodecKind::Runtime) => codec_id
            .filter(|id| !id.is_empty())
            .map_or(LlmCodecIdentity::Opaque, LlmCodecIdentity::Runtime),
        Some(LlmCodecKind::Opaque) | None => LlmCodecIdentity::Opaque,
    }
}

fn event_payload(
    payload: Option<nemo_relay_worker_proto::v1::invoke_request::Payload>,
) -> Result<Event> {
    match payload {
        Some(nemo_relay_worker_proto::v1::invoke_request::Payload::Event(value)) => {
            Ok(decode_json_envelope::<Event>(&value)?)
        }
        _ => Err(WorkerSdkError::InvalidInput(
            "expected event payload".into(),
        )),
    }
}

fn tool_payload(
    payload: Option<nemo_relay_worker_proto::v1::invoke_request::Payload>,
) -> Result<ToolPayload> {
    match payload {
        Some(nemo_relay_worker_proto::v1::invoke_request::Payload::Tool(value)) => {
            let json = required_json::<Json>(value.value, "tool value")?;
            Ok(ToolPayload {
                tool_name: value.tool_name,
                value: json,
            })
        }
        _ => Err(WorkerSdkError::InvalidInput("expected tool payload".into())),
    }
}

fn llm_payload(
    payload: Option<nemo_relay_worker_proto::v1::invoke_request::Payload>,
) -> Result<LlmPayload> {
    match payload {
        Some(nemo_relay_worker_proto::v1::invoke_request::Payload::Llm(value)) => Ok(LlmPayload {
            model_name: value.model_name,
            request: value.request,
            annotated_request: value.annotated_request,
            response: value.response,
            sanitize_context: value.sanitize_context,
        }),
        _ => Err(WorkerSdkError::InvalidInput("expected llm payload".into())),
    }
}

fn required_json<T: serde::de::DeserializeOwned>(
    value: Option<JsonEnvelope>,
    field: &str,
) -> Result<T> {
    let value = value.ok_or_else(|| WorkerSdkError::InvalidInput(format!("{field} is missing")))?;
    Ok(decode_json_envelope::<T>(&value)?)
}

fn decode_expected_json_envelope<T: serde::de::DeserializeOwned>(
    value: &JsonEnvelope,
    field: &str,
    expected_schema: &str,
) -> Result<T> {
    if value.schema != expected_schema {
        return Err(WorkerSdkError::InvalidInput(format!(
            "{field} has schema {:?}; expected {expected_schema:?}",
            value.schema
        )));
    }
    Ok(decode_json_envelope(value)?)
}

fn empty_response() -> InvokeResponse {
    InvokeResponse {
        result: Some(nemo_relay_worker_proto::v1::invoke_response::Result::Empty(
            EmptyResult {},
        )),
    }
}

fn json_response(value: Json) -> InvokeResponse {
    InvokeResponse {
        result: Some(nemo_relay_worker_proto::v1::invoke_response::Result::Json(
            JsonResult {
                value: Some(infallible_json_envelope(JSON_SCHEMA, &value)),
                error: None,
            },
        )),
    }
}

fn guardrail_response(reason: Option<String>) -> InvokeResponse {
    InvokeResponse {
        result: Some(
            nemo_relay_worker_proto::v1::invoke_response::Result::Guardrail(GuardrailResult {
                block_reason: reason.unwrap_or_default(),
            }),
        ),
    }
}

fn llm_request_response(outcome: LlmRequestInterceptOutcome) -> Result<InvokeResponse> {
    Ok(InvokeResponse {
        result: Some(
            nemo_relay_worker_proto::v1::invoke_response::Result::LlmRequest(
                LlmRequestInterceptResult {
                    outcome: Some(json_envelope(
                        nemo_relay_types::api::llm::LLM_REQUEST_INTERCEPT_OUTCOME_SCHEMA,
                        &outcome,
                    )?),
                },
            ),
        ),
    })
}

fn tool_execution_response(outcome: ToolExecutionInterceptOutcome) -> Result<InvokeResponse> {
    Ok(InvokeResponse {
        result: Some(
            nemo_relay_worker_proto::v1::invoke_response::Result::ToolExecution(
                ToolExecutionInterceptResult {
                    outcome: Some(tool_execution_outcome_to_proto(outcome)?),
                },
            ),
        ),
    })
}

fn tool_execution_result_to_sdk(
    result: ToolExecutionResultResponse,
) -> Result<ToolExecutionResult> {
    if let Some(error) = result.error {
        return Err(worker_error_to_sdk(error));
    }
    let value = result
        .value
        .ok_or_else(|| WorkerSdkError::InvalidInput("tool execution result is missing".into()))?;
    tool_execution_result_from_proto(value)
}

fn tool_execution_outcome_to_proto(
    outcome: ToolExecutionInterceptOutcome,
) -> Result<ProtoToolExecutionInterceptOutcome> {
    Ok(ProtoToolExecutionInterceptOutcome {
        result: Some(json_value(&outcome.result)?),
        annotation: outcome
            .annotation
            .as_ref()
            .filter(|value| !value.is_null())
            .map(json_value)
            .transpose()?,
        pending_marks: (!outcome.pending_marks.is_empty())
            .then(|| json_value(&outcome.pending_marks))
            .transpose()?,
    })
}

fn tool_execution_result_from_proto(
    value: ProtoToolExecutionResult,
) -> Result<ToolExecutionResult> {
    let result = value.result.ok_or_else(|| {
        WorkerSdkError::InvalidInput("tool execution result.result is missing".into())
    })?;
    let annotation = value
        .annotation
        .as_ref()
        .map(decode_json_value)
        .transpose()?
        .filter(|value: &Json| !value.is_null());
    Ok(ToolExecutionResult {
        result: decode_json_value(&result)?,
        annotation,
    })
}

fn stream_chunk_to_json(chunk: StreamChunk) -> Result<Json> {
    match chunk.item {
        Some(nemo_relay_worker_proto::v1::stream_chunk::Item::Value(value)) => {
            Ok(decode_json_envelope::<Json>(&value)?)
        }
        Some(nemo_relay_worker_proto::v1::stream_chunk::Item::Error(error)) => {
            Err(worker_error_to_sdk(error))
        }
        None => Err(WorkerSdkError::InvalidInput("empty stream chunk".into())),
    }
}

fn json_result_to_sdk(result: JsonResult) -> Result<Json> {
    if let Some(error) = result.error {
        return Err(worker_error_to_sdk(error));
    }
    required_json(result.value, "json result")
}

fn decode_typed_json_result<T: serde::de::DeserializeOwned>(
    result: JsonResult,
    expected_schema: &str,
) -> Result<T> {
    if let Some(error) = result.error {
        return Err(worker_error_to_sdk(error));
    }
    let value = result
        .value
        .ok_or_else(|| WorkerSdkError::InvalidInput("json result is missing".into()))?;
    decode_expected_json_envelope(&value, "json result", expected_schema)
}

fn optional_json_envelope(value: Option<Json>) -> Result<Option<JsonEnvelope>> {
    value
        .as_ref()
        .map(|value| json_envelope(JSON_SCHEMA, value).map_err(WorkerSdkError::from))
        .transpose()
}

fn optional_typed_json_envelope<T: serde::Serialize>(
    schema: &str,
    value: Option<&T>,
) -> Result<Option<JsonEnvelope>> {
    value
        .map(|value| json_envelope(schema, value).map_err(WorkerSdkError::from))
        .transpose()
}

fn severity_wire_value(value: &LogSeverity) -> Result<String> {
    serde_json::to_value(value)?
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| {
            WorkerSdkError::InvalidInput("LogSeverity did not serialize as a string".into())
        })
}

fn infallible_json_envelope<T: serde::Serialize>(schema: &str, value: &T) -> JsonEnvelope {
    json_envelope(schema, value).expect("Relay DTOs and serde_json::Value are JSON serializable")
}

fn sdk_error_to_worker(error: WorkerSdkError) -> WorkerError {
    WorkerError {
        code: "worker.error".into(),
        message: error.to_string(),
        retryable: false,
    }
}

fn cancelled_worker_error() -> WorkerError {
    WorkerError {
        code: "worker.cancelled".into(),
        message: "worker invocation was cancelled".into(),
        retryable: false,
    }
}

fn cancelled_invoke_response() -> InvokeResponse {
    InvokeResponse {
        result: Some(nemo_relay_worker_proto::v1::invoke_response::Result::Error(
            cancelled_worker_error(),
        )),
    }
}

fn cancelled_stream_chunk() -> StreamChunk {
    StreamChunk {
        item: Some(nemo_relay_worker_proto::v1::stream_chunk::Item::Error(
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
    WorkerSdkError::Callback(format!("{}: {}", error.code, error.message))
}

fn status_from_sdk(error: WorkerSdkError) -> Status {
    Status::internal(error.to_string())
}

fn ack_to_result(ok: bool, error: Option<WorkerError>) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(error
            .map(worker_error_to_sdk)
            .unwrap_or_else(|| WorkerSdkError::Callback("host call failed".into())))
    }
}

struct ScopedJsonStream {
    inner: JsonStream,
    scope: Option<ScopeContext>,
}

impl ScopedJsonStream {
    fn new(inner: JsonStream, scope: Option<ScopeContext>) -> Self {
        Self { inner, scope }
    }
}

impl Stream for ScopedJsonStream {
    type Item = Result<Json>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let scope = this.scope.clone();
        TASK_SCOPE_CONTEXT.sync_scope(scope.clone(), || {
            with_thread_scope(&scope, || this.inner.as_mut().poll_next(cx))
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
    f()
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
        ScopeType::Agent => nemo_relay_worker_proto::v1::ScopeType::Agent,
        ScopeType::Function => nemo_relay_worker_proto::v1::ScopeType::Function,
        ScopeType::Tool => nemo_relay_worker_proto::v1::ScopeType::Tool,
        ScopeType::Llm => nemo_relay_worker_proto::v1::ScopeType::Llm,
        ScopeType::Retriever => nemo_relay_worker_proto::v1::ScopeType::Retriever,
        ScopeType::Embedder => nemo_relay_worker_proto::v1::ScopeType::Embedder,
        ScopeType::Reranker => nemo_relay_worker_proto::v1::ScopeType::Reranker,
        ScopeType::Guardrail => nemo_relay_worker_proto::v1::ScopeType::Guardrail,
        ScopeType::Evaluator => nemo_relay_worker_proto::v1::ScopeType::Evaluator,
        ScopeType::Custom => nemo_relay_worker_proto::v1::ScopeType::Custom,
        ScopeType::Unknown => nemo_relay_worker_proto::v1::ScopeType::Unknown,
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
