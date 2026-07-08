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
use std::collections::{HashMap, HashSet};
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
pub use nemo_relay_types::api::event::{Event, EventSanitizeFields, PendingMarkSpec};
pub use nemo_relay_types::api::llm::{LlmRequest, LlmRequestInterceptOutcome};
pub use nemo_relay_types::api::scope::ScopeType;
pub use nemo_relay_types::api::tool::ToolExecutionInterceptOutcome;
use nemo_relay_types::codec::request::AnnotatedLlmRequest;
pub use nemo_relay_types::plugin::{ConfigDiagnostic, DiagnosticLevel};
use nemo_relay_worker_proto::v1::plugin_worker_server::{PluginWorker, PluginWorkerServer};
use nemo_relay_worker_proto::v1::relay_host_runtime_client::RelayHostRuntimeClient;
use nemo_relay_worker_proto::v1::{
    CancelInvocationRequest, CreateScopeStackRequest, DropScopeStackRequest, EmitMarkRequest,
    EmptyResult, GuardrailResult, HandshakeRequest, HandshakeResponse, HealthRequest,
    HealthResponse, InvokeRequest, InvokeResponse, JsonEnvelope, JsonResult, LlmNextRequest,
    LlmRequestInterceptResult, LlmStreamNextRequest, PopScopeRequest, PushScopeRequest,
    RegisterRequest, RegisterResponse, Registration, RegistrationSurface, ScopeContext,
    ShutdownRequest, StreamChunk, ToolExecutionInterceptResult, ToolNextRequest, ValidateRequest,
    ValidateResponse, WorkerAck, WorkerError,
};
use nemo_relay_worker_proto::{WORKER_PROTOCOL_GRPC_V1, decode_json_envelope, json_envelope};
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

/// Boxed future returned by async worker callbacks.
pub type BoxFutureResult<T> = Pin<Box<dyn Future<Output = Result<T>> + Send>>;

/// Boxed JSON stream returned by streaming worker callbacks.
pub type JsonStream = Pin<Box<dyn tokio_stream::Stream<Item = Result<Json>> + Send>>;

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
    Arc<dyn Fn(&Event, EventSanitizeFields) -> EventSanitizeFields + Send + Sync>;
type ToolSanitizeFn = Arc<dyn Fn(&str, Json) -> Json + Send + Sync>;
type ToolConditionalFn = Arc<dyn Fn(&str, &Json) -> Result<Option<String>> + Send + Sync>;
type ToolRequestFn = Arc<dyn Fn(&str, Json) -> Result<Json> + Send + Sync>;
type ToolExecutionFn = Arc<
    dyn Fn(&str, Json, ToolNext) -> BoxFutureResult<ToolExecutionInterceptOutcome> + Send + Sync,
>;
type LlmSanitizeRequestFn = Arc<dyn Fn(LlmRequest) -> LlmRequest + Send + Sync>;
type LlmSanitizeResponseFn = Arc<dyn Fn(Json) -> Json + Send + Sync>;
type LlmConditionalFn = Arc<dyn Fn(&LlmRequest) -> Result<Option<String>> + Send + Sync>;
type LlmRequestFn = Arc<
    dyn Fn(&str, LlmRequest, Option<AnnotatedLlmRequest>) -> Result<LlmRequestInterceptOutcome>
        + Send
        + Sync,
>;
type LlmExecutionFn = Arc<dyn Fn(&str, LlmRequest, LlmNext) -> BoxFutureResult<Json> + Send + Sync>;
type LlmStreamExecutionFn =
    Arc<dyn Fn(&str, LlmRequest, LlmStreamNext) -> BoxFutureResult<JsonStream> + Send + Sync>;

#[derive(Default)]
struct WorkerHandlers {
    registrations: Vec<Registration>,
    subscribers: HashMap<String, SubscriberFn>,
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

    fn register_event_sanitizer<F>(
        &mut self,
        name: &str,
        priority: i32,
        surface: RegistrationSurface,
        callback: F,
    ) where
        F: Fn(&Event, EventSanitizeFields) -> EventSanitizeFields + Send + Sync + 'static,
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
        sanitizers.insert(name.into(), Arc::new(callback));
    }

    /// Registers a mark event sanitizer.
    pub fn register_mark_sanitize_guardrail<F>(&mut self, name: &str, priority: i32, callback: F)
    where
        F: Fn(&Event, EventSanitizeFields) -> EventSanitizeFields + Send + Sync + 'static,
    {
        self.register_event_sanitizer(
            name,
            priority,
            RegistrationSurface::MarkSanitizeGuardrail,
            callback,
        );
    }

    /// Registers a scope-start event sanitizer.
    pub fn register_scope_sanitize_start_guardrail<F>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) where
        F: Fn(&Event, EventSanitizeFields) -> EventSanitizeFields + Send + Sync + 'static,
    {
        self.register_event_sanitizer(
            name,
            priority,
            RegistrationSurface::ScopeSanitizeStartGuardrail,
            callback,
        );
    }

    /// Registers a scope-end event sanitizer.
    pub fn register_scope_sanitize_end_guardrail<F>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) where
        F: Fn(&Event, EventSanitizeFields) -> EventSanitizeFields + Send + Sync + 'static,
    {
        self.register_event_sanitizer(
            name,
            priority,
            RegistrationSurface::ScopeSanitizeEndGuardrail,
            callback,
        );
    }

    /// Registers a tool sanitize-request guardrail.
    pub fn register_tool_sanitize_request_guardrail<F>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) where
        F: Fn(&str, Json) -> Json + Send + Sync + 'static,
    {
        self.push_registration(
            name,
            RegistrationSurface::ToolSanitizeRequestGuardrail,
            priority,
            false,
        );
        self.handlers
            .tool_sanitize_requests
            .insert(name.into(), Arc::new(callback));
    }

    /// Registers a tool sanitize-response guardrail.
    pub fn register_tool_sanitize_response_guardrail<F>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) where
        F: Fn(&str, Json) -> Json + Send + Sync + 'static,
    {
        self.push_registration(
            name,
            RegistrationSurface::ToolSanitizeResponseGuardrail,
            priority,
            false,
        );
        self.handlers
            .tool_sanitize_responses
            .insert(name.into(), Arc::new(callback));
    }

    /// Registers a tool conditional-execution guardrail.
    pub fn register_tool_conditional_execution_guardrail<F>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) where
        F: Fn(&str, &Json) -> Result<Option<String>> + Send + Sync + 'static,
    {
        self.push_registration(
            name,
            RegistrationSurface::ToolConditionalExecutionGuardrail,
            priority,
            false,
        );
        self.handlers
            .tool_conditionals
            .insert(name.into(), Arc::new(callback));
    }

    /// Registers a tool request intercept.
    pub fn register_tool_request_intercept<F>(
        &mut self,
        name: &str,
        priority: i32,
        break_chain: bool,
        callback: F,
    ) where
        F: Fn(&str, Json) -> Result<Json> + Send + Sync + 'static,
    {
        self.push_registration(
            name,
            RegistrationSurface::ToolRequestIntercept,
            priority,
            break_chain,
        );
        self.handlers
            .tool_requests
            .insert(name.into(), Arc::new(callback));
    }

    /// Registers a tool execution intercept.
    ///
    /// The callback returns a [`ToolExecutionInterceptOutcome`]. Calling
    /// [`ToolNext::call`] continues the chain and returns only the raw
    /// downstream result JSON; Relay retains downstream pending marks.
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
    pub fn register_llm_sanitize_request_guardrail<F>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) where
        F: Fn(LlmRequest) -> LlmRequest + Send + Sync + 'static,
    {
        self.push_registration(
            name,
            RegistrationSurface::LlmSanitizeRequestGuardrail,
            priority,
            false,
        );
        self.handlers
            .llm_sanitize_requests
            .insert(name.into(), Arc::new(callback));
    }

    /// Registers an LLM sanitize-response guardrail.
    pub fn register_llm_sanitize_response_guardrail<F>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) where
        F: Fn(Json) -> Json + Send + Sync + 'static,
    {
        self.push_registration(
            name,
            RegistrationSurface::LlmSanitizeResponseGuardrail,
            priority,
            false,
        );
        self.handlers
            .llm_sanitize_responses
            .insert(name.into(), Arc::new(callback));
    }

    /// Registers an LLM conditional-execution guardrail.
    pub fn register_llm_conditional_execution_guardrail<F>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) where
        F: Fn(&LlmRequest) -> Result<Option<String>> + Send + Sync + 'static,
    {
        self.push_registration(
            name,
            RegistrationSurface::LlmConditionalExecutionGuardrail,
            priority,
            false,
        );
        self.handlers
            .llm_conditionals
            .insert(name.into(), Arc::new(callback));
    }

    /// Registers an LLM request intercept.
    pub fn register_llm_request_intercept<F>(
        &mut self,
        name: &str,
        priority: i32,
        break_chain: bool,
        callback: F,
    ) where
        F: Fn(&str, LlmRequest, Option<AnnotatedLlmRequest>) -> Result<LlmRequestInterceptOutcome>
            + Send
            + Sync
            + 'static,
    {
        self.push_registration(
            name,
            RegistrationSurface::LlmRequestIntercept,
            priority,
            break_chain,
        );
        self.handlers
            .llm_requests
            .insert(name.into(), Arc::new(callback));
    }

    /// Registers an LLM execution intercept.
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
}

impl PluginRuntime {
    /// Emits a mark event through the host runtime.
    pub async fn emit_mark(
        &self,
        name: &str,
        data: Option<Json>,
        metadata: Option<Json>,
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
            }))
            .await
            .map_err(|err| WorkerSdkError::Transport(err.to_string()))?
            .into_inner();
        ack_to_result(response.ok, response.error)
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
    pub async fn call(&self, value: Json) -> Result<Json> {
        let mut client = self.runtime.host_client().await?;
        let response = client
            .tool_next(Request::new(ToolNextRequest {
                activation_id: self.runtime.activation_id.clone(),
                auth_token: self.runtime.auth_token.clone(),
                continuation_id: self.continuation_id.clone(),
                value: Some(json_envelope("nemo.relay.Json@1", &value)?),
            }))
            .await
            .map_err(|err| WorkerSdkError::Transport(err.to_string()))?
            .into_inner();
        json_result_to_sdk(response)
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
    pub async fn call(&self, request: LlmRequest) -> Result<Json> {
        let mut client = self.runtime.host_client().await?;
        let response = client
            .llm_next(Request::new(LlmNextRequest {
                activation_id: self.runtime.activation_id.clone(),
                auth_token: self.runtime.auth_token.clone(),
                continuation_id: self.continuation_id.clone(),
                request: Some(json_envelope("nemo.relay.LlmRequest@1", &request)?),
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
    pub async fn call(&self, request: LlmRequest) -> Result<JsonStream> {
        let scope = self.runtime.current_scope_context();
        let mut client = self.runtime.host_client().await?;
        let response = client
            .llm_stream_next(Request::new(LlmStreamNextRequest {
                activation_id: self.runtime.activation_id.clone(),
                auth_token: self.runtime.auth_token.clone(),
                continuation_id: self.continuation_id.clone(),
                request: Some(json_envelope("nemo.relay.LlmRequest@1", &request)?),
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
            }));
        }
        if let Err(err) = validate_unique_registrations(&ctx.handlers.registrations) {
            return Ok(Response::new(RegisterResponse {
                registrations: Vec::new(),
                error: Some(sdk_error_to_worker(err)),
            }));
        }
        let registrations = ctx.handlers.registrations.clone();
        *self
            .handlers
            .lock()
            .map_err(|err| Status::internal(format!("handler lock poisoned: {err}")))? =
            ctx.handlers;
        Ok(Response::new(RegisterResponse {
            registrations,
            error: None,
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
                            match json_envelope("nemo.relay.Json@1", &value) {
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
            RegistrationSurface::Subscriber => {
                let event = event_payload(request.payload)?;
                let handler = self.subscriber(&request.registration_name)?;
                with_thread_scope(&scope, || handler(&event));
                Ok(empty_response())
            }
            RegistrationSurface::MarkSanitizeGuardrail
            | RegistrationSurface::ScopeSanitizeStartGuardrail
            | RegistrationSurface::ScopeSanitizeEndGuardrail => {
                let event = event_payload(request.payload)?;
                let fields = event.sanitize_fields();
                let handler = self.event_sanitizer(surface, &request.registration_name)?;
                let fields = with_thread_scope(&scope, || handler(&event, fields));
                Ok(json_response(
                    serde_json::to_value(fields)
                        .expect("event sanitize fields are JSON serializable"),
                ))
            }
            RegistrationSurface::ToolSanitizeRequestGuardrail => {
                let payload = tool_payload(request.payload)?;
                let handler = self.tool_sanitize_request(&request.registration_name)?;
                Ok(json_response(with_thread_scope(&scope, || {
                    handler(&payload.tool_name, payload.value)
                })))
            }
            RegistrationSurface::ToolSanitizeResponseGuardrail => {
                let payload = tool_payload(request.payload)?;
                let handler = self.tool_sanitize_response(&request.registration_name)?;
                Ok(json_response(with_thread_scope(&scope, || {
                    handler(&payload.tool_name, payload.value)
                })))
            }
            RegistrationSurface::ToolConditionalExecutionGuardrail => {
                let payload = tool_payload(request.payload)?;
                let handler = self.tool_conditional(&request.registration_name)?;
                Ok(guardrail_response(with_thread_scope(&scope, || {
                    handler(&payload.tool_name, &payload.value)
                })?))
            }
            RegistrationSurface::ToolRequestIntercept => {
                let payload = tool_payload(request.payload)?;
                let handler = self.tool_request(&request.registration_name)?;
                Ok(json_response(with_thread_scope(&scope, || {
                    handler(&payload.tool_name, payload.value)
                })?))
            }
            RegistrationSurface::ToolExecutionIntercept => {
                let payload = tool_payload(request.payload)?;
                let handler = self.tool_execution(&request.registration_name)?;
                let next = ToolNext {
                    runtime: self.runtime.clone(),
                    continuation_id: request.continuation_id,
                };
                let future =
                    with_thread_scope(&scope, || handler(&payload.tool_name, payload.value, next));
                Ok(tool_execution_response(future.await?)?)
            }
            RegistrationSurface::LlmSanitizeRequestGuardrail => {
                let payload = llm_payload(request.payload)?;
                let request_value = required_json::<LlmRequest>(payload.request, "llm request")?;
                let handler = self.llm_sanitize_request(&request.registration_name)?;
                let request = with_thread_scope(&scope, || handler(request_value));
                Ok(json_response(
                    serde_json::to_value(request).expect("LLM request is JSON serializable"),
                ))
            }
            RegistrationSurface::LlmSanitizeResponseGuardrail => {
                let payload = llm_payload(request.payload)?;
                let response = required_json::<Json>(payload.response, "llm response")?;
                let handler = self.llm_sanitize_response(&request.registration_name)?;
                Ok(json_response(with_thread_scope(&scope, || {
                    handler(response)
                })))
            }
            RegistrationSurface::LlmConditionalExecutionGuardrail => {
                let payload = llm_payload(request.payload)?;
                let request_value = required_json::<LlmRequest>(payload.request, "llm request")?;
                let handler = self.llm_conditional(&request.registration_name)?;
                Ok(guardrail_response(with_thread_scope(&scope, || {
                    handler(&request_value)
                })?))
            }
            RegistrationSurface::LlmRequestIntercept => {
                let payload = llm_payload(request.payload)?;
                let request_value = required_json::<LlmRequest>(payload.request, "llm request")?;
                let annotated = payload
                    .annotated_request
                    .map(|value| decode_json_envelope::<AnnotatedLlmRequest>(&value))
                    .transpose()?;
                let handler = self.llm_request(&request.registration_name)?;
                let outcome = with_thread_scope(&scope, || {
                    handler(&payload.model_name, request_value, annotated)
                })?;
                Ok(llm_request_response(outcome)?)
            }
            RegistrationSurface::LlmExecutionIntercept => {
                let payload = llm_payload(request.payload)?;
                let request_value = required_json::<LlmRequest>(payload.request, "llm request")?;
                let handler = self.llm_execution(&request.registration_name)?;
                let next = LlmNext {
                    runtime: self.runtime.clone(),
                    continuation_id: request.continuation_id,
                };
                let future =
                    with_thread_scope(&scope, || handler(&payload.model_name, request_value, next));
                Ok(json_response(future.await?))
            }
            RegistrationSurface::LlmStreamExecutionIntercept | RegistrationSurface::Unspecified => {
                Err(WorkerSdkError::InvalidInput(
                    "surface must use InvokeStream or is unspecified".into(),
                ))
            }
        }
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
                value: Some(infallible_json_envelope("nemo.relay.Json@1", &value)),
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
                        "nemo.relay.LlmRequestInterceptOutcome@1",
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
                    outcome: Some(json_envelope(
                        "nemo.relay.ToolExecutionInterceptOutcome@1",
                        &outcome,
                    )?),
                },
            ),
        ),
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

fn optional_json_envelope(value: Option<Json>) -> Result<Option<JsonEnvelope>> {
    value
        .as_ref()
        .map(|value| json_envelope("nemo.relay.Json@1", value).map_err(WorkerSdkError::from))
        .transpose()
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
