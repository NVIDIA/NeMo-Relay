// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frozen native API v1.0. Add capabilities in new modules, never by changing
//! these records, signatures, marker traits, or enum discriminants.

use crate::Json;
use crate::future::FfiFuture;
pub use crate::handles::*;

/// Boundary error, interpreted by the adapters in its originating library.
#[repr(C)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    /// Stable operation/classification code; unknown codes remain errors.
    pub code: String,
    /// Human-readable diagnostic detail.
    pub message: String,
    /// Optional upstream HTTP error status.
    pub http_status: Option<u16>,
}
impl Error {
    /// Construct an error without an HTTP status.
    pub fn new(code: &str, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            http_status: None,
        }
    }
}
/// Local Rust result, converted to an owned wire result by the adapters.
pub type Result<T> = std::result::Result<T, Error>;

/// Required middleware/subscriber surfaces in the baseline.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Surface {
    /// Synchronous event subscriber.
    Subscriber,
    /// Event metadata injection.
    EventMetadataInjector,
    /// Mark field sanitization.
    MarkSanitize,
    /// Scope-start field sanitization.
    ScopeSanitizeStart,
    /// Scope-end field sanitization.
    ScopeSanitizeEnd,
    /// Tool input sanitization.
    ToolSanitizeRequest,
    /// Tool output sanitization.
    ToolSanitizeResponse,
    /// Tool execution admission.
    ToolConditionalExecution,
    /// Tool request interception.
    ToolRequestIntercept,
    /// Tool execution interception.
    ToolExecutionIntercept,
    /// LLM input sanitization.
    LlmSanitizeRequest,
    /// LLM output sanitization.
    LlmSanitizeResponse,
    /// LLM execution admission.
    LlmConditionalExecution,
    /// LLM request interception.
    LlmRequestIntercept,
    /// LLM execution interception.
    LlmExecutionIntercept,
    /// Streaming LLM interception.
    LlmStreamExecutionIntercept,
    /// Conditional middleware gate.
    ConditionalMiddleware,
}

/// Host-owned canonical tool result, independent of its application payload.
#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct ToolExecutionResult {
    /// Application-owned result.
    pub result: Json,
    /// Optional opaque annotation.
    pub annotation: Option<Json>,
}

/// Canonical intercept result including deferred lifecycle marks.
#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct ToolExecutionOutcome {
    /// Application-visible result and annotation.
    pub result: ToolExecutionResult,
    /// Ordered mark specifications, retained by the lifecycle owner.
    pub pending_marks: Vec<Json>,
}

/// Configuration diagnostic; severity uses the frozen warning/error vocabulary.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct ConfigDiagnostic {
    /// Whether the diagnostic prevents activation.
    pub is_error: bool,
    /// Machine-readable code.
    pub code: String,
    /// Optional component identifier.
    pub component: Option<String>,
    /// Optional configuration field.
    pub field: Option<String>,
    /// Human-readable diagnostic detail.
    pub message: String,
}

/// One invocation's closed metadata and open application payload.
#[repr(C)]
pub struct Invocation {
    /// Registered callback surface.
    pub surface: Surface,
    /// Tool, model, or registration name as appropriate for the surface.
    pub name: String,
    /// Lossless payload, without JSON text encoding.
    pub value: Json,
    /// Provider-issued tool-call correlation identifier.
    pub tool_call_id: Option<String>,
    /// Invocation-scoped host capabilities.
    pub context: Context,
}

/// Runtime capabilities are scoped to an invocation or component, not a library.
#[repr(C)]
pub struct Context {
    /// Retained host scope binding. Restoration is required on every poll.
    pub scope: ScopeBox,
    /// Optional request codec capability.
    pub request_codec: Option<CodecBox>,
    /// Optional response codec capability.
    pub response_codec: Option<CodecBox>,
}

/// Captured scope ownership; no host runtime structures cross the boundary.
pub trait Scope: Send + Sync {
    /// Bind this scope on the calling thread and return an RAII restoration lease.
    fn enter(&self) -> Result<ScopeLeaseBox>;
    /// Inspect the effective stack, including the absence of an active stack.
    fn snapshot(&self) -> Result<Option<Json>>;
}

/// Dropping the lease restores its originating library's prior thread binding.
pub trait ScopeLease {
    /// Explicit idempotent restoration before drop.
    fn restore(&mut self) -> Result<()>;
}

/// Revocable request or response codec capability.
pub trait Codec: Send + Sync {
    /// Stable codec identity, including none/runtime/opaque classifications.
    fn identity(&self) -> Result<Json>;
    /// Decode an application payload into its annotated representation.
    fn decode(&self, value: Json) -> Result<Json>;
    /// Encode changes while retaining the original application representation.
    fn encode(&self, annotated: Json, original: Json) -> Result<Json>;
}

/// Repeated or concurrent downstream execution.
pub trait Continuation: Send + Sync {
    /// Invoke a unary downstream operation.
    fn call(&self, value: Json) -> FfiFuture<Result<Json>>;
    /// Invoke a canonical tool continuation without an untyped result envelope.
    fn call_tool(&self, value: Json) -> FfiFuture<Result<ToolExecutionResult>>;
    /// Open a downstream stream.
    fn open_stream(&self, value: Json) -> FfiFuture<Result<StreamBox>>;
}

/// Pull-based stream. `None` is terminal and errors remain distinguishable.
pub trait Stream: Send + Sync {
    /// Pull one item; dropping the returned future cancels that pull.
    fn pull(&mut self) -> FfiFuture<Result<Option<Json>>>;
    /// Cancel and await tracked stream cleanup; repeated close is harmless.
    fn close(&mut self) -> FfiFuture<Result<()>>;
}

/// A plugin-owned callback. Adapters enforce the declared registration surface.
pub trait Callback: Send + Sync {
    /// Invoke unary middleware on the callback's owning executor.
    fn invoke(
        &self,
        invocation: Invocation,
        next: Option<ContinuationBox>,
    ) -> FfiFuture<Result<Json>>;
    /// Invoke tool middleware with a canonical typed result and deferred marks.
    fn invoke_tool(
        &self,
        invocation: Invocation,
        next: ContinuationBox,
    ) -> FfiFuture<Result<ToolExecutionOutcome>>;
    /// Invoke a synchronous subscriber or raw synchronous callback.
    fn invoke_sync(&self, invocation: Invocation) -> Result<Json>;
    /// Invoke streaming middleware on its owning executor.
    fn open_stream(
        &self,
        invocation: Invocation,
        next: ContinuationBox,
    ) -> FfiFuture<Result<StreamBox>>;
}

/// Declarative component-owned registration, applied atomically by the host.
#[repr(C)]
pub struct Registration {
    /// Component-local registration name.
    pub name: String,
    /// Surface implemented by the callback.
    pub surface: Surface,
    /// Middleware ordering priority.
    pub priority: i32,
    /// Whether request interception terminates its chain.
    pub break_chain: bool,
    /// Conditional gate target kinds and name, otherwise null.
    pub options: Json,
    /// Callback owner, released during component teardown.
    pub callback: CallbackBox,
}

/// Explicit component lifecycle, independent of resident library code.
pub trait Component: Send + Sync {
    /// Stop admission, cancel, drain, and release the component executor.
    fn shutdown(&self) -> FfiFuture<Result<()>>;
}

/// Registration result; failed host application must shut down this owner.
#[repr(C)]
pub struct RegisteredComponent {
    /// Author registration failure. Ownership is still returned for ordered,
    /// retryable rollback; pre-construction failures use the outer result error.
    pub error: Option<Error>,
    /// Registrations to install in the existing host transaction.
    pub registrations: Vec<Registration>,
    /// SDK executor and asynchronous cleanup owner.
    pub component: ComponentBox,
}

/// Per-component host authority. Operation names are frozen in the SDK modules.
pub trait Runtime: Send + Sync {
    /// Synchronous scope, telemetry, registry, codec, and diagnostic operations.
    fn call_sync(&self, operation: String, arguments: Json) -> Result<Json>;
    /// Operations requiring host-owned async execution.
    fn call(&self, operation: String, arguments: Json) -> FfiFuture<Result<Json>>;
    /// Bounded diagnostic authoring and separately authorized whole-context reads.
    fn diagnostics(&self) -> crate::diagnostics::DiagnosticsBox;
    /// Capture actual host scope/event/publication ownership on the calling thread.
    fn capture_scope(&self) -> Result<ScopeBox>;
    /// Transfer cleanup ownership before starting registration work. Exactly one
    /// owner may be bound during activation; retained capabilities cannot rebind.
    fn bind_component(&self, component: ComponentBox) -> Result<()>;
    /// Apply an activation-time callback through the current host transaction.
    fn register_callback(&self, registration: Registration) -> Result<()>;
    /// Register a component-owned dynamic conditional gate; other surfaces fail.
    fn register_gate(&self, registration: Registration) -> Result<String>;
}

/// Plugin authoring adapter, constructed only after checking metadata/modules.
pub trait Plugin: Send + Sync {
    /// Whether the plugin permits multiple configured components.
    fn allows_multiple_components(&self) -> bool;
    /// Validate one component without installing callbacks.
    fn validate(&self, config: Json) -> Vec<ConfigDiagnostic>;
    /// Register one component on its owning executor.
    fn register(&self, config: Json, runtime: RuntimeBox)
    -> FfiFuture<Result<RegisteredComponent>>;
}

/// Exact module name used by checked native discovery.
pub const FACTORY_MODULE: &str = "nemo.relay.native.factory.v1.0";

/// Construct a fresh definition after the host verifies the complete contract.
pub trait Factory: Send + Sync {
    /// Create one plugin definition without admitting component work.
    fn create(&self) -> Result<PluginBox>;
}
