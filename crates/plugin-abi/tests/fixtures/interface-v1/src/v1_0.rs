// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frozen native API v1.0. Add capabilities in new modules, never by changing
//! these records, signatures, marker traits, or enum discriminants.

use crate::Json;
use abi_stable::{StableAbi, library::RootModule, sabi_trait, sabi_types::VersionStrings, std_types::{RBox, ROption, RResult, RString, RVec}};
use async_ffi::FfiFuture;

/// Boundary error, interpreted by the adapters in its originating library.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, PartialEq, Eq)]
pub struct Error {
    /// Stable operation/classification code; unknown codes remain errors.
    pub code: RString,
    /// Human-readable diagnostic detail.
    pub message: RString,
    /// Optional upstream HTTP error status.
    pub http_status: ROption<u16>,
}
impl Error {
    /// Construct an error without an HTTP status.
    pub fn new(code: &str, message: impl Into<RString>) -> Self {
        Self {code: code.into(), message: message.into(), http_status: ROption::RNone}
    }
}
/// ABI-safe result, without thread-local last-error state.
pub type Result<T> = RResult<T, Error>;

/// Required middleware/subscriber surfaces in the baseline.
#[repr(u8)]
#[derive(StableAbi, Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(StableAbi, Debug, Clone, PartialEq)]
pub struct ToolExecutionResult {
    /// Application-owned result.
    pub result: Json,
    /// Optional opaque annotation.
    pub annotation: ROption<Json>,
}

/// Canonical intercept result including deferred lifecycle marks.
#[repr(C)]
#[derive(StableAbi, Debug, Clone, PartialEq)]
pub struct ToolExecutionOutcome {
    /// Application-visible result and annotation.
    pub result: ToolExecutionResult,
    /// Ordered mark specifications, retained by the lifecycle owner.
    pub pending_marks: RVec<Json>,
}

/// Configuration diagnostic; severity uses the frozen warning/error vocabulary.
#[repr(C)]
#[derive(StableAbi, Debug, Clone)]
pub struct ConfigDiagnostic {
    /// Whether the diagnostic prevents activation.
    pub is_error: bool,
    /// Machine-readable code.
    pub code: RString,
    /// Optional component identifier.
    pub component: ROption<RString>,
    /// Optional configuration field.
    pub field: ROption<RString>,
    /// Human-readable diagnostic detail.
    pub message: RString,
}

/// One invocation's closed metadata and open application payload.
#[repr(C)]
#[derive(StableAbi)]
pub struct Invocation {
    /// Registered callback surface.
    pub surface: Surface,
    /// Tool, model, or registration name as appropriate for the surface.
    pub name: RString,
    /// Lossless payload, without JSON text encoding.
    pub value: Json,
    /// Provider-issued tool-call correlation identifier.
    pub tool_call_id: ROption<RString>,
    /// Invocation-scoped host capabilities.
    pub context: Context,
}

/// Runtime capabilities are scoped to an invocation or component, not a library.
#[repr(C)]
#[derive(StableAbi)]
pub struct Context {
    /// Retained host scope binding. Restoration is required on every poll.
    pub scope: ScopeBox,
    /// Optional request codec capability.
    pub request_codec: ROption<CodecBox>,
    /// Optional response codec capability.
    pub response_codec: ROption<CodecBox>,
}

/// Captured scope ownership; no host runtime structures cross the boundary.
#[sabi_trait]
pub trait Scope: Send + Sync {
    /// Bind this scope on the calling thread and return an RAII restoration lease.
    fn enter(&self) -> Result<ScopeLeaseBox>;
    /// Inspect the effective stack, including the absence of an active stack.
    #[sabi(last_prefix_field)]
    fn snapshot(&self) -> Result<ROption<Json>>;
}
/// Owned scope capability.
pub type ScopeBox = Scope_TO<'static, RBox<()>>;

/// Dropping the lease restores its originating library's prior thread binding.
#[sabi_trait]
pub trait ScopeLease {
    /// Explicit idempotent restoration before drop.
    #[sabi(last_prefix_field)]
    fn restore(&mut self) -> Result<()>;
}
/// Owned binding restoration lease; never move an entered lease across threads.
pub type ScopeLeaseBox = ScopeLease_TO<'static, RBox<()>>;

/// Revocable request or response codec capability.
#[sabi_trait]
pub trait Codec: Send + Sync {
    /// Stable codec identity, including none/runtime/opaque classifications.
    fn identity(&self) -> Result<Json>;
    /// Decode an application payload into its annotated representation.
    fn decode(&self, value: Json) -> Result<Json>;
    /// Encode changes while retaining the original application representation.
    #[sabi(last_prefix_field)]
    fn encode(&self, annotated: Json, original: Json) -> Result<Json>;
}
/// Owned codec capability.
pub type CodecBox = Codec_TO<'static, RBox<()>>;

/// Repeated or concurrent downstream execution.
#[sabi_trait]
pub trait Continuation: Send + Sync {
    /// Invoke a unary downstream operation.
    fn call(&self, value: Json) -> FfiFuture<Result<Json>>;
    /// Invoke a canonical tool continuation without an untyped result envelope.
    fn call_tool(&self, value: Json) -> FfiFuture<Result<ToolExecutionResult>>;
    /// Open a downstream stream.
    #[sabi(last_prefix_field)]
    fn open_stream(&self, value: Json) -> FfiFuture<Result<StreamBox>>;
}
/// Owned downstream continuation.
pub type ContinuationBox = Continuation_TO<'static, RBox<()>>;

/// Pull-based stream. `None` is terminal and errors remain distinguishable.
#[sabi_trait]
pub trait Stream: Send {
    /// Pull one item; dropping the returned future cancels that pull.
    fn pull(&mut self) -> FfiFuture<Result<ROption<Json>>>;
    /// Cancel and await tracked stream cleanup; repeated close is harmless.
    #[sabi(last_prefix_field)]
    fn close(&mut self) -> FfiFuture<Result<()>>;
}
/// Owned stream state.
pub type StreamBox = Stream_TO<'static, RBox<()>>;

/// A plugin-owned callback. Adapters enforce the declared registration surface.
#[sabi_trait]
pub trait Callback: Send + Sync {
    /// Invoke unary middleware on the callback's owning executor.
    fn invoke(&self, invocation: Invocation, next: ROption<ContinuationBox>) -> FfiFuture<Result<Json>>;
    /// Invoke tool middleware with a canonical typed result and deferred marks.
    fn invoke_tool(&self, invocation: Invocation, next: ContinuationBox) -> FfiFuture<Result<ToolExecutionOutcome>>;
    /// Invoke a synchronous subscriber or raw synchronous callback.
    fn invoke_sync(&self, invocation: Invocation) -> Result<Json>;
    /// Invoke streaming middleware on its owning executor.
    #[sabi(last_prefix_field)]
    fn open_stream(&self, invocation: Invocation, next: ContinuationBox) -> FfiFuture<Result<StreamBox>>;
}
/// Owned plugin callback.
pub type CallbackBox = Callback_TO<'static, RBox<()>>;

/// Declarative component-owned registration, applied atomically by the host.
#[repr(C)]
#[derive(StableAbi)]
pub struct Registration {
    /// Component-local registration name.
    pub name: RString,
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
#[sabi_trait]
pub trait Component: Send + Sync {
    /// Stop admission, cancel, drain, and release the component executor.
    #[sabi(last_prefix_field)]
    fn shutdown(&self) -> FfiFuture<Result<()>>;
}
/// Owned component lifecycle capability.
pub type ComponentBox = Component_TO<'static, RBox<()>>;

/// Registration result; failed host application must shut down this owner.
#[repr(C)]
#[derive(StableAbi)]
pub struct RegisteredComponent {
    /// Author registration failure. Ownership is still returned for ordered,
    /// retryable rollback; pre-construction failures use the outer result error.
    pub error: ROption<Error>,
    /// Registrations to install in the existing host transaction.
    pub registrations: RVec<Registration>,
    /// SDK executor and asynchronous cleanup owner.
    pub component: ComponentBox,
}

/// Per-component host authority. Operation names are frozen in the SDK modules.
#[sabi_trait]
pub trait Runtime: Send + Sync {
    /// Synchronous scope, telemetry, registry, codec, and diagnostic operations.
    fn call_sync(&self, operation: RString, arguments: Json) -> Result<Json>;
    /// Operations requiring host-owned async execution.
    fn call(&self, operation: RString, arguments: Json) -> FfiFuture<Result<Json>>;
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
    #[sabi(last_prefix_field)]
    fn register_gate(&self, registration: Registration) -> Result<RString>;
}
/// Owned, revocable component runtime authority.
pub type RuntimeBox = Runtime_TO<'static, RBox<()>>;

/// Plugin authoring adapter, constructed only after checking metadata/modules.
#[sabi_trait]
pub trait Plugin: Send + Sync {
    /// Whether the plugin permits multiple configured components.
    fn allows_multiple_components(&self) -> bool;
    /// Validate one component without installing callbacks.
    fn validate(&self, config: Json) -> RVec<ConfigDiagnostic>;
    /// Register one component on its owning executor.
    #[sabi(last_prefix_field)]
    fn register(&self, config: Json, runtime: RuntimeBox) -> FfiFuture<Result<RegisteredComponent>>;
}
/// Owned plugin definition adapter.
pub type PluginBox = Plugin_TO<'static, RBox<()>>;

/// Checked factory module for the entire v1.0 capability set.
#[repr(C)]
#[derive(StableAbi)]
#[sabi(kind(Prefix(prefix_ref = Factory_Ref)))]
pub struct Factory {
    /// Construct an adapter without starting a component executor.
    #[sabi(last_prefix_field)]
    pub create: extern "C" fn() -> Result<PluginBox>,
}
impl RootModule for Factory_Ref {
    abi_stable::declare_root_module_statics! {Factory_Ref}
    const BASE_NAME: &'static str = "nemo_relay_plugin";
    const NAME: &'static str = "nemo.relay.native.factory.v1.0";
    const VERSION_STRINGS: VersionStrings = VersionStrings::new("1.0.0");
}
/// Exact module name used by the frozen bootstrap query.
pub const FACTORY_MODULE: &str = "nemo.relay.native.factory.v1.0";
