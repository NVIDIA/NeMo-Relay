// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![deny(rustdoc::broken_intra_doc_links, rustdoc::private_intra_doc_links)]

//! Stable native plugin ABI and Rust authoring helpers for NeMo Relay.
//!
//! This crate intentionally does not depend on the `nemo-relay` runtime crate.
//! Native plugins built with it communicate with a host through versioned
//! C-compatible tables and host-owned string handles.

mod async_sdk;

pub use async_sdk::{LlmJsonAsyncStream, LlmNext, LlmStreamNext, NativeExecutorConfig, ToolNext};

use std::ffi::{c_char, c_void};
use std::marker::{PhantomData, PhantomPinned};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;
use std::sync::{Arc, Mutex};

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
pub use nemo_relay_types::plugin::{ConfigDiagnostic, DiagnosticLevel};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Map;

/// Native plugin ABI version supported by this crate.
///
/// Version 4 adds completion-scoped codecs, pull-based LLM streams, extended
/// mark emission, runtime diagnostics, and activation-owned runtime-registration
/// discovery and dynamic conditional middleware guardrail control. Hosts retain
/// frozen version-3 and version-2 tables for already-built plugins that target
/// those layouts.
pub const NEMO_RELAY_NATIVE_ABI_VERSION: u32 = 4;
/// ABI version that introduced completion-based asynchronous middleware.
pub const NEMO_RELAY_NATIVE_ABI_VERSION_ASYNC_MIDDLEWARE: u32 = 3;

/// Legacy native plugin ABI accepted by Relay hosts for compatibility.
pub const NEMO_RELAY_NATIVE_ABI_VERSION_LEGACY: u32 = 2;

/// Per-call request codec context delivered to an LLM sanitizer.
pub struct LlmSanitizeRequestContext<'a> {
    /// Identity of the active codec.
    pub codec: LlmCodecIdentity,
    resolved: Option<LlmSanitizeRequestCodec<'a>>,
}
// SAFETY: this context is constructed only by the async SDK from a retained
// completion capability; callback-scoped native contexts never construct it.
unsafe impl Send for LlmSanitizeRequestContext<'_> {}

/// Per-call response codec context delivered to an LLM sanitizer.
pub struct LlmSanitizeResponseContext<'a> {
    /// Identity of the active codec.
    pub codec: LlmCodecIdentity,
    resolved: Option<LlmSanitizeResponseCodec<'a>>,
}
// SAFETY: this context is constructed only by the async SDK from a retained
// completion capability; callback-scoped native contexts never construct it.
unsafe impl Send for LlmSanitizeResponseContext<'_> {}

/// Status codes returned by stable native ABI functions.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NemoRelayStatus {
    /// Operation completed successfully.
    Ok = 0,
    /// A resource with the given name already exists.
    AlreadyExists = 1,
    /// The requested resource was not found.
    NotFound = 2,
    /// The scope stack is empty.
    ScopeStackEmpty = 3,
    /// A guardrail rejected the operation.
    GuardrailRejected = 4,
    /// An internal runtime error occurred.
    Internal = 5,
    /// A required pointer argument was null.
    NullPointer = 6,
    /// A JSON string argument could not be parsed.
    InvalidJson = 7,
    /// A string argument contained invalid UTF-8.
    InvalidUtf8 = 8,
    /// A function argument had an invalid value.
    InvalidArg = 9,
    /// A stream reached end-of-stream and has no chunk to return.
    StreamEnd = 10,
    /// A bounded stream queue is full; retry this operation after it advances.
    Backpressured = 11,
}

/// Opaque host-owned UTF-8 string or JSON byte buffer.
#[repr(C)]
pub struct NemoRelayNativeString {
    _private: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}

/// Opaque callback-scoped request codec capability owned by the host.
#[repr(C)]
pub struct NemoRelayNativeLlmRequestCodec {
    _private: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}

/// Opaque callback-scoped response codec capability owned by the host.
#[repr(C)]
pub struct NemoRelayNativeLlmResponseCodec {
    _private: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}

/// Discriminator for the codec supplied to an LLM sanitizer over the native ABI.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NemoRelayNativeLlmCodecKind {
    /// No codec was active for this call.
    None = 0,
    /// A Relay built-in codec was active.
    BuiltIn = 1,
    /// A runtime-registered codec was active.
    Runtime = 2,
    /// A codec was active but has no registered identity.
    Opaque = 3,
}

/// Per-call LLM sanitizer context passed over the native ABI.
///
/// `codec_id` is borrowed for the duration of the callback. It is null for
/// [`NemoRelayNativeLlmCodecKind::None`] and
/// [`NemoRelayNativeLlmCodecKind::Opaque`]. For `BuiltIn`, it is one of the
/// stable built-in codec IDs; for `Runtime`, it is the registered codec ID.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NemoRelayNativeLlmSanitizeRequestContext {
    /// Discriminator for the active codec.
    pub codec_kind: NemoRelayNativeLlmCodecKind,
    /// Optional borrowed codec identifier.
    pub codec_id: *const NemoRelayNativeString,
    /// Borrowed request codec capability, or null when no codec is active.
    pub codec: *const NemoRelayNativeLlmRequestCodec,
}

/// Per-call response sanitizer context passed over the native ABI.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NemoRelayNativeLlmSanitizeResponseContext {
    /// Discriminator for the active codec.
    pub codec_kind: NemoRelayNativeLlmCodecKind,
    /// Optional borrowed codec identifier.
    pub codec_id: *const NemoRelayNativeString,
    /// Borrowed response codec capability, or null when no codec is active.
    pub codec: *const NemoRelayNativeLlmResponseCodec,
}

/// Safe completion-backed request codec facade for typed native plugins.
pub struct LlmSanitizeRequestCodec<'a> {
    async_host: NemoRelayNativeHostApiV4,
    completion: *const NemoRelayNativeAsyncCompletion,
    completion_release: unsafe extern "C" fn(*const NemoRelayNativeAsyncCompletion),
    _lifetime: PhantomData<&'a NemoRelayNativeLlmRequestCodec>,
}
// SAFETY: this type has no borrowed-handle construction path. The async SDK
// retains the completion capability until this facade is dropped.
unsafe impl Send for LlmSanitizeRequestCodec<'_> {}
// SAFETY: calls use the immutable host API and retained completion capability,
// which the host permits from concurrent plugin tasks.
unsafe impl Sync for LlmSanitizeRequestCodec<'_> {}

impl Drop for LlmSanitizeRequestCodec<'_> {
    fn drop(&mut self) {
        unsafe { (self.completion_release)(self.completion) };
    }
}

impl LlmSanitizeRequestCodec<'_> {
    /// Decode an opaque request into Relay's normalized request model.
    pub fn decode(&self, request: &LlmRequest) -> Result<AnnotatedLlmRequest> {
        native_codec_call(&self.async_host.v3.v1, |out| unsafe {
            let request = HostString::from_json(&self.async_host.v3.v1, request)
                .ok_or_else(|| "failed to serialize LLM request".to_string())?;
            let status = (self.async_host.async_completion_llm_request_codec_decode)(
                self.completion,
                request.as_ptr(),
                out,
            );
            codec_status(&self.async_host.v3.v1, status)
        })
    }

    /// Encode normalized changes onto the original opaque request.
    pub fn encode(
        &self,
        annotated: &AnnotatedLlmRequest,
        original: &LlmRequest,
    ) -> Result<LlmRequest> {
        native_codec_call(&self.async_host.v3.v1, |out| unsafe {
            let annotated = HostString::from_json(&self.async_host.v3.v1, annotated)
                .ok_or_else(|| "failed to serialize annotated request".to_string())?;
            let original = HostString::from_json(&self.async_host.v3.v1, original)
                .ok_or_else(|| "failed to serialize original request".to_string())?;
            let status = (self.async_host.async_completion_llm_request_codec_encode)(
                self.completion,
                annotated.as_ptr(),
                original.as_ptr(),
                out,
            );
            codec_status(&self.async_host.v3.v1, status)
        })
    }
}

/// Safe completion-backed response codec facade for typed native plugins.
pub struct LlmSanitizeResponseCodec<'a> {
    async_host: NemoRelayNativeHostApiV4,
    completion: *const NemoRelayNativeAsyncCompletion,
    completion_release: unsafe extern "C" fn(*const NemoRelayNativeAsyncCompletion),
    _lifetime: PhantomData<&'a NemoRelayNativeLlmResponseCodec>,
}
// SAFETY: this type has no borrowed-handle construction path. The async SDK
// retains the completion capability until this facade is dropped.
unsafe impl Send for LlmSanitizeResponseCodec<'_> {}
// SAFETY: calls use the immutable host API and the retained completion
// capability, which the host permits from concurrent plugin tasks.
unsafe impl Sync for LlmSanitizeResponseCodec<'_> {}

impl Drop for LlmSanitizeResponseCodec<'_> {
    fn drop(&mut self) {
        unsafe { (self.completion_release)(self.completion) };
    }
}

impl LlmSanitizeResponseCodec<'_> {
    /// Decode an opaque response into Relay's normalized response model.
    pub fn decode(&self, response: &Json) -> Result<AnnotatedLlmResponse> {
        native_codec_call(&self.async_host.v3.v1, |out| unsafe {
            let response = HostString::from_json(&self.async_host.v3.v1, response)
                .ok_or_else(|| "failed to serialize LLM response".to_string())?;
            let status = (self.async_host.async_completion_llm_response_codec_decode)(
                self.completion,
                response.as_ptr(),
                out,
            );
            codec_status(&self.async_host.v3.v1, status)
        })
    }
}

impl<'a> LlmSanitizeRequestContext<'a> {
    /// Resolve the active request codec capability.
    #[must_use]
    pub fn resolve_codec(&self) -> Option<&LlmSanitizeRequestCodec<'a>> {
        self.resolved.as_ref()
    }
}

impl<'a> LlmSanitizeResponseContext<'a> {
    /// Resolve the active response codec capability.
    #[must_use]
    pub fn resolve_codec(&self) -> Option<&LlmSanitizeResponseCodec<'a>> {
        self.resolved.as_ref()
    }
}

/// Opaque plugin registration context borrowed from the host during registration.
#[repr(C)]
pub struct NemoRelayNativePluginContext {
    _private: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}

/// Opaque activation-owned runtime capability used by native plugins.
#[repr(C)]
pub struct NemoRelayNativePluginRuntime {
    _private: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}

/// Opaque host-owned scope handle.
#[repr(C)]
pub struct NemoRelayNativeScopeHandle {
    _private: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}

/// Opaque host-owned scope stack handle.
#[repr(C)]
pub struct NemoRelayNativeScopeStack {
    _private: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}

/// Opaque host-owned captured scope-stack binding.
#[repr(C)]
pub struct NemoRelayNativeScopeStackBinding {
    _private: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}

/// Scope category used by native plugins when opening scopes.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NemoRelayNativeScopeType {
    /// Top-level agent scope.
    Agent = 0,
    /// Generic function scope.
    Function = 1,
    /// Tool invocation scope.
    Tool = 2,
    /// LLM call scope.
    Llm = 3,
    /// Retriever scope.
    Retriever = 4,
    /// Embedder scope.
    Embedder = 5,
    /// Reranker scope.
    Reranker = 6,
    /// Guardrail evaluation scope.
    Guardrail = 7,
    /// Evaluator scope.
    Evaluator = 8,
    /// User-defined custom scope.
    Custom = 9,
    /// Unknown or unspecified scope type.
    Unknown = 10,
}

/// Optional destructor for user data captured by native callbacks.
pub type NemoRelayNativeFreeFn = Option<unsafe extern "C" fn(user_data: *mut c_void)>;

/// Native callback executed while a host scope stack is temporarily active.
pub type NemoRelayNativeWithScopeStackCb =
    unsafe extern "C" fn(user_data: *mut c_void) -> NemoRelayStatus;

/// Runtime-provided continuation for tool execution intercepts.
///
/// On success, `out_json` contains canonical [`ToolExecutionResult`] JSON. The
/// returned host-owned string must be released with the active host table's
/// `string_free` hook.
pub type NemoRelayNativeToolNextFn = unsafe extern "C" fn(
    args_json: *const NemoRelayNativeString,
    next_ctx: *mut c_void,
    out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus;

/// Runtime-provided continuation for LLM execution intercepts.
pub type NemoRelayNativeLlmNextFn = unsafe extern "C" fn(
    request_json: *const NemoRelayNativeString,
    next_ctx: *mut c_void,
    out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus;

/// Native stream poll callback.
///
/// Return [`NemoRelayStatus::Ok`] with `out_json` set for one chunk,
/// [`NemoRelayStatus::StreamEnd`] with `out_json` null at end of stream, or an
/// error status for stream failure.
pub type NemoRelayNativeLlmStreamPollFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus;

/// Optional native stream cancellation callback.
pub type NemoRelayNativeLlmStreamCancelFn =
    Option<unsafe extern "C" fn(user_data: *mut c_void) -> NemoRelayStatus>;

/// Optional native stream destructor callback.
pub type NemoRelayNativeLlmStreamDropFn = Option<unsafe extern "C" fn(user_data: *mut c_void)>;

/// Native LLM JSON stream handle table.
#[repr(C)]
pub struct NemoRelayNativeLlmStreamV1 {
    /// Size of this struct as seen by the producer.
    pub struct_size: usize,
    /// Stream state passed back to poll/cancel/drop callbacks.
    pub user_data: *mut c_void,
    /// Polls the next stream chunk.
    pub next: Option<NemoRelayNativeLlmStreamPollFn>,
    /// Cancels an in-flight stream when a consumer stops before stream end.
    pub cancel: NemoRelayNativeLlmStreamCancelFn,
    /// Drops stream state after stream completion, error, or cancellation.
    pub drop: NemoRelayNativeLlmStreamDropFn,
}

impl Default for NemoRelayNativeLlmStreamV1 {
    fn default() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>(),
            user_data: ptr::null_mut(),
            next: None,
            cancel: None,
            drop: None,
        }
    }
}

/// Runtime-provided continuation for LLM stream execution intercepts.
pub type NemoRelayNativeLlmStreamNextFn = unsafe extern "C" fn(
    request_json: *const NemoRelayNativeString,
    next_ctx: *mut c_void,
    out_stream: *mut NemoRelayNativeLlmStreamV1,
) -> NemoRelayStatus;

/// Native event subscriber callback.
pub type NemoRelayNativeEventSubscriberCb = unsafe extern "C" fn(
    user_data: *mut c_void,
    event_json: *const NemoRelayNativeString,
) -> NemoRelayStatus;

/// Native event observability-field sanitizer callback.
pub type NemoRelayNativeEventSanitizeCb = unsafe extern "C" fn(
    user_data: *mut c_void,
    event_json: *const NemoRelayNativeString,
    fields_json: *const NemoRelayNativeString,
    out_fields_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus;

/// Native JSON transform callback for tool request/response sanitizers and tool request intercepts.
pub type NemoRelayNativeToolJsonCb = unsafe extern "C" fn(
    user_data: *mut c_void,
    name: *const NemoRelayNativeString,
    payload_json: *const NemoRelayNativeString,
    out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus;

/// Native tool conditional-execution callback.
pub type NemoRelayNativeToolConditionalCb = unsafe extern "C" fn(
    user_data: *mut c_void,
    name: *const NemoRelayNativeString,
    args_json: *const NemoRelayNativeString,
    out_reason: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus;

/// Native callback deciding whether a matching runtime registration is eligible.
///
/// `kinds_json` contains a JSON set of configured registration kinds and
/// `registration_name` is the target's effective name. Return a host-allocated
/// reason through `out_reason` to disable the target, or leave it null to keep
/// the target enabled. The input strings are borrowed for the callback only.
/// Relay treats a non-OK status or invalid output as an allow decision.
pub type NemoRelayNativeConditionalMiddlewareCb = unsafe extern "C" fn(
    user_data: *mut c_void,
    kinds_json: *const NemoRelayNativeString,
    registration_name: *const NemoRelayNativeString,
    out_reason: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus;

/// Native tool execution intercept callback.
///
/// A successful callback must set `out_outcome_json` to canonical
/// [`ToolExecutionInterceptOutcome`] JSON allocated through the host. The
/// `next_ctx` capability is valid only while this callback is active and must
/// not be retained.
pub type NemoRelayNativeToolExecutionCb = unsafe extern "C" fn(
    user_data: *mut c_void,
    name: *const NemoRelayNativeString,
    args_json: *const NemoRelayNativeString,
    next_fn: NemoRelayNativeToolNextFn,
    next_ctx: *mut c_void,
    out_outcome_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus;

/// Native LLM request sanitizer callback. Return a successful null output to
/// omit the observability payload and annotation. `request_json` is borrowed,
/// but may be written directly to `out_request_json` as a pass-through; the
/// host releases an aliased input/output once. Any other non-null output must
/// be host-allocated and transfers ownership to the host.
pub type NemoRelayNativeLlmSanitizeRequestCb = unsafe extern "C" fn(
    user_data: *mut c_void,
    request_json: *const NemoRelayNativeString,
    context: NemoRelayNativeLlmSanitizeRequestContext,
    out_request_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus;

/// Native LLM response sanitizer callback. Return a successful null output to
/// omit the observability payload and annotation. `payload_json` is borrowed,
/// but may be written directly to `out_json` as a pass-through; the host
/// releases an aliased input/output once. Any other non-null output must be
/// host-allocated and transfers ownership to the host.
pub type NemoRelayNativeLlmSanitizeResponseCb = unsafe extern "C" fn(
    user_data: *mut c_void,
    payload_json: *const NemoRelayNativeString,
    context: NemoRelayNativeLlmSanitizeResponseContext,
    out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus;

/// Native LLM conditional-execution callback.
pub type NemoRelayNativeLlmConditionalCb = unsafe extern "C" fn(
    user_data: *mut c_void,
    request_json: *const NemoRelayNativeString,
    out_reason: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus;

/// Native LLM request intercept callback.
pub type NemoRelayNativeLlmRequestInterceptCb = unsafe extern "C" fn(
    user_data: *mut c_void,
    name: *const NemoRelayNativeString,
    request_json: *const NemoRelayNativeString,
    annotated_json: *const NemoRelayNativeString,
    out_outcome_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus;

/// Native LLM execution intercept callback.
pub type NemoRelayNativeLlmExecutionCb = unsafe extern "C" fn(
    user_data: *mut c_void,
    name: *const NemoRelayNativeString,
    request_json: *const NemoRelayNativeString,
    next_fn: NemoRelayNativeLlmNextFn,
    next_ctx: *mut c_void,
    out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus;

/// Native LLM stream execution intercept callback.
pub type NemoRelayNativeLlmStreamExecutionCb = unsafe extern "C" fn(
    user_data: *mut c_void,
    name: *const NemoRelayNativeString,
    request_json: *const NemoRelayNativeString,
    next_fn: NemoRelayNativeLlmStreamNextFn,
    next_ctx: *mut c_void,
    out_stream: *mut NemoRelayNativeLlmStreamV1,
) -> NemoRelayStatus;

/// Native plugin validation callback.
pub type NemoRelayNativePluginValidateFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    plugin_config_json: *const NemoRelayNativeString,
    out_diagnostics_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus;

/// Native plugin registration callback.
pub type NemoRelayNativePluginRegisterFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    plugin_config_json: *const NemoRelayNativeString,
    ctx: *mut NemoRelayNativePluginContext,
) -> NemoRelayStatus;

/// Native plugin drop callback.
pub type NemoRelayNativePluginDropFn = Option<unsafe extern "C" fn(user_data: *mut c_void)>;

/// Versioned host API table passed to native plugin entry symbols.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NemoRelayNativeHostApiV1 {
    /// ABI version implemented by this table.
    pub abi_version: u32,
    /// Size of this struct as seen by the host.
    pub struct_size: usize,
    /// Null-terminated host Relay version string.
    pub relay_version: *const c_char,
    /// Allocates a host-owned string from UTF-8 bytes.
    pub string_new: unsafe extern "C" fn(
        data: *const u8,
        len: usize,
        out: *mut *mut NemoRelayNativeString,
    ) -> NemoRelayStatus,
    /// Returns the string data pointer for a host-owned string.
    pub string_data: unsafe extern "C" fn(value: *const NemoRelayNativeString) -> *const u8,
    /// Returns the byte length for a host-owned string.
    pub string_len: unsafe extern "C" fn(value: *const NemoRelayNativeString) -> usize,
    /// Frees a host-owned string.
    pub string_free: unsafe extern "C" fn(value: *mut NemoRelayNativeString),
    /// Clears the host thread-local native ABI error message.
    pub last_error_clear: unsafe extern "C" fn(),
    /// Sets the host thread-local native ABI error message.
    pub last_error_set: unsafe extern "C" fn(message: *const NemoRelayNativeString),
    /// Decodes an LLM request through a callback-scoped codec capability.
    pub llm_request_codec_decode: unsafe extern "C" fn(
        codec: *const NemoRelayNativeLlmRequestCodec,
        request_json: *const NemoRelayNativeString,
        out: *mut *mut NemoRelayNativeString,
    ) -> NemoRelayStatus,
    /// Encodes normalized request changes through a callback-scoped codec capability.
    pub llm_request_codec_encode: unsafe extern "C" fn(
        codec: *const NemoRelayNativeLlmRequestCodec,
        annotated_json: *const NemoRelayNativeString,
        original_json: *const NemoRelayNativeString,
        out: *mut *mut NemoRelayNativeString,
    ) -> NemoRelayStatus,
    /// Decodes an LLM response through a callback-scoped codec capability.
    pub llm_response_codec_decode: unsafe extern "C" fn(
        codec: *const NemoRelayNativeLlmResponseCodec,
        response_json: *const NemoRelayNativeString,
        out: *mut *mut NemoRelayNativeString,
    ) -> NemoRelayStatus,
    /// Registers an event subscriber through the plugin context.
    pub plugin_context_register_subscriber: unsafe extern "C" fn(
        ctx: *mut NemoRelayNativePluginContext,
        name: *const NemoRelayNativeString,
        cb: NemoRelayNativeEventSubscriberCb,
        user_data: *mut c_void,
        free_fn: NemoRelayNativeFreeFn,
    ) -> NemoRelayStatus,
    /// Registers a tool sanitize-request guardrail through the plugin context.
    pub plugin_context_register_tool_sanitize_request_guardrail:
        unsafe extern "C" fn(
            ctx: *mut NemoRelayNativePluginContext,
            name: *const NemoRelayNativeString,
            priority: i32,
            cb: NemoRelayNativeToolJsonCb,
            user_data: *mut c_void,
            free_fn: NemoRelayNativeFreeFn,
        ) -> NemoRelayStatus,
    /// Registers a tool sanitize-response guardrail through the plugin context.
    pub plugin_context_register_tool_sanitize_response_guardrail:
        unsafe extern "C" fn(
            ctx: *mut NemoRelayNativePluginContext,
            name: *const NemoRelayNativeString,
            priority: i32,
            cb: NemoRelayNativeToolJsonCb,
            user_data: *mut c_void,
            free_fn: NemoRelayNativeFreeFn,
        ) -> NemoRelayStatus,
    /// Registers a tool conditional-execution guardrail through the plugin context.
    pub plugin_context_register_tool_conditional_execution_guardrail:
        unsafe extern "C" fn(
            ctx: *mut NemoRelayNativePluginContext,
            name: *const NemoRelayNativeString,
            priority: i32,
            cb: NemoRelayNativeToolConditionalCb,
            user_data: *mut c_void,
            free_fn: NemoRelayNativeFreeFn,
        ) -> NemoRelayStatus,
    /// Registers a tool request intercept through the plugin context.
    pub plugin_context_register_tool_request_intercept: unsafe extern "C" fn(
        ctx: *mut NemoRelayNativePluginContext,
        name: *const NemoRelayNativeString,
        priority: i32,
        break_chain: bool,
        cb: NemoRelayNativeToolJsonCb,
        user_data: *mut c_void,
        free_fn: NemoRelayNativeFreeFn,
    )
        -> NemoRelayStatus,
    /// Registers a tool execution intercept through the plugin context.
    pub plugin_context_register_tool_execution_intercept: unsafe extern "C" fn(
        ctx: *mut NemoRelayNativePluginContext,
        name: *const NemoRelayNativeString,
        priority: i32,
        cb: NemoRelayNativeToolExecutionCb,
        user_data: *mut c_void,
        free_fn: NemoRelayNativeFreeFn,
    )
        -> NemoRelayStatus,
    /// Registers an LLM sanitize-request guardrail through the plugin context.
    pub plugin_context_register_llm_sanitize_request_guardrail:
        unsafe extern "C" fn(
            ctx: *mut NemoRelayNativePluginContext,
            name: *const NemoRelayNativeString,
            priority: i32,
            cb: NemoRelayNativeLlmSanitizeRequestCb,
            user_data: *mut c_void,
            free_fn: NemoRelayNativeFreeFn,
        ) -> NemoRelayStatus,
    /// Registers an LLM sanitize-response guardrail through the plugin context.
    pub plugin_context_register_llm_sanitize_response_guardrail:
        unsafe extern "C" fn(
            ctx: *mut NemoRelayNativePluginContext,
            name: *const NemoRelayNativeString,
            priority: i32,
            cb: NemoRelayNativeLlmSanitizeResponseCb,
            user_data: *mut c_void,
            free_fn: NemoRelayNativeFreeFn,
        ) -> NemoRelayStatus,
    /// Registers an LLM conditional-execution guardrail through the plugin context.
    pub plugin_context_register_llm_conditional_execution_guardrail:
        unsafe extern "C" fn(
            ctx: *mut NemoRelayNativePluginContext,
            name: *const NemoRelayNativeString,
            priority: i32,
            cb: NemoRelayNativeLlmConditionalCb,
            user_data: *mut c_void,
            free_fn: NemoRelayNativeFreeFn,
        ) -> NemoRelayStatus,
    /// Registers an LLM request intercept through the plugin context.
    pub plugin_context_register_llm_request_intercept: unsafe extern "C" fn(
        ctx: *mut NemoRelayNativePluginContext,
        name: *const NemoRelayNativeString,
        priority: i32,
        break_chain: bool,
        cb: NemoRelayNativeLlmRequestInterceptCb,
        user_data: *mut c_void,
        free_fn: NemoRelayNativeFreeFn,
    ) -> NemoRelayStatus,
    /// Registers an LLM execution intercept through the plugin context.
    pub plugin_context_register_llm_execution_intercept: unsafe extern "C" fn(
        ctx: *mut NemoRelayNativePluginContext,
        name: *const NemoRelayNativeString,
        priority: i32,
        cb: NemoRelayNativeLlmExecutionCb,
        user_data: *mut c_void,
        free_fn: NemoRelayNativeFreeFn,
    )
        -> NemoRelayStatus,
    /// Registers an LLM stream execution intercept through the plugin context.
    pub plugin_context_register_llm_stream_execution_intercept:
        unsafe extern "C" fn(
            ctx: *mut NemoRelayNativePluginContext,
            name: *const NemoRelayNativeString,
            priority: i32,
            cb: NemoRelayNativeLlmStreamExecutionCb,
            user_data: *mut c_void,
            free_fn: NemoRelayNativeFreeFn,
        ) -> NemoRelayStatus,
    /// Frees a host-owned scope handle.
    pub scope_handle_free: unsafe extern "C" fn(handle: *mut NemoRelayNativeScopeHandle),
    /// Retrieves the current scope handle from the active stack.
    pub scope_get_current:
        unsafe extern "C" fn(out: *mut *mut NemoRelayNativeScopeHandle) -> NemoRelayStatus,
    /// Pushes a scope, emits its start event, and returns its handle.
    pub scope_push: unsafe extern "C" fn(
        name: *const NemoRelayNativeString,
        scope_type: NemoRelayNativeScopeType,
        parent: *const NemoRelayNativeScopeHandle,
        attributes: u32,
        data_json: *const NemoRelayNativeString,
        metadata_json: *const NemoRelayNativeString,
        input_json: *const NemoRelayNativeString,
        timestamp_unix_micros: *const i64,
        out: *mut *mut NemoRelayNativeScopeHandle,
    ) -> NemoRelayStatus,
    /// Pops a scope handle, emits its end event, and clears scope-local registrations.
    pub scope_pop: unsafe extern "C" fn(
        handle: *const NemoRelayNativeScopeHandle,
        output_json: *const NemoRelayNativeString,
        metadata_json: *const NemoRelayNativeString,
        timestamp_unix_micros: *const i64,
    ) -> NemoRelayStatus,
    /// Emits a mark event under the current or provided parent scope.
    pub emit_mark: unsafe extern "C" fn(
        name: *const NemoRelayNativeString,
        parent: *const NemoRelayNativeScopeHandle,
        data_json: *const NemoRelayNativeString,
        metadata_json: *const NemoRelayNativeString,
        timestamp_unix_micros: *const i64,
    ) -> NemoRelayStatus,
    /// Creates a new independent scope stack with its own root scope.
    pub scope_stack_create:
        unsafe extern "C" fn(out: *mut *mut NemoRelayNativeScopeStack) -> NemoRelayStatus,
    /// Frees a host-owned scope stack handle.
    pub scope_stack_free: unsafe extern "C" fn(stack: *mut NemoRelayNativeScopeStack),
    /// Binds a scope stack to the current OS thread.
    pub scope_stack_set_thread:
        unsafe extern "C" fn(stack: *const NemoRelayNativeScopeStack) -> NemoRelayStatus,
    /// Captures the current thread-local scope-stack binding.
    pub scope_stack_capture_thread:
        unsafe extern "C" fn(out: *mut *mut NemoRelayNativeScopeStackBinding) -> NemoRelayStatus,
    /// Restores and frees a captured thread-local scope-stack binding.
    pub scope_stack_restore_thread:
        unsafe extern "C" fn(binding: *mut NemoRelayNativeScopeStackBinding) -> NemoRelayStatus,
    /// Frees a captured thread-local binding without restoring it.
    pub scope_stack_binding_free:
        unsafe extern "C" fn(binding: *mut NemoRelayNativeScopeStackBinding),
    /// Returns whether the current context has an explicitly active scope stack.
    pub scope_stack_active: unsafe extern "C" fn() -> bool,
    /// Runs a callback with the provided scope stack visible to host runtime APIs.
    pub scope_stack_with_current: unsafe extern "C" fn(
        stack: *const NemoRelayNativeScopeStack,
        cb: NemoRelayNativeWithScopeStackCb,
        user_data: *mut c_void,
    ) -> NemoRelayStatus,
    /// Registers a mark event sanitizer through the plugin context.
    pub plugin_context_register_mark_sanitize_guardrail: unsafe extern "C" fn(
        ctx: *mut NemoRelayNativePluginContext,
        name: *const NemoRelayNativeString,
        priority: i32,
        cb: NemoRelayNativeEventSanitizeCb,
        user_data: *mut c_void,
        free_fn: NemoRelayNativeFreeFn,
    )
        -> NemoRelayStatus,
    /// Registers a scope-start event sanitizer through the plugin context.
    pub plugin_context_register_scope_sanitize_start_guardrail:
        unsafe extern "C" fn(
            ctx: *mut NemoRelayNativePluginContext,
            name: *const NemoRelayNativeString,
            priority: i32,
            cb: NemoRelayNativeEventSanitizeCb,
            user_data: *mut c_void,
            free_fn: NemoRelayNativeFreeFn,
        ) -> NemoRelayStatus,
    /// Registers a scope-end event sanitizer through the plugin context.
    pub plugin_context_register_scope_sanitize_end_guardrail:
        unsafe extern "C" fn(
            ctx: *mut NemoRelayNativePluginContext,
            name: *const NemoRelayNativeString,
            priority: i32,
            cb: NemoRelayNativeEventSanitizeCb,
            user_data: *mut c_void,
            free_fn: NemoRelayNativeFreeFn,
        ) -> NemoRelayStatus,
}

/// Middleware surface selected by the native async registration hook.
///
/// The host only exposes this through the ABI-v3 extension table.  It keeps
/// every asynchronous callback shape uniform while allowing the host to
/// deserialize the surface-specific invocation and result payloads.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NemoRelayNativeAsyncMiddlewareKind {
    /// Tool start-event request sanitizer.
    ToolSanitizeRequest = 0,
    /// Tool end-event response sanitizer.
    ToolSanitizeResponse = 1,
    /// Tool execution admission guardrail.
    ToolConditionalExecution = 2,
    /// Tool request rewrite intercept.
    ToolRequestIntercept = 3,
    /// Tool execution intercept with a continuation.
    ToolExecutionIntercept = 4,
    /// LLM start-event request sanitizer.
    LlmSanitizeRequest = 5,
    /// LLM end-event response sanitizer.
    LlmSanitizeResponse = 6,
    /// LLM execution admission guardrail.
    LlmConditionalExecution = 7,
    /// LLM request rewrite intercept.
    LlmRequestIntercept = 8,
    /// LLM execution intercept with a continuation.
    LlmExecutionIntercept = 9,
    /// Reserved legacy discriminant for streaming LLM execution intercepts.
    ///
    /// Hosts reject this kind from the generic completion-based registration
    /// hook. Use `plugin_context_register_async_stream_middleware` so chunks
    /// remain incremental.
    LlmStreamExecutionIntercept = 10,
    /// Mark event sanitizer.
    MarkSanitize = 11,
    /// Scope-start event sanitizer.
    ScopeSanitizeStart = 12,
    /// Scope-end event sanitizer.
    ScopeSanitizeEnd = 13,
    /// Event metadata injector.
    EventMetadataInjector = 14,
}

impl TryFrom<u32> for NemoRelayNativeAsyncMiddlewareKind {
    type Error = ();

    fn try_from(value: u32) -> std::result::Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::ToolSanitizeRequest),
            1 => Ok(Self::ToolSanitizeResponse),
            2 => Ok(Self::ToolConditionalExecution),
            3 => Ok(Self::ToolRequestIntercept),
            4 => Ok(Self::ToolExecutionIntercept),
            5 => Ok(Self::LlmSanitizeRequest),
            6 => Ok(Self::LlmSanitizeResponse),
            7 => Ok(Self::LlmConditionalExecution),
            8 => Ok(Self::LlmRequestIntercept),
            9 => Ok(Self::LlmExecutionIntercept),
            10 => Ok(Self::LlmStreamExecutionIntercept),
            11 => Ok(Self::MarkSanitize),
            12 => Ok(Self::ScopeSanitizeStart),
            13 => Ok(Self::ScopeSanitizeEnd),
            14 => Ok(Self::EventMetadataInjector),
            _ => Err(()),
        }
    }
}

/// Indicates whether an asynchronous native callback settled before returning.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NemoRelayNativeAsyncCallbackState {
    /// The callback settled its completion before returning.
    Complete = 0,
    /// The callback retained its completion for later settlement.
    Pending = 1,
}

impl TryFrom<u32> for NemoRelayNativeAsyncCallbackState {
    type Error = ();

    fn try_from(value: u32) -> std::result::Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Complete),
            1 => Ok(Self::Pending),
            _ => Err(()),
        }
    }
}

/// Opaque one-shot completion retained by a pending native callback.
#[repr(C)]
pub struct NemoRelayNativeAsyncCompletion {
    _private: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}

/// Opaque native execution continuation supplied only to execution intercepts.
#[repr(C)]
pub struct NemoRelayNativeAsyncNext {
    _private: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}

/// Opaque incremental output channel supplied to native async stream intercepts.
#[repr(C)]
pub struct NemoRelayNativeAsyncStream {
    _private: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}

/// Opaque pull-based downstream LLM stream owned by the host.
#[repr(C)]
pub struct NemoRelayNativeLlmAsyncStream {
    _private: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}

/// Receives the result of asynchronously opening a downstream LLM stream.
///
/// Exactly one of `stream` and `error` is non-null. A non-null stream is an
/// owned plugin reference and must be released exactly once.
pub type NemoRelayNativeAsyncLlmStreamOpenCb = unsafe extern "C" fn(
    user_data: *mut c_void,
    stream: *const NemoRelayNativeLlmAsyncStream,
    error: *const NemoRelayNativeString,
);

/// Receives one item from a pull-based downstream LLM stream.
///
/// A chunk has non-null `chunk_json` and `done = false`; clean completion has
/// both strings null and `done = true`; failure has non-null `error` and
/// `done = true`. Only one pull may be outstanding per stream.
pub type NemoRelayNativeAsyncLlmStreamPullCb = unsafe extern "C" fn(
    user_data: *mut c_void,
    chunk_json: *const NemoRelayNativeString,
    error: *const NemoRelayNativeString,
    done: bool,
);

/// Receives one downstream stream item. `chunk_json` is non-null for a chunk,
/// `error` is non-null for failure or consumer cancellation, and `done` marks
/// clean completion. Unless the callback itself returns `false`, the host
/// invokes one terminal callback so the plugin can reclaim `user_data`.
/// Return `false` to cancel downstream production after the current callback;
/// in that case, reclaim `user_data` before returning because no later callback
/// is made.
pub type NemoRelayNativeAsyncNextStreamCb = unsafe extern "C" fn(
    user_data: *mut c_void,
    chunk_json: *const NemoRelayNativeString,
    error: *const NemoRelayNativeString,
    done: bool,
) -> bool;

/// Receives one completion from a unary execution-continuation invocation.
///
/// Exactly one of `value_json` and `error` is non-null. The callback owns its
/// `user_data` and is invoked exactly once after a successful
/// `async_next_invoke_result` call, including when the owning interceptor
/// settles and cancels unfinished downstream work. For a tool continuation,
/// `value_json` contains canonical [`ToolExecutionResult`] JSON.
pub type NemoRelayNativeAsyncNextResultCb = unsafe extern "C" fn(
    user_data: *mut c_void,
    value_json: *const NemoRelayNativeString,
    error: *const NemoRelayNativeString,
);

/// Incremental native LLM stream intercept callback.
///
/// The callback owns `next` and `stream` and must release each exactly once.
/// It may push chunks before returning or retain the handles and return
/// `Pending`; no implicit timeout is applied. Relay can invoke separate
/// middleware calls concurrently without stable OS-thread affinity. Retained
/// handles may be used from a plugin-owned thread, while callbacks supplied to
/// `async_next_invoke_stream` run on a Relay runtime worker. The output stream
/// owns the callback lifetime: `next` may be invoked repeatedly or concurrently
/// until that stream finishes, rejects, or is cancelled, and each invocation
/// has independent callback state. Relay rejects or cancels unfinished and
/// later invocations after settlement. The plugin must synchronize shared
/// `user_data` and callback state and serialize each handle's final release
/// after its last operation returns.
pub type NemoRelayNativeAsyncStreamMiddlewareCb = unsafe extern "C" fn(
    user_data: *mut c_void,
    invocation_json: *const NemoRelayNativeString,
    next: *const NemoRelayNativeAsyncNext,
    stream: *const NemoRelayNativeAsyncStream,
) -> u32;

/// Completion-based native middleware callback.
///
/// `invocation_json` is borrowed for the call. A callback that returns
/// [`NemoRelayNativeAsyncCallbackState::Pending`] as a `u32` owns one
/// completion reference and must settle it then call the v3
/// `async_completion_release` hook. The host validates the returned
/// discriminant. When `next` is non-null, the callback owns that handle for
/// the invocation and must call `async_next_release` exactly once after its
/// final use, regardless of whether it returns `Complete` or `Pending`. The
/// host never reclaims a `next` handle after handing it to the callback.
/// `next` is null for non-execution middleware. Relay invokes the callback on
/// the Tokio runtime worker polling that middleware invocation, without stable
/// OS-thread affinity; separate invocations may run concurrently. After
/// returning `Pending`, retained completion and `next` handles may be used from
/// a plugin-owned thread until the completion settles. Every `next` operation
/// must finish before resolving or rejecting the completion; Relay rejects or
/// cancels unfinished and later continuation calls. The plugin must synchronize
/// shared `user_data` and callback state and serialize each handle's final
/// release after its last operation returns. A tool-execution callback must
/// resolve its completion with canonical [`ToolExecutionInterceptOutcome`]
/// JSON.
pub type NemoRelayNativeAsyncMiddlewareCb = unsafe extern "C" fn(
    user_data: *mut c_void,
    invocation_json: *const NemoRelayNativeString,
    next: *const NemoRelayNativeAsyncNext,
    completion: *const NemoRelayNativeAsyncCompletion,
) -> u32;

/// ABI-v3 host extension appended to [`NemoRelayNativeHostApiV1`].
///
/// Its first field is the complete v1/v2 table, so legacy plugins can keep
/// treating the pointer as a [`NemoRelayNativeHostApiV1`].
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NemoRelayNativeHostApiV3 {
    /// Compatibility prefix for ABI-v1/v2 plugins.
    pub v1: NemoRelayNativeHostApiV1,
    /// Resolves an async callback completion with a JSON value.
    ///
    /// Tool-execution middleware must supply canonical
    /// [`ToolExecutionInterceptOutcome`] JSON.
    pub async_completion_resolve_json: unsafe extern "C" fn(
        completion: *const NemoRelayNativeAsyncCompletion,
        value_json: *const NemoRelayNativeString,
    ) -> NemoRelayStatus,
    /// Rejects an async callback completion with a UTF-8 message.
    pub async_completion_reject: unsafe extern "C" fn(
        completion: *const NemoRelayNativeAsyncCompletion,
        message: *const NemoRelayNativeString,
    ) -> NemoRelayStatus,
    /// Returns true after the awaiting runtime has cancelled the invocation.
    pub async_completion_is_cancelled:
        unsafe extern "C" fn(completion: *const NemoRelayNativeAsyncCompletion) -> bool,
    /// Releases the callback-owned reference after a pending completion settles.
    pub async_completion_release:
        unsafe extern "C" fn(completion: *const NemoRelayNativeAsyncCompletion),
    /// Invokes an execution continuation and settles a supplied completion.
    ///
    /// Cancellation of that completion aborts an in-flight continuation. This
    /// legacy convenience hook is one-shot because its result settles the
    /// middleware completion; use `async_next_invoke_result` for repeated or
    /// concurrent calls. For a tool continuation, the host resolves the
    /// completion with canonical [`ToolExecutionInterceptOutcome`] JSON.
    pub async_next_invoke: unsafe extern "C" fn(
        next: *const NemoRelayNativeAsyncNext,
        invocation_json: *const NemoRelayNativeString,
        completion: *const NemoRelayNativeAsyncCompletion,
    ) -> NemoRelayStatus,
    /// Releases the callback-owned continuation reference.
    ///
    /// Execution callbacks must call this exactly once after their final use
    /// for both `Complete` and `Pending` return states.
    pub async_next_release: unsafe extern "C" fn(next: *const NemoRelayNativeAsyncNext),
    /// Registers a completion-based asynchronous middleware surface.
    ///
    /// `kind` must be a valid [`NemoRelayNativeAsyncMiddlewareKind`]
    /// discriminant. The host rejects unknown `u32` values and
    /// [`NemoRelayNativeAsyncMiddlewareKind::LlmStreamExecutionIntercept`],
    /// which must use `plugin_context_register_async_stream_middleware`.
    pub plugin_context_register_async_middleware: unsafe extern "C" fn(
        ctx: *mut NemoRelayNativePluginContext,
        kind: u32,
        name: *const NemoRelayNativeString,
        priority: i32,
        break_chain: bool,
        cb: NemoRelayNativeAsyncMiddlewareCb,
        user_data: *mut c_void,
        free_fn: NemoRelayNativeFreeFn,
    ) -> NemoRelayStatus,
    /// Pushes one JSON chunk to an incremental native stream without blocking.
    ///
    /// A full bounded host queue returns [`NemoRelayStatus::Backpressured`];
    /// retry this same logical chunk after the consumer advances.
    pub async_stream_push_json: unsafe extern "C" fn(
        stream: *const NemoRelayNativeAsyncStream,
        chunk_json: *const NemoRelayNativeString,
    ) -> NemoRelayStatus,
    /// Finishes an incremental native stream successfully.
    pub async_stream_finish:
        unsafe extern "C" fn(stream: *const NemoRelayNativeAsyncStream) -> NemoRelayStatus,
    /// Rejects an incremental native stream without blocking.
    ///
    /// A full bounded queue returns [`NemoRelayStatus::Backpressured`]; retry
    /// this same rejection after the consumer advances.
    pub async_stream_reject: unsafe extern "C" fn(
        stream: *const NemoRelayNativeAsyncStream,
        message: *const NemoRelayNativeString,
    ) -> NemoRelayStatus,
    /// Returns true when the consumer cancelled or released the stream.
    pub async_stream_is_cancelled:
        unsafe extern "C" fn(stream: *const NemoRelayNativeAsyncStream) -> bool,
    /// Releases the callback-owned incremental stream reference.
    pub async_stream_release: unsafe extern "C" fn(stream: *const NemoRelayNativeAsyncStream),
    /// Invokes a downstream stream and reports chunks incrementally.
    ///
    /// The host reports consumer cancellation through one terminal callback
    /// with a non-null error. If a result callback returns `false`, it must
    /// reclaim its own `user_data` before returning because no terminal
    /// callback follows. This hook may be called repeatedly or concurrently
    /// with independent callback state while the output stream remains active.
    pub async_next_invoke_stream: unsafe extern "C" fn(
        next: *const NemoRelayNativeAsyncNext,
        invocation_json: *const NemoRelayNativeString,
        stream: *const NemoRelayNativeAsyncStream,
        cb: NemoRelayNativeAsyncNextStreamCb,
        user_data: *mut c_void,
    ) -> NemoRelayStatus,
    /// Registers an incremental asynchronous LLM stream intercept.
    pub plugin_context_register_async_stream_middleware: unsafe extern "C" fn(
        ctx: *mut NemoRelayNativePluginContext,
        name: *const NemoRelayNativeString,
        priority: i32,
        cb: NemoRelayNativeAsyncStreamMiddlewareCb,
        user_data: *mut c_void,
        free_fn: NemoRelayNativeFreeFn,
    )
        -> NemoRelayStatus,
    /// Invokes a unary execution continuation with an independent result sink.
    ///
    /// Unlike the legacy completion-coupled `async_next_invoke`, this hook may
    /// be called repeatedly or concurrently with distinct `user_data`. For a
    /// tool continuation, the result callback receives canonical
    /// [`ToolExecutionResult`] JSON.
    pub async_next_invoke_result: unsafe extern "C" fn(
        next: *const NemoRelayNativeAsyncNext,
        invocation_json: *const NemoRelayNativeString,
        cb: NemoRelayNativeAsyncNextResultCb,
        user_data: *mut c_void,
    ) -> NemoRelayStatus,
}

/// Extended native mark-emission function introduced in ABI v4.
pub type NemoRelayNativeEmitMarkV2Fn = unsafe extern "C" fn(
    name: *const NemoRelayNativeString,
    parent: *const NemoRelayNativeScopeHandle,
    data_json: *const NemoRelayNativeString,
    metadata_json: *const NemoRelayNativeString,
    data_schema_json: *const NemoRelayNativeString,
    severity: *const NemoRelayNativeString,
    timestamp_unix_micros: *const i64,
) -> NemoRelayStatus;

/// Reads the active host runtime-diagnostics snapshot as canonical JSON.
pub type NemoRelayNativeGetRuntimeDiagnosticsFn =
    unsafe extern "C" fn(out_json: *mut *mut NemoRelayNativeString) -> NemoRelayStatus;

/// ABI-v4 host extension for typed asynchronous middleware, mark options,
/// diagnostics, and activation-owned dynamic gate control.
///
/// The complete ABI-v3 table is the prefix, preserving layout compatibility.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NemoRelayNativeHostApiV4 {
    /// Frozen ABI-v3 compatibility prefix.
    pub v3: NemoRelayNativeHostApiV3,
    /// Decodes an LLM request using the request codec attached to `completion`.
    pub async_completion_llm_request_codec_decode: unsafe extern "C" fn(
        completion: *const NemoRelayNativeAsyncCompletion,
        request_json: *const NemoRelayNativeString,
        out: *mut *mut NemoRelayNativeString,
    ) -> NemoRelayStatus,
    /// Encodes an annotated LLM request using the request codec attached to `completion`.
    pub async_completion_llm_request_codec_encode: unsafe extern "C" fn(
        completion: *const NemoRelayNativeAsyncCompletion,
        annotated_json: *const NemoRelayNativeString,
        original_json: *const NemoRelayNativeString,
        out: *mut *mut NemoRelayNativeString,
    ) -> NemoRelayStatus,
    /// Decodes an LLM response using the response codec attached to `completion`.
    pub async_completion_llm_response_codec_decode: unsafe extern "C" fn(
        completion: *const NemoRelayNativeAsyncCompletion,
        response_json: *const NemoRelayNativeString,
        out: *mut *mut NemoRelayNativeString,
    ) -> NemoRelayStatus,
    /// Opens an independent pull-based downstream LLM stream.
    pub async_next_open_llm_stream: unsafe extern "C" fn(
        next: *const NemoRelayNativeAsyncNext,
        request_json: *const NemoRelayNativeString,
        cb: NemoRelayNativeAsyncLlmStreamOpenCb,
        user_data: *mut c_void,
    ) -> NemoRelayStatus,
    /// Requests one item from a pull-based downstream LLM stream.
    pub async_llm_stream_pull: unsafe extern "C" fn(
        stream: *const NemoRelayNativeLlmAsyncStream,
        cb: NemoRelayNativeAsyncLlmStreamPullCb,
        user_data: *mut c_void,
    ) -> NemoRelayStatus,
    /// Cancels a pull-based downstream LLM stream.
    pub async_llm_stream_cancel:
        unsafe extern "C" fn(stream: *const NemoRelayNativeLlmAsyncStream) -> NemoRelayStatus,
    /// Releases the plugin-owned stream reference exactly once.
    pub async_llm_stream_release:
        unsafe extern "C" fn(stream: *const NemoRelayNativeLlmAsyncStream),
    /// Retains a completion capability for a codec facade that outlives the
    /// callback's original completion reference.
    pub async_completion_retain:
        unsafe extern "C" fn(completion: *const NemoRelayNativeAsyncCompletion) -> NemoRelayStatus,
    /// Returns whether the output queue is currently full.
    ///
    /// Use [`NemoRelayStatus::Backpressured`] from the individual push or
    /// rejection operation to decide whether that operation must be retried.
    pub async_stream_is_backpressured:
        unsafe extern "C" fn(stream: *const NemoRelayNativeAsyncStream) -> bool,
    /// Emits a mark with optional data-schema and telemetry-severity fields.
    pub emit_mark_v2: NemoRelayNativeEmitMarkV2Fn,
    /// Returns a bounded snapshot of active host runtime diagnostics.
    pub get_runtime_diagnostics: NemoRelayNativeGetRuntimeDiagnosticsFn,
    /// Creates an activation-owned runtime capability from a registration context.
    ///
    /// On success, `out` receives one owned reference. Every owned reference must
    /// be released exactly once with `plugin_runtime_release`.
    pub plugin_context_runtime: unsafe extern "C" fn(
        ctx: *mut NemoRelayNativePluginContext,
        out: *mut *const NemoRelayNativePluginRuntime,
    ) -> NemoRelayStatus,
    /// Retains a runtime capability for a cloned SDK handle.
    ///
    /// A successful call creates one additional owned reference.
    pub plugin_runtime_retain:
        unsafe extern "C" fn(runtime: *const NemoRelayNativePluginRuntime) -> NemoRelayStatus,
    /// Releases one owned runtime capability reference.
    pub plugin_runtime_release: unsafe extern "C" fn(runtime: *const NemoRelayNativePluginRuntime),
    /// Lists global runtime registrations as JSON.
    pub plugin_runtime_list_registrations: unsafe extern "C" fn(
        runtime: *const NemoRelayNativePluginRuntime,
        kinds_json: *const NemoRelayNativeString,
        out_json: *mut *mut NemoRelayNativeString,
    ) -> NemoRelayStatus,
    /// Legacy constant-reason gate registration retained for v4 compatibility.
    pub plugin_runtime_register_conditional_middleware_guardrail:
        unsafe extern "C" fn(
            runtime: *const NemoRelayNativePluginRuntime,
            name: *const NemoRelayNativeString,
            kinds_json: *const NemoRelayNativeString,
            registration_name: *const NemoRelayNativeString,
            reason: *const NemoRelayNativeString,
            out_handle: *mut *mut NemoRelayNativeString,
        ) -> NemoRelayStatus,
    /// Deregisters an activation-owned gate by opaque handle.
    pub plugin_runtime_deregister_conditional_middleware_guardrail:
        unsafe extern "C" fn(
            runtime: *const NemoRelayNativePluginRuntime,
            handle: *const NemoRelayNativeString,
            out_removed: *mut bool,
        ) -> NemoRelayStatus,
    /// Legacy constant-reason component gate retained for v4 compatibility.
    pub plugin_context_register_conditional_middleware_guardrail:
        unsafe extern "C" fn(
            ctx: *mut NemoRelayNativePluginContext,
            name: *const NemoRelayNativeString,
            kinds_json: *const NemoRelayNativeString,
            registration_name: *const NemoRelayNativeString,
            reason: *const NemoRelayNativeString,
        ) -> NemoRelayStatus,
    /// Registers an activation-owned callback gate and returns its handle.
    ///
    /// Relay invokes `cb` synchronously and consumes `user_data` on every return
    /// path. A null callback reason leaves the target enabled.
    pub plugin_runtime_register_conditional_middleware_guardrail_callback:
        unsafe extern "C" fn(
            runtime: *const NemoRelayNativePluginRuntime,
            name: *const NemoRelayNativeString,
            kinds_json: *const NemoRelayNativeString,
            registration_name: *const NemoRelayNativeString,
            cb: NemoRelayNativeConditionalMiddlewareCb,
            user_data: *mut c_void,
            free_fn: NemoRelayNativeFreeFn,
            out_handle: *mut *mut NemoRelayNativeString,
        ) -> NemoRelayStatus,
    /// Declares a callback gate during component registration.
    ///
    /// Relay invokes `cb` synchronously and consumes `user_data` on every return
    /// path. A null callback reason leaves the target enabled.
    pub plugin_context_register_conditional_middleware_guardrail_callback:
        unsafe extern "C" fn(
            ctx: *mut NemoRelayNativePluginContext,
            name: *const NemoRelayNativeString,
            kinds_json: *const NemoRelayNativeString,
            registration_name: *const NemoRelayNativeString,
            cb: NemoRelayNativeConditionalMiddlewareCb,
            user_data: *mut c_void,
            free_fn: NemoRelayNativeFreeFn,
        ) -> NemoRelayStatus,
}

unsafe impl Send for NemoRelayNativeHostApiV3 {}
unsafe impl Sync for NemoRelayNativeHostApiV3 {}
// SAFETY: the v4 host table is immutable after construction. Its function
// pointers and inherited host metadata may be invoked from any plugin thread.
unsafe impl Send for NemoRelayNativeHostApiV4 {}
unsafe impl Sync for NemoRelayNativeHostApiV4 {}

// The host API table is immutable after construction. Function pointers and
// the null-terminated version string pointer are safe to share across threads.
unsafe impl Send for NemoRelayNativeHostApiV1 {}
unsafe impl Sync for NemoRelayNativeHostApiV1 {}

/// Versioned plugin descriptor returned by native plugin entry symbols.
#[repr(C)]
pub struct NemoRelayNativePluginV1 {
    /// Size of this struct as seen by the plugin.
    pub struct_size: usize,
    /// Host-owned plugin kind string.
    pub plugin_kind: *mut NemoRelayNativeString,
    /// Whether this plugin kind supports multiple configured components.
    pub allows_multiple_components: bool,
    /// Plugin-owned state pointer passed to callbacks.
    pub user_data: *mut c_void,
    /// Optional validation callback.
    pub validate: Option<NemoRelayNativePluginValidateFn>,
    /// Required registration callback.
    pub register: Option<NemoRelayNativePluginRegisterFn>,
    /// Optional plugin-owned state destructor.
    pub drop: NemoRelayNativePluginDropFn,
}

impl Default for NemoRelayNativePluginV1 {
    fn default() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>(),
            plugin_kind: ptr::null_mut(),
            allows_multiple_components: true,
            user_data: ptr::null_mut(),
            validate: None,
            register: None,
            drop: None,
        }
    }
}

/// Native entry symbol type loaded by the host.
pub type NemoRelayNativePluginEntry = unsafe extern "C" fn(
    host: *const NemoRelayNativeHostApiV1,
    out: *mut NemoRelayNativePluginV1,
) -> NemoRelayStatus;

/// Result type used by the Rust native plugin SDK.
pub type Result<T> = std::result::Result<T, String>;

/// One bounded runtime diagnostic reported by the Relay host.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, serde::Deserialize)]
pub struct RuntimeDiagnostic {
    /// Stable identifier for the diagnostic condition.
    pub code: String,
    /// Most recently recorded message for this condition.
    pub message: String,
    /// Total number of occurrences recorded for this condition.
    pub count: u64,
}

/// Bounded snapshot of active host runtime diagnostics.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, serde::Deserialize)]
pub struct RuntimeDiagnostics {
    entries: Vec<RuntimeDiagnostic>,
}

/// Opaque activation-owned handle for a native plugin's dynamic gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConditionalMiddlewareGuardrailHandle(String);

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

/// Synchronous JSON chunk stream used by native LLM stream intercept helpers.
pub type LlmJsonStream = Box<dyn Iterator<Item = Result<Json>> + Send>;

/// Cloneable high-level runtime handle for host APIs available to native plugins.
pub struct PluginRuntime {
    host: NemoRelayNativeHostApiV1,
    emit_mark_v2: Option<NemoRelayNativeEmitMarkV2Fn>,
    get_runtime_diagnostics: Option<NemoRelayNativeGetRuntimeDiagnosticsFn>,
    v4: Option<NemoRelayNativeHostApiV4>,
    capability: *const NemoRelayNativePluginRuntime,
}

// SAFETY: PluginRuntime holds an immutable host table and a retained,
// thread-safe host capability. Clone and Drop use the host's atomic reference
// management operations.
unsafe impl Send for PluginRuntime {}
unsafe impl Sync for PluginRuntime {}

impl Clone for PluginRuntime {
    fn clone(&self) -> Self {
        let mut capability = self.capability;
        if let (Some(v4), false) = (self.v4, self.capability.is_null())
            && unsafe { (v4.plugin_runtime_retain)(self.capability) } != NemoRelayStatus::Ok
        {
            capability = ptr::null();
        }
        Self {
            host: self.host,
            emit_mark_v2: self.emit_mark_v2,
            get_runtime_diagnostics: self.get_runtime_diagnostics,
            v4: self.v4,
            capability,
        }
    }
}

impl Drop for PluginRuntime {
    fn drop(&mut self) {
        if let (Some(v4), false) = (self.v4, self.capability.is_null()) {
            unsafe { (v4.plugin_runtime_release)(self.capability) };
        }
    }
}

impl PluginRuntime {
    /// Creates a runtime handle from the host ABI table.
    pub fn new(host: &NemoRelayNativeHostApiV1) -> Self {
        let v4 = (host.abi_version >= NEMO_RELAY_NATIVE_ABI_VERSION
            && host.struct_size >= std::mem::size_of::<NemoRelayNativeHostApiV4>())
        .then(|| unsafe { *(host as *const _ as *const NemoRelayNativeHostApiV4) });
        Self {
            host: *host,
            emit_mark_v2: v4.map(|host| host.emit_mark_v2),
            get_runtime_diagnostics: v4.map(|host| host.get_runtime_diagnostics),
            v4,
            capability: ptr::null(),
        }
    }

    fn from_context(
        host: &NemoRelayNativeHostApiV1,
        ctx: *mut NemoRelayNativePluginContext,
    ) -> Self {
        let mut runtime = Self::new(host);
        let Some(v4) = runtime.v4 else {
            return runtime;
        };
        let mut capability = ptr::null();
        if unsafe { (v4.plugin_context_runtime)(ctx, &mut capability) } == NemoRelayStatus::Ok {
            runtime.capability = capability;
        }
        runtime
    }

    /// Lists global gateable runtime registrations.
    pub fn list_runtime_registrations(
        &self,
        kinds: Option<&std::collections::BTreeSet<RuntimeRegistrationKind>>,
    ) -> Result<Vec<RuntimeRegistrationIdentity>> {
        let v4 = self.runtime_v4()?;
        let kinds = match kinds {
            Some(kinds) => Some(
                HostString::from_json(&self.host, kinds)
                    .ok_or_else(|| "failed to serialize runtime registration kinds".to_string())?,
            ),
            None => None,
        };
        let mut out = ptr::null_mut();
        let status = unsafe {
            (v4.plugin_runtime_list_registrations)(
                self.capability,
                kinds.as_ref().map_or(ptr::null(), HostString::as_ptr),
                &mut out,
            )
        };
        status_result(&self.host, status, "list runtime registrations")?;
        take_host_json(&self.host, out)
    }

    /// Registers an activation-owned callback eligibility gate.
    ///
    /// Return `Some(reason)` to disable the matching target or `None` to leave
    /// it enabled.
    pub fn register_conditional_middleware_guardrail<F>(
        &self,
        name: &str,
        kinds: &std::collections::BTreeSet<RuntimeRegistrationKind>,
        registration_name: &str,
        callback: F,
    ) -> Result<ConditionalMiddlewareGuardrailHandle>
    where
        F: Fn(&std::collections::BTreeSet<RuntimeRegistrationKind>, &str) -> Option<String>
            + Send
            + Sync
            + 'static,
    {
        let v4 = self.runtime_v4()?;
        let name = HostString::new(&self.host, name)
            .ok_or_else(|| "failed to allocate gate name".to_string())?;
        let kinds = HostString::from_json(&self.host, kinds)
            .ok_or_else(|| "failed to serialize runtime registration kinds".to_string())?;
        let registration_name = HostString::new(&self.host, registration_name)
            .ok_or_else(|| "failed to allocate target name".to_string())?;
        let user_data = typed_callback_user_data(&self.host, callback);
        let mut out = ptr::null_mut();
        let status = unsafe {
            (v4.plugin_runtime_register_conditional_middleware_guardrail_callback)(
                self.capability,
                name.as_ptr(),
                kinds.as_ptr(),
                registration_name.as_ptr(),
                typed_conditional_middleware_trampoline::<F>,
                user_data,
                Some(drop_typed_callback::<F>),
                &mut out,
            )
        };
        finish_typed_registration(
            &self.host,
            status,
            user_data,
            "register conditional middleware guardrail",
        )?;
        take_host_string(&self.host, out).map(ConditionalMiddlewareGuardrailHandle)
    }

    /// Deregisters an activation-owned eligibility gate.
    pub fn deregister_conditional_middleware_guardrail(
        &self,
        handle: &ConditionalMiddlewareGuardrailHandle,
    ) -> Result<bool> {
        let v4 = self.runtime_v4()?;
        let handle = HostString::new(&self.host, &handle.0)
            .ok_or_else(|| "failed to allocate gate handle".to_string())?;
        let mut removed = false;
        let status = unsafe {
            (v4.plugin_runtime_deregister_conditional_middleware_guardrail)(
                self.capability,
                handle.as_ptr(),
                &mut removed,
            )
        };
        status_result(
            &self.host,
            status,
            "deregister conditional middleware guardrail",
        )?;
        Ok(removed)
    }

    fn runtime_v4(&self) -> Result<NemoRelayNativeHostApiV4> {
        self.v4
            .filter(|_| !self.capability.is_null())
            .ok_or_else(|| "host does not support activation-owned runtime gate control".into())
    }

    /// Returns the underlying host ABI table.
    pub fn host_api(&self) -> &NemoRelayNativeHostApiV1 {
        &self.host
    }

    /// Retrieves the current scope handle.
    pub fn current_scope(&self) -> Result<ScopeHandle<'_>> {
        current_scope(&self.host)
    }

    /// Pushes a scope and emits its start event.
    pub fn push_scope(
        &self,
        name: &str,
        scope_type: ScopeType,
        data: Option<&Json>,
        metadata: Option<&Json>,
        input: Option<&Json>,
    ) -> Result<ScopeHandle<'_>> {
        push_scope(&self.host, name, scope_type.into(), data, metadata, input)
    }

    /// Pops a scope and emits its end event.
    pub fn pop_scope(
        &self,
        handle: &ScopeHandle<'_>,
        output: Option<&Json>,
        metadata: Option<&Json>,
    ) -> Result<()> {
        pop_scope(&self.host, handle, output, metadata)
    }

    /// Opens a scope that is popped automatically when the guard is closed or dropped.
    pub fn scope(
        &self,
        name: &str,
        scope_type: ScopeType,
        data: Option<&Json>,
        metadata: Option<&Json>,
        input: Option<&Json>,
    ) -> Result<ScopeGuard<'_>> {
        let handle = self.push_scope(name, scope_type, data, metadata, input)?;
        Ok(ScopeGuard {
            runtime: self,
            handle: Some(handle),
        })
    }

    /// Emits a mark event under the current scope.
    pub fn emit_mark(
        &self,
        name: &str,
        data: Option<&Json>,
        metadata: Option<&Json>,
    ) -> Result<()> {
        emit_mark(&self.host, name, data, metadata)
    }

    /// Emits a mark event with an optional data schema and telemetry severity.
    ///
    /// Hosts without the ABI-v4 mark extension accept this call only when both
    /// new options are absent, in which case the legacy function is used.
    pub fn emit_mark_with_options(
        &self,
        name: &str,
        data: Option<&Json>,
        metadata: Option<&Json>,
        data_schema: Option<&DataSchema>,
        severity: Option<LogSeverity>,
    ) -> Result<()> {
        match self.emit_mark_v2 {
            Some(emit_mark_v2) => emit_mark_v2_call(
                &self.host,
                emit_mark_v2,
                name,
                data,
                metadata,
                data_schema,
                severity,
            ),
            None if data_schema.is_none() && severity.is_none() => {
                emit_mark(&self.host, name, data, metadata)
            }
            None => Err("mark data_schema and severity require native host ABI v4".into()),
        }
    }

    /// Emits a validated Relay metric-measurement mark under the current scope.
    pub fn emit_metric(
        &self,
        name: &str,
        measurements: Vec<MetricMeasurement>,
        metadata: Option<&Json>,
    ) -> Result<()> {
        let envelope = MetricEnvelope { measurements };
        envelope.validate().map_err(|err| err.to_string())?;
        let data = serde_json::to_value(envelope)
            .map_err(|err| format!("failed to serialize metric mark: {err}"))?;
        let data_schema = DataSchema::builder()
            .name(METRIC_DATA_SCHEMA_NAME)
            .version(METRIC_DATA_SCHEMA_VERSION)
            .build();
        self.emit_mark_with_options(name, Some(&data), metadata, Some(&data_schema), None)
    }

    /// Return a bounded snapshot of active host runtime diagnostics.
    pub fn runtime_diagnostics(&self) -> Result<RuntimeDiagnostics> {
        let Some(get_runtime_diagnostics) = self.get_runtime_diagnostics else {
            return Err(
                "runtime diagnostics require the native host ABI v4 diagnostics extension".into(),
            );
        };
        native_json_call(&self.host, "runtime diagnostics", |out| {
            let status = unsafe { get_runtime_diagnostics(out) };
            codec_status(&self.host, status)
        })
    }

    /// Creates a new independent scope stack.
    pub fn create_scope_stack(&self) -> Result<ScopeStack<'_>> {
        create_scope_stack(&self.host)
    }

    /// Captures the current thread-local scope-stack binding.
    pub fn capture_scope_stack_thread(&self) -> Result<ScopeStackBinding<'_>> {
        capture_scope_stack_thread(&self.host)
    }

    /// Returns whether the current context has an explicitly active scope stack.
    pub fn scope_stack_active(&self) -> bool {
        unsafe { (self.host.scope_stack_active)() }
    }

    /// Binds `stack` to the current OS thread until the returned guard is dropped.
    pub fn bind_scope_stack_thread<'a>(
        &'a self,
        stack: &'a ScopeStack<'a>,
    ) -> Result<ThreadScopeStackGuard<'a>> {
        let previous = self.capture_scope_stack_thread()?;
        let status = stack.set_thread();
        if status == NemoRelayStatus::Ok {
            Ok(ThreadScopeStackGuard {
                previous: Some(previous),
            })
        } else {
            let _ = previous.restore();
            Err(format!("scope_stack_set_thread failed: {status:?}"))
        }
    }
}

impl From<ScopeType> for NemoRelayNativeScopeType {
    fn from(value: ScopeType) -> Self {
        match value {
            ScopeType::Agent => Self::Agent,
            ScopeType::Function => Self::Function,
            ScopeType::Tool => Self::Tool,
            ScopeType::Llm => Self::Llm,
            ScopeType::Retriever => Self::Retriever,
            ScopeType::Embedder => Self::Embedder,
            ScopeType::Reranker => Self::Reranker,
            ScopeType::Guardrail => Self::Guardrail,
            ScopeType::Evaluator => Self::Evaluator,
            ScopeType::Custom => Self::Custom,
            ScopeType::Unknown => Self::Unknown,
        }
    }
}

/// RAII guard for a host scope opened by [`PluginRuntime::scope`].
///
/// A guard may move between threads only while its scope stack is bound on the
/// destination thread. Async middleware restores that binding around each poll;
/// tasks created with `tokio::spawn` do not inherit it and must not own a guard.
pub struct ScopeGuard<'a> {
    runtime: &'a PluginRuntime,
    handle: Option<ScopeHandle<'a>>,
}
unsafe impl Send for ScopeGuard<'_> {}

impl<'a> ScopeGuard<'a> {
    /// Returns the active scope handle.
    pub fn handle(&self) -> Option<&ScopeHandle<'a>> {
        self.handle.as_ref()
    }

    /// Pops the scope with optional output and metadata.
    pub fn close(&mut self, output: Option<&Json>, metadata: Option<&Json>) -> Result<()> {
        let Some(handle) = self.handle.as_ref() else {
            return Ok(());
        };
        self.runtime.pop_scope(handle, output, metadata)?;
        self.handle.take();
        Ok(())
    }
}

impl Drop for ScopeGuard<'_> {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = self.runtime.pop_scope(&handle, None, None);
        }
    }
}

/// RAII guard that restores the previous thread-local scope stack on drop.
pub struct ThreadScopeStackGuard<'a> {
    previous: Option<ScopeStackBinding<'a>>,
}

impl ThreadScopeStackGuard<'_> {
    /// Restores the previous thread-local scope stack immediately.
    pub fn restore(mut self) -> Result<()> {
        let Some(previous) = self.previous.take() else {
            return Ok(());
        };
        let status = previous.restore();
        if status == NemoRelayStatus::Ok {
            Ok(())
        } else {
            Err(format!("scope_stack_restore_thread failed: {status:?}"))
        }
    }
}

impl Drop for ThreadScopeStackGuard<'_> {
    fn drop(&mut self) {
        if let Some(previous) = self.previous.take() {
            let _ = previous.restore();
        }
    }
}

/// Host- or plugin-owned stream returned across the native LLM stream ABI.
pub struct LlmStream {
    host: NemoRelayNativeHostApiV1,
    raw: NemoRelayNativeLlmStreamV1,
    finished: bool,
}

// The host ABI table is Send, and stream ownership is exclusive through this wrapper.
unsafe impl Send for LlmStream {}

impl LlmStream {
    /// Creates a typed stream wrapper from a raw stream table.
    ///
    /// # Safety
    /// `raw` must contain callbacks and `user_data` produced by the same host
    /// and must not be used again after it is moved into this wrapper.
    pub unsafe fn from_raw(
        host: &NemoRelayNativeHostApiV1,
        mut raw: NemoRelayNativeLlmStreamV1,
    ) -> Result<Self> {
        let expected_size = std::mem::size_of::<NemoRelayNativeLlmStreamV1>();
        if raw.struct_size != expected_size {
            if raw.struct_size >= expected_size {
                unsafe { drop_raw_llm_stream(&mut raw) };
            }
            return Err(format!(
                "unsupported LLM stream struct size: {}",
                raw.struct_size
            ));
        }
        if raw.next.is_none() {
            unsafe { drop_raw_llm_stream(&mut raw) };
            return Err("LLM stream next callback was null".into());
        }
        Ok(Self {
            host: *host,
            raw,
            finished: false,
        })
    }

    /// Polls the next stream chunk.
    pub fn next_chunk(&mut self) -> Result<Option<Json>> {
        if self.finished {
            return Ok(None);
        }
        let next = self
            .raw
            .next
            .expect("LLM stream next callback is validated on construction");
        let mut out = ptr::null_mut();
        let status = unsafe { next(self.raw.user_data, &mut out) };
        match status {
            NemoRelayStatus::Ok => {
                if out.is_null() {
                    self.finished = true;
                    return Err("LLM stream returned null chunk".into());
                }
                let result = read_json_value(&self.host, out, "LLM stream chunk");
                unsafe { (self.host.string_free)(out) };
                match result {
                    Ok(chunk) => Ok(Some(chunk)),
                    Err(status) => {
                        self.finished = true;
                        Err(format!("LLM stream returned invalid JSON: {status:?}"))
                    }
                }
            }
            NemoRelayStatus::StreamEnd => {
                if !out.is_null() {
                    unsafe { (self.host.string_free)(out) };
                }
                self.finished = true;
                Ok(None)
            }
            other => {
                if !out.is_null() {
                    unsafe { (self.host.string_free)(out) };
                }
                self.finished = true;
                Err(format!("LLM stream failed: {other:?}"))
            }
        }
    }

    /// Cancels the stream if it has not reached end-of-stream.
    pub fn cancel(&mut self) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        if let Some(cancel) = self.raw.cancel {
            let status = unsafe { cancel(self.raw.user_data) };
            if status != NemoRelayStatus::Ok {
                return Err(format!("LLM stream cancel failed: {status:?}"));
            }
        }
        self.finished = true;
        Ok(())
    }
}

impl Iterator for LlmStream {
    type Item = Result<Json>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_chunk() {
            Ok(Some(chunk)) => Some(Ok(chunk)),
            Ok(None) => None,
            Err(message) => Some(Err(message)),
        }
    }
}

unsafe fn drop_raw_llm_stream(raw: &mut NemoRelayNativeLlmStreamV1) {
    if let Some(drop_fn) = raw.drop.take() {
        unsafe { drop_fn(raw.user_data) };
    }
    raw.user_data = ptr::null_mut();
}

impl Drop for LlmStream {
    fn drop(&mut self) {
        if !self.finished {
            if let Some(cancel) = self.raw.cancel {
                let _ = unsafe { cancel(self.raw.user_data) };
            }
            self.finished = true;
        }
        unsafe { drop_raw_llm_stream(&mut self.raw) };
    }
}

/// Host-owned scope handle returned by native scope APIs.
pub struct ScopeHandle<'a> {
    host: &'a NemoRelayNativeHostApiV1,
    ptr: *mut NemoRelayNativeScopeHandle,
}
unsafe impl Send for ScopeHandle<'_> {}

impl<'a> ScopeHandle<'a> {
    /// Returns the raw ABI pointer.
    pub fn as_ptr(&self) -> *const NemoRelayNativeScopeHandle {
        self.ptr
    }
}

impl Drop for ScopeHandle<'_> {
    fn drop(&mut self) {
        unsafe { (self.host.scope_handle_free)(self.ptr) };
    }
}

/// Host-owned isolated scope stack returned by native scope-stack APIs.
pub struct ScopeStack<'a> {
    host: &'a NemoRelayNativeHostApiV1,
    ptr: *mut NemoRelayNativeScopeStack,
}
unsafe impl Send for ScopeStack<'_> {}

impl<'a> ScopeStack<'a> {
    /// Returns the raw ABI pointer.
    pub fn as_ptr(&self) -> *const NemoRelayNativeScopeStack {
        self.ptr
    }

    /// Binds this stack to the current executor thread.
    ///
    /// Prefer [`PluginRuntime::bind_scope_stack_thread`] for synchronous code.
    /// Async middleware may pair this with a captured binding across an await.
    /// Capture the previous binding first and restore it after the future completes.
    pub fn set_thread(&self) -> NemoRelayStatus {
        unsafe { (self.host.scope_stack_set_thread)(self.ptr) }
    }

    /// Executes `f` while this stack is visible to host runtime APIs.
    pub fn with_current<F>(&self, f: F) -> Result<()>
    where
        F: FnOnce() -> Result<()>,
    {
        struct State<F> {
            f: Option<F>,
            error: Option<String>,
        }

        unsafe extern "C" fn trampoline<F>(user_data: *mut c_void) -> NemoRelayStatus
        where
            F: FnOnce() -> Result<()>,
        {
            if user_data.is_null() {
                return NemoRelayStatus::NullPointer;
            }
            let state = unsafe { &mut *(user_data as *mut State<F>) };
            let result = catch_unwind(AssertUnwindSafe(|| {
                let Some(f) = state.f.take() else {
                    return Err("scope-stack callback was already consumed".to_string());
                };
                f()
            }));
            match result {
                Ok(Ok(())) => NemoRelayStatus::Ok,
                Ok(Err(message)) => {
                    state.error = Some(message);
                    NemoRelayStatus::Internal
                }
                Err(_) => {
                    state.error = Some("scope-stack callback panicked".into());
                    NemoRelayStatus::Internal
                }
            }
        }

        let mut state = State {
            f: Some(f),
            error: None,
        };
        let status = unsafe {
            (self.host.scope_stack_with_current)(
                self.ptr,
                trampoline::<F>,
                (&mut state as *mut State<_>).cast(),
            )
        };
        if status == NemoRelayStatus::Ok {
            Ok(())
        } else {
            Err(state
                .error
                .unwrap_or_else(|| format!("scope_stack_with_current failed: {status:?}")))
        }
    }
}

impl Drop for ScopeStack<'_> {
    fn drop(&mut self) {
        unsafe { (self.host.scope_stack_free)(self.ptr) };
    }
}

/// Captured thread-local scope-stack binding.
pub struct ScopeStackBinding<'a> {
    host: &'a NemoRelayNativeHostApiV1,
    ptr: *mut NemoRelayNativeScopeStackBinding,
}
unsafe impl Send for ScopeStackBinding<'_> {}

impl<'a> ScopeStackBinding<'a> {
    /// Restores and consumes this binding.
    pub fn restore(mut self) -> NemoRelayStatus {
        let ptr = std::mem::replace(&mut self.ptr, ptr::null_mut());
        unsafe { (self.host.scope_stack_restore_thread)(ptr) }
    }
}

impl Drop for ScopeStackBinding<'_> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { (self.host.scope_stack_binding_free)(self.ptr) };
        }
    }
}

/// Retrieves the current scope handle.
pub fn current_scope(host: &NemoRelayNativeHostApiV1) -> Result<ScopeHandle<'_>> {
    let mut out = ptr::null_mut();
    let status = unsafe { (host.scope_get_current)(&mut out) };
    if status == NemoRelayStatus::Ok && !out.is_null() {
        Ok(ScopeHandle { host, ptr: out })
    } else {
        Err(format!("scope_get_current failed: {status:?}"))
    }
}

/// Pushes a scope and emits its start event.
pub fn push_scope<'a>(
    host: &'a NemoRelayNativeHostApiV1,
    name: &str,
    scope_type: NemoRelayNativeScopeType,
    data: Option<&Json>,
    metadata: Option<&Json>,
    input: Option<&Json>,
) -> Result<ScopeHandle<'a>> {
    let name =
        HostString::new(host, name).ok_or_else(|| "failed to allocate scope name".to_string())?;
    let data = OptionalHostJson::new(host, data)?;
    let metadata = OptionalHostJson::new(host, metadata)?;
    let input = OptionalHostJson::new(host, input)?;
    let mut out = ptr::null_mut();
    let status = unsafe {
        (host.scope_push)(
            name.as_ptr(),
            scope_type,
            ptr::null(),
            0,
            data.as_ptr(),
            metadata.as_ptr(),
            input.as_ptr(),
            ptr::null(),
            &mut out,
        )
    };
    if status == NemoRelayStatus::Ok && !out.is_null() {
        Ok(ScopeHandle { host, ptr: out })
    } else {
        Err(format!("scope_push failed: {status:?}"))
    }
}

/// Pops a scope and emits its end event.
pub fn pop_scope(
    host: &NemoRelayNativeHostApiV1,
    handle: &ScopeHandle<'_>,
    output: Option<&Json>,
    metadata: Option<&Json>,
) -> Result<()> {
    let output = OptionalHostJson::new(host, output)?;
    let metadata = OptionalHostJson::new(host, metadata)?;
    let status = unsafe {
        (host.scope_pop)(
            handle.as_ptr(),
            output.as_ptr(),
            metadata.as_ptr(),
            ptr::null(),
        )
    };
    if status == NemoRelayStatus::Ok {
        Ok(())
    } else {
        Err(format!("scope_pop failed: {status:?}"))
    }
}

/// Emits a mark event under the current scope.
pub fn emit_mark(
    host: &NemoRelayNativeHostApiV1,
    name: &str,
    data: Option<&Json>,
    metadata: Option<&Json>,
) -> Result<()> {
    let name =
        HostString::new(host, name).ok_or_else(|| "failed to allocate mark name".to_string())?;
    let data = OptionalHostJson::new(host, data)?;
    let metadata = OptionalHostJson::new(host, metadata)?;
    let status = unsafe {
        (host.emit_mark)(
            name.as_ptr(),
            ptr::null(),
            data.as_ptr(),
            metadata.as_ptr(),
            ptr::null(),
        )
    };
    if status == NemoRelayStatus::Ok {
        Ok(())
    } else {
        Err(format!("emit_mark failed: {status:?}"))
    }
}

#[allow(clippy::too_many_arguments)] // Mirrors the append-only native ABI function.
fn emit_mark_v2_call(
    host: &NemoRelayNativeHostApiV1,
    emit_mark_v2: NemoRelayNativeEmitMarkV2Fn,
    name: &str,
    data: Option<&Json>,
    metadata: Option<&Json>,
    data_schema: Option<&DataSchema>,
    severity: Option<LogSeverity>,
) -> Result<()> {
    let name =
        HostString::new(host, name).ok_or_else(|| "failed to allocate mark name".to_string())?;
    let data = OptionalHostJson::new(host, data)?;
    let metadata = OptionalHostJson::new(host, metadata)?;
    let data_schema = data_schema
        .map(|value| {
            HostString::from_json(host, value)
                .ok_or_else(|| "failed to serialize mark data schema".to_string())
        })
        .transpose()?;
    let severity = severity
        .map(|value| {
            serde_json::to_value(value)
                .map_err(|err| format!("failed to serialize mark severity: {err}"))
                .and_then(|value| {
                    value
                        .as_str()
                        .ok_or_else(|| "mark severity did not serialize as a string".to_string())
                        .and_then(|value| {
                            HostString::new(host, value)
                                .ok_or_else(|| "failed to allocate mark severity".to_string())
                        })
                })
        })
        .transpose()?;
    let status = unsafe {
        emit_mark_v2(
            name.as_ptr(),
            ptr::null(),
            data.as_ptr(),
            metadata.as_ptr(),
            data_schema
                .as_ref()
                .map(HostString::as_ptr)
                .unwrap_or(ptr::null()),
            severity
                .as_ref()
                .map(HostString::as_ptr)
                .unwrap_or(ptr::null()),
            ptr::null(),
        )
    };
    if status == NemoRelayStatus::Ok {
        Ok(())
    } else {
        Err(format!("emit_mark_v2 failed: {status:?}"))
    }
}

/// Creates a new independent scope stack.
pub fn create_scope_stack(host: &NemoRelayNativeHostApiV1) -> Result<ScopeStack<'_>> {
    let mut out = ptr::null_mut();
    let status = unsafe { (host.scope_stack_create)(&mut out) };
    if status == NemoRelayStatus::Ok && !out.is_null() {
        Ok(ScopeStack { host, ptr: out })
    } else {
        Err(format!("scope_stack_create failed: {status:?}"))
    }
}

/// Captures the current thread-local scope-stack binding.
pub fn capture_scope_stack_thread(
    host: &NemoRelayNativeHostApiV1,
) -> Result<ScopeStackBinding<'_>> {
    let mut out = ptr::null_mut();
    let status = unsafe { (host.scope_stack_capture_thread)(&mut out) };
    if status == NemoRelayStatus::Ok && !out.is_null() {
        Ok(ScopeStackBinding { host, ptr: out })
    } else {
        Err(format!("scope_stack_capture_thread failed: {status:?}"))
    }
}

/// Trait implemented by Rust native plugins.
pub trait NativePlugin: Send + 'static {
    /// Returns the stable plugin kind.
    fn plugin_kind(&self) -> &str;

    /// Returns whether the plugin allows multiple configured components.
    fn allows_multiple_components(&self) -> bool {
        true
    }

    /// Configures the SDK-owned Tokio executor used by typed middleware.
    ///
    /// This supplies the plugin-wide default. Relay applies an optional
    /// component-local `[plugins.dynamic.config.executor]` override when it
    /// registers each component.
    fn executor_config(&self) -> NativeExecutorConfig {
        NativeExecutorConfig::default()
    }

    /// Resolves the executor configuration for one component registration.
    ///
    /// Override this only when the plugin needs custom component configuration
    /// rules. The default recognizes `executor.worker_threads` and validates
    /// that it is a positive integer.
    fn executor_config_for_component(
        &self,
        plugin_config: &Map<String, Json>,
    ) -> Result<NativeExecutorConfig> {
        self.executor_config().with_component_config(plugin_config)
    }

    /// Validates one component-local JSON config object.
    fn validate(&self, plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        self.executor_config_for_component(plugin_config)
            .err()
            .map(|message| ConfigDiagnostic {
                level: DiagnosticLevel::Error,
                code: "native_executor_config.invalid".into(),
                component: None,
                field: Some("executor.worker_threads".into()),
                message,
            })
            .into_iter()
            .collect()
    }

    /// Registers runtime behavior through the component-scoped plugin context.
    fn register(
        &mut self,
        plugin_config: &Map<String, Json>,
        ctx: &mut PluginContext<'_>,
    ) -> Result<()>;
}

/// Borrowed safe wrapper around a host plugin registration context.
pub struct PluginContext<'a> {
    host: &'a NemoRelayNativeHostApiV1,
    raw: *mut NemoRelayNativePluginContext,
    executor: Arc<async_sdk::NativeExecutor>,
}

#[allow(clippy::not_unsafe_ptr_arg_deref)]
impl<'a> PluginContext<'a> {
    /// Creates a plugin context wrapper from raw ABI parts.
    ///
    /// # Safety
    /// `host` and `raw` must remain valid for the lifetime of this wrapper.
    pub unsafe fn from_raw(
        host: &'a NemoRelayNativeHostApiV1,
        raw: *mut NemoRelayNativePluginContext,
    ) -> Self {
        Self {
            host,
            raw,
            executor: async_sdk::NativeExecutor::new(NativeExecutorConfig::default(), "standalone"),
        }
    }

    unsafe fn from_raw_with_executor(
        host: &'a NemoRelayNativeHostApiV1,
        raw: *mut NemoRelayNativePluginContext,
        executor: Arc<async_sdk::NativeExecutor>,
    ) -> Self {
        Self {
            host,
            raw,
            executor,
        }
    }

    /// Returns the host ABI table backing this registration context.
    pub fn host_api(&self) -> &'a NemoRelayNativeHostApiV1 {
        self.host
    }

    /// Returns a cloneable high-level runtime handle.
    pub fn runtime(&self) -> PluginRuntime {
        PluginRuntime::from_context(self.host, self.raw)
    }

    /// Declares a callback conditional middleware guardrail for activation.
    ///
    /// Return `Some(reason)` to disable the matching target or `None` to leave
    /// it enabled.
    pub fn register_conditional_middleware_guardrail<F>(
        &mut self,
        name: &str,
        kinds: &std::collections::BTreeSet<RuntimeRegistrationKind>,
        registration_name: &str,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(&std::collections::BTreeSet<RuntimeRegistrationKind>, &str) -> Option<String>
            + Send
            + Sync
            + 'static,
    {
        if self.host.abi_version < NEMO_RELAY_NATIVE_ABI_VERSION
            || self.host.struct_size < std::mem::size_of::<NemoRelayNativeHostApiV4>()
        {
            return Err("host does not support conditional middleware guardrails".into());
        }
        let v4 = unsafe { &*(self.host as *const _ as *const NemoRelayNativeHostApiV4) };
        let name = HostString::new(self.host, name)
            .ok_or_else(|| "failed to allocate gate name".to_string())?;
        let kinds = HostString::from_json(self.host, kinds)
            .ok_or_else(|| "failed to serialize runtime registration kinds".to_string())?;
        let registration_name = HostString::new(self.host, registration_name)
            .ok_or_else(|| "failed to allocate target name".to_string())?;
        let user_data = typed_callback_user_data(self.host, callback);
        let status = unsafe {
            (v4.plugin_context_register_conditional_middleware_guardrail_callback)(
                self.raw,
                name.as_ptr(),
                kinds.as_ptr(),
                registration_name.as_ptr(),
                typed_conditional_middleware_trampoline::<F>,
                user_data,
                Some(drop_typed_callback::<F>),
            )
        };
        finish_typed_registration(
            self.host,
            status,
            user_data,
            "register conditional middleware guardrail",
        )
    }

    /// Registers a typed event subscriber callback.
    pub fn register_subscriber<F>(&mut self, name: &str, callback: F) -> Result<()>
    where
        F: Fn(&Event) + Send + Sync + 'static,
    {
        let user_data = typed_callback_user_data(self.host, callback);
        let status = unsafe {
            self.register_subscriber_raw(
                name,
                typed_subscriber_trampoline::<F>,
                user_data,
                Some(drop_typed_callback::<F>),
            )
        };
        finish_typed_registration(self.host, status, user_data, "subscriber")
    }

    /// Registers a raw event subscriber callback.
    ///
    /// # Safety
    /// `cb`, `user_data`, and `free_fn` must remain valid for every host
    /// callback invocation until the host deregisters the callback or calls
    /// `free_fn`. `free_fn` must match the allocation behind `user_data`.
    pub unsafe fn register_subscriber_raw(
        &mut self,
        name: &str,
        cb: NemoRelayNativeEventSubscriberCb,
        user_data: *mut c_void,
        free_fn: NemoRelayNativeFreeFn,
    ) -> NemoRelayStatus {
        self.with_name_and_callback(name, user_data, free_fn, |host, name| unsafe {
            (host.plugin_context_register_subscriber)(self.raw, name, cb, user_data, free_fn)
        })
    }

    /// Registers a raw mark event sanitizer callback.
    ///
    /// # Safety
    /// `cb`, `user_data`, and `free_fn` must remain valid for every host
    /// callback invocation until the host deregisters the callback or calls
    /// `free_fn`. `free_fn` must match the allocation behind `user_data`.
    pub unsafe fn register_mark_sanitize_guardrail_raw(
        &mut self,
        name: &str,
        priority: i32,
        cb: NemoRelayNativeEventSanitizeCb,
        user_data: *mut c_void,
        free_fn: NemoRelayNativeFreeFn,
    ) -> NemoRelayStatus {
        self.with_name_and_callback(name, user_data, free_fn, |host, name| unsafe {
            (host.plugin_context_register_mark_sanitize_guardrail)(
                self.raw, name, priority, cb, user_data, free_fn,
            )
        })
    }

    /// Registers a raw scope-start event sanitizer callback.
    ///
    /// # Safety
    /// `cb`, `user_data`, and `free_fn` must remain valid for every host
    /// callback invocation until the host deregisters the callback or calls
    /// `free_fn`. `free_fn` must match the allocation behind `user_data`.
    pub unsafe fn register_scope_sanitize_start_guardrail_raw(
        &mut self,
        name: &str,
        priority: i32,
        cb: NemoRelayNativeEventSanitizeCb,
        user_data: *mut c_void,
        free_fn: NemoRelayNativeFreeFn,
    ) -> NemoRelayStatus {
        self.with_name_and_callback(name, user_data, free_fn, |host, name| unsafe {
            (host.plugin_context_register_scope_sanitize_start_guardrail)(
                self.raw, name, priority, cb, user_data, free_fn,
            )
        })
    }

    /// Registers a raw scope-end event sanitizer callback.
    ///
    /// # Safety
    /// `cb`, `user_data`, and `free_fn` must remain valid for every host
    /// callback invocation until the host deregisters the callback or calls
    /// `free_fn`. `free_fn` must match the allocation behind `user_data`.
    pub unsafe fn register_scope_sanitize_end_guardrail_raw(
        &mut self,
        name: &str,
        priority: i32,
        cb: NemoRelayNativeEventSanitizeCb,
        user_data: *mut c_void,
        free_fn: NemoRelayNativeFreeFn,
    ) -> NemoRelayStatus {
        self.with_name_and_callback(name, user_data, free_fn, |host, name| unsafe {
            (host.plugin_context_register_scope_sanitize_end_guardrail)(
                self.raw, name, priority, cb, user_data, free_fn,
            )
        })
    }

    /// Registers a raw tool sanitize-request guardrail callback.
    ///
    /// # Safety
    /// `cb`, `user_data`, and `free_fn` must remain valid for every host
    /// callback invocation until the host deregisters the callback or calls
    /// `free_fn`. `free_fn` must match the allocation behind `user_data`.
    pub unsafe fn register_tool_sanitize_request_guardrail_raw(
        &mut self,
        name: &str,
        priority: i32,
        cb: NemoRelayNativeToolJsonCb,
        user_data: *mut c_void,
        free_fn: NemoRelayNativeFreeFn,
    ) -> NemoRelayStatus {
        self.with_name_and_callback(name, user_data, free_fn, |host, name| unsafe {
            (host.plugin_context_register_tool_sanitize_request_guardrail)(
                self.raw, name, priority, cb, user_data, free_fn,
            )
        })
    }

    /// Registers a raw tool sanitize-response guardrail callback.
    ///
    /// # Safety
    /// `cb`, `user_data`, and `free_fn` must remain valid for every host
    /// callback invocation until the host deregisters the callback or calls
    /// `free_fn`. `free_fn` must match the allocation behind `user_data`.
    pub unsafe fn register_tool_sanitize_response_guardrail_raw(
        &mut self,
        name: &str,
        priority: i32,
        cb: NemoRelayNativeToolJsonCb,
        user_data: *mut c_void,
        free_fn: NemoRelayNativeFreeFn,
    ) -> NemoRelayStatus {
        self.with_name_and_callback(name, user_data, free_fn, |host, name| unsafe {
            (host.plugin_context_register_tool_sanitize_response_guardrail)(
                self.raw, name, priority, cb, user_data, free_fn,
            )
        })
    }

    /// Registers a raw tool conditional-execution guardrail callback.
    ///
    /// # Safety
    /// `cb`, `user_data`, and `free_fn` must remain valid for every host
    /// callback invocation until the host deregisters the callback or calls
    /// `free_fn`. `free_fn` must match the allocation behind `user_data`.
    pub unsafe fn register_tool_conditional_execution_guardrail_raw(
        &mut self,
        name: &str,
        priority: i32,
        cb: NemoRelayNativeToolConditionalCb,
        user_data: *mut c_void,
        free_fn: NemoRelayNativeFreeFn,
    ) -> NemoRelayStatus {
        self.with_name_and_callback(name, user_data, free_fn, |host, name| unsafe {
            (host.plugin_context_register_tool_conditional_execution_guardrail)(
                self.raw, name, priority, cb, user_data, free_fn,
            )
        })
    }

    /// Registers a raw tool request intercept callback.
    ///
    /// # Safety
    /// `cb`, `user_data`, and `free_fn` must remain valid for every host
    /// callback invocation until the host deregisters the callback or calls
    /// `free_fn`. `free_fn` must match the allocation behind `user_data`.
    pub unsafe fn register_tool_request_intercept_raw(
        &mut self,
        name: &str,
        priority: i32,
        break_chain: bool,
        cb: NemoRelayNativeToolJsonCb,
        user_data: *mut c_void,
        free_fn: NemoRelayNativeFreeFn,
    ) -> NemoRelayStatus {
        self.with_name_and_callback(name, user_data, free_fn, |host, name| unsafe {
            (host.plugin_context_register_tool_request_intercept)(
                self.raw,
                name,
                priority,
                break_chain,
                cb,
                user_data,
                free_fn,
            )
        })
    }

    /// Registers a raw tool execution intercept callback.
    ///
    /// # Safety
    /// `cb`, `user_data`, and `free_fn` must remain valid for every host
    /// callback invocation until the host deregisters the callback or calls
    /// `free_fn`. `free_fn` must match the allocation behind `user_data`.
    pub unsafe fn register_tool_execution_intercept_raw(
        &mut self,
        name: &str,
        priority: i32,
        cb: NemoRelayNativeToolExecutionCb,
        user_data: *mut c_void,
        free_fn: NemoRelayNativeFreeFn,
    ) -> NemoRelayStatus {
        self.with_name_and_callback(name, user_data, free_fn, |host, name| unsafe {
            (host.plugin_context_register_tool_execution_intercept)(
                self.raw, name, priority, cb, user_data, free_fn,
            )
        })
    }

    /// Registers a raw LLM sanitize-request guardrail callback.
    ///
    /// # Safety
    /// `cb`, `user_data`, and `free_fn` must remain valid for every host
    /// callback invocation until the host deregisters the callback or calls
    /// `free_fn`. `free_fn` must match the allocation behind `user_data`.
    pub unsafe fn register_llm_sanitize_request_guardrail_raw(
        &mut self,
        name: &str,
        priority: i32,
        cb: NemoRelayNativeLlmSanitizeRequestCb,
        user_data: *mut c_void,
        free_fn: NemoRelayNativeFreeFn,
    ) -> NemoRelayStatus {
        self.with_name_and_callback(name, user_data, free_fn, |host, name| unsafe {
            (host.plugin_context_register_llm_sanitize_request_guardrail)(
                self.raw, name, priority, cb, user_data, free_fn,
            )
        })
    }

    /// Registers a raw LLM sanitize-response guardrail callback.
    ///
    /// # Safety
    /// `cb`, `user_data`, and `free_fn` must remain valid for every host
    /// callback invocation until the host deregisters the callback or calls
    /// `free_fn`. `free_fn` must match the allocation behind `user_data`.
    pub unsafe fn register_llm_sanitize_response_guardrail_raw(
        &mut self,
        name: &str,
        priority: i32,
        cb: NemoRelayNativeLlmSanitizeResponseCb,
        user_data: *mut c_void,
        free_fn: NemoRelayNativeFreeFn,
    ) -> NemoRelayStatus {
        self.with_name_and_callback(name, user_data, free_fn, |host, name| unsafe {
            (host.plugin_context_register_llm_sanitize_response_guardrail)(
                self.raw, name, priority, cb, user_data, free_fn,
            )
        })
    }

    /// Registers a raw LLM conditional-execution guardrail callback.
    ///
    /// # Safety
    /// `cb`, `user_data`, and `free_fn` must remain valid for every host
    /// callback invocation until the host deregisters the callback or calls
    /// `free_fn`. `free_fn` must match the allocation behind `user_data`.
    pub unsafe fn register_llm_conditional_execution_guardrail_raw(
        &mut self,
        name: &str,
        priority: i32,
        cb: NemoRelayNativeLlmConditionalCb,
        user_data: *mut c_void,
        free_fn: NemoRelayNativeFreeFn,
    ) -> NemoRelayStatus {
        self.with_name_and_callback(name, user_data, free_fn, |host, name| unsafe {
            (host.plugin_context_register_llm_conditional_execution_guardrail)(
                self.raw, name, priority, cb, user_data, free_fn,
            )
        })
    }

    /// Registers a raw LLM request intercept callback.
    ///
    /// # Safety
    /// `cb`, `user_data`, and `free_fn` must remain valid for every host
    /// callback invocation until the host deregisters the callback or calls
    /// `free_fn`. `free_fn` must match the allocation behind `user_data`.
    pub unsafe fn register_llm_request_intercept_raw(
        &mut self,
        name: &str,
        priority: i32,
        break_chain: bool,
        cb: NemoRelayNativeLlmRequestInterceptCb,
        user_data: *mut c_void,
        free_fn: NemoRelayNativeFreeFn,
    ) -> NemoRelayStatus {
        self.with_name_and_callback(name, user_data, free_fn, |host, name| unsafe {
            (host.plugin_context_register_llm_request_intercept)(
                self.raw,
                name,
                priority,
                break_chain,
                cb,
                user_data,
                free_fn,
            )
        })
    }

    /// Registers a raw LLM execution intercept callback.
    ///
    /// # Safety
    /// `cb`, `user_data`, and `free_fn` must remain valid for every host
    /// callback invocation until the host deregisters the callback or calls
    /// `free_fn`. `free_fn` must match the allocation behind `user_data`.
    pub unsafe fn register_llm_execution_intercept_raw(
        &mut self,
        name: &str,
        priority: i32,
        cb: NemoRelayNativeLlmExecutionCb,
        user_data: *mut c_void,
        free_fn: NemoRelayNativeFreeFn,
    ) -> NemoRelayStatus {
        self.with_name_and_callback(name, user_data, free_fn, |host, name| unsafe {
            (host.plugin_context_register_llm_execution_intercept)(
                self.raw, name, priority, cb, user_data, free_fn,
            )
        })
    }

    /// Registers a raw LLM stream execution intercept callback.
    ///
    /// # Safety
    /// `cb`, `user_data`, and `free_fn` must remain valid for every host
    /// callback invocation until the host deregisters the callback or calls
    /// `free_fn`. `free_fn` must match the allocation behind `user_data`.
    pub unsafe fn register_llm_stream_execution_intercept_raw(
        &mut self,
        name: &str,
        priority: i32,
        cb: NemoRelayNativeLlmStreamExecutionCb,
        user_data: *mut c_void,
        free_fn: NemoRelayNativeFreeFn,
    ) -> NemoRelayStatus {
        self.with_name_and_callback(name, user_data, free_fn, |host, name| unsafe {
            (host.plugin_context_register_llm_stream_execution_intercept)(
                self.raw, name, priority, cb, user_data, free_fn,
            )
        })
    }

    /// Registers completion-based asynchronous middleware through the ABI-v3
    /// extension table.
    ///
    /// Plugins built against older hosts receive [`NemoRelayStatus::InvalidArg`]
    /// instead of attempting to read beyond the legacy host table.
    ///
    /// # Safety
    /// `cb`, `user_data`, and `free_fn` must remain valid until the host
    /// deregisters the callback or invokes `free_fn`. This call consumes the
    /// `user_data` ownership even when it rejects the host ABI. A callback returning
    /// `Pending` must settle and release its completion/next references.
    /// [`NemoRelayNativeAsyncMiddlewareKind::LlmStreamExecutionIntercept`] is
    /// rejected; use [`Self::register_async_stream_middleware_raw`] instead.
    #[allow(clippy::too_many_arguments)] // Mirrors the native C ABI registration callback.
    pub unsafe fn register_async_middleware_raw(
        &mut self,
        kind: NemoRelayNativeAsyncMiddlewareKind,
        name: &str,
        priority: i32,
        break_chain: bool,
        cb: NemoRelayNativeAsyncMiddlewareCb,
        user_data: *mut c_void,
        free_fn: NemoRelayNativeFreeFn,
    ) -> NemoRelayStatus {
        if self.host.abi_version < NEMO_RELAY_NATIVE_ABI_VERSION_ASYNC_MIDDLEWARE
            || self.host.struct_size < std::mem::size_of::<NemoRelayNativeHostApiV3>()
        {
            if let Some(free_fn) = free_fn {
                unsafe { free_fn(user_data) };
            }
            return NemoRelayStatus::InvalidArg;
        }
        let host = unsafe { &*(self.host as *const _ as *const NemoRelayNativeHostApiV3) };
        self.with_name_and_callback(name, user_data, free_fn, |_, name| unsafe {
            (host.plugin_context_register_async_middleware)(
                self.raw,
                kind as u32,
                name,
                priority,
                break_chain,
                cb,
                user_data,
                free_fn,
            )
        })
    }

    /// Registers an incremental completion-based LLM stream intercept.
    ///
    /// # Safety
    /// The callback and user data must remain valid until deregistration or
    /// `free_fn`; this call consumes `user_data` even when it rejects the host
    /// ABI. Callback-owned `next` and `stream` handles must each be
    /// released exactly once. Stream pushes and rejection are nonblocking:
    /// Retry only [`NemoRelayStatus::Backpressured`] operations. The output
    /// stream owns the callback lifetime. `next` may be invoked
    /// repeatedly or concurrently until that stream settles; Relay then
    /// rejects or cancels unfinished and later calls.
    pub unsafe fn register_async_stream_middleware_raw(
        &mut self,
        name: &str,
        priority: i32,
        cb: NemoRelayNativeAsyncStreamMiddlewareCb,
        user_data: *mut c_void,
        free_fn: NemoRelayNativeFreeFn,
    ) -> NemoRelayStatus {
        if self.host.abi_version < NEMO_RELAY_NATIVE_ABI_VERSION_ASYNC_MIDDLEWARE
            || self.host.struct_size < std::mem::size_of::<NemoRelayNativeHostApiV3>()
        {
            if let Some(free_fn) = free_fn {
                unsafe { free_fn(user_data) };
            }
            return NemoRelayStatus::InvalidArg;
        }
        let host = unsafe { &*(self.host as *const _ as *const NemoRelayNativeHostApiV3) };
        self.with_name_and_callback(name, user_data, free_fn, |_, name| unsafe {
            (host.plugin_context_register_async_stream_middleware)(
                self.raw, name, priority, cb, user_data, free_fn,
            )
        })
    }

    fn with_name_and_callback(
        &self,
        name: &str,
        user_data: *mut c_void,
        free_fn: NemoRelayNativeFreeFn,
        f: impl FnOnce(&NemoRelayNativeHostApiV1, *const NemoRelayNativeString) -> NemoRelayStatus,
    ) -> NemoRelayStatus {
        let name = match HostString::try_new(self.host, name) {
            Ok(name) => name,
            Err(status) => {
                if let Some(free_fn) = free_fn {
                    unsafe { free_fn(user_data) };
                }
                return status;
            }
        };
        f(self.host, name.as_ptr())
    }
}

struct TypedCallback<F> {
    host: NemoRelayNativeHostApiV1,
    callback: F,
}

fn typed_callback_user_data<F>(host: &NemoRelayNativeHostApiV1, callback: F) -> *mut c_void {
    Box::into_raw(Box::new(TypedCallback {
        host: *host,
        callback,
    })) as *mut c_void
}

unsafe extern "C" fn drop_typed_callback<F>(user_data: *mut c_void) {
    if !user_data.is_null() {
        let callback = unsafe { Box::from_raw(user_data as *mut TypedCallback<F>) };
        let host = callback.host;
        if catch_unwind(AssertUnwindSafe(|| drop(callback))).is_err() {
            set_last_error(&host, "native plugin typed callback state drop panicked");
        }
    }
}

fn finish_typed_registration(
    host: &NemoRelayNativeHostApiV1,
    status: NemoRelayStatus,
    user_data: *mut c_void,
    label: &str,
) -> Result<()> {
    let _ = user_data;
    if status == NemoRelayStatus::Ok {
        Ok(())
    } else {
        Err(status_error(host, status, label))
    }
}

fn status_error(host: &NemoRelayNativeHostApiV1, status: NemoRelayStatus, label: &str) -> String {
    debug_assert_ne!(status, NemoRelayStatus::Ok);
    set_last_error(host, &format!("{label} failed: {status:?}"));
    format!("{label} failed: {status:?}")
}

fn status_result(
    host: &NemoRelayNativeHostApiV1,
    status: NemoRelayStatus,
    label: &str,
) -> Result<()> {
    if status == NemoRelayStatus::Ok {
        Ok(())
    } else {
        Err(status_error(host, status, label))
    }
}

fn callback_panic(host: &NemoRelayNativeHostApiV1, label: &str) -> NemoRelayStatus {
    set_last_error(host, &format!("{label} panicked"));
    NemoRelayStatus::Internal
}

unsafe extern "C" fn typed_subscriber_trampoline<F>(
    user_data: *mut c_void,
    event_json: *const NemoRelayNativeString,
) -> NemoRelayStatus
where
    F: Fn(&Event) + Send + Sync + 'static,
{
    if user_data.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    let state = unsafe { &*(user_data as *const TypedCallback<F>) };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let event: Event = read_json_value(&state.host, event_json, "event")?;
        (state.callback)(&event);
        Ok::<_, NemoRelayStatus>(())
    }));
    match result {
        Ok(Ok(())) => NemoRelayStatus::Ok,
        Ok(Err(status)) => status,
        Err(_) => callback_panic(&state.host, "subscriber callback"),
    }
}

unsafe extern "C" fn typed_conditional_middleware_trampoline<F>(
    user_data: *mut c_void,
    kinds_json: *const NemoRelayNativeString,
    registration_name: *const NemoRelayNativeString,
    out_reason: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus
where
    F: Fn(&std::collections::BTreeSet<RuntimeRegistrationKind>, &str) -> Option<String>
        + Send
        + Sync
        + 'static,
{
    if user_data.is_null() || out_reason.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *out_reason = ptr::null_mut() };
    let state = unsafe { &*(user_data as *const TypedCallback<F>) };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let kinds = read_json_value(&state.host, kinds_json, "runtime registration kinds")?;
        let registration_name =
            read_required_host_string(&state.host, registration_name, "registration name")?;
        let Some(reason) = (state.callback)(&kinds, &registration_name) else {
            return Ok(NemoRelayStatus::Ok);
        };
        let status = unsafe { (state.host.string_new)(reason.as_ptr(), reason.len(), out_reason) };
        Ok(status)
    }));
    match result {
        Ok(Ok(status)) => status,
        Ok(Err(status)) => status,
        Err(_) => callback_panic(&state.host, "conditional middleware callback"),
    }
}

struct HostString<'a> {
    host: &'a NemoRelayNativeHostApiV1,
    ptr: *mut NemoRelayNativeString,
}
unsafe impl Send for HostString<'_> {}

impl<'a> HostString<'a> {
    fn try_new(
        host: &'a NemoRelayNativeHostApiV1,
        value: &str,
    ) -> std::result::Result<Self, NemoRelayStatus> {
        let mut out = ptr::null_mut();
        let status = unsafe { (host.string_new)(value.as_ptr(), value.len(), &mut out) };
        if status != NemoRelayStatus::Ok {
            return Err(status);
        }
        if out.is_null() {
            return Err(NemoRelayStatus::Internal);
        }
        Ok(Self { host, ptr: out })
    }

    fn new(host: &'a NemoRelayNativeHostApiV1, value: &str) -> Option<Self> {
        Self::try_new(host, value).ok()
    }

    fn from_json<T: Serialize>(host: &'a NemoRelayNativeHostApiV1, value: &T) -> Option<Self> {
        serde_json::to_string(value)
            .ok()
            .and_then(|json| Self::new(host, &json))
    }

    fn as_ptr(&self) -> *const NemoRelayNativeString {
        self.ptr
    }
}

impl Drop for HostString<'_> {
    fn drop(&mut self) {
        unsafe { (self.host.string_free)(self.ptr) };
    }
}

fn codec_status(host: &NemoRelayNativeHostApiV1, status: NemoRelayStatus) -> Result<()> {
    if status == NemoRelayStatus::Ok {
        Ok(())
    } else {
        Err(status_error(host, status, "LLM codec operation"))
    }
}

fn native_codec_call<T: DeserializeOwned>(
    host: &NemoRelayNativeHostApiV1,
    call: impl FnOnce(*mut *mut NemoRelayNativeString) -> Result<()>,
) -> Result<T> {
    native_json_call(host, "LLM codec operation", call)
}

fn native_json_call<T: DeserializeOwned>(
    host: &NemoRelayNativeHostApiV1,
    operation: &str,
    call: impl FnOnce(*mut *mut NemoRelayNativeString) -> Result<()>,
) -> Result<T> {
    let mut out = ptr::null_mut();
    call(&mut out)?;
    if out.is_null() {
        return Err(format!("{operation} returned null"));
    }
    let out = HostString { host, ptr: out };
    let text = read_host_string(host, out.as_ptr())
        .map_err(|_| format!("{operation} returned invalid UTF-8"))?;
    serde_json::from_str(&text).map_err(|error| format!("invalid {operation} result: {error}"))
}

struct OptionalHostJson<'a>(Option<HostString<'a>>);

impl<'a> OptionalHostJson<'a> {
    fn new(host: &'a NemoRelayNativeHostApiV1, value: Option<&Json>) -> Result<Self> {
        match value {
            Some(value) => HostString::from_json(host, value)
                .map(|value| Self(Some(value)))
                .ok_or_else(|| "failed to allocate JSON host string".into()),
            None => Ok(Self(None)),
        }
    }

    fn as_ptr(&self) -> *const NemoRelayNativeString {
        self.0
            .as_ref()
            .map(HostString::as_ptr)
            .unwrap_or(ptr::null())
    }
}

enum OwnedHostApi {
    V1(NemoRelayNativeHostApiV1),
    V3(NemoRelayNativeHostApiV3),
    V4(NemoRelayNativeHostApiV4),
}

impl OwnedHostApi {
    unsafe fn copy_from(host: &NemoRelayNativeHostApiV1) -> Self {
        if host.abi_version >= NEMO_RELAY_NATIVE_ABI_VERSION
            && host.struct_size >= std::mem::size_of::<NemoRelayNativeHostApiV4>()
        {
            Self::V4(unsafe { *(host as *const _ as *const NemoRelayNativeHostApiV4) })
        } else if host.abi_version >= NEMO_RELAY_NATIVE_ABI_VERSION_ASYNC_MIDDLEWARE
            && host.struct_size >= std::mem::size_of::<NemoRelayNativeHostApiV3>()
        {
            Self::V3(unsafe { *(host as *const _ as *const NemoRelayNativeHostApiV3) })
        } else {
            Self::V1(*host)
        }
    }

    fn v1(&self) -> &NemoRelayNativeHostApiV1 {
        match self {
            Self::V1(host) => host,
            Self::V3(host) => &host.v1,
            Self::V4(host) => &host.v3.v1,
        }
    }
}

struct PluginState<P> {
    host: OwnedHostApi,
    plugin: Mutex<P>,
}

unsafe extern "C" fn drop_plugin_state<P: NativePlugin>(user_data: *mut c_void) {
    if !user_data.is_null() {
        let state = unsafe { Box::from_raw(user_data as *mut PluginState<P>) };
        let host = *state.host.v1();
        if catch_unwind(AssertUnwindSafe(|| drop(state))).is_err() {
            set_last_error(&host, "native plugin state drop panicked");
        }
    }
}

unsafe extern "C" fn validate_trampoline<P: NativePlugin>(
    user_data: *mut c_void,
    plugin_config_json: *const NemoRelayNativeString,
    out_diagnostics_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    if user_data.is_null() || out_diagnostics_json.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *out_diagnostics_json = ptr::null_mut() };
    let state = unsafe { &*(user_data as *const PluginState<P>) };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let host = state.host.v1();
        let config = match read_json_object(host, plugin_config_json) {
            Ok(config) => config,
            Err(status) => return status,
        };
        let plugin = match state.plugin.lock() {
            Ok(plugin) => plugin,
            Err(_) => {
                set_last_error(host, "native plugin state lock poisoned");
                return NemoRelayStatus::Internal;
            }
        };
        let diagnostics = plugin.validate(&config);
        write_json(host, &diagnostics, out_diagnostics_json)
    }));
    result.unwrap_or_else(|_| {
        set_last_error(state.host.v1(), "native plugin validate callback panicked");
        NemoRelayStatus::Internal
    })
}

unsafe extern "C" fn register_trampoline<P: NativePlugin>(
    user_data: *mut c_void,
    plugin_config_json: *const NemoRelayNativeString,
    ctx: *mut NemoRelayNativePluginContext,
) -> NemoRelayStatus {
    if user_data.is_null() || ctx.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    let state = unsafe { &*(user_data as *const PluginState<P>) };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let host = state.host.v1();
        let config = match read_json_object(host, plugin_config_json) {
            Ok(config) => config,
            Err(status) => return status,
        };
        let mut plugin = match state.plugin.lock() {
            Ok(plugin) => plugin,
            Err(_) => {
                set_last_error(host, "native plugin state lock poisoned");
                return NemoRelayStatus::Internal;
            }
        };
        let executor_config = match plugin.executor_config_for_component(&config) {
            Ok(config) => config,
            Err(error) => {
                set_last_error(host, &error);
                return NemoRelayStatus::InvalidArg;
            }
        };
        let mut ctx = unsafe {
            PluginContext::from_raw_with_executor(
                host,
                ctx,
                async_sdk::NativeExecutor::new(executor_config, plugin.plugin_kind()),
            )
        };
        match plugin.register(&config, &mut ctx) {
            Ok(()) => NemoRelayStatus::Ok,
            Err(message) => {
                set_last_error(host, &message);
                NemoRelayStatus::Internal
            }
        }
    }));
    result.unwrap_or_else(|_| {
        set_last_error(state.host.v1(), "native plugin register callback panicked");
        NemoRelayStatus::Internal
    })
}

fn read_json_object(
    host: &NemoRelayNativeHostApiV1,
    value: *const NemoRelayNativeString,
) -> std::result::Result<Map<String, Json>, NemoRelayStatus> {
    let value: Json = read_json_value(host, value, "plugin config")?;
    match value {
        Json::Object(map) => Ok(map),
        _ => {
            set_last_error(host, "plugin config must be a JSON object");
            Err(NemoRelayStatus::InvalidJson)
        }
    }
}

fn read_json_value<T: DeserializeOwned>(
    host: &NemoRelayNativeHostApiV1,
    value: *const NemoRelayNativeString,
    label: &str,
) -> std::result::Result<T, NemoRelayStatus> {
    let text = read_required_host_string(host, value, label)?;
    serde_json::from_str::<T>(&text).map_err(|error| {
        set_last_error(host, &format!("{label} was invalid JSON: {error}"));
        NemoRelayStatus::InvalidJson
    })
}

#[derive(Debug)]
enum HostStringReadError {
    Null,
    InvalidUtf8,
}

fn read_required_host_string(
    host: &NemoRelayNativeHostApiV1,
    value: *const NemoRelayNativeString,
    label: &str,
) -> std::result::Result<String, NemoRelayStatus> {
    match read_host_string(host, value) {
        Ok(value) => Ok(value),
        Err(HostStringReadError::Null) => {
            set_last_error(host, &format!("{label} was null"));
            Err(NemoRelayStatus::NullPointer)
        }
        Err(HostStringReadError::InvalidUtf8) => {
            set_last_error(host, &format!("{label} contained invalid UTF-8"));
            Err(NemoRelayStatus::InvalidUtf8)
        }
    }
}

fn read_host_string(
    host: &NemoRelayNativeHostApiV1,
    value: *const NemoRelayNativeString,
) -> std::result::Result<String, HostStringReadError> {
    if value.is_null() {
        return Err(HostStringReadError::Null);
    }
    let len = unsafe { (host.string_len)(value) };
    let data = unsafe { (host.string_data)(value) };
    if data.is_null() && len > 0 {
        return Err(HostStringReadError::InvalidUtf8);
    }
    let bytes = if len == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(data, len) }
    };
    std::str::from_utf8(bytes)
        .map(str::to_owned)
        .map_err(|_| HostStringReadError::InvalidUtf8)
}

fn take_host_string(
    host: &NemoRelayNativeHostApiV1,
    value: *mut NemoRelayNativeString,
) -> Result<String> {
    let result = read_host_string(host, value)
        .map_err(|error| format!("host returned an invalid string: {error:?}"));
    if !value.is_null() {
        unsafe { (host.string_free)(value) };
    }
    result
}

fn take_host_json<T: DeserializeOwned>(
    host: &NemoRelayNativeHostApiV1,
    value: *mut NemoRelayNativeString,
) -> Result<T> {
    let text = take_host_string(host, value)?;
    serde_json::from_str(&text).map_err(|error| format!("host returned invalid JSON: {error}"))
}

fn write_json<T: Serialize>(
    host: &NemoRelayNativeHostApiV1,
    value: &T,
    out: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    if out.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *out = ptr::null_mut() };
    let json = serde_json::to_value(value).expect("Relay DTOs and serde_json::Value serialize");
    let Some(handle) = HostString::from_json(host, &json) else {
        set_last_error(host, "failed to allocate host string");
        return NemoRelayStatus::Internal;
    };
    unsafe { *out = handle.ptr };
    std::mem::forget(handle);
    NemoRelayStatus::Ok
}

fn set_last_error(host: &NemoRelayNativeHostApiV1, message: &str) {
    if let Some(message) = HostString::new(host, message) {
        unsafe { (host.last_error_set)(message.as_ptr()) };
    }
}

/// Sets a host last-error message from generated entry symbols.
///
/// # Safety
/// `host` must be null or point to a valid [`NemoRelayNativeHostApiV1`].
#[doc(hidden)]
pub unsafe fn __set_last_error_from_entry(host: *const NemoRelayNativeHostApiV1, message: &str) {
    if !host.is_null() {
        set_last_error(unsafe { &*host }, message);
    }
}

/// Initializes a native plugin descriptor for a Rust SDK plugin value.
///
/// # Safety
/// `host` must point to a valid [`NemoRelayNativeHostApiV1`] for the duration
/// of the call, and `out` must point to writable memory for one
/// [`NemoRelayNativePluginV1`] descriptor.
pub unsafe fn export_plugin<P: NativePlugin>(
    host: *const NemoRelayNativeHostApiV1,
    out: *mut NemoRelayNativePluginV1,
    plugin: P,
) -> NemoRelayStatus {
    if host.is_null() || out.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *out = NemoRelayNativePluginV1::default() };
    let host_ref = unsafe { &*host };
    export_plugin_checked(host_ref, out, || plugin)
}

/// Initializes a native plugin descriptor from a constructor callback.
///
/// # Safety
/// `host` must point to a valid [`NemoRelayNativeHostApiV1`] for the duration
/// of the call, and `out` must point to writable memory for one
/// [`NemoRelayNativePluginV1`] descriptor.
#[doc(hidden)]
pub unsafe fn __export_plugin_from_constructor<P, F>(
    host: *const NemoRelayNativeHostApiV1,
    out: *mut NemoRelayNativePluginV1,
    constructor: F,
) -> NemoRelayStatus
where
    P: NativePlugin,
    F: FnOnce() -> P,
{
    if host.is_null() || out.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *out = NemoRelayNativePluginV1::default() };
    let host_ref = unsafe { &*host };
    export_plugin_checked(host_ref, out, constructor)
}

fn export_plugin_checked<P, F>(
    host_ref: &NemoRelayNativeHostApiV1,
    out: *mut NemoRelayNativePluginV1,
    constructor: F,
) -> NemoRelayStatus
where
    P: NativePlugin,
    F: FnOnce() -> P,
{
    let supported_abi = (NEMO_RELAY_NATIVE_ABI_VERSION_LEGACY..=NEMO_RELAY_NATIVE_ABI_VERSION)
        .contains(&host_ref.abi_version);
    if !supported_abi {
        return NemoRelayStatus::InvalidArg;
    }
    if host_ref.struct_size < std::mem::size_of::<NemoRelayNativeHostApiV1>() {
        return NemoRelayStatus::InvalidArg;
    }

    let plugin = constructor();
    let kind = plugin.plugin_kind().to_owned();
    let allows_multiple_components = plugin.allows_multiple_components();
    let Some(kind_handle) = HostString::new(host_ref, &kind) else {
        return NemoRelayStatus::Internal;
    };
    let state = Box::new(PluginState {
        host: unsafe { OwnedHostApi::copy_from(host_ref) },
        plugin: Mutex::new(plugin),
    });
    unsafe {
        *out = NemoRelayNativePluginV1 {
            struct_size: std::mem::size_of::<NemoRelayNativePluginV1>(),
            plugin_kind: kind_handle.ptr,
            allows_multiple_components,
            user_data: Box::into_raw(state) as *mut c_void,
            validate: Some(validate_trampoline::<P>),
            register: Some(register_trampoline::<P>),
            drop: Some(drop_plugin_state::<P>),
        };
    }
    std::mem::forget(kind_handle);
    NemoRelayStatus::Ok
}

/// Exports a concrete plugin constructor as a native plugin entry symbol body.
#[macro_export]
macro_rules! nemo_relay_plugin {
    ($symbol:ident, $constructor:expr) => {
        #[doc = "Native plugin entry symbol generated by `nemo_relay_plugin!`."]
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $symbol(
            host: *const $crate::NemoRelayNativeHostApiV1,
            out: *mut $crate::NemoRelayNativePluginV1,
        ) -> $crate::NemoRelayStatus {
            match ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| unsafe {
                $crate::__export_plugin_from_constructor(host, out, $constructor)
            })) {
                Ok(status) => status,
                Err(_) => {
                    unsafe {
                        $crate::__set_last_error_from_entry(
                            host,
                            "native plugin entry callback panicked",
                        )
                    };
                    $crate::NemoRelayStatus::Internal
                }
            }
        }
    };
}
