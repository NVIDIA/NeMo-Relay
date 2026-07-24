// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![deny(rustdoc::broken_intra_doc_links, rustdoc::private_intra_doc_links)]

//! Stable native plugin ABI and Rust authoring helpers for NeMo Relay.
//!
//! This crate intentionally does not depend on the `nemo-relay` runtime crate.
//! Native plugins built with it communicate with a host through versioned
//! C-compatible tables and host-owned string handles.

use std::ffi::{c_char, c_void};
use std::marker::{PhantomData, PhantomPinned};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;
use std::sync::Mutex;

pub use nemo_relay_types::Json;
pub use nemo_relay_types::api::event::{
    CategoryProfile, DataSchema, Event, EventCategory, EventSanitizeFields, PendingMarkSpec,
    ScopeCategory,
};
pub use nemo_relay_types::api::llm::{LlmAttributes, LlmRequest, LlmRequestInterceptOutcome};
pub use nemo_relay_types::api::scope::{HandleAttributes, ScopeAttributes, ScopeType};
pub use nemo_relay_types::api::tool::{ToolAttributes, ToolExecutionInterceptOutcome};
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
pub const NEMO_RELAY_NATIVE_ABI_VERSION: u32 = 2;

/// Built-in LLM codec identities available to native plugins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BuiltinLlmCodec {
    /// OpenAI Chat Completions.
    #[serde(rename = "openai_chat")]
    OpenAiChat,
    /// OpenAI Responses.
    #[serde(rename = "openai_responses")]
    OpenAiResponses,
    /// Anthropic Messages.
    #[serde(rename = "anthropic_messages")]
    AnthropicMessages,
}

/// Per-call LLM codec identity delivered to native plugins.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, serde::Deserialize)]
#[serde(tag = "kind", content = "id", rename_all = "snake_case")]
pub enum LlmCodecIdentity {
    /// No codec was active.
    #[default]
    None,
    /// A Relay built-in codec was active.
    #[serde(rename = "builtin")]
    BuiltIn(BuiltinLlmCodec),
    /// A runtime-registered codec was active, identified by its stable ID.
    Runtime(String),
    /// A codec was active but has no registered identity.
    Opaque,
}

/// Per-call codec context delivered to LLM sanitizer callbacks.
#[derive(Debug, Clone, Default, Serialize, serde::Deserialize)]
pub struct LlmSanitizeContext {
    /// Identity of the active codec.
    pub codec: LlmCodecIdentity,
}

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
}

/// Opaque host-owned UTF-8 string or JSON byte buffer.
#[repr(C)]
pub struct NemoRelayNativeString {
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
pub struct NemoRelayNativeLlmSanitizeContext {
    /// Discriminator for the active codec.
    pub codec_kind: NemoRelayNativeLlmCodecKind,
    /// Optional borrowed codec identifier.
    pub codec_id: *const NemoRelayNativeString,
}

/// Opaque plugin registration context borrowed from the host during registration.
#[repr(C)]
pub struct NemoRelayNativePluginContext {
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

/// Native tool execution intercept callback.
pub type NemoRelayNativeToolExecutionCb = unsafe extern "C" fn(
    user_data: *mut c_void,
    name: *const NemoRelayNativeString,
    args_json: *const NemoRelayNativeString,
    next_fn: NemoRelayNativeToolNextFn,
    next_ctx: *mut c_void,
    out_outcome_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus;

/// Native LLM request sanitizer callback. Return a successful null output to
/// omit the observability payload and annotation.
pub type NemoRelayNativeLlmSanitizeRequestCb = unsafe extern "C" fn(
    user_data: *mut c_void,
    request_json: *const NemoRelayNativeString,
    context: NemoRelayNativeLlmSanitizeContext,
    out_request_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus;

/// Native LLM response sanitizer callback. Return a successful null output to
/// omit the observability payload and annotation.
pub type NemoRelayNativeLlmSanitizeResponseCb = unsafe extern "C" fn(
    user_data: *mut c_void,
    payload_json: *const NemoRelayNativeString,
    context: NemoRelayNativeLlmSanitizeContext,
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

/// Synchronous JSON chunk stream used by native LLM stream intercept helpers.
pub type LlmJsonStream = Box<dyn Iterator<Item = Result<Json>> + Send>;

/// Cloneable high-level runtime handle for host APIs available to native plugins.
#[derive(Clone)]
pub struct PluginRuntime {
    host: NemoRelayNativeHostApiV1,
}

impl PluginRuntime {
    /// Creates a runtime handle from the host ABI table.
    pub fn new(host: &NemoRelayNativeHostApiV1) -> Self {
        Self { host: *host }
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
pub struct ScopeGuard<'a> {
    runtime: &'a PluginRuntime,
    handle: Option<ScopeHandle<'a>>,
}

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

/// Typed continuation passed to tool execution intercepts.
pub struct ToolNext<'a> {
    host: &'a NemoRelayNativeHostApiV1,
    next_fn: NemoRelayNativeToolNextFn,
    next_ctx: *mut c_void,
}

impl ToolNext<'_> {
    /// Continues the tool execution chain with replacement arguments.
    pub fn call(&self, args: Json) -> Result<Json> {
        let args = HostString::from_json(self.host, &args)
            .ok_or_else(|| "failed to allocate tool next args".to_string())?;
        let mut out = ptr::null_mut();
        let status = unsafe { (self.next_fn)(args.as_ptr(), self.next_ctx, &mut out) };
        if status != NemoRelayStatus::Ok {
            return Err(format!("tool next failed: {status:?}"));
        }
        if out.is_null() {
            return Err("tool next returned null output".into());
        }
        let result = read_json_value(self.host, out, "tool next result");
        unsafe { (self.host.string_free)(out) };
        result.map_err(|status| format!("tool next returned invalid JSON: {status:?}"))
    }
}

/// Typed continuation passed to LLM execution intercepts.
pub struct LlmNext<'a> {
    host: &'a NemoRelayNativeHostApiV1,
    next_fn: NemoRelayNativeLlmNextFn,
    next_ctx: *mut c_void,
}

impl LlmNext<'_> {
    /// Continues the LLM execution chain with a replacement request.
    pub fn call(&self, request: LlmRequest) -> Result<Json> {
        let request = HostString::from_json(self.host, &request)
            .ok_or_else(|| "failed to allocate LLM next request".to_string())?;
        let mut out = ptr::null_mut();
        let status = unsafe { (self.next_fn)(request.as_ptr(), self.next_ctx, &mut out) };
        if status != NemoRelayStatus::Ok {
            return Err(format!("llm next failed: {status:?}"));
        }
        if out.is_null() {
            return Err("llm next returned null output".into());
        }
        let result = read_json_value(self.host, out, "llm next result");
        unsafe { (self.host.string_free)(out) };
        result.map_err(|status| format!("llm next returned invalid JSON: {status:?}"))
    }
}

/// Typed continuation passed to LLM stream execution intercepts.
pub struct LlmStreamNext<'a> {
    host: &'a NemoRelayNativeHostApiV1,
    next_fn: NemoRelayNativeLlmStreamNextFn,
    next_ctx: *mut c_void,
}

impl LlmStreamNext<'_> {
    /// Continues the LLM stream execution chain with a replacement request.
    pub fn call(&self, request: LlmRequest) -> Result<LlmStream> {
        let request = HostString::from_json(self.host, &request)
            .ok_or_else(|| "failed to allocate LLM stream next request".to_string())?;
        let mut raw = NemoRelayNativeLlmStreamV1::default();
        let status = unsafe { (self.next_fn)(request.as_ptr(), self.next_ctx, &mut raw) };
        if status != NemoRelayStatus::Ok {
            return Err(format!("llm stream next failed: {status:?}"));
        }
        unsafe { LlmStream::from_raw(self.host, raw) }
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

impl<'a> ScopeStack<'a> {
    /// Returns the raw ABI pointer.
    pub fn as_ptr(&self) -> *const NemoRelayNativeScopeStack {
        self.ptr
    }

    fn set_thread(&self) -> NemoRelayStatus {
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

    /// Validates one component-local JSON config object.
    fn validate(&self, _plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        vec![]
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
        Self { host, raw }
    }

    /// Returns the host ABI table backing this registration context.
    pub fn host_api(&self) -> &'a NemoRelayNativeHostApiV1 {
        self.host
    }

    /// Returns a cloneable high-level runtime handle.
    pub fn runtime(&self) -> PluginRuntime {
        PluginRuntime::new(self.host)
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
        finish_typed_registration::<F>(self.host, status, user_data, "subscriber")
    }

    fn register_event_sanitizer<F>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
        register: unsafe extern "C" fn(
            *mut NemoRelayNativePluginContext,
            *const NemoRelayNativeString,
            i32,
            NemoRelayNativeEventSanitizeCb,
            *mut c_void,
            NemoRelayNativeFreeFn,
        ) -> NemoRelayStatus,
        label: &str,
    ) -> Result<()>
    where
        F: Fn(&Event, EventSanitizeFields) -> EventSanitizeFields + Send + Sync + 'static,
    {
        let user_data = typed_callback_user_data(self.host, callback);
        let status = self.with_name(name, |_, name| unsafe {
            register(
                self.raw,
                name,
                priority,
                typed_event_sanitize_trampoline::<F>,
                user_data,
                Some(drop_typed_callback::<F>),
            )
        });
        finish_typed_registration::<F>(self.host, status, user_data, label)
    }

    /// Registers a typed mark event sanitizer.
    pub fn register_mark_sanitize_guardrail<F>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(&Event, EventSanitizeFields) -> EventSanitizeFields + Send + Sync + 'static,
    {
        self.register_event_sanitizer(
            name,
            priority,
            callback,
            self.host.plugin_context_register_mark_sanitize_guardrail,
            "mark sanitize guardrail",
        )
    }

    /// Registers a typed scope-start event sanitizer.
    pub fn register_scope_sanitize_start_guardrail<F>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(&Event, EventSanitizeFields) -> EventSanitizeFields + Send + Sync + 'static,
    {
        self.register_event_sanitizer(
            name,
            priority,
            callback,
            self.host
                .plugin_context_register_scope_sanitize_start_guardrail,
            "scope-start sanitize guardrail",
        )
    }

    /// Registers a typed scope-end event sanitizer.
    pub fn register_scope_sanitize_end_guardrail<F>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(&Event, EventSanitizeFields) -> EventSanitizeFields + Send + Sync + 'static,
    {
        self.register_event_sanitizer(
            name,
            priority,
            callback,
            self.host
                .plugin_context_register_scope_sanitize_end_guardrail,
            "scope-end sanitize guardrail",
        )
    }

    /// Registers a typed tool sanitize-request guardrail.
    pub fn register_tool_sanitize_request_guardrail<F>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(&str, Json) -> Json + Send + Sync + 'static,
    {
        let user_data = typed_callback_user_data(self.host, callback);
        let status = unsafe {
            self.register_tool_sanitize_request_guardrail_raw(
                name,
                priority,
                typed_tool_sanitize_trampoline::<F>,
                user_data,
                Some(drop_typed_callback::<F>),
            )
        };
        finish_typed_registration::<F>(
            self.host,
            status,
            user_data,
            "tool sanitize request guardrail",
        )
    }

    /// Registers a typed tool sanitize-response guardrail.
    pub fn register_tool_sanitize_response_guardrail<F>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(&str, Json) -> Json + Send + Sync + 'static,
    {
        let user_data = typed_callback_user_data(self.host, callback);
        let status = unsafe {
            self.register_tool_sanitize_response_guardrail_raw(
                name,
                priority,
                typed_tool_sanitize_trampoline::<F>,
                user_data,
                Some(drop_typed_callback::<F>),
            )
        };
        finish_typed_registration::<F>(
            self.host,
            status,
            user_data,
            "tool sanitize response guardrail",
        )
    }

    /// Registers a typed tool conditional-execution guardrail.
    pub fn register_tool_conditional_execution_guardrail<F>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(&str, &Json) -> Result<Option<String>> + Send + Sync + 'static,
    {
        let user_data = typed_callback_user_data(self.host, callback);
        let status = unsafe {
            self.register_tool_conditional_execution_guardrail_raw(
                name,
                priority,
                typed_tool_conditional_trampoline::<F>,
                user_data,
                Some(drop_typed_callback::<F>),
            )
        };
        finish_typed_registration::<F>(
            self.host,
            status,
            user_data,
            "tool conditional execution guardrail",
        )
    }

    /// Registers a typed tool request intercept.
    pub fn register_tool_request_intercept<F>(
        &mut self,
        name: &str,
        priority: i32,
        break_chain: bool,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(&str, Json) -> Result<Json> + Send + Sync + 'static,
    {
        let user_data = typed_callback_user_data(self.host, callback);
        let status = unsafe {
            self.register_tool_request_intercept_raw(
                name,
                priority,
                break_chain,
                typed_tool_intercept_trampoline::<F>,
                user_data,
                Some(drop_typed_callback::<F>),
            )
        };
        finish_typed_registration::<F>(self.host, status, user_data, "tool request intercept")
    }

    /// Registers a typed tool execution intercept.
    ///
    /// The callback returns a [`ToolExecutionInterceptOutcome`]. Calling
    /// [`ToolNext::call`] continues the chain and returns only the raw
    /// downstream result JSON; Relay retains downstream pending marks.
    pub fn register_tool_execution_intercept<F>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: for<'next> Fn(&str, Json, ToolNext<'next>) -> Result<ToolExecutionInterceptOutcome>
            + Send
            + Sync
            + 'static,
    {
        let user_data = typed_callback_user_data(self.host, callback);
        let status = unsafe {
            self.register_tool_execution_intercept_raw(
                name,
                priority,
                typed_tool_execution_trampoline::<F>,
                user_data,
                Some(drop_typed_callback::<F>),
            )
        };
        finish_typed_registration::<F>(self.host, status, user_data, "tool execution intercept")
    }

    /// Registers a typed LLM sanitize-request guardrail.
    pub fn register_llm_sanitize_request_guardrail<F>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(LlmRequest, LlmSanitizeContext) -> Option<LlmRequest> + Send + Sync + 'static,
    {
        let user_data = typed_callback_user_data(self.host, callback);
        let status = unsafe {
            self.register_llm_sanitize_request_guardrail_raw(
                name,
                priority,
                typed_llm_sanitize_request_trampoline::<F>,
                user_data,
                Some(drop_typed_callback::<F>),
            )
        };
        finish_typed_registration::<F>(
            self.host,
            status,
            user_data,
            "llm sanitize request guardrail",
        )
    }

    /// Registers a typed LLM sanitize-response guardrail.
    pub fn register_llm_sanitize_response_guardrail<F>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(Json, LlmSanitizeContext) -> Option<Json> + Send + Sync + 'static,
    {
        let user_data = typed_callback_user_data(self.host, callback);
        let status = unsafe {
            self.register_llm_sanitize_response_guardrail_raw(
                name,
                priority,
                typed_llm_sanitize_response_trampoline::<F>,
                user_data,
                Some(drop_typed_callback::<F>),
            )
        };
        finish_typed_registration::<F>(
            self.host,
            status,
            user_data,
            "llm sanitize response guardrail",
        )
    }

    /// Registers a typed LLM conditional-execution guardrail.
    pub fn register_llm_conditional_execution_guardrail<F>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(&LlmRequest) -> Result<Option<String>> + Send + Sync + 'static,
    {
        let user_data = typed_callback_user_data(self.host, callback);
        let status = unsafe {
            self.register_llm_conditional_execution_guardrail_raw(
                name,
                priority,
                typed_llm_conditional_trampoline::<F>,
                user_data,
                Some(drop_typed_callback::<F>),
            )
        };
        finish_typed_registration::<F>(
            self.host,
            status,
            user_data,
            "llm conditional execution guardrail",
        )
    }

    /// Registers a typed LLM request intercept.
    pub fn register_llm_request_intercept<F>(
        &mut self,
        name: &str,
        priority: i32,
        break_chain: bool,
        callback: F,
    ) -> Result<()>
    where
        F: Fn(&str, LlmRequest, Option<AnnotatedLlmRequest>) -> Result<LlmRequestInterceptOutcome>
            + Send
            + Sync
            + 'static,
    {
        let user_data = typed_callback_user_data(self.host, callback);
        let status = unsafe {
            self.register_llm_request_intercept_raw(
                name,
                priority,
                break_chain,
                typed_llm_request_intercept_trampoline::<F>,
                user_data,
                Some(drop_typed_callback::<F>),
            )
        };
        finish_typed_registration::<F>(self.host, status, user_data, "llm request intercept")
    }

    /// Registers a typed LLM execution intercept.
    pub fn register_llm_execution_intercept<F>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: for<'next> Fn(&str, LlmRequest, LlmNext<'next>) -> Result<Json> + Send + Sync + 'static,
    {
        let user_data = typed_callback_user_data(self.host, callback);
        let status = unsafe {
            self.register_llm_execution_intercept_raw(
                name,
                priority,
                typed_llm_execution_trampoline::<F>,
                user_data,
                Some(drop_typed_callback::<F>),
            )
        };
        finish_typed_registration::<F>(self.host, status, user_data, "llm execution intercept")
    }

    /// Registers a typed LLM stream execution intercept.
    ///
    /// Native ABI v1 represents stream execution as one JSON result. The host
    /// wraps that result as a one-chunk stream.
    pub fn register_llm_stream_execution_intercept<F>(
        &mut self,
        name: &str,
        priority: i32,
        callback: F,
    ) -> Result<()>
    where
        F: for<'next> Fn(&str, LlmRequest, LlmStreamNext<'next>) -> Result<LlmJsonStream>
            + Send
            + Sync
            + 'static,
    {
        let user_data = typed_callback_user_data(self.host, callback);
        let status = unsafe {
            self.register_llm_stream_execution_intercept_raw(
                name,
                priority,
                typed_llm_stream_execution_trampoline::<F>,
                user_data,
                Some(drop_typed_callback::<F>),
            )
        };
        finish_typed_registration::<F>(
            self.host,
            status,
            user_data,
            "llm stream execution intercept",
        )
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
        self.with_name(name, |host, name| unsafe {
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
        self.with_name(name, |host, name| unsafe {
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
        self.with_name(name, |host, name| unsafe {
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
        self.with_name(name, |host, name| unsafe {
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
        self.with_name(name, |host, name| unsafe {
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
        self.with_name(name, |host, name| unsafe {
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
        self.with_name(name, |host, name| unsafe {
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
        self.with_name(name, |host, name| unsafe {
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
        self.with_name(name, |host, name| unsafe {
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
        self.with_name(name, |host, name| unsafe {
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
        self.with_name(name, |host, name| unsafe {
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
        self.with_name(name, |host, name| unsafe {
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
        self.with_name(name, |host, name| unsafe {
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
        self.with_name(name, |host, name| unsafe {
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
        self.with_name(name, |host, name| unsafe {
            (host.plugin_context_register_llm_stream_execution_intercept)(
                self.raw, name, priority, cb, user_data, free_fn,
            )
        })
    }

    fn with_name(
        &self,
        name: &str,
        f: impl FnOnce(&NemoRelayNativeHostApiV1, *const NemoRelayNativeString) -> NemoRelayStatus,
    ) -> NemoRelayStatus {
        let name = match HostString::try_new(self.host, name) {
            Ok(name) => name,
            Err(status) => return status,
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

fn finish_typed_registration<F>(
    host: &NemoRelayNativeHostApiV1,
    status: NemoRelayStatus,
    user_data: *mut c_void,
    label: &str,
) -> Result<()> {
    if status == NemoRelayStatus::Ok {
        Ok(())
    } else {
        unsafe { drop_typed_callback::<F>(user_data) };
        Err(status_error(host, status, label))
    }
}

fn status_error(host: &NemoRelayNativeHostApiV1, status: NemoRelayStatus, label: &str) -> String {
    debug_assert_ne!(status, NemoRelayStatus::Ok);
    set_last_error(host, &format!("{label} failed: {status:?}"));
    format!("{label} failed: {status:?}")
}

fn callback_error(host: &NemoRelayNativeHostApiV1, message: String) -> NemoRelayStatus {
    set_last_error(host, &message);
    NemoRelayStatus::Internal
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

unsafe extern "C" fn typed_event_sanitize_trampoline<F>(
    user_data: *mut c_void,
    event_json: *const NemoRelayNativeString,
    fields_json: *const NemoRelayNativeString,
    out_fields_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus
where
    F: Fn(&Event, EventSanitizeFields) -> EventSanitizeFields + Send + Sync + 'static,
{
    if user_data.is_null() || out_fields_json.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *out_fields_json = ptr::null_mut() };
    let state = unsafe { &*(user_data as *const TypedCallback<F>) };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let event: Event = read_json_value(&state.host, event_json, "event")?;
        let fields: EventSanitizeFields =
            read_json_value(&state.host, fields_json, "event sanitize fields")?;
        let output = (state.callback)(&event, fields);
        Ok::<_, NemoRelayStatus>(write_json(&state.host, &output, out_fields_json))
    }));
    match result {
        Ok(Ok(status)) => status,
        Ok(Err(status)) => status,
        Err(_) => callback_panic(&state.host, "event sanitize callback"),
    }
}

unsafe extern "C" fn typed_tool_sanitize_trampoline<F>(
    user_data: *mut c_void,
    name: *const NemoRelayNativeString,
    payload_json: *const NemoRelayNativeString,
    out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus
where
    F: Fn(&str, Json) -> Json + Send + Sync + 'static,
{
    if user_data.is_null() || out_json.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *out_json = ptr::null_mut() };
    let state = unsafe { &*(user_data as *const TypedCallback<F>) };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let name = read_required_host_string(&state.host, name, "tool name")?;
        let payload: Json = read_json_value(&state.host, payload_json, "tool payload")?;
        let output = (state.callback)(&name, payload);
        Ok::<_, NemoRelayStatus>(write_json(&state.host, &output, out_json))
    }));
    match result {
        Ok(Ok(status)) => status,
        Ok(Err(status)) => status,
        Err(_) => callback_panic(&state.host, "tool sanitize callback"),
    }
}

unsafe extern "C" fn typed_tool_intercept_trampoline<F>(
    user_data: *mut c_void,
    name: *const NemoRelayNativeString,
    payload_json: *const NemoRelayNativeString,
    out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus
where
    F: Fn(&str, Json) -> Result<Json> + Send + Sync + 'static,
{
    if user_data.is_null() || out_json.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *out_json = ptr::null_mut() };
    let state = unsafe { &*(user_data as *const TypedCallback<F>) };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let name = read_required_host_string(&state.host, name, "tool name")?;
        let payload: Json = read_json_value(&state.host, payload_json, "tool payload")?;
        match (state.callback)(&name, payload) {
            Ok(output) => Ok::<_, NemoRelayStatus>(write_json(&state.host, &output, out_json)),
            Err(message) => Ok(callback_error(&state.host, message)),
        }
    }));
    match result {
        Ok(Ok(status)) => status,
        Ok(Err(status)) => status,
        Err(_) => callback_panic(&state.host, "tool intercept callback"),
    }
}

unsafe extern "C" fn typed_tool_conditional_trampoline<F>(
    user_data: *mut c_void,
    name: *const NemoRelayNativeString,
    args_json: *const NemoRelayNativeString,
    out_reason: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus
where
    F: Fn(&str, &Json) -> Result<Option<String>> + Send + Sync + 'static,
{
    if user_data.is_null() || out_reason.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *out_reason = ptr::null_mut() };
    let state = unsafe { &*(user_data as *const TypedCallback<F>) };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let name = read_required_host_string(&state.host, name, "tool name")?;
        let args: Json = read_json_value(&state.host, args_json, "tool args")?;
        match (state.callback)(&name, &args) {
            Ok(Some(reason)) => {
                let reason =
                    HostString::new(&state.host, &reason).ok_or(NemoRelayStatus::Internal)?;
                unsafe { *out_reason = reason.ptr };
                std::mem::forget(reason);
                Ok(NemoRelayStatus::Ok)
            }
            Ok(None) => {
                unsafe { *out_reason = ptr::null_mut() };
                Ok(NemoRelayStatus::Ok)
            }
            Err(message) => Ok(callback_error(&state.host, message)),
        }
    }));
    match result {
        Ok(Ok(status)) => status,
        Ok(Err(status)) => status,
        Err(_) => callback_panic(&state.host, "tool conditional callback"),
    }
}

unsafe extern "C" fn typed_tool_execution_trampoline<F>(
    user_data: *mut c_void,
    name: *const NemoRelayNativeString,
    args_json: *const NemoRelayNativeString,
    next_fn: NemoRelayNativeToolNextFn,
    next_ctx: *mut c_void,
    out_outcome_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus
where
    F: for<'next> Fn(&str, Json, ToolNext<'next>) -> Result<ToolExecutionInterceptOutcome>
        + Send
        + Sync
        + 'static,
{
    if user_data.is_null() || out_outcome_json.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *out_outcome_json = ptr::null_mut() };
    let state = unsafe { &*(user_data as *const TypedCallback<F>) };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let name = read_required_host_string(&state.host, name, "tool name")?;
        let args: Json = read_json_value(&state.host, args_json, "tool args")?;
        let next = ToolNext {
            host: &state.host,
            next_fn,
            next_ctx,
        };
        match (state.callback)(&name, args, next) {
            Ok(outcome) => {
                let Some(outcome) = HostString::from_json(&state.host, &outcome) else {
                    set_last_error(&state.host, "failed to allocate tool execution outcome");
                    return Ok(NemoRelayStatus::Internal);
                };
                unsafe { *out_outcome_json = outcome.ptr };
                std::mem::forget(outcome);
                Ok(NemoRelayStatus::Ok)
            }
            Err(message) => Ok(callback_error(&state.host, message)),
        }
    }));
    match result {
        Ok(Ok(status)) => status,
        Ok(Err(status)) => status,
        Err(_) => callback_panic(&state.host, "tool execution callback"),
    }
}

unsafe extern "C" fn typed_llm_sanitize_request_trampoline<F>(
    user_data: *mut c_void,
    request_json: *const NemoRelayNativeString,
    context: NemoRelayNativeLlmSanitizeContext,
    out_request_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus
where
    F: Fn(LlmRequest, LlmSanitizeContext) -> Option<LlmRequest> + Send + Sync + 'static,
{
    if user_data.is_null() || out_request_json.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *out_request_json = ptr::null_mut() };
    let state = unsafe { &*(user_data as *const TypedCallback<F>) };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let context = llm_sanitize_context_from_native(&state.host, context)?;
        let request: LlmRequest = read_json_value(&state.host, request_json, "LLM request")?;
        match (state.callback)(request, context) {
            Some(output) => {
                Ok::<_, NemoRelayStatus>(write_json(&state.host, &output, out_request_json))
            }
            None => Ok(NemoRelayStatus::Ok),
        }
    }));
    match result {
        Ok(Ok(status)) => status,
        Ok(Err(status)) => status,
        Err(_) => callback_panic(&state.host, "LLM sanitize request callback"),
    }
}

unsafe extern "C" fn typed_llm_sanitize_response_trampoline<F>(
    user_data: *mut c_void,
    payload_json: *const NemoRelayNativeString,
    context: NemoRelayNativeLlmSanitizeContext,
    out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus
where
    F: Fn(Json, LlmSanitizeContext) -> Option<Json> + Send + Sync + 'static,
{
    if user_data.is_null() || out_json.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *out_json = ptr::null_mut() };
    let state = unsafe { &*(user_data as *const TypedCallback<F>) };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let context = llm_sanitize_context_from_native(&state.host, context)?;
        let payload: Json = read_json_value(&state.host, payload_json, "LLM response")?;
        match (state.callback)(payload, context) {
            Some(output) => Ok::<_, NemoRelayStatus>(write_json(&state.host, &output, out_json)),
            None => Ok(NemoRelayStatus::Ok),
        }
    }));
    match result {
        Ok(Ok(status)) => status,
        Ok(Err(status)) => status,
        Err(_) => callback_panic(&state.host, "LLM sanitize response callback"),
    }
}

unsafe extern "C" fn typed_llm_conditional_trampoline<F>(
    user_data: *mut c_void,
    request_json: *const NemoRelayNativeString,
    out_reason: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus
where
    F: Fn(&LlmRequest) -> Result<Option<String>> + Send + Sync + 'static,
{
    if user_data.is_null() || out_reason.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *out_reason = ptr::null_mut() };
    let state = unsafe { &*(user_data as *const TypedCallback<F>) };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let request: LlmRequest = read_json_value(&state.host, request_json, "LLM request")?;
        match (state.callback)(&request) {
            Ok(Some(reason)) => {
                let reason =
                    HostString::new(&state.host, &reason).ok_or(NemoRelayStatus::Internal)?;
                unsafe { *out_reason = reason.ptr };
                std::mem::forget(reason);
                Ok(NemoRelayStatus::Ok)
            }
            Ok(None) => {
                unsafe { *out_reason = ptr::null_mut() };
                Ok(NemoRelayStatus::Ok)
            }
            Err(message) => Ok(callback_error(&state.host, message)),
        }
    }));
    match result {
        Ok(Ok(status)) => status,
        Ok(Err(status)) => status,
        Err(_) => callback_panic(&state.host, "LLM conditional callback"),
    }
}

unsafe extern "C" fn typed_llm_request_intercept_trampoline<F>(
    user_data: *mut c_void,
    name: *const NemoRelayNativeString,
    request_json: *const NemoRelayNativeString,
    annotated_json: *const NemoRelayNativeString,
    out_outcome_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus
where
    F: Fn(&str, LlmRequest, Option<AnnotatedLlmRequest>) -> Result<LlmRequestInterceptOutcome>
        + Send
        + Sync
        + 'static,
{
    if user_data.is_null() || out_outcome_json.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    unsafe {
        *out_outcome_json = ptr::null_mut();
    }
    let state = unsafe { &*(user_data as *const TypedCallback<F>) };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let name = read_required_host_string(&state.host, name, "LLM name")?;
        let request: LlmRequest = read_json_value(&state.host, request_json, "LLM request")?;
        let annotated: Option<AnnotatedLlmRequest> =
            read_optional_json_value(&state.host, annotated_json, "annotated LLM request")?;
        match (state.callback)(&name, request, annotated) {
            Ok(outcome) => {
                let Some(outcome) = HostString::from_json(&state.host, &outcome) else {
                    set_last_error(&state.host, "failed to allocate LLM request outcome");
                    return Ok(NemoRelayStatus::Internal);
                };
                unsafe {
                    *out_outcome_json = outcome.ptr;
                }
                std::mem::forget(outcome);
                Ok(NemoRelayStatus::Ok)
            }
            Err(message) => Ok(callback_error(&state.host, message)),
        }
    }));
    match result {
        Ok(Ok(status)) => status,
        Ok(Err(status)) => status,
        Err(_) => callback_panic(&state.host, "LLM request intercept callback"),
    }
}

unsafe extern "C" fn typed_llm_execution_trampoline<F>(
    user_data: *mut c_void,
    name: *const NemoRelayNativeString,
    request_json: *const NemoRelayNativeString,
    next_fn: NemoRelayNativeLlmNextFn,
    next_ctx: *mut c_void,
    out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus
where
    F: for<'next> Fn(&str, LlmRequest, LlmNext<'next>) -> Result<Json> + Send + Sync + 'static,
{
    if user_data.is_null() || out_json.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *out_json = ptr::null_mut() };
    let state = unsafe { &*(user_data as *const TypedCallback<F>) };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let name = read_required_host_string(&state.host, name, "LLM name")?;
        let request: LlmRequest = read_json_value(&state.host, request_json, "LLM request")?;
        let next = LlmNext {
            host: &state.host,
            next_fn,
            next_ctx,
        };
        match (state.callback)(&name, request, next) {
            Ok(output) => Ok::<_, NemoRelayStatus>(write_json(&state.host, &output, out_json)),
            Err(message) => Ok(callback_error(&state.host, message)),
        }
    }));
    match result {
        Ok(Ok(status)) => status,
        Ok(Err(status)) => status,
        Err(_) => callback_panic(&state.host, "LLM execution callback"),
    }
}

struct TypedLlmJsonStream {
    host: NemoRelayNativeHostApiV1,
    state: Mutex<TypedLlmJsonStreamState>,
}

struct TypedLlmJsonStreamState {
    iter: LlmJsonStream,
    finished: bool,
}

fn native_stream_from_iter(
    host: &NemoRelayNativeHostApiV1,
    iter: LlmJsonStream,
) -> NemoRelayNativeLlmStreamV1 {
    let state = Box::new(TypedLlmJsonStream {
        host: *host,
        state: Mutex::new(TypedLlmJsonStreamState {
            iter,
            finished: false,
        }),
    });
    NemoRelayNativeLlmStreamV1 {
        struct_size: std::mem::size_of::<NemoRelayNativeLlmStreamV1>(),
        user_data: Box::into_raw(state).cast(),
        next: Some(poll_typed_llm_json_stream),
        cancel: Some(cancel_typed_llm_json_stream),
        drop: Some(drop_typed_llm_json_stream),
    }
}

unsafe extern "C" fn poll_typed_llm_json_stream(
    user_data: *mut c_void,
    out_json: *mut *mut NemoRelayNativeString,
) -> NemoRelayStatus {
    if user_data.is_null() || out_json.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *out_json = ptr::null_mut() };
    let stream = unsafe { &*(user_data as *const TypedLlmJsonStream) };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut state = match stream.state.lock() {
            Ok(state) => state,
            Err(_) => {
                set_last_error(&stream.host, "native plugin stream state lock poisoned");
                return NemoRelayStatus::Internal;
            }
        };
        if state.finished {
            return NemoRelayStatus::StreamEnd;
        }
        match state.iter.next() {
            Some(Ok(chunk)) => {
                let status = write_json(&stream.host, &chunk, out_json);
                if status != NemoRelayStatus::Ok {
                    state.finished = true;
                }
                status
            }
            Some(Err(message)) => {
                state.finished = true;
                callback_error(&stream.host, message)
            }
            None => {
                state.finished = true;
                NemoRelayStatus::StreamEnd
            }
        }
    }));
    result.unwrap_or_else(|_| callback_panic(&stream.host, "LLM stream callback"))
}

unsafe extern "C" fn cancel_typed_llm_json_stream(user_data: *mut c_void) -> NemoRelayStatus {
    if user_data.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    let stream = unsafe { &*(user_data as *const TypedLlmJsonStream) };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut state = match stream.state.lock() {
            Ok(state) => state,
            Err(_) => {
                set_last_error(&stream.host, "native plugin stream state lock poisoned");
                return NemoRelayStatus::Internal;
            }
        };
        state.finished = true;
        NemoRelayStatus::Ok
    }));
    result.unwrap_or_else(|_| callback_panic(&stream.host, "LLM stream cancel callback"))
}

unsafe extern "C" fn drop_typed_llm_json_stream(user_data: *mut c_void) {
    if !user_data.is_null() {
        let stream = unsafe { Box::from_raw(user_data as *mut TypedLlmJsonStream) };
        let host = stream.host;
        if catch_unwind(AssertUnwindSafe(|| drop(stream))).is_err() {
            set_last_error(&host, "native plugin LLM stream state drop panicked");
        }
    }
}

unsafe extern "C" fn typed_llm_stream_execution_trampoline<F>(
    user_data: *mut c_void,
    name: *const NemoRelayNativeString,
    request_json: *const NemoRelayNativeString,
    next_fn: NemoRelayNativeLlmStreamNextFn,
    next_ctx: *mut c_void,
    out_stream: *mut NemoRelayNativeLlmStreamV1,
) -> NemoRelayStatus
where
    F: for<'next> Fn(&str, LlmRequest, LlmStreamNext<'next>) -> Result<LlmJsonStream>
        + Send
        + Sync
        + 'static,
{
    if user_data.is_null() || out_stream.is_null() {
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *out_stream = NemoRelayNativeLlmStreamV1::default() };
    let state = unsafe { &*(user_data as *const TypedCallback<F>) };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let name = read_required_host_string(&state.host, name, "LLM name")?;
        let request: LlmRequest = read_json_value(&state.host, request_json, "LLM request")?;
        let next = LlmStreamNext {
            host: &state.host,
            next_fn,
            next_ctx,
        };
        match (state.callback)(&name, request, next) {
            Ok(stream) => {
                unsafe { *out_stream = native_stream_from_iter(&state.host, stream) };
                Ok::<_, NemoRelayStatus>(NemoRelayStatus::Ok)
            }
            Err(message) => Ok(callback_error(&state.host, message)),
        }
    }));
    match result {
        Ok(Ok(status)) => status,
        Ok(Err(status)) => status,
        Err(_) => callback_panic(&state.host, "LLM stream execution callback"),
    }
}

struct HostString<'a> {
    host: &'a NemoRelayNativeHostApiV1,
    ptr: *mut NemoRelayNativeString,
}

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

struct PluginState<P> {
    host: NemoRelayNativeHostApiV1,
    plugin: Mutex<P>,
}

unsafe extern "C" fn drop_plugin_state<P: NativePlugin>(user_data: *mut c_void) {
    if !user_data.is_null() {
        let state = unsafe { Box::from_raw(user_data as *mut PluginState<P>) };
        let host = state.host;
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
        let config = match read_json_object(&state.host, plugin_config_json) {
            Ok(config) => config,
            Err(status) => return status,
        };
        let plugin = match state.plugin.lock() {
            Ok(plugin) => plugin,
            Err(_) => {
                set_last_error(&state.host, "native plugin state lock poisoned");
                return NemoRelayStatus::Internal;
            }
        };
        let diagnostics = plugin.validate(&config);
        write_json(&state.host, &diagnostics, out_diagnostics_json)
    }));
    result.unwrap_or_else(|_| {
        set_last_error(&state.host, "native plugin validate callback panicked");
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
        let config = match read_json_object(&state.host, plugin_config_json) {
            Ok(config) => config,
            Err(status) => return status,
        };
        let mut ctx = unsafe { PluginContext::from_raw(&state.host, ctx) };
        let mut plugin = match state.plugin.lock() {
            Ok(plugin) => plugin,
            Err(_) => {
                set_last_error(&state.host, "native plugin state lock poisoned");
                return NemoRelayStatus::Internal;
            }
        };
        match plugin.register(&config, &mut ctx) {
            Ok(()) => NemoRelayStatus::Ok,
            Err(message) => {
                set_last_error(&state.host, &message);
                NemoRelayStatus::Internal
            }
        }
    }));
    result.unwrap_or_else(|_| {
        set_last_error(&state.host, "native plugin register callback panicked");
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

fn read_optional_json_value<T: DeserializeOwned>(
    host: &NemoRelayNativeHostApiV1,
    value: *const NemoRelayNativeString,
    label: &str,
) -> std::result::Result<Option<T>, NemoRelayStatus> {
    if value.is_null() {
        Ok(None)
    } else {
        read_json_value(host, value, label).map(Some)
    }
}

fn llm_sanitize_context_from_native(
    host: &NemoRelayNativeHostApiV1,
    context: NemoRelayNativeLlmSanitizeContext,
) -> std::result::Result<LlmSanitizeContext, NemoRelayStatus> {
    let codec = match context.codec_kind {
        NemoRelayNativeLlmCodecKind::None => LlmCodecIdentity::None,
        NemoRelayNativeLlmCodecKind::Opaque => LlmCodecIdentity::Opaque,
        NemoRelayNativeLlmCodecKind::BuiltIn => {
            let id = read_required_host_string(host, context.codec_id, "LLM built-in codec ID")?;
            let builtin = match id.as_str() {
                "openai_chat" => BuiltinLlmCodec::OpenAiChat,
                "openai_responses" => BuiltinLlmCodec::OpenAiResponses,
                "anthropic_messages" => BuiltinLlmCodec::AnthropicMessages,
                _ => {
                    set_last_error(host, &format!("unknown built-in LLM codec ID: {id}"));
                    return Err(NemoRelayStatus::InvalidArg);
                }
            };
            LlmCodecIdentity::BuiltIn(builtin)
        }
        NemoRelayNativeLlmCodecKind::Runtime => LlmCodecIdentity::Runtime(
            read_required_host_string(host, context.codec_id, "LLM runtime codec ID")?,
        ),
    };
    Ok(LlmSanitizeContext { codec })
}

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
    if host_ref.abi_version != NEMO_RELAY_NATIVE_ABI_VERSION {
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
        host: *host_ref,
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
