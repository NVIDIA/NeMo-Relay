// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! C-compatible types exposed through the FFI boundary.
//!
//! This module defines opaque handle wrappers, enumerations, accessor functions,
//! and free functions for all types that cross the C FFI boundary. Each opaque
//! struct wraps a corresponding core type and is heap-allocated; the C consumer
//! sees only an opaque pointer. All returned C strings must be freed with
//! [`crate::convert::nemo_relay_string_free`], and all handles must be freed
//! with their corresponding `nemo_relay_*_free` function.

use libc::c_char;
use nemo_relay::api::runtime::{ScopeStackHandle, ThreadScopeStackBinding};
use nemo_relay::plugin::PluginRegistrationContext;
use serde_json::Value as Json;

use nemo_relay::api::event::Event;
#[cfg(test)]
use nemo_relay::api::llm::LlmAttributes;
use nemo_relay::api::llm::{LlmHandle, LlmRequest};
#[cfg(test)]
use nemo_relay::api::scope::ScopeAttributes;
use nemo_relay::api::scope::{ScopeHandle, ScopeType};
#[cfg(test)]
use nemo_relay::api::tool::ToolAttributes;
use nemo_relay::api::tool::ToolHandle;
use nemo_relay::codec::traits::{LlmCodec, LlmResponseCodec};
use nemo_relay_adaptive::AdaptiveRuntime;

use crate::convert::{json_to_c_string, str_to_c_string};
use crate::error::set_last_error;
#[cfg(test)]
use crate::{api, convert};

// ---------------------------------------------------------------------------
// Opaque handle wrappers — each wraps a core type in a Box on the heap.
// The C consumer sees only `*mut FfiScopeHandle` etc.
// ---------------------------------------------------------------------------

/// Opaque handle representing an active execution scope.
pub struct FfiScopeHandle(pub ScopeHandle);
/// Opaque handle representing an active tool call.
pub struct FfiToolHandle(pub ToolHandle);
/// Opaque handle representing an active LLM call.
pub struct FfiLLMHandle(pub LlmHandle);
/// Opaque wrapper around an LLM request (headers, content).
pub struct FfiLLMRequest(pub LlmRequest);
/// Opaque wrapper around a lifecycle event emitted by the runtime.
pub struct FfiEvent(pub Event);
/// Opaque handle to an isolated scope stack for per-request/per-task isolation.
pub struct FfiScopeStack(pub ScopeStackHandle);
/// Opaque handle to a captured thread-local scope stack binding.
pub struct FfiThreadScopeStackBinding(pub ThreadScopeStackBinding);
/// Opaque ATIF exporter handle.
pub struct FfiAtifExporter(pub nemo_relay::observability::atif::AtifExporter);
/// Opaque ATOF JSONL exporter handle.
pub struct FfiAtofExporter(pub nemo_relay::observability::atof::AtofExporter);
/// Opaque OpenTelemetry subscriber handle.
pub struct FfiOpenTelemetrySubscriber(pub nemo_relay::observability::otel::OpenTelemetrySubscriber);
/// Opaque OpenInference subscriber handle.
pub struct FfiOpenInferenceSubscriber(
    pub nemo_relay::observability::openinference::OpenInferenceSubscriber,
);
/// Opaque owned adaptive runtime handle.
pub struct FfiAdaptiveRuntime(pub std::sync::Mutex<Option<AdaptiveRuntime>>);
/// Opaque plugin registration context.
///
/// This wrapper contains a borrowed raw pointer to an
/// `nemo_relay::plugin::PluginRegistrationContext`, not an owned heap allocation.
/// It is only valid for the duration of the plugin registration callback that receives
/// it. C callers must not store the pointer, use it after the callback returns, or attempt to
/// free or drop it.
///
/// There is intentionally no `nemo_relay_plugin_context_free` function because this FFI
/// wrapper does not own the underlying registration context.
pub struct FfiPluginContext(pub *mut PluginRegistrationContext);

/// Opaque handle carrying both request and response codec trait objects.
///
/// Created by `nemo_relay_openai_chat_codec_new` (and similar constructors).
/// Freed by `nemo_relay_codec_free`. The handle carries two `Arc`s pointing
/// to the same underlying codec instance: one for the `LlmCodec` trait and
/// one for the `LlmResponseCodec` trait.
pub struct FfiCodecHandle {
    #[allow(dead_code)]
    pub(crate) codec: std::sync::Arc<dyn LlmCodec>,
    pub(crate) response_codec: std::sync::Arc<dyn LlmResponseCodec>,
}

// ---------------------------------------------------------------------------
// Enums exposed to C
// ---------------------------------------------------------------------------

/// The type of scope in the agent execution hierarchy.
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum NemoRelayScopeType {
    /// Top-level agent scope.
    Agent = 0,
    /// Generic function scope.
    Function = 1,
    /// Tool invocation scope.
    Tool = 2,
    /// LLM call scope.
    Llm = 3,
    /// Retriever scope (e.g., RAG lookup).
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

impl From<NemoRelayScopeType> for ScopeType {
    fn from(v: NemoRelayScopeType) -> Self {
        match v {
            NemoRelayScopeType::Agent => ScopeType::Agent,
            NemoRelayScopeType::Function => ScopeType::Function,
            NemoRelayScopeType::Tool => ScopeType::Tool,
            NemoRelayScopeType::Llm => ScopeType::Llm,
            NemoRelayScopeType::Retriever => ScopeType::Retriever,
            NemoRelayScopeType::Embedder => ScopeType::Embedder,
            NemoRelayScopeType::Reranker => ScopeType::Reranker,
            NemoRelayScopeType::Guardrail => ScopeType::Guardrail,
            NemoRelayScopeType::Evaluator => ScopeType::Evaluator,
            NemoRelayScopeType::Custom => ScopeType::Custom,
            NemoRelayScopeType::Unknown => ScopeType::Unknown,
        }
    }
}

impl From<ScopeType> for NemoRelayScopeType {
    fn from(v: ScopeType) -> Self {
        match v {
            ScopeType::Agent => NemoRelayScopeType::Agent,
            ScopeType::Function => NemoRelayScopeType::Function,
            ScopeType::Tool => NemoRelayScopeType::Tool,
            ScopeType::Llm => NemoRelayScopeType::Llm,
            ScopeType::Retriever => NemoRelayScopeType::Retriever,
            ScopeType::Embedder => NemoRelayScopeType::Embedder,
            ScopeType::Reranker => NemoRelayScopeType::Reranker,
            ScopeType::Guardrail => NemoRelayScopeType::Guardrail,
            ScopeType::Evaluator => NemoRelayScopeType::Evaluator,
            ScopeType::Custom => NemoRelayScopeType::Custom,
            ScopeType::Unknown => NemoRelayScopeType::Unknown,
        }
    }
}

// ---------------------------------------------------------------------------
// Free functions for opaque handles
// ---------------------------------------------------------------------------

/// Free a scope handle previously returned by the runtime.
///
/// # Safety
/// `ptr` must be a valid pointer returned by an `nemo_relay_*` function, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_handle_free(ptr: *mut FfiScopeHandle) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

/// Free a tool handle previously returned by the runtime.
///
/// # Safety
/// `ptr` must be a valid pointer returned by an `nemo_relay_*` function, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_tool_handle_free(ptr: *mut FfiToolHandle) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

/// Free an LLM handle previously returned by the runtime.
///
/// # Safety
/// `ptr` must be a valid pointer returned by an `nemo_relay_*` function, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_llm_handle_free(ptr: *mut FfiLLMHandle) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

/// Free an LLM request object.
///
/// # Safety
/// `ptr` must be a valid pointer returned by an `nemo_relay_*` function, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_llm_request_free(ptr: *mut FfiLLMRequest) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

/// Free an event object.
///
/// # Safety
/// `ptr` must be a valid pointer returned by an `nemo_relay_*` function, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_event_free(ptr: *mut FfiEvent) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

/// Free a scope stack handle previously returned by `nemo_relay_scope_stack_create`.
///
/// # Safety
/// `ptr` must be a valid pointer returned by `nemo_relay_scope_stack_create`, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_stack_free(ptr: *mut FfiScopeStack) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

/// Free an ATIF exporter handle previously returned by `nemo_relay_atif_exporter_create`.
///
/// # Safety
/// `ptr` must be a valid pointer returned by `nemo_relay_atif_exporter_create`, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_atif_exporter_free(ptr: *mut FfiAtifExporter) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

/// Free an ATOF JSONL exporter handle previously returned by `nemo_relay_atof_exporter_create`.
///
/// # Safety
/// `ptr` must be a valid pointer returned by `nemo_relay_atof_exporter_create`, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_atof_exporter_free(ptr: *mut FfiAtofExporter) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

/// Free an OpenTelemetry subscriber handle previously returned by
/// `nemo_relay_otel_subscriber_create`.
///
/// # Safety
/// `ptr` must be a valid pointer returned by `nemo_relay_otel_subscriber_create`, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_otel_subscriber_free(ptr: *mut FfiOpenTelemetrySubscriber) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

/// Free an OpenInference subscriber handle previously returned by
/// `nemo_relay_openinference_subscriber_create`.
///
/// # Safety
/// `ptr` must be a valid pointer returned by
/// `nemo_relay_openinference_subscriber_create`, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_openinference_subscriber_free(
    ptr: *mut FfiOpenInferenceSubscriber,
) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

/// Free an adaptive runtime handle previously returned by
/// `nemo_relay_adaptive_runtime_create`.
///
/// # Safety
/// `ptr` must be a valid pointer returned by `nemo_relay_adaptive_runtime_create`, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_adaptive_runtime_free(ptr: *mut FfiAdaptiveRuntime) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

/// Free a codec handle previously returned by one of the codec constructor
/// functions (`nemo_relay_openai_chat_codec_new`, etc.).
///
/// # Safety
/// `handle` must be a valid pointer returned by one of the codec constructor
/// functions, or null. Double-free is undefined behavior.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_codec_free(handle: *mut FfiCodecHandle) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle) });
    }
}

// ---------------------------------------------------------------------------
// Accessor functions for ScopeHandle
// ---------------------------------------------------------------------------

/// Return the UUID of a scope handle as a C string. Caller must free the result
/// with `nemo_relay_string_free`. Returns null if `ptr` is null.
///
/// # Safety
/// `ptr` must be a valid `FfiScopeHandle` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_handle_uuid(ptr: *const FfiScopeHandle) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    str_to_c_string(&unsafe { &*ptr }.0.uuid.to_string())
}

/// Return the name of a scope handle as a C string. Caller must free the result.
/// Returns null if `ptr` is null.
///
/// # Safety
/// `ptr` must be a valid `FfiScopeHandle` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_handle_name(ptr: *const FfiScopeHandle) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    str_to_c_string(&unsafe { &*ptr }.0.name)
}

/// Return the scope type of a scope handle. Returns `Unknown` if `ptr` is null.
///
/// # Safety
/// `ptr` must be a valid `FfiScopeHandle` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_handle_scope_type(
    ptr: *const FfiScopeHandle,
) -> NemoRelayScopeType {
    if ptr.is_null() {
        return NemoRelayScopeType::Unknown;
    }
    unsafe { &*ptr }.0.scope_type.into()
}

/// Return the bitfield attributes of a scope handle. Returns 0 if `ptr` is null.
///
/// # Safety
/// `ptr` must be a valid `FfiScopeHandle` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_handle_attributes(ptr: *const FfiScopeHandle) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    unsafe { &*ptr }.0.attributes.bits()
}

/// Return the parent scope UUID as a C string, or null if there is no parent.
/// Caller must free the result with `nemo_relay_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiScopeHandle` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_handle_parent_uuid(
    ptr: *const FfiScopeHandle,
) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match &unsafe { &*ptr }.0.parent_uuid {
        Some(u) => str_to_c_string(&u.to_string()),
        None => std::ptr::null_mut(),
    }
}

/// Return the scope data as a JSON C string, or null if no data is set.
/// Caller must free the result with `nemo_relay_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiScopeHandle` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_handle_data(ptr: *const FfiScopeHandle) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match &unsafe { &*ptr }.0.data {
        Some(d) => json_to_c_string(d),
        None => std::ptr::null_mut(),
    }
}

/// Return the scope metadata as a JSON C string, or null if no metadata is set.
/// Caller must free the result with `nemo_relay_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiScopeHandle` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_scope_handle_metadata(
    ptr: *const FfiScopeHandle,
) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match &unsafe { &*ptr }.0.metadata {
        Some(m) => json_to_c_string(m),
        None => std::ptr::null_mut(),
    }
}

// ---------------------------------------------------------------------------
// Accessor functions for ToolHandle
// ---------------------------------------------------------------------------

/// Return the UUID of a tool handle as a C string. Caller must free the result.
///
/// # Safety
/// `ptr` must be a valid `FfiToolHandle` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_tool_handle_uuid(ptr: *const FfiToolHandle) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    str_to_c_string(&unsafe { &*ptr }.0.uuid.to_string())
}

/// Return the name of a tool handle as a C string. Caller must free the result.
///
/// # Safety
/// `ptr` must be a valid `FfiToolHandle` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_tool_handle_name(ptr: *const FfiToolHandle) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    str_to_c_string(&unsafe { &*ptr }.0.name)
}

/// Return the bitfield attributes of a tool handle. Returns 0 if `ptr` is null.
///
/// # Safety
/// `ptr` must be a valid `FfiToolHandle` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_tool_handle_attributes(ptr: *const FfiToolHandle) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    unsafe { &*ptr }.0.attributes.bits()
}

/// Return the parent scope UUID of a tool handle, or null if none.
/// Caller must free the result with `nemo_relay_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiToolHandle` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_tool_handle_parent_uuid(
    ptr: *const FfiToolHandle,
) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match &unsafe { &*ptr }.0.parent_uuid {
        Some(u) => str_to_c_string(&u.to_string()),
        None => std::ptr::null_mut(),
    }
}

// ---------------------------------------------------------------------------
// Accessor functions for LlmHandle
// ---------------------------------------------------------------------------

/// Return the UUID of an LLM handle as a C string. Caller must free the result.
///
/// # Safety
/// `ptr` must be a valid `FfiLLMHandle` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_llm_handle_uuid(ptr: *const FfiLLMHandle) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    str_to_c_string(&unsafe { &*ptr }.0.uuid.to_string())
}

/// Return the name of an LLM handle as a C string. Caller must free the result.
///
/// # Safety
/// `ptr` must be a valid `FfiLLMHandle` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_llm_handle_name(ptr: *const FfiLLMHandle) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    str_to_c_string(&unsafe { &*ptr }.0.name)
}

/// Return the bitfield attributes of an LLM handle. Returns 0 if `ptr` is null.
///
/// # Safety
/// `ptr` must be a valid `FfiLLMHandle` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_llm_handle_attributes(ptr: *const FfiLLMHandle) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    unsafe { &*ptr }.0.attributes.bits()
}

/// Return the parent scope UUID of an LLM handle, or null if none.
/// Caller must free the result with `nemo_relay_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiLLMHandle` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_llm_handle_parent_uuid(
    ptr: *const FfiLLMHandle,
) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match &unsafe { &*ptr }.0.parent_uuid {
        Some(u) => str_to_c_string(&u.to_string()),
        None => std::ptr::null_mut(),
    }
}

// ---------------------------------------------------------------------------
// LlmRequest construction + accessors
// ---------------------------------------------------------------------------

/// Create a new LLM request object. Returns a heap-allocated `FfiLLMRequest`
/// that must be freed with `nemo_relay_llm_request_free`. Returns null on
/// invalid input.
///
/// # Parameters
/// - `headers_json`: JSON object of headers/metadata, or null.
/// - `content_json`: JSON request content payload, or null.
///
/// # Safety
/// All string arguments must be valid null-terminated C strings or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_llm_request_new(
    headers_json: *const c_char,
    content_json: *const c_char,
) -> *mut FfiLLMRequest {
    let headers = match crate::convert::c_str_to_json(headers_json) {
        Some(Json::Object(m)) => m,
        _ => serde_json::Map::new(),
    };
    let content = crate::convert::c_str_to_json(content_json).unwrap_or(Json::Null);

    Box::into_raw(Box::new(FfiLLMRequest(LlmRequest { headers, content })))
}

/// Return the headers of an LLM request as a JSON C string. Caller must free the result.
///
/// # Safety
/// `ptr` must be a valid `FfiLLMRequest` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_llm_request_headers(ptr: *const FfiLLMRequest) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    json_to_c_string(&Json::Object(unsafe { &*ptr }.0.headers.clone()))
}

/// Return the content of an LLM request as a JSON C string. Caller must free the result.
///
/// # Safety
/// `ptr` must be a valid `FfiLLMRequest` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_llm_request_content(ptr: *const FfiLLMRequest) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    json_to_c_string(&unsafe { &*ptr }.0.content)
}

// ---------------------------------------------------------------------------
// Event accessors
// ---------------------------------------------------------------------------

/// Return the UUID of an event as a C string. Caller must free the result.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_event_uuid(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    str_to_c_string(&unsafe { &*ptr }.0.uuid().to_string())
}

/// Return the name of an event as a C string, or null if unnamed.
/// Caller must free the result with `nemo_relay_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_event_name(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    str_to_c_string(unsafe { &*ptr }.0.name())
}

/// Return the event discriminator as a C string.
/// Caller must free the result with `nemo_relay_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_event_kind(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    str_to_c_string(unsafe { &*ptr }.0.kind())
}

/// Return the canonical subscriber event JSON as a C string.
/// Caller must free the result with `nemo_relay_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_event_json(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match unsafe { &*ptr }.0.try_to_json_value() {
        Ok(value) => json_to_c_string(&value),
        Err(error) => {
            set_last_error(&error.to_string());
            std::ptr::null_mut()
        }
    }
}

/// Return the ATOF version as a C string.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_event_atof_version(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match &unsafe { &*ptr }.0 {
        Event::Scope(event) => str_to_c_string(&event.base.atof_version),
        Event::Mark(event) => str_to_c_string(&event.base.atof_version),
    }
}

/// Return the ATOF scope category as a C string, or null for mark events.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_event_scope_category(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match unsafe { &*ptr }.0.scope_category() {
        Some(nemo_relay::api::event::ScopeCategory::Start) => str_to_c_string("start"),
        Some(nemo_relay::api::event::ScopeCategory::End) => str_to_c_string("end"),
        None => std::ptr::null_mut(),
    }
}

/// Return the ATOF category as a C string, or null if absent.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_event_category(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match unsafe { &*ptr }.0.category() {
        Some(category) => str_to_c_string(category.as_str()),
        None => std::ptr::null_mut(),
    }
}

/// Return ATOF attributes as a JSON string array.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_event_attributes_json(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match unsafe { &*ptr }.0.attributes() {
        Some(attributes) => json_to_c_string(&serde_json::json!(attributes)),
        None => std::ptr::null_mut(),
    }
}

/// Return the ATOF category profile as a JSON C string, or null if absent.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_event_category_profile(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match unsafe { &*ptr }.0.category_profile() {
        Some(profile) => {
            let value = serde_json::to_value(profile).unwrap_or(Json::Null);
            json_to_c_string(&value)
        }
        None => std::ptr::null_mut(),
    }
}

/// Return the Agent Trajectory Observability Format (ATOF) data schema as a
/// JSON C string, or null if absent.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_event_data_schema(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match unsafe { &*ptr }.0.data_schema() {
        Some(schema) => {
            let value = serde_json::to_value(schema).unwrap_or(Json::Null);
            json_to_c_string(&value)
        }
        None => std::ptr::null_mut(),
    }
}

/// Return the raw attribute bitfield for an event, or 0 if it has none.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_event_attributes(ptr: *const FfiEvent) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    0
}

/// Return the event data as a JSON C string, or null if no data is set.
/// Caller must free the result with `nemo_relay_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_event_data(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match unsafe { &*ptr }.0.data() {
        Some(d) => json_to_c_string(d),
        None => std::ptr::null_mut(),
    }
}

/// Return the event metadata as a JSON C string, or null if no metadata is set.
/// Caller must free the result with `nemo_relay_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_event_metadata(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match unsafe { &*ptr }.0.metadata() {
        Some(m) => json_to_c_string(m),
        None => std::ptr::null_mut(),
    }
}

/// Return the event timestamp as an RFC 3339 C string. Caller must free the result.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_event_timestamp(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    str_to_c_string(&unsafe { &*ptr }.0.timestamp().to_rfc3339())
}

/// Return the event input as a JSON C string, or null if no input is set.
/// Caller must free the result with `nemo_relay_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_event_input(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match unsafe { &*ptr }.0.input() {
        Some(d) => json_to_c_string(d),
        None => std::ptr::null_mut(),
    }
}

/// Return the event output as a JSON C string, or null if no output is set.
/// Caller must free the result with `nemo_relay_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_event_output(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match unsafe { &*ptr }.0.output() {
        Some(d) => json_to_c_string(d),
        None => std::ptr::null_mut(),
    }
}

/// Return the event model name as a C string, or null if no model name is set.
/// Caller must free the result with `nemo_relay_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_event_model_name(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match unsafe { &*ptr }.0.model_name() {
        Some(s) => str_to_c_string(s),
        None => std::ptr::null_mut(),
    }
}

/// Return the event tool call ID as a C string, or null if no tool call ID is set.
/// Caller must free the result with `nemo_relay_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_event_tool_call_id(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match unsafe { &*ptr }.0.tool_call_id() {
        Some(s) => str_to_c_string(s),
        None => std::ptr::null_mut(),
    }
}

/// Return the event parent UUID as a C string, or null if no parent UUID is set.
/// Caller must free the result with `nemo_relay_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_event_parent_uuid(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match unsafe { &*ptr }.0.parent_uuid() {
        Some(u) => str_to_c_string(&u.to_string()),
        None => std::ptr::null_mut(),
    }
}

/// Return the event scope type as a C string, or null if no scope type is set.
/// Caller must free the result with `nemo_relay_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_event_scope_type(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match unsafe { &*ptr }.0.scope_type() {
        Some(st) => str_to_c_string(st.as_str()),
        None => std::ptr::null_mut(),
    }
}

/// Return the annotated request from an LLM start event as a JSON C string,
/// or null if not available (non-LLM events, or no codec was active).
/// Caller must free the result with `nemo_relay_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_event_annotated_request(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match unsafe { &*ptr }.0.annotated_request() {
        Some(a) => {
            let value = serde_json::to_value(a.as_ref()).unwrap_or_default();
            json_to_c_string(&value)
        }
        None => std::ptr::null_mut(),
    }
}

/// Return the annotated response from an LLM end event as a JSON C string,
/// or null if not available (non-LLM events, or no response codec was active).
/// Caller must free the result with `nemo_relay_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_event_annotated_response(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match unsafe { &*ptr }.0.annotated_response() {
        Some(a) => {
            let value = serde_json::to_value(a.as_ref()).unwrap_or_default();
            json_to_c_string(&value)
        }
        None => std::ptr::null_mut(),
    }
}

#[cfg(test)]
#[path = "../../tests/unit/types_tests.rs"]
mod tests;
