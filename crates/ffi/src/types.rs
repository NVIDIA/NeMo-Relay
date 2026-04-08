// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! C-compatible types exposed through the FFI boundary.
//!
//! This module defines opaque handle wrappers, enumerations, accessor functions,
//! and free functions for all types that cross the C FFI boundary. Each opaque
//! struct wraps a corresponding core type and is heap-allocated; the C consumer
//! sees only an opaque pointer. All returned C strings must be freed with
//! [`crate::convert::nat_nexus_string_free`], and all handles must be freed
//! with their corresponding `nat_nexus_*_free` function.

use libc::c_char;
use serde_json::Value as Json;

use nvidia_nat_nexus_core::types as core_types;

use crate::convert::{json_to_c_string, str_to_c_string};

// ---------------------------------------------------------------------------
// Opaque handle wrappers — each wraps a core type in a Box on the heap.
// The C consumer sees only `*mut FfiScopeHandle` etc.
// ---------------------------------------------------------------------------

/// Opaque handle representing an active execution scope.
pub struct FfiScopeHandle(pub core_types::ScopeHandle);
/// Opaque handle representing an active tool call.
pub struct FfiToolHandle(pub core_types::ToolHandle);
/// Opaque handle representing an active LLM call.
pub struct FfiLLMHandle(pub core_types::LLMHandle);
/// Opaque wrapper around an LLM request (headers, content).
pub struct FfiLLMRequest(pub core_types::LLMRequest);
/// Opaque wrapper around a lifecycle event emitted by the runtime.
pub struct FfiEvent(pub core_types::Event);
/// Opaque handle to an isolated scope stack for per-request/per-task isolation.
pub struct FfiScopeStack(pub nvidia_nat_nexus_core::ScopeStackHandle);
/// Opaque ATIF exporter handle.
pub struct FfiAtifExporter(pub nvidia_nat_nexus_core::atif::AtifExporter);
/// Opaque OpenTelemetry subscriber handle.
pub struct FfiOpenTelemetrySubscriber(pub nvidia_nat_nexus_otel::OpenTelemetrySubscriber);
/// Opaque OpenInference subscriber handle.
pub struct FfiOpenInferenceSubscriber(pub nvidia_nat_nexus_openinference::OpenInferenceSubscriber);
/// Opaque optimizer runtime handle.
pub struct FfiOptimizerRuntime(pub nvidia_nat_nexus_optimizer::OptimizerRuntime);
/// Opaque optimizer hosted plugin context.
///
/// This wrapper contains a borrowed raw pointer to an
/// `nvidia_nat_nexus_optimizer::HostedRegistrationContext`, not an owned heap allocation.
/// It is only valid for the duration of the hosted plugin registration callback that receives
/// it. C callers must not store the pointer, use it after the callback returns, or attempt to
/// free or drop it.
///
/// There is intentionally no `nat_nexus_optimizer_plugin_context_free` function because this FFI
/// wrapper does not own the underlying registration context.
pub struct FfiOptimizerPluginContext(
    pub *mut nvidia_nat_nexus_optimizer::HostedRegistrationContext,
);

// ---------------------------------------------------------------------------
// Enums exposed to C
// ---------------------------------------------------------------------------

/// The type of scope in the agent execution hierarchy.
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum NatNexusScopeType {
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

impl From<NatNexusScopeType> for core_types::ScopeType {
    fn from(v: NatNexusScopeType) -> Self {
        match v {
            NatNexusScopeType::Agent => core_types::ScopeType::Agent,
            NatNexusScopeType::Function => core_types::ScopeType::Function,
            NatNexusScopeType::Tool => core_types::ScopeType::Tool,
            NatNexusScopeType::Llm => core_types::ScopeType::Llm,
            NatNexusScopeType::Retriever => core_types::ScopeType::Retriever,
            NatNexusScopeType::Embedder => core_types::ScopeType::Embedder,
            NatNexusScopeType::Reranker => core_types::ScopeType::Reranker,
            NatNexusScopeType::Guardrail => core_types::ScopeType::Guardrail,
            NatNexusScopeType::Evaluator => core_types::ScopeType::Evaluator,
            NatNexusScopeType::Custom => core_types::ScopeType::Custom,
            NatNexusScopeType::Unknown => core_types::ScopeType::Unknown,
        }
    }
}

impl From<core_types::ScopeType> for NatNexusScopeType {
    fn from(v: core_types::ScopeType) -> Self {
        match v {
            core_types::ScopeType::Agent => NatNexusScopeType::Agent,
            core_types::ScopeType::Function => NatNexusScopeType::Function,
            core_types::ScopeType::Tool => NatNexusScopeType::Tool,
            core_types::ScopeType::Llm => NatNexusScopeType::Llm,
            core_types::ScopeType::Retriever => NatNexusScopeType::Retriever,
            core_types::ScopeType::Embedder => NatNexusScopeType::Embedder,
            core_types::ScopeType::Reranker => NatNexusScopeType::Reranker,
            core_types::ScopeType::Guardrail => NatNexusScopeType::Guardrail,
            core_types::ScopeType::Evaluator => NatNexusScopeType::Evaluator,
            core_types::ScopeType::Custom => NatNexusScopeType::Custom,
            core_types::ScopeType::Unknown => NatNexusScopeType::Unknown,
        }
    }
}

// ---------------------------------------------------------------------------
// Free functions for opaque handles
// ---------------------------------------------------------------------------

/// Free a scope handle previously returned by the runtime.
///
/// # Safety
/// `ptr` must be a valid pointer returned by an `nat_nexus_*` function, or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_scope_handle_free(ptr: *mut FfiScopeHandle) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

/// Free a tool handle previously returned by the runtime.
///
/// # Safety
/// `ptr` must be a valid pointer returned by an `nat_nexus_*` function, or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_tool_handle_free(ptr: *mut FfiToolHandle) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

/// Free an LLM handle previously returned by the runtime.
///
/// # Safety
/// `ptr` must be a valid pointer returned by an `nat_nexus_*` function, or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_llm_handle_free(ptr: *mut FfiLLMHandle) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

/// Free an LLM request object.
///
/// # Safety
/// `ptr` must be a valid pointer returned by an `nat_nexus_*` function, or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_llm_request_free(ptr: *mut FfiLLMRequest) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

/// Free an event object.
///
/// # Safety
/// `ptr` must be a valid pointer returned by an `nat_nexus_*` function, or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_event_free(ptr: *mut FfiEvent) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

/// Free a scope stack handle previously returned by `nat_nexus_scope_stack_create`.
///
/// # Safety
/// `ptr` must be a valid pointer returned by `nat_nexus_scope_stack_create`, or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_scope_stack_free(ptr: *mut FfiScopeStack) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

/// Free an ATIF exporter handle previously returned by `nat_nexus_atif_exporter_create`.
///
/// # Safety
/// `ptr` must be a valid pointer returned by `nat_nexus_atif_exporter_create`, or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_atif_exporter_free(ptr: *mut FfiAtifExporter) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

/// Free an OpenTelemetry subscriber handle previously returned by
/// `nat_nexus_otel_subscriber_create`.
///
/// # Safety
/// `ptr` must be a valid pointer returned by `nat_nexus_otel_subscriber_create`, or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_otel_subscriber_free(ptr: *mut FfiOpenTelemetrySubscriber) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

/// Free an OpenInference subscriber handle previously returned by
/// `nat_nexus_openinference_subscriber_create`.
///
/// # Safety
/// `ptr` must be a valid pointer returned by
/// `nat_nexus_openinference_subscriber_create`, or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_openinference_subscriber_free(
    ptr: *mut FfiOpenInferenceSubscriber,
) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

/// Free an optimizer runtime handle previously returned by `nat_nexus_optimizer_runtime_create`.
///
/// # Safety
/// `ptr` must be a valid pointer returned by `nat_nexus_optimizer_runtime_create`, or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_optimizer_runtime_free(ptr: *mut FfiOptimizerRuntime) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

// ---------------------------------------------------------------------------
// Accessor functions for ScopeHandle
// ---------------------------------------------------------------------------

/// Return the UUID of a scope handle as a C string. Caller must free the result
/// with `nat_nexus_string_free`. Returns null if `ptr` is null.
///
/// # Safety
/// `ptr` must be a valid `FfiScopeHandle` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_scope_handle_uuid(ptr: *const FfiScopeHandle) -> *mut c_char {
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
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_scope_handle_name(ptr: *const FfiScopeHandle) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    str_to_c_string(&unsafe { &*ptr }.0.name)
}

/// Return the scope type of a scope handle. Returns `Unknown` if `ptr` is null.
///
/// # Safety
/// `ptr` must be a valid `FfiScopeHandle` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_scope_handle_scope_type(
    ptr: *const FfiScopeHandle,
) -> NatNexusScopeType {
    if ptr.is_null() {
        return NatNexusScopeType::Unknown;
    }
    unsafe { &*ptr }.0.scope_type.into()
}

/// Return the bitfield attributes of a scope handle. Returns 0 if `ptr` is null.
///
/// # Safety
/// `ptr` must be a valid `FfiScopeHandle` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_scope_handle_attributes(ptr: *const FfiScopeHandle) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    unsafe { &*ptr }.0.attributes.bits()
}

/// Return the parent scope UUID as a C string, or null if there is no parent.
/// Caller must free the result with `nat_nexus_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiScopeHandle` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_scope_handle_parent_uuid(
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
/// Caller must free the result with `nat_nexus_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiScopeHandle` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_scope_handle_data(ptr: *const FfiScopeHandle) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match &unsafe { &*ptr }.0.data {
        Some(d) => json_to_c_string(d),
        None => std::ptr::null_mut(),
    }
}

/// Return the scope metadata as a JSON C string, or null if no metadata is set.
/// Caller must free the result with `nat_nexus_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiScopeHandle` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_scope_handle_metadata(
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
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_tool_handle_uuid(ptr: *const FfiToolHandle) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    str_to_c_string(&unsafe { &*ptr }.0.uuid.to_string())
}

/// Return the name of a tool handle as a C string. Caller must free the result.
///
/// # Safety
/// `ptr` must be a valid `FfiToolHandle` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_tool_handle_name(ptr: *const FfiToolHandle) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    str_to_c_string(&unsafe { &*ptr }.0.name)
}

/// Return the bitfield attributes of a tool handle. Returns 0 if `ptr` is null.
///
/// # Safety
/// `ptr` must be a valid `FfiToolHandle` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_tool_handle_attributes(ptr: *const FfiToolHandle) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    unsafe { &*ptr }.0.attributes.bits()
}

/// Return the parent scope UUID of a tool handle, or null if none.
/// Caller must free the result with `nat_nexus_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiToolHandle` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_tool_handle_parent_uuid(
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
// Accessor functions for LLMHandle
// ---------------------------------------------------------------------------

/// Return the UUID of an LLM handle as a C string. Caller must free the result.
///
/// # Safety
/// `ptr` must be a valid `FfiLLMHandle` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_llm_handle_uuid(ptr: *const FfiLLMHandle) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    str_to_c_string(&unsafe { &*ptr }.0.uuid.to_string())
}

/// Return the name of an LLM handle as a C string. Caller must free the result.
///
/// # Safety
/// `ptr` must be a valid `FfiLLMHandle` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_llm_handle_name(ptr: *const FfiLLMHandle) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    str_to_c_string(&unsafe { &*ptr }.0.name)
}

/// Return the bitfield attributes of an LLM handle. Returns 0 if `ptr` is null.
///
/// # Safety
/// `ptr` must be a valid `FfiLLMHandle` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_llm_handle_attributes(ptr: *const FfiLLMHandle) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    unsafe { &*ptr }.0.attributes.bits()
}

/// Return the parent scope UUID of an LLM handle, or null if none.
/// Caller must free the result with `nat_nexus_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiLLMHandle` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_llm_handle_parent_uuid(ptr: *const FfiLLMHandle) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match &unsafe { &*ptr }.0.parent_uuid {
        Some(u) => str_to_c_string(&u.to_string()),
        None => std::ptr::null_mut(),
    }
}

// ---------------------------------------------------------------------------
// LLMRequest construction + accessors
// ---------------------------------------------------------------------------

/// Create a new LLM request object. Returns a heap-allocated `FfiLLMRequest`
/// that must be freed with `nat_nexus_llm_request_free`. Returns null on
/// invalid input.
///
/// # Parameters
/// - `headers_json`: JSON object of headers/metadata, or null.
/// - `content_json`: JSON request content payload, or null.
///
/// # Safety
/// All string arguments must be valid null-terminated C strings or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_llm_request_new(
    headers_json: *const c_char,
    content_json: *const c_char,
) -> *mut FfiLLMRequest {
    let headers = match crate::convert::c_str_to_json(headers_json) {
        Some(Json::Object(m)) => m,
        _ => serde_json::Map::new(),
    };
    let content = crate::convert::c_str_to_json(content_json).unwrap_or(Json::Null);

    Box::into_raw(Box::new(FfiLLMRequest(core_types::LLMRequest {
        headers,
        content,
    })))
}

/// Return the headers of an LLM request as a JSON C string. Caller must free the result.
///
/// # Safety
/// `ptr` must be a valid `FfiLLMRequest` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_llm_request_headers(ptr: *const FfiLLMRequest) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    json_to_c_string(&Json::Object(unsafe { &*ptr }.0.headers.clone()))
}

/// Return the content of an LLM request as a JSON C string. Caller must free the result.
///
/// # Safety
/// `ptr` must be a valid `FfiLLMRequest` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_llm_request_content(ptr: *const FfiLLMRequest) -> *mut c_char {
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
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_event_uuid(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    str_to_c_string(&unsafe { &*ptr }.0.uuid().to_string())
}

/// Return the name of an event as a C string, or null if unnamed.
/// Caller must free the result with `nat_nexus_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_event_name(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    str_to_c_string(unsafe { &*ptr }.0.name())
}

/// Return the event discriminator as a C string.
/// Caller must free the result with `nat_nexus_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_event_kind(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    str_to_c_string(unsafe { &*ptr }.0.kind())
}

/// Return the raw attribute bitfield for an event, or 0 if it has none.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_event_attributes(ptr: *const FfiEvent) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    match unsafe { &*ptr }.0.attributes() {
        Some(core_types::HandleAttributes::Scope(attributes)) => attributes.bits(),
        Some(core_types::HandleAttributes::Tool(attributes)) => attributes.bits(),
        Some(core_types::HandleAttributes::Llm(attributes)) => attributes.bits(),
        None => 0,
    }
}

/// Return the event data as a JSON C string, or null if no data is set.
/// Caller must free the result with `nat_nexus_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_event_data(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match unsafe { &*ptr }.0.data() {
        Some(d) => json_to_c_string(d),
        None => std::ptr::null_mut(),
    }
}

/// Return the event metadata as a JSON C string, or null if no metadata is set.
/// Caller must free the result with `nat_nexus_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_event_metadata(ptr: *const FfiEvent) -> *mut c_char {
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
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_event_timestamp(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    str_to_c_string(&unsafe { &*ptr }.0.timestamp().to_rfc3339())
}

/// Return the event input as a JSON C string, or null if no input is set.
/// Caller must free the result with `nat_nexus_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_event_input(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match unsafe { &*ptr }.0.input() {
        Some(d) => json_to_c_string(d),
        None => std::ptr::null_mut(),
    }
}

/// Return the event output as a JSON C string, or null if no output is set.
/// Caller must free the result with `nat_nexus_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_event_output(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match unsafe { &*ptr }.0.output() {
        Some(d) => json_to_c_string(d),
        None => std::ptr::null_mut(),
    }
}

/// Return the event model name as a C string, or null if no model name is set.
/// Caller must free the result with `nat_nexus_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_event_model_name(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match unsafe { &*ptr }.0.model_name() {
        Some(s) => str_to_c_string(s),
        None => std::ptr::null_mut(),
    }
}

/// Return the event tool call ID as a C string, or null if no tool call ID is set.
/// Caller must free the result with `nat_nexus_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_event_tool_call_id(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match unsafe { &*ptr }.0.tool_call_id() {
        Some(s) => str_to_c_string(s),
        None => std::ptr::null_mut(),
    }
}

/// Return the event parent UUID as a C string, or null if no parent UUID is set.
/// Caller must free the result with `nat_nexus_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_event_parent_uuid(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match unsafe { &*ptr }.0.parent_uuid() {
        Some(u) => str_to_c_string(&u.to_string()),
        None => std::ptr::null_mut(),
    }
}

/// Return the event scope type as a C string, or null if no scope type is set.
/// Caller must free the result with `nat_nexus_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nat_nexus_event_scope_type(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match unsafe { &*ptr }.0.scope_type() {
        Some(st) => str_to_c_string(&format!("{:?}", st)),
        None => std::ptr::null_mut(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::{CStr, CString};

    use serde_json::json;
    use uuid::Uuid;

    fn take_string(ptr: *mut c_char) -> Option<String> {
        if ptr.is_null() {
            return None;
        }
        let value = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap().to_string();
        unsafe { crate::convert::nat_nexus_string_free(ptr) };
        Some(value)
    }

    #[test]
    fn test_scope_handle_accessors_and_null_metadata_guard() {
        assert!(unsafe { nat_nexus_scope_handle_metadata(std::ptr::null()) }.is_null());

        let parent_uuid = Uuid::new_v4();
        let handle = FfiScopeHandle(core_types::ScopeHandle::new(
            "scope".into(),
            core_types::ScopeType::Tool,
            core_types::ScopeAttributes::PARALLEL,
            Some(parent_uuid),
            Some(json!({"data": true})),
            Some(json!({"meta": true})),
        ));

        assert_eq!(
            take_string(unsafe { nat_nexus_scope_handle_name(&handle) }),
            Some("scope".into())
        );
        assert_eq!(
            unsafe { nat_nexus_scope_handle_scope_type(&handle) } as i32,
            NatNexusScopeType::Tool as i32
        );
        assert_eq!(
            unsafe { nat_nexus_scope_handle_attributes(&handle) },
            core_types::ScopeAttributes::PARALLEL.bits()
        );
        assert_eq!(
            take_string(unsafe { nat_nexus_scope_handle_parent_uuid(&handle) }),
            Some(parent_uuid.to_string())
        );
        assert_eq!(
            take_string(unsafe { nat_nexus_scope_handle_data(&handle) }),
            Some(r#"{"data":true}"#.into())
        );
        assert_eq!(
            take_string(unsafe { nat_nexus_scope_handle_metadata(&handle) }),
            Some(r#"{"meta":true}"#.into())
        );
    }

    #[test]
    fn test_scope_type_conversions_and_handle_null_guards() {
        let scope_types = [
            (NatNexusScopeType::Agent, core_types::ScopeType::Agent),
            (NatNexusScopeType::Function, core_types::ScopeType::Function),
            (NatNexusScopeType::Tool, core_types::ScopeType::Tool),
            (NatNexusScopeType::Llm, core_types::ScopeType::Llm),
            (
                NatNexusScopeType::Retriever,
                core_types::ScopeType::Retriever,
            ),
            (NatNexusScopeType::Embedder, core_types::ScopeType::Embedder),
            (NatNexusScopeType::Reranker, core_types::ScopeType::Reranker),
            (
                NatNexusScopeType::Guardrail,
                core_types::ScopeType::Guardrail,
            ),
            (
                NatNexusScopeType::Evaluator,
                core_types::ScopeType::Evaluator,
            ),
            (NatNexusScopeType::Custom, core_types::ScopeType::Custom),
            (NatNexusScopeType::Unknown, core_types::ScopeType::Unknown),
        ];

        for (ffi, core) in scope_types {
            let round_trip: NatNexusScopeType = core.into();
            assert_eq!(round_trip as i32, ffi as i32);
            let back: core_types::ScopeType = ffi.into();
            assert_eq!(back, core);
        }

        assert!(unsafe { nat_nexus_scope_handle_uuid(std::ptr::null()) }.is_null());
        assert!(unsafe { nat_nexus_scope_handle_name(std::ptr::null()) }.is_null());
        assert_eq!(
            unsafe { nat_nexus_scope_handle_scope_type(std::ptr::null()) } as i32,
            NatNexusScopeType::Unknown as i32
        );
        assert_eq!(
            unsafe { nat_nexus_scope_handle_attributes(std::ptr::null()) },
            0
        );
        assert!(unsafe { nat_nexus_scope_handle_parent_uuid(std::ptr::null()) }.is_null());
        assert!(unsafe { nat_nexus_scope_handle_data(std::ptr::null()) }.is_null());
        assert!(unsafe { nat_nexus_scope_handle_metadata(std::ptr::null()) }.is_null());
    }

    #[test]
    fn test_tool_and_llm_handle_accessors_and_null_guards() {
        let parent_uuid = Uuid::new_v4();
        let tool = FfiToolHandle(core_types::ToolHandle::new(
            "tool".into(),
            core_types::ToolAttributes::LOCAL,
            Some(parent_uuid),
            None,
            None,
        ));
        assert_eq!(
            take_string(unsafe { nat_nexus_tool_handle_uuid(&tool) }),
            Some(tool.0.uuid.to_string())
        );
        assert_eq!(
            take_string(unsafe { nat_nexus_tool_handle_name(&tool) }),
            Some("tool".into())
        );
        assert_eq!(
            unsafe { nat_nexus_tool_handle_attributes(&tool) },
            core_types::ToolAttributes::LOCAL.bits()
        );
        assert_eq!(
            take_string(unsafe { nat_nexus_tool_handle_parent_uuid(&tool) }),
            Some(parent_uuid.to_string())
        );

        let llm = FfiLLMHandle(core_types::LLMHandle::new(
            "llm".into(),
            core_types::LLMAttributes::STATELESS | core_types::LLMAttributes::STREAMING,
            Some(parent_uuid),
            None,
            None,
        ));
        assert_eq!(
            take_string(unsafe { nat_nexus_llm_handle_uuid(&llm) }),
            Some(llm.0.uuid.to_string())
        );
        assert_eq!(
            take_string(unsafe { nat_nexus_llm_handle_name(&llm) }),
            Some("llm".into())
        );
        assert_eq!(
            unsafe { nat_nexus_llm_handle_attributes(&llm) },
            (core_types::LLMAttributes::STATELESS | core_types::LLMAttributes::STREAMING).bits()
        );
        assert_eq!(
            take_string(unsafe { nat_nexus_llm_handle_parent_uuid(&llm) }),
            Some(parent_uuid.to_string())
        );

        assert!(unsafe { nat_nexus_tool_handle_uuid(std::ptr::null()) }.is_null());
        assert!(unsafe { nat_nexus_tool_handle_name(std::ptr::null()) }.is_null());
        assert_eq!(
            unsafe { nat_nexus_tool_handle_attributes(std::ptr::null()) },
            0
        );
        assert!(unsafe { nat_nexus_tool_handle_parent_uuid(std::ptr::null()) }.is_null());

        assert!(unsafe { nat_nexus_llm_handle_uuid(std::ptr::null()) }.is_null());
        assert!(unsafe { nat_nexus_llm_handle_name(std::ptr::null()) }.is_null());
        assert_eq!(
            unsafe { nat_nexus_llm_handle_attributes(std::ptr::null()) },
            0
        );
        assert!(unsafe { nat_nexus_llm_handle_parent_uuid(std::ptr::null()) }.is_null());
    }

    #[test]
    fn test_llm_request_null_inputs_event_null_guards_and_free_nulls() {
        let request_ptr = unsafe { nat_nexus_llm_request_new(std::ptr::null(), std::ptr::null()) };
        assert!(!request_ptr.is_null());
        assert_eq!(
            take_string(unsafe { nat_nexus_llm_request_headers(request_ptr) }),
            Some("{}".into())
        );
        assert_eq!(
            take_string(unsafe { nat_nexus_llm_request_content(request_ptr) }),
            Some("null".into())
        );
        unsafe { nat_nexus_llm_request_free(request_ptr) };

        assert!(unsafe { nat_nexus_llm_request_headers(std::ptr::null()) }.is_null());
        assert!(unsafe { nat_nexus_llm_request_content(std::ptr::null()) }.is_null());
        assert!(unsafe { nat_nexus_event_uuid(std::ptr::null()) }.is_null());
        assert!(unsafe { nat_nexus_event_name(std::ptr::null()) }.is_null());
        assert!(unsafe { nat_nexus_event_kind(std::ptr::null()) }.is_null());
        assert_eq!(unsafe { nat_nexus_event_attributes(std::ptr::null()) }, 0);
        assert!(unsafe { nat_nexus_event_data(std::ptr::null()) }.is_null());
        assert!(unsafe { nat_nexus_event_metadata(std::ptr::null()) }.is_null());
        assert!(unsafe { nat_nexus_event_timestamp(std::ptr::null()) }.is_null());
        assert!(unsafe { nat_nexus_event_input(std::ptr::null()) }.is_null());
        assert!(unsafe { nat_nexus_event_output(std::ptr::null()) }.is_null());
        assert!(unsafe { nat_nexus_event_model_name(std::ptr::null()) }.is_null());
        assert!(unsafe { nat_nexus_event_tool_call_id(std::ptr::null()) }.is_null());
        assert!(unsafe { nat_nexus_event_parent_uuid(std::ptr::null()) }.is_null());
        assert!(unsafe { nat_nexus_event_scope_type(std::ptr::null()) }.is_null());

        unsafe {
            nat_nexus_scope_handle_free(std::ptr::null_mut());
            nat_nexus_tool_handle_free(std::ptr::null_mut());
            nat_nexus_llm_handle_free(std::ptr::null_mut());
            nat_nexus_llm_request_free(std::ptr::null_mut());
            nat_nexus_event_free(std::ptr::null_mut());
            nat_nexus_scope_stack_free(std::ptr::null_mut());
            nat_nexus_atif_exporter_free(std::ptr::null_mut());
            nat_nexus_otel_subscriber_free(std::ptr::null_mut());
            nat_nexus_openinference_subscriber_free(std::ptr::null_mut());
            nat_nexus_optimizer_runtime_free(std::ptr::null_mut());
        }
    }

    #[test]
    fn test_llm_request_and_event_accessors() {
        let headers = CString::new(r#"{"header":"value"}"#).unwrap();
        let content = CString::new(r#"{"prompt":"hi"}"#).unwrap();
        let request_ptr = unsafe { nat_nexus_llm_request_new(headers.as_ptr(), content.as_ptr()) };
        assert!(!request_ptr.is_null());
        assert_eq!(
            take_string(unsafe { nat_nexus_llm_request_headers(request_ptr) }),
            Some(r#"{"header":"value"}"#.into())
        );
        assert_eq!(
            take_string(unsafe { nat_nexus_llm_request_content(request_ptr) }),
            Some(r#"{"prompt":"hi"}"#.into())
        );
        unsafe { nat_nexus_llm_request_free(request_ptr) };

        let parent_uuid = Uuid::new_v4();
        let scope_event = core_types::Event::scope_start(
            Some(parent_uuid),
            Uuid::new_v4(),
            "ffi-event",
            Some(json!({"data": 1})),
            Some(json!({"meta": 2})),
            core_types::ScopeAttributes::empty(),
            core_types::ScopeType::Guardrail,
        );
        let ffi_event = FfiEvent(scope_event.clone());

        assert_eq!(
            take_string(unsafe { nat_nexus_event_kind(&ffi_event) }),
            Some("ScopeStart".into())
        );
        assert_eq!(
            take_string(unsafe { nat_nexus_event_name(&ffi_event) }),
            Some("ffi-event".into())
        );
        assert_eq!(
            take_string(unsafe { nat_nexus_event_data(&ffi_event) }),
            Some(r#"{"data":1}"#.into())
        );
        assert_eq!(
            take_string(unsafe { nat_nexus_event_metadata(&ffi_event) }),
            Some(r#"{"meta":2}"#.into())
        );
        assert_eq!(
            take_string(unsafe { nat_nexus_event_scope_type(&ffi_event) }),
            Some("Guardrail".into())
        );
        assert_eq!(
            take_string(unsafe { nat_nexus_event_parent_uuid(&ffi_event) }),
            scope_event.parent_uuid().map(|uuid| uuid.to_string())
        );
        assert!(
            take_string(unsafe { nat_nexus_event_timestamp(&ffi_event) })
                .unwrap()
                .contains('T')
        );

        assert_eq!(
            take_string(unsafe { nat_nexus_event_input(&ffi_event) }),
            None
        );
        assert_eq!(
            take_string(unsafe { nat_nexus_event_output(&ffi_event) }),
            None
        );
        assert_eq!(
            take_string(unsafe { nat_nexus_event_model_name(&ffi_event) }),
            None
        );
        assert_eq!(
            take_string(unsafe { nat_nexus_event_tool_call_id(&ffi_event) }),
            None
        );

        let llm_event = core_types::Event::llm_start(
            Some(parent_uuid),
            Uuid::new_v4(),
            "ffi-llm",
            None,
            None,
            core_types::LLMAttributes::empty(),
            Some(json!({"input": true})),
            Some("model".into()),
        );
        let ffi_llm_event = FfiEvent(llm_event);
        assert_eq!(
            take_string(unsafe { nat_nexus_event_input(&ffi_llm_event) }),
            Some(r#"{"input":true}"#.into())
        );
        assert_eq!(
            take_string(unsafe { nat_nexus_event_model_name(&ffi_llm_event) }),
            Some("model".into())
        );
        assert_eq!(
            take_string(unsafe { nat_nexus_event_scope_type(&ffi_llm_event) }),
            None
        );

        let tool_event = core_types::Event::tool_end(
            Some(parent_uuid),
            Uuid::new_v4(),
            "ffi-tool",
            None,
            None,
            core_types::ToolAttributes::empty(),
            Some(json!({"output": true})),
            Some("tool-call-id".into()),
        );
        let ffi_tool_event = FfiEvent(tool_event);
        assert_eq!(
            take_string(unsafe { nat_nexus_event_output(&ffi_tool_event) }),
            Some(r#"{"output":true}"#.into())
        );
        assert_eq!(
            take_string(unsafe { nat_nexus_event_tool_call_id(&ffi_tool_event) }),
            Some("tool-call-id".into())
        );
        assert_eq!(
            take_string(unsafe { nat_nexus_event_scope_type(&ffi_tool_event) }),
            None
        );

        let mark_event =
            core_types::Event::mark(Some(parent_uuid), Uuid::new_v4(), "ffi-mark", None, None);
        let ffi_mark_event = FfiEvent(mark_event);
        assert_eq!(
            take_string(unsafe { nat_nexus_event_scope_type(&ffi_mark_event) }),
            None
        );
    }
}
