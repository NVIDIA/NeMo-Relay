// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! C-compatible types exposed through the FFI boundary.
//!
//! This module defines opaque handle wrappers, enumerations, accessor functions,
//! and free functions for all types that cross the C FFI boundary. Each opaque
//! struct wraps a corresponding core type and is heap-allocated; the C consumer
//! sees only an opaque pointer. All returned C strings must be freed with
//! [`crate::convert::nvmagic_string_free`], and all handles must be freed
//! with their corresponding `nvmagic_*_free` function.

use libc::c_char;
use serde_json::Value as Json;

use nvmagic_core::types as core_types;

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
pub struct FfiScopeStack(pub nvmagic_core::ScopeStackHandle);
/// Opaque ATIF exporter handle.
pub struct FfiAtifExporter(pub nvmagic_core::atif::AtifExporter);

// ---------------------------------------------------------------------------
// Enums exposed to C
// ---------------------------------------------------------------------------

/// The type of scope in the agent execution hierarchy.
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum NvMagicScopeType {
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

impl From<NvMagicScopeType> for core_types::ScopeType {
    fn from(v: NvMagicScopeType) -> Self {
        match v {
            NvMagicScopeType::Agent => core_types::ScopeType::Agent,
            NvMagicScopeType::Function => core_types::ScopeType::Function,
            NvMagicScopeType::Tool => core_types::ScopeType::Tool,
            NvMagicScopeType::Llm => core_types::ScopeType::Llm,
            NvMagicScopeType::Retriever => core_types::ScopeType::Retriever,
            NvMagicScopeType::Embedder => core_types::ScopeType::Embedder,
            NvMagicScopeType::Reranker => core_types::ScopeType::Reranker,
            NvMagicScopeType::Guardrail => core_types::ScopeType::Guardrail,
            NvMagicScopeType::Evaluator => core_types::ScopeType::Evaluator,
            NvMagicScopeType::Custom => core_types::ScopeType::Custom,
            NvMagicScopeType::Unknown => core_types::ScopeType::Unknown,
        }
    }
}

impl From<core_types::ScopeType> for NvMagicScopeType {
    fn from(v: core_types::ScopeType) -> Self {
        match v {
            core_types::ScopeType::Agent => NvMagicScopeType::Agent,
            core_types::ScopeType::Function => NvMagicScopeType::Function,
            core_types::ScopeType::Tool => NvMagicScopeType::Tool,
            core_types::ScopeType::Llm => NvMagicScopeType::Llm,
            core_types::ScopeType::Retriever => NvMagicScopeType::Retriever,
            core_types::ScopeType::Embedder => NvMagicScopeType::Embedder,
            core_types::ScopeType::Reranker => NvMagicScopeType::Reranker,
            core_types::ScopeType::Guardrail => NvMagicScopeType::Guardrail,
            core_types::ScopeType::Evaluator => NvMagicScopeType::Evaluator,
            core_types::ScopeType::Custom => NvMagicScopeType::Custom,
            core_types::ScopeType::Unknown => NvMagicScopeType::Unknown,
        }
    }
}

/// The type of lifecycle event emitted by the runtime.
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum NvMagicEventType {
    /// A scope or operation has started.
    Start = 0,
    /// A scope or operation has ended.
    End = 1,
    /// A point-in-time marker event.
    Mark = 2,
}

impl From<core_types::EventType> for NvMagicEventType {
    fn from(v: core_types::EventType) -> Self {
        match v {
            core_types::EventType::Start => NvMagicEventType::Start,
            core_types::EventType::End => NvMagicEventType::End,
            core_types::EventType::Mark => NvMagicEventType::Mark,
        }
    }
}

// ---------------------------------------------------------------------------
// Free functions for opaque handles
// ---------------------------------------------------------------------------

/// Free a scope handle previously returned by the runtime.
///
/// # Safety
/// `ptr` must be a valid pointer returned by an `nvmagic_*` function, or null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_scope_handle_free(ptr: *mut FfiScopeHandle) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

/// Free a tool handle previously returned by the runtime.
///
/// # Safety
/// `ptr` must be a valid pointer returned by an `nvmagic_*` function, or null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_tool_handle_free(ptr: *mut FfiToolHandle) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

/// Free an LLM handle previously returned by the runtime.
///
/// # Safety
/// `ptr` must be a valid pointer returned by an `nvmagic_*` function, or null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_llm_handle_free(ptr: *mut FfiLLMHandle) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

/// Free an LLM request object.
///
/// # Safety
/// `ptr` must be a valid pointer returned by an `nvmagic_*` function, or null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_llm_request_free(ptr: *mut FfiLLMRequest) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

/// Free an event object.
///
/// # Safety
/// `ptr` must be a valid pointer returned by an `nvmagic_*` function, or null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_event_free(ptr: *mut FfiEvent) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

/// Free a scope stack handle previously returned by `nvmagic_scope_stack_create`.
///
/// # Safety
/// `ptr` must be a valid pointer returned by `nvmagic_scope_stack_create`, or null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_scope_stack_free(ptr: *mut FfiScopeStack) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

/// Free an ATIF exporter handle previously returned by `nvmagic_atif_exporter_create`.
///
/// # Safety
/// `ptr` must be a valid pointer returned by `nvmagic_atif_exporter_create`, or null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_atif_exporter_free(ptr: *mut FfiAtifExporter) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

// ---------------------------------------------------------------------------
// Accessor functions for ScopeHandle
// ---------------------------------------------------------------------------

/// Return the UUID of a scope handle as a C string. Caller must free the result
/// with `nvmagic_string_free`. Returns null if `ptr` is null.
///
/// # Safety
/// `ptr` must be a valid `FfiScopeHandle` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_scope_handle_uuid(ptr: *const FfiScopeHandle) -> *mut c_char {
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
pub unsafe extern "C" fn nvmagic_scope_handle_name(ptr: *const FfiScopeHandle) -> *mut c_char {
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
pub unsafe extern "C" fn nvmagic_scope_handle_scope_type(
    ptr: *const FfiScopeHandle,
) -> NvMagicScopeType {
    if ptr.is_null() {
        return NvMagicScopeType::Unknown;
    }
    unsafe { &*ptr }.0.scope_type.into()
}

/// Return the bitfield attributes of a scope handle. Returns 0 if `ptr` is null.
///
/// # Safety
/// `ptr` must be a valid `FfiScopeHandle` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_scope_handle_attributes(ptr: *const FfiScopeHandle) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    unsafe { &*ptr }.0.attributes.bits()
}

/// Return the parent scope UUID as a C string, or null if there is no parent.
/// Caller must free the result with `nvmagic_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiScopeHandle` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_scope_handle_parent_uuid(
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
/// Caller must free the result with `nvmagic_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiScopeHandle` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_scope_handle_data(ptr: *const FfiScopeHandle) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match &unsafe { &*ptr }.0.data {
        Some(d) => json_to_c_string(d),
        None => std::ptr::null_mut(),
    }
}

/// Return the scope metadata as a JSON C string, or null if no metadata is set.
/// Caller must free the result with `nvmagic_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiScopeHandle` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_scope_handle_metadata(ptr: *const FfiScopeHandle) -> *mut c_char {
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
pub unsafe extern "C" fn nvmagic_tool_handle_uuid(ptr: *const FfiToolHandle) -> *mut c_char {
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
pub unsafe extern "C" fn nvmagic_tool_handle_name(ptr: *const FfiToolHandle) -> *mut c_char {
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
pub unsafe extern "C" fn nvmagic_tool_handle_attributes(ptr: *const FfiToolHandle) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    unsafe { &*ptr }.0.attributes.bits()
}

/// Return the parent scope UUID of a tool handle, or null if none.
/// Caller must free the result with `nvmagic_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiToolHandle` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_tool_handle_parent_uuid(ptr: *const FfiToolHandle) -> *mut c_char {
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
pub unsafe extern "C" fn nvmagic_llm_handle_uuid(ptr: *const FfiLLMHandle) -> *mut c_char {
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
pub unsafe extern "C" fn nvmagic_llm_handle_name(ptr: *const FfiLLMHandle) -> *mut c_char {
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
pub unsafe extern "C" fn nvmagic_llm_handle_attributes(ptr: *const FfiLLMHandle) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    unsafe { &*ptr }.0.attributes.bits()
}

/// Return the parent scope UUID of an LLM handle, or null if none.
/// Caller must free the result with `nvmagic_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiLLMHandle` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_llm_handle_parent_uuid(ptr: *const FfiLLMHandle) -> *mut c_char {
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
/// that must be freed with `nvmagic_llm_request_free`. Returns null on
/// invalid input.
///
/// # Parameters
/// - `headers_json`: JSON object of headers/metadata, or null.
/// - `content_json`: JSON request content payload, or null.
///
/// # Safety
/// All string arguments must be valid null-terminated C strings or null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_llm_request_new(
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
pub unsafe extern "C" fn nvmagic_llm_request_headers(ptr: *const FfiLLMRequest) -> *mut c_char {
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
pub unsafe extern "C" fn nvmagic_llm_request_content(ptr: *const FfiLLMRequest) -> *mut c_char {
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
pub unsafe extern "C" fn nvmagic_event_uuid(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    str_to_c_string(&unsafe { &*ptr }.0.uuid.to_string())
}

/// Return the name of an event as a C string, or null if unnamed.
/// Caller must free the result with `nvmagic_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_event_name(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match &unsafe { &*ptr }.0.name {
        Some(n) => str_to_c_string(n),
        None => std::ptr::null_mut(),
    }
}

/// Return the event type. Returns `Mark` if `ptr` is null.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_event_type(ptr: *const FfiEvent) -> NvMagicEventType {
    if ptr.is_null() {
        return NvMagicEventType::Mark;
    }
    unsafe { &*ptr }.0.event_type.into()
}

/// Return the event data as a JSON C string, or null if no data is set.
/// Caller must free the result with `nvmagic_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_event_data(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match &unsafe { &*ptr }.0.data {
        Some(d) => json_to_c_string(d),
        None => std::ptr::null_mut(),
    }
}

/// Return the event metadata as a JSON C string, or null if no metadata is set.
/// Caller must free the result with `nvmagic_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_event_metadata(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match &unsafe { &*ptr }.0.metadata {
        Some(m) => json_to_c_string(m),
        None => std::ptr::null_mut(),
    }
}

/// Return the event timestamp as an RFC 3339 C string. Caller must free the result.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_event_timestamp(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    str_to_c_string(&unsafe { &*ptr }.0.timestamp.to_rfc3339())
}

/// Return the event input as a JSON C string, or null if no input is set.
/// Caller must free the result with `nvmagic_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_event_input(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match &unsafe { &*ptr }.0.input {
        Some(d) => json_to_c_string(d),
        None => std::ptr::null_mut(),
    }
}

/// Return the event output as a JSON C string, or null if no output is set.
/// Caller must free the result with `nvmagic_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_event_output(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match &unsafe { &*ptr }.0.output {
        Some(d) => json_to_c_string(d),
        None => std::ptr::null_mut(),
    }
}

/// Return the event model name as a C string, or null if no model name is set.
/// Caller must free the result with `nvmagic_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_event_model_name(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match &unsafe { &*ptr }.0.model_name {
        Some(s) => str_to_c_string(s),
        None => std::ptr::null_mut(),
    }
}

/// Return the event tool call ID as a C string, or null if no tool call ID is set.
/// Caller must free the result with `nvmagic_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_event_tool_call_id(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match &unsafe { &*ptr }.0.tool_call_id {
        Some(s) => str_to_c_string(s),
        None => std::ptr::null_mut(),
    }
}

/// Return the event root UUID as a C string, or null if no root UUID is set.
/// Caller must free the result with `nvmagic_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_event_root_uuid(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match &unsafe { &*ptr }.0.root_uuid {
        Some(u) => str_to_c_string(&u.to_string()),
        None => std::ptr::null_mut(),
    }
}

/// Return the event parent UUID as a C string, or null if no parent UUID is set.
/// Caller must free the result with `nvmagic_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_event_parent_uuid(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match &unsafe { &*ptr }.0.parent_uuid {
        Some(u) => str_to_c_string(&u.to_string()),
        None => std::ptr::null_mut(),
    }
}

/// Return the event scope type as a C string, or null if no scope type is set.
/// Caller must free the result with `nvmagic_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nvmagic_event_scope_type(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match &unsafe { &*ptr }.0.scope_type {
        Some(st) => str_to_c_string(&format!("{:?}", st)),
        None => std::ptr::null_mut(),
    }
}
