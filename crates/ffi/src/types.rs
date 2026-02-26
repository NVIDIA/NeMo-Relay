//! C-compatible types exposed through the FFI boundary.
//!
//! This module defines opaque handle wrappers, enumerations, accessor functions,
//! and free functions for all types that cross the C FFI boundary. Each opaque
//! struct wraps a corresponding core type and is heap-allocated; the C consumer
//! sees only an opaque pointer. All returned C strings must be freed with
//! [`crate::convert::nv_agentrt_string_free`], and all handles must be freed
//! with their corresponding `nv_agentrt_*_free` function.

use libc::c_char;
use serde_json::Value as Json;

use nvagentrt_core::types as core_types;

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
/// Opaque wrapper around an LLM HTTP request (method, URL, headers, body).
pub struct FfiLLMRequest(pub core_types::LLMRequest);
/// Opaque wrapper around a lifecycle event emitted by the runtime.
pub struct FfiEvent(pub core_types::Event);
/// Opaque wrapper around a server-sent event (SSE) used in LLM streaming.
pub struct FfiSseEvent(pub core_types::SseEvent);

// ---------------------------------------------------------------------------
// Enums exposed to C
// ---------------------------------------------------------------------------

/// The type of scope in the agent execution hierarchy.
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum NvAgentRtScopeType {
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

impl From<NvAgentRtScopeType> for core_types::ScopeType {
    fn from(v: NvAgentRtScopeType) -> Self {
        match v {
            NvAgentRtScopeType::Agent => core_types::ScopeType::Agent,
            NvAgentRtScopeType::Function => core_types::ScopeType::Function,
            NvAgentRtScopeType::Tool => core_types::ScopeType::Tool,
            NvAgentRtScopeType::Llm => core_types::ScopeType::Llm,
            NvAgentRtScopeType::Retriever => core_types::ScopeType::Retriever,
            NvAgentRtScopeType::Embedder => core_types::ScopeType::Embedder,
            NvAgentRtScopeType::Reranker => core_types::ScopeType::Reranker,
            NvAgentRtScopeType::Guardrail => core_types::ScopeType::Guardrail,
            NvAgentRtScopeType::Evaluator => core_types::ScopeType::Evaluator,
            NvAgentRtScopeType::Custom => core_types::ScopeType::Custom,
            NvAgentRtScopeType::Unknown => core_types::ScopeType::Unknown,
        }
    }
}

impl From<core_types::ScopeType> for NvAgentRtScopeType {
    fn from(v: core_types::ScopeType) -> Self {
        match v {
            core_types::ScopeType::Agent => NvAgentRtScopeType::Agent,
            core_types::ScopeType::Function => NvAgentRtScopeType::Function,
            core_types::ScopeType::Tool => NvAgentRtScopeType::Tool,
            core_types::ScopeType::Llm => NvAgentRtScopeType::Llm,
            core_types::ScopeType::Retriever => NvAgentRtScopeType::Retriever,
            core_types::ScopeType::Embedder => NvAgentRtScopeType::Embedder,
            core_types::ScopeType::Reranker => NvAgentRtScopeType::Reranker,
            core_types::ScopeType::Guardrail => NvAgentRtScopeType::Guardrail,
            core_types::ScopeType::Evaluator => NvAgentRtScopeType::Evaluator,
            core_types::ScopeType::Custom => NvAgentRtScopeType::Custom,
            core_types::ScopeType::Unknown => NvAgentRtScopeType::Unknown,
        }
    }
}

/// The type of lifecycle event emitted by the runtime.
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum NvAgentRtEventType {
    /// A scope or operation has started.
    Start = 0,
    /// A scope or operation has ended.
    End = 1,
    /// A point-in-time marker event.
    Mark = 2,
}

impl From<core_types::EventType> for NvAgentRtEventType {
    fn from(v: core_types::EventType) -> Self {
        match v {
            core_types::EventType::Start => NvAgentRtEventType::Start,
            core_types::EventType::End => NvAgentRtEventType::End,
            core_types::EventType::Mark => NvAgentRtEventType::Mark,
        }
    }
}

// ---------------------------------------------------------------------------
// Free functions for opaque handles
// ---------------------------------------------------------------------------

/// Free a scope handle previously returned by the runtime.
///
/// # Safety
/// `ptr` must be a valid pointer returned by an `nv_agentrt_*` function, or null.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_scope_handle_free(ptr: *mut FfiScopeHandle) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

/// Free a tool handle previously returned by the runtime.
///
/// # Safety
/// `ptr` must be a valid pointer returned by an `nv_agentrt_*` function, or null.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_tool_handle_free(ptr: *mut FfiToolHandle) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

/// Free an LLM handle previously returned by the runtime.
///
/// # Safety
/// `ptr` must be a valid pointer returned by an `nv_agentrt_*` function, or null.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_llm_handle_free(ptr: *mut FfiLLMHandle) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

/// Free an LLM request object.
///
/// # Safety
/// `ptr` must be a valid pointer returned by an `nv_agentrt_*` function, or null.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_llm_request_free(ptr: *mut FfiLLMRequest) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

/// Free an event object.
///
/// # Safety
/// `ptr` must be a valid pointer returned by an `nv_agentrt_*` function, or null.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_event_free(ptr: *mut FfiEvent) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

/// Free an SSE event object.
///
/// # Safety
/// `ptr` must be a valid pointer returned by an `nv_agentrt_*` function, or null.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_sse_event_free(ptr: *mut FfiSseEvent) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
    }
}

// ---------------------------------------------------------------------------
// Accessor functions for ScopeHandle
// ---------------------------------------------------------------------------

/// Return the UUID of a scope handle as a C string. Caller must free the result
/// with `nv_agentrt_string_free`. Returns null if `ptr` is null.
///
/// # Safety
/// `ptr` must be a valid `FfiScopeHandle` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_scope_handle_uuid(ptr: *const FfiScopeHandle) -> *mut c_char {
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
pub unsafe extern "C" fn nv_agentrt_scope_handle_name(ptr: *const FfiScopeHandle) -> *mut c_char {
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
pub unsafe extern "C" fn nv_agentrt_scope_handle_scope_type(
    ptr: *const FfiScopeHandle,
) -> NvAgentRtScopeType {
    if ptr.is_null() {
        return NvAgentRtScopeType::Unknown;
    }
    unsafe { &*ptr }.0.scope_type.into()
}

/// Return the bitfield attributes of a scope handle. Returns 0 if `ptr` is null.
///
/// # Safety
/// `ptr` must be a valid `FfiScopeHandle` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_scope_handle_attributes(ptr: *const FfiScopeHandle) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    unsafe { &*ptr }.0.attributes.bits()
}

/// Return the parent scope UUID as a C string, or null if there is no parent.
/// Caller must free the result with `nv_agentrt_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiScopeHandle` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_scope_handle_parent_uuid(
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
/// Caller must free the result with `nv_agentrt_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiScopeHandle` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_scope_handle_data(ptr: *const FfiScopeHandle) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match &unsafe { &*ptr }.0.data {
        Some(d) => json_to_c_string(d),
        None => std::ptr::null_mut(),
    }
}

/// Return the scope metadata as a JSON C string, or null if no metadata is set.
/// Caller must free the result with `nv_agentrt_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiScopeHandle` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_scope_handle_metadata(
    ptr: *const FfiScopeHandle,
) -> *mut c_char {
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
pub unsafe extern "C" fn nv_agentrt_tool_handle_uuid(ptr: *const FfiToolHandle) -> *mut c_char {
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
pub unsafe extern "C" fn nv_agentrt_tool_handle_name(ptr: *const FfiToolHandle) -> *mut c_char {
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
pub unsafe extern "C" fn nv_agentrt_tool_handle_attributes(ptr: *const FfiToolHandle) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    unsafe { &*ptr }.0.attributes.bits()
}

/// Return the parent scope UUID of a tool handle, or null if none.
/// Caller must free the result with `nv_agentrt_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiToolHandle` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_tool_handle_parent_uuid(
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
pub unsafe extern "C" fn nv_agentrt_llm_handle_uuid(ptr: *const FfiLLMHandle) -> *mut c_char {
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
pub unsafe extern "C" fn nv_agentrt_llm_handle_name(ptr: *const FfiLLMHandle) -> *mut c_char {
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
pub unsafe extern "C" fn nv_agentrt_llm_handle_attributes(ptr: *const FfiLLMHandle) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    unsafe { &*ptr }.0.attributes.bits()
}

/// Return the parent scope UUID of an LLM handle, or null if none.
/// Caller must free the result with `nv_agentrt_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiLLMHandle` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_llm_handle_parent_uuid(
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
// LLMRequest construction + accessors
// ---------------------------------------------------------------------------

/// Create a new LLM request object. Returns a heap-allocated `FfiLLMRequest`
/// that must be freed with `nv_agentrt_llm_request_free`. Returns null on
/// invalid input.
///
/// # Parameters
/// - `method`: HTTP method (e.g., "POST").
/// - `url`: The endpoint URL.
/// - `headers_json`: JSON object of HTTP headers, or null.
/// - `body_json`: JSON request body, or null.
///
/// # Safety
/// All string arguments must be valid null-terminated C strings or null.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_llm_request_new(
    method: *const c_char,
    url: *const c_char,
    headers_json: *const c_char,
    body_json: *const c_char,
) -> *mut FfiLLMRequest {
    let method = match crate::convert::c_str_to_string(method) {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    let url = match crate::convert::c_str_to_string(url) {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };
    let headers = match crate::convert::c_str_to_json(headers_json) {
        Some(Json::Object(m)) => m,
        _ => serde_json::Map::new(),
    };
    let body = crate::convert::c_str_to_json(body_json).unwrap_or(Json::Null);

    Box::into_raw(Box::new(FfiLLMRequest(core_types::LLMRequest {
        method,
        url,
        headers,
        body,
    })))
}

/// Return the HTTP method of an LLM request as a C string. Caller must free the result.
///
/// # Safety
/// `ptr` must be a valid `FfiLLMRequest` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_llm_request_method(ptr: *const FfiLLMRequest) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    str_to_c_string(&unsafe { &*ptr }.0.method)
}

/// Return the URL of an LLM request as a C string. Caller must free the result.
///
/// # Safety
/// `ptr` must be a valid `FfiLLMRequest` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_llm_request_url(ptr: *const FfiLLMRequest) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    str_to_c_string(&unsafe { &*ptr }.0.url)
}

/// Return the headers of an LLM request as a JSON C string. Caller must free the result.
///
/// # Safety
/// `ptr` must be a valid `FfiLLMRequest` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_llm_request_headers(ptr: *const FfiLLMRequest) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    json_to_c_string(&Json::Object(unsafe { &*ptr }.0.headers.clone()))
}

/// Return the body of an LLM request as a JSON C string. Caller must free the result.
///
/// # Safety
/// `ptr` must be a valid `FfiLLMRequest` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_llm_request_body(ptr: *const FfiLLMRequest) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    json_to_c_string(&unsafe { &*ptr }.0.body)
}

// ---------------------------------------------------------------------------
// Event accessors
// ---------------------------------------------------------------------------

/// Return the UUID of an event as a C string. Caller must free the result.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_event_uuid(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    str_to_c_string(&unsafe { &*ptr }.0.uuid.to_string())
}

/// Return the name of an event as a C string, or null if unnamed.
/// Caller must free the result with `nv_agentrt_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_event_name(ptr: *const FfiEvent) -> *mut c_char {
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
pub unsafe extern "C" fn nv_agentrt_event_type(ptr: *const FfiEvent) -> NvAgentRtEventType {
    if ptr.is_null() {
        return NvAgentRtEventType::Mark;
    }
    unsafe { &*ptr }.0.event_type.into()
}

/// Return the event data as a JSON C string, or null if no data is set.
/// Caller must free the result with `nv_agentrt_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_event_data(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    match &unsafe { &*ptr }.0.data {
        Some(d) => json_to_c_string(d),
        None => std::ptr::null_mut(),
    }
}

/// Return the event metadata as a JSON C string, or null if no metadata is set.
/// Caller must free the result with `nv_agentrt_string_free`.
///
/// # Safety
/// `ptr` must be a valid `FfiEvent` pointer or null.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_event_metadata(ptr: *const FfiEvent) -> *mut c_char {
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
pub unsafe extern "C" fn nv_agentrt_event_timestamp(ptr: *const FfiEvent) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }
    str_to_c_string(&unsafe { &*ptr }.0.timestamp.to_rfc3339())
}
