//! Top-level FFI API functions exported as `extern "C"`.
//!
//! Each function clears the thread-local error before executing and returns an
//! [`NvAgentRtStatus`]. On failure, call [`nv_agentrt_last_error`] to retrieve
//! the error message.

use std::sync::OnceLock;

use libc::c_char;
use nvagentrt_core as core;
use nvagentrt_core::types as core_types;
use tokio::runtime::Runtime;
use tokio_stream::StreamExt;

use crate::callable::*;
use crate::convert::*;
use crate::error::*;
use crate::types::*;

// ---------------------------------------------------------------------------
// Tokio runtime singleton (for blocking on async functions)
// ---------------------------------------------------------------------------

fn tokio_runtime() -> &'static Runtime {
    static RT: OnceLock<Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed to create tokio runtime")
    })
}

// ---------------------------------------------------------------------------
// Scope / handle operations
// ---------------------------------------------------------------------------

/// Retrieve the current scope handle from the thread-local scope stack.
///
/// # Parameters
/// - `out`: On success, receives a heap-allocated `FfiScopeHandle` that must be
///   freed with `nv_agentrt_scope_handle_free`.
///
/// # Safety
/// `out` must be a valid, non-null pointer.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_get_handle(out: *mut *mut FfiScopeHandle) -> NvAgentRtStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NvAgentRtStatus::NullPointer;
    }
    match core::nv_agentrt_get_handle() {
        Ok(h) => {
            unsafe { *out = Box::into_raw(Box::new(FfiScopeHandle(h))) };
            NvAgentRtStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// Push a new scope onto the scope stack.
///
/// # Parameters
/// - `name`: Null-terminated scope name.
/// - `scope_type`: The type of scope to create.
/// - `parent`: Optional parent scope handle, or null for auto-parenting.
/// - `attributes`: Bitfield of scope attributes.
/// - `out`: On success, receives a heap-allocated `FfiScopeHandle`.
///
/// # Safety
/// `name` must be a valid C string. `out` must be non-null. `parent` may be null.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_push_scope(
    name: *const c_char,
    scope_type: NvAgentRtScopeType,
    parent: *const FfiScopeHandle,
    attributes: u32,
    out: *mut *mut FfiScopeHandle,
) -> NvAgentRtStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NvAgentRtStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let parent_ref = if parent.is_null() {
        None
    } else {
        Some(&unsafe { &*parent }.0)
    };
    let attrs = core_types::ScopeAttributes::from_bits_truncate(attributes);

    match core::nv_agentrt_push_scope(&name, scope_type.into(), parent_ref, attrs) {
        Ok(h) => {
            unsafe { *out = Box::into_raw(Box::new(FfiScopeHandle(h))) };
            NvAgentRtStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// Pop a scope from the scope stack by its handle.
///
/// # Parameters
/// - `handle`: The scope handle to pop.
///
/// # Safety
/// `handle` must be a valid, non-null `FfiScopeHandle` pointer.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_pop_scope(handle: *const FfiScopeHandle) -> NvAgentRtStatus {
    clear_last_error();
    if handle.is_null() {
        set_last_error("handle is null");
        return NvAgentRtStatus::NullPointer;
    }
    match core::nv_agentrt_pop_scope(&unsafe { &*handle }.0.uuid) {
        Ok(()) => NvAgentRtStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Emit a named lifecycle event.
///
/// # Parameters
/// - `name`: Null-terminated event name.
/// - `parent`: Optional parent scope handle, or null.
/// - `data_json`: Optional JSON data payload, or null.
/// - `metadata_json`: Optional JSON metadata payload, or null.
///
/// # Safety
/// `name` must be a valid C string. Other pointer args may be null.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_event(
    name: *const c_char,
    parent: *const FfiScopeHandle,
    data_json: *const c_char,
    metadata_json: *const c_char,
) -> NvAgentRtStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let parent_ref = if parent.is_null() {
        None
    } else {
        Some(&unsafe { &*parent }.0)
    };
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NvAgentRtStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NvAgentRtStatus::InvalidJson,
    };

    match core::nv_agentrt_event(&name, parent_ref, data, metadata) {
        Ok(()) => NvAgentRtStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// Tool lifecycle
// ---------------------------------------------------------------------------

/// Begin a tool call, running pre-call guardrails and intercepts.
///
/// # Parameters
/// - `name`: Null-terminated tool name.
/// - `args_json`: Tool arguments as a JSON C string.
/// - `parent`: Optional parent scope handle, or null.
/// - `attributes`: Bitfield of tool attributes.
/// - `data_json`: Optional JSON data, or null.
/// - `metadata_json`: Optional JSON metadata, or null.
/// - `out`: On success, receives a heap-allocated `FfiToolHandle`.
///
/// # Safety
/// `name` and `args_json` must be valid C strings. `out` must be non-null.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_tool_call(
    name: *const c_char,
    args_json: *const c_char,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    out: *mut *mut FfiToolHandle,
) -> NvAgentRtStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NvAgentRtStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let args = match c_str_to_json(args_json) {
        Some(a) => a,
        None => return NvAgentRtStatus::InvalidJson,
    };
    let parent_ref = if parent.is_null() {
        None
    } else {
        Some(&unsafe { &*parent }.0)
    };
    let attrs = core_types::ToolAttributes::from_bits_truncate(attributes);
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NvAgentRtStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NvAgentRtStatus::InvalidJson,
    };

    match core::nv_agentrt_tool_call(&name, args, parent_ref, attrs, data, metadata) {
        Ok(h) => {
            unsafe { *out = Box::into_raw(Box::new(FfiToolHandle(h))) };
            NvAgentRtStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// End a tool call, running post-call guardrails and intercepts.
///
/// # Parameters
/// - `handle`: The tool handle from `nv_agentrt_tool_call`.
/// - `result_json`: Tool result as a JSON C string.
/// - `data_json`: Optional JSON data, or null.
/// - `metadata_json`: Optional JSON metadata, or null.
///
/// # Safety
/// `handle` and `result_json` must be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_tool_call_end(
    handle: *const FfiToolHandle,
    result_json: *const c_char,
    data_json: *const c_char,
    metadata_json: *const c_char,
) -> NvAgentRtStatus {
    clear_last_error();
    if handle.is_null() {
        set_last_error("handle is null");
        return NvAgentRtStatus::NullPointer;
    }
    let result = match c_str_to_json(result_json) {
        Some(r) => r,
        None => return NvAgentRtStatus::InvalidJson,
    };
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NvAgentRtStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NvAgentRtStatus::InvalidJson,
    };

    match core::nv_agentrt_tool_call_end(&unsafe { &*handle }.0, result, data, metadata) {
        Ok(()) => NvAgentRtStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Execute a tool call end-to-end: begin, invoke the callback, apply guardrails
/// and intercepts, then end. Blocks the calling thread until completion.
///
/// # Parameters
/// - `name`: Null-terminated tool name.
/// - `args_json`: Tool arguments as a JSON C string.
/// - `func`: C callback that performs the actual tool execution.
/// - `func_user_data`: Opaque pointer passed to `func`.
/// - `func_free`: Optional destructor for `func_user_data`.
/// - `parent`: Optional parent scope handle, or null.
/// - `attributes`: Bitfield of tool attributes.
/// - `data_json`: Optional JSON data, or null.
/// - `metadata_json`: Optional JSON metadata, or null.
/// - `out`: On success, receives the result as a JSON C string. Caller must free
///   with `nv_agentrt_string_free`.
///
/// # Safety
/// `name`, `args_json`, and `out` must be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_tool_call_execute(
    name: *const c_char,
    args_json: *const c_char,
    func: NvAgentRtToolExecCb,
    func_user_data: *mut libc::c_void,
    func_free: NvAgentRtFreeFn,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    out: *mut *mut c_char,
) -> NvAgentRtStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NvAgentRtStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let args = match c_str_to_json(args_json) {
        Some(a) => a,
        None => return NvAgentRtStatus::InvalidJson,
    };
    let parent_handle = if parent.is_null() {
        None
    } else {
        Some(unsafe { &*parent }.0.clone())
    };
    let attrs = core_types::ToolAttributes::from_bits_truncate(attributes);
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NvAgentRtStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NvAgentRtStatus::InvalidJson,
    };

    let exec_fn = wrap_tool_exec_fn(func, func_user_data, func_free);

    let result = tokio_runtime().block_on(async {
        core::nv_agentrt_tool_call_execute(
            &name,
            args,
            exec_fn,
            parent_handle,
            attrs,
            data,
            metadata,
        )
        .await
    });

    match result {
        Ok(json) => {
            unsafe { *out = json_to_c_string(&json) };
            NvAgentRtStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// LLM lifecycle
// ---------------------------------------------------------------------------

/// Begin an LLM call, running pre-call guardrails and intercepts.
///
/// # Parameters
/// - `name`: Null-terminated LLM provider name.
/// - `request`: The LLM request object (created via `nv_agentrt_llm_request_new`).
/// - `parent`: Optional parent scope handle, or null.
/// - `attributes`: Bitfield of LLM attributes.
/// - `data_json`: Optional JSON data, or null.
/// - `metadata_json`: Optional JSON metadata, or null.
/// - `out`: On success, receives a heap-allocated `FfiLLMHandle`.
///
/// # Safety
/// `name`, `request`, and `out` must be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_llm_call(
    name: *const c_char,
    request: *const FfiLLMRequest,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    out: *mut *mut FfiLLMHandle,
) -> NvAgentRtStatus {
    clear_last_error();
    if out.is_null() || request.is_null() {
        set_last_error("null pointer argument");
        return NvAgentRtStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let parent_ref = if parent.is_null() {
        None
    } else {
        Some(&unsafe { &*parent }.0)
    };
    let attrs = core_types::LLMAttributes::from_bits_truncate(attributes);
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NvAgentRtStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NvAgentRtStatus::InvalidJson,
    };

    match core::nv_agentrt_llm_call(
        &name,
        &unsafe { &*request }.0,
        parent_ref,
        attrs,
        data,
        metadata,
    ) {
        Ok(h) => {
            unsafe { *out = Box::into_raw(Box::new(FfiLLMHandle(h))) };
            NvAgentRtStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// End an LLM call, running post-call guardrails and intercepts.
///
/// # Parameters
/// - `handle`: The LLM handle from `nv_agentrt_llm_call`.
/// - `response_json`: LLM response as a JSON C string.
/// - `data_json`: Optional JSON data, or null.
/// - `metadata_json`: Optional JSON metadata, or null.
///
/// # Safety
/// `handle` and `response_json` must be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_llm_call_end(
    handle: *const FfiLLMHandle,
    response_json: *const c_char,
    data_json: *const c_char,
    metadata_json: *const c_char,
) -> NvAgentRtStatus {
    clear_last_error();
    if handle.is_null() {
        set_last_error("handle is null");
        return NvAgentRtStatus::NullPointer;
    }
    let response = match c_str_to_json(response_json) {
        Some(r) => r,
        None => return NvAgentRtStatus::InvalidJson,
    };
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NvAgentRtStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NvAgentRtStatus::InvalidJson,
    };

    match core::nv_agentrt_llm_call_end(&unsafe { &*handle }.0, response, data, metadata) {
        Ok(()) => NvAgentRtStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Execute an LLM call end-to-end: begin, invoke the callback, apply guardrails
/// and intercepts, then end. Blocks the calling thread until completion.
///
/// # Parameters
/// - `name`: Null-terminated LLM provider name.
/// - `request`: The LLM request object.
/// - `func`: C callback that performs the actual LLM call.
/// - `func_user_data`: Opaque pointer passed to `func`.
/// - `func_free`: Optional destructor for `func_user_data`.
/// - `parent`: Optional parent scope handle, or null.
/// - `attributes`: Bitfield of LLM attributes.
/// - `data_json`: Optional JSON data, or null.
/// - `metadata_json`: Optional JSON metadata, or null.
/// - `out`: On success, receives the response as a JSON C string. Caller must
///   free with `nv_agentrt_string_free`.
///
/// # Safety
/// `name`, `request`, and `out` must be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_llm_call_execute(
    name: *const c_char,
    request: *const FfiLLMRequest,
    func: NvAgentRtLlmExecCb,
    func_user_data: *mut libc::c_void,
    func_free: NvAgentRtFreeFn,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    out: *mut *mut c_char,
) -> NvAgentRtStatus {
    clear_last_error();
    if out.is_null() || request.is_null() {
        set_last_error("null pointer argument");
        return NvAgentRtStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let req = unsafe { &*request }.0.clone();
    let parent_handle = if parent.is_null() {
        None
    } else {
        Some(unsafe { &*parent }.0.clone())
    };
    let attrs = core_types::LLMAttributes::from_bits_truncate(attributes);
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NvAgentRtStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NvAgentRtStatus::InvalidJson,
    };

    let exec_fn = wrap_llm_exec_fn(func, func_user_data, func_free);

    let result = tokio_runtime().block_on(async {
        core::nv_agentrt_llm_call_execute(&name, req, exec_fn, parent_handle, attrs, data, metadata)
            .await
    });

    match result {
        Ok(json) => {
            unsafe { *out = json_to_c_string(&json) };
            NvAgentRtStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// Stream
// ---------------------------------------------------------------------------

/// Opaque stream handle for consuming LLM streaming responses chunk by chunk.
/// Use `nv_agentrt_stream_next` to poll and `nv_agentrt_stream_free` to release.
pub struct FfiStream {
    receiver: tokio::sync::Mutex<tokio::sync::mpsc::Receiver<nvagentrt_core::Result<String>>>,
}

/// Execute a streaming LLM call end-to-end. Returns a stream handle that can
/// be polled with `nv_agentrt_stream_next`. Blocks until the stream is set up.
///
/// # Parameters
/// - `name`: Null-terminated LLM provider name.
/// - `request`: The LLM request object.
/// - `func`: C callback that performs the actual LLM call.
/// - `func_user_data`: Opaque pointer passed to `func`.
/// - `func_free`: Optional destructor for `func_user_data`.
/// - `parent`: Optional parent scope handle, or null.
/// - `attributes`: Bitfield of LLM attributes.
/// - `data_json`: Optional JSON data, or null.
/// - `metadata_json`: Optional JSON metadata, or null.
/// - `out`: On success, receives a heap-allocated `FfiStream`.
///
/// # Safety
/// `name`, `request`, and `out` must be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_llm_stream_call_execute(
    name: *const c_char,
    request: *const FfiLLMRequest,
    func: NvAgentRtLlmExecCb,
    func_user_data: *mut libc::c_void,
    func_free: NvAgentRtFreeFn,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    out: *mut *mut FfiStream,
) -> NvAgentRtStatus {
    clear_last_error();
    if out.is_null() || request.is_null() {
        set_last_error("null pointer argument");
        return NvAgentRtStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let req = unsafe { &*request }.0.clone();
    let parent_handle = if parent.is_null() {
        None
    } else {
        Some(unsafe { &*parent }.0.clone())
    };
    let attrs = core_types::LLMAttributes::from_bits_truncate(attributes);
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NvAgentRtStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NvAgentRtStatus::InvalidJson,
    };

    let exec_fn = wrap_llm_stream_exec_fn(func, func_user_data, func_free);

    let result = tokio_runtime().block_on(async {
        core::nv_agentrt_llm_stream_call_execute(
            &name,
            req,
            exec_fn,
            parent_handle,
            attrs,
            data,
            metadata,
        )
        .await
    });

    match result {
        Ok(rust_stream) => {
            let (tx, rx) = tokio::sync::mpsc::channel(32);
            tokio_runtime().spawn(async move {
                let mut stream = rust_stream;
                while let Some(item) = stream.next().await {
                    if tx.send(item).await.is_err() {
                        break;
                    }
                }
            });
            let ffi_stream = Box::new(FfiStream {
                receiver: tokio::sync::Mutex::new(rx),
            });
            unsafe { *out = Box::into_raw(ffi_stream) };
            NvAgentRtStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// Poll the next chunk from a streaming LLM response. Blocks until a chunk is
/// available.
///
/// # Returns
/// - `1`: A chunk was written to `*out_chunk`. Caller must free with
///   `nv_agentrt_string_free`.
/// - `0`: The stream is complete (no more chunks).
/// - `-1`: An error occurred. Call `nv_agentrt_last_error` for details.
///
/// # Safety
/// `stream` and `out_chunk` must be valid, non-null pointers.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_stream_next(
    stream: *mut FfiStream,
    out_chunk: *mut *mut c_char,
) -> i32 {
    if stream.is_null() || out_chunk.is_null() {
        return -1;
    }
    let stream = unsafe { &*stream };
    let result = tokio_runtime().block_on(async {
        let mut guard = stream.receiver.lock().await;
        guard.recv().await
    });
    match result {
        None => 0, // stream done
        Some(Ok(text)) => {
            unsafe { *out_chunk = str_to_c_string(&text) };
            1
        }
        Some(Err(e)) => {
            set_last_error(&e.to_string());
            -1
        }
    }
}

/// Free a stream handle and release its resources.
///
/// # Safety
/// `stream` must be a valid `FfiStream` pointer returned by
/// `nv_agentrt_llm_stream_call_execute`, or null.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_stream_free(stream: *mut FfiStream) {
    if !stream.is_null() {
        drop(unsafe { Box::from_raw(stream) });
    }
}

// ---------------------------------------------------------------------------
// Tool guardrail registrations
// ---------------------------------------------------------------------------

macro_rules! ffi_guardrail_tool_api {
    ($(#[$reg_doc:meta])* $register_name:ident,
     $(#[$dereg_doc:meta])* $deregister_name:ident,
     $core_register:path, $core_deregister:path, $wrapper:ident) => {
        $(#[$reg_doc])*
        #[no_mangle]
        pub unsafe extern "C" fn $register_name(
            name: *const c_char,
            priority: i32,
            cb: NvAgentRtToolSanitizeCb,
            user_data: *mut libc::c_void,
            free_fn: NvAgentRtFreeFn,
        ) -> NvAgentRtStatus {
            clear_last_error();
            let name = match c_str_to_string(name) {
                Ok(s) => s,
                Err(status) => return status,
            };
            let wrapped = $wrapper(cb, user_data, free_fn);
            match $core_register(&name, priority, wrapped) {
                Ok(()) => NvAgentRtStatus::Ok,
                Err(e) => status_from_error(&e),
            }
        }

        $(#[$dereg_doc])*
        #[no_mangle]
        pub unsafe extern "C" fn $deregister_name(
            name: *const c_char,
        ) -> NvAgentRtStatus {
            clear_last_error();
            let name = match c_str_to_string(name) {
                Ok(s) => s,
                Err(status) => return status,
            };
            match $core_deregister(&name) {
                Ok(_) => NvAgentRtStatus::Ok,
                Err(e) => status_from_error(&e),
            }
        }
    };
}

ffi_guardrail_tool_api!(
    /// Register a tool request sanitization guardrail. The callback can inspect
    /// and modify tool arguments before the tool executes.
    ///
    /// # Parameters
    /// - `name`: Unique guardrail name.
    /// - `priority`: Execution priority (lower runs first).
    /// - `cb`: Sanitize callback that receives tool name and args JSON, returns sanitized args JSON.
    /// - `user_data`: Opaque pointer passed to `cb`.
    /// - `free_fn`: Optional destructor for `user_data`.
    ///
    /// # Safety
    /// `name` must be a valid C string. `cb` must be a valid function pointer.
    nv_agentrt_register_tool_sanitize_request_guardrail,
    /// Deregister a tool request sanitization guardrail by name.
    ///
    /// # Safety
    /// `name` must be a valid C string.
    nv_agentrt_deregister_tool_sanitize_request_guardrail,
    core::nv_agentrt_register_tool_sanitize_request_guardrail,
    core::nv_agentrt_deregister_tool_sanitize_request_guardrail,
    wrap_tool_sanitize_fn
);

ffi_guardrail_tool_api!(
    /// Register a tool response sanitization guardrail. The callback can inspect
    /// and modify tool results after the tool executes.
    ///
    /// # Parameters
    /// - `name`: Unique guardrail name.
    /// - `priority`: Execution priority (lower runs first).
    /// - `cb`: Sanitize callback that receives tool name and result JSON, returns sanitized result JSON.
    /// - `user_data`: Opaque pointer passed to `cb`.
    /// - `free_fn`: Optional destructor for `user_data`.
    ///
    /// # Safety
    /// `name` must be a valid C string. `cb` must be a valid function pointer.
    nv_agentrt_register_tool_sanitize_response_guardrail,
    /// Deregister a tool response sanitization guardrail by name.
    ///
    /// # Safety
    /// `name` must be a valid C string.
    nv_agentrt_deregister_tool_sanitize_response_guardrail,
    core::nv_agentrt_register_tool_sanitize_response_guardrail,
    core::nv_agentrt_deregister_tool_sanitize_response_guardrail,
    wrap_tool_sanitize_fn
);

/// Register a tool conditional execution guardrail. The callback decides whether
/// a tool call should proceed. Returns an error message to reject, or null to allow.
///
/// # Parameters
/// - `name`: Unique guardrail name.
/// - `priority`: Execution priority (lower runs first).
/// - `cb`: Conditional callback.
/// - `user_data`: Opaque pointer passed to `cb`.
/// - `free_fn`: Optional destructor for `user_data`.
///
/// # Safety
/// `name` must be a valid C string. `cb` must be a valid function pointer.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_register_tool_conditional_execution_guardrail(
    name: *const c_char,
    priority: i32,
    cb: NvAgentRtToolConditionalCb,
    user_data: *mut libc::c_void,
    free_fn: NvAgentRtFreeFn,
) -> NvAgentRtStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_tool_conditional_fn(cb, user_data, free_fn);
    match core::nv_agentrt_register_tool_conditional_execution_guardrail(&name, priority, wrapped) {
        Ok(()) => NvAgentRtStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a tool conditional execution guardrail by name.
///
/// # Safety
/// `name` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_deregister_tool_conditional_execution_guardrail(
    name: *const c_char,
) -> NvAgentRtStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nv_agentrt_deregister_tool_conditional_execution_guardrail(&name) {
        Ok(_) => NvAgentRtStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// Tool intercept registrations
// ---------------------------------------------------------------------------

macro_rules! ffi_intercept_tool_api {
    ($(#[$reg_doc:meta])* $register_name:ident,
     $(#[$dereg_doc:meta])* $deregister_name:ident,
     $core_register:path, $core_deregister:path, $wrapper:ident) => {
        $(#[$reg_doc])*
        #[no_mangle]
        pub unsafe extern "C" fn $register_name(
            name: *const c_char,
            priority: i32,
            break_chain: bool,
            cb: NvAgentRtToolSanitizeCb,
            user_data: *mut libc::c_void,
            free_fn: NvAgentRtFreeFn,
        ) -> NvAgentRtStatus {
            clear_last_error();
            let name = match c_str_to_string(name) {
                Ok(s) => s,
                Err(status) => return status,
            };
            let wrapped = $wrapper(cb, user_data, free_fn);
            match $core_register(&name, priority, break_chain, wrapped) {
                Ok(()) => NvAgentRtStatus::Ok,
                Err(e) => status_from_error(&e),
            }
        }

        $(#[$dereg_doc])*
        #[no_mangle]
        pub unsafe extern "C" fn $deregister_name(
            name: *const c_char,
        ) -> NvAgentRtStatus {
            clear_last_error();
            let name = match c_str_to_string(name) {
                Ok(s) => s,
                Err(status) => return status,
            };
            match $core_deregister(&name) {
                Ok(_) => NvAgentRtStatus::Ok,
                Err(e) => status_from_error(&e),
            }
        }
    };
}

ffi_intercept_tool_api!(
    /// Register a tool request intercept. The callback can transform tool
    /// arguments before execution. Runs after request guardrails in the
    /// middleware pipeline.
    ///
    /// # Parameters
    /// - `name`: Unique intercept name.
    /// - `priority`: Execution priority (lower runs first).
    /// - `break_chain`: If true, stop processing further intercepts after this one.
    /// - `cb`: Transform callback that receives tool name and args JSON, returns modified args JSON.
    /// - `user_data`: Opaque pointer passed to `cb`.
    /// - `free_fn`: Optional destructor for `user_data`.
    ///
    /// # Safety
    /// `name` must be a valid C string. `cb` must be a valid function pointer.
    nv_agentrt_register_tool_request_intercept,
    /// Deregister a tool request intercept by name.
    ///
    /// # Safety
    /// `name` must be a valid C string.
    nv_agentrt_deregister_tool_request_intercept,
    core::nv_agentrt_register_tool_request_intercept,
    core::nv_agentrt_deregister_tool_request_intercept,
    wrap_tool_sanitize_fn
);

ffi_intercept_tool_api!(
    /// Register a tool response intercept. The callback can transform tool
    /// results after execution. Runs before response guardrails in the
    /// middleware pipeline.
    ///
    /// # Parameters
    /// - `name`: Unique intercept name.
    /// - `priority`: Execution priority (lower runs first).
    /// - `break_chain`: If true, stop processing further intercepts after this one.
    /// - `cb`: Transform callback that receives tool name and result JSON, returns modified result JSON.
    /// - `user_data`: Opaque pointer passed to `cb`.
    /// - `free_fn`: Optional destructor for `user_data`.
    ///
    /// # Safety
    /// `name` must be a valid C string. `cb` must be a valid function pointer.
    nv_agentrt_register_tool_response_intercept,
    /// Deregister a tool response intercept by name.
    ///
    /// # Safety
    /// `name` must be a valid C string.
    nv_agentrt_deregister_tool_response_intercept,
    core::nv_agentrt_register_tool_response_intercept,
    core::nv_agentrt_deregister_tool_response_intercept,
    wrap_tool_sanitize_fn
);

/// Register a tool execution intercept. When the condition callback returns true,
/// the execution callback replaces the default tool execution.
///
/// # Parameters
/// - `name`: Unique intercept name.
/// - `priority`: Execution priority (lower runs first).
/// - `cond_cb`: Condition callback that decides if this intercept applies.
/// - `cond_user_data`: Opaque pointer for the condition callback.
/// - `cond_free`: Optional destructor for `cond_user_data`.
/// - `exec_cb`: Execution callback that replaces the tool call.
/// - `exec_user_data`: Opaque pointer for the execution callback.
/// - `exec_free`: Optional destructor for `exec_user_data`.
///
/// # Safety
/// `name` must be a valid C string. Callback pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_register_tool_execution_intercept(
    name: *const c_char,
    priority: i32,
    cond_cb: NvAgentRtToolExecConditionalCb,
    cond_user_data: *mut libc::c_void,
    cond_free: NvAgentRtFreeFn,
    exec_cb: NvAgentRtToolExecCb,
    exec_user_data: *mut libc::c_void,
    exec_free: NvAgentRtFreeFn,
) -> NvAgentRtStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let cond = wrap_tool_exec_conditional_fn(cond_cb, cond_user_data, cond_free);
    let exec = wrap_tool_exec_fn(exec_cb, exec_user_data, exec_free);
    match core::nv_agentrt_register_tool_execution_intercept(&name, priority, cond, exec) {
        Ok(()) => NvAgentRtStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister a tool execution intercept by name.
///
/// # Safety
/// `name` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_deregister_tool_execution_intercept(
    name: *const c_char,
) -> NvAgentRtStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nv_agentrt_deregister_tool_execution_intercept(&name) {
        Ok(_) => NvAgentRtStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// LLM guardrail registrations
// ---------------------------------------------------------------------------

/// Register an LLM request sanitization guardrail. The callback can modify or
/// replace the LLM request before it is sent.
///
/// # Parameters
/// - `name`: Unique guardrail name.
/// - `priority`: Execution priority (lower runs first).
/// - `cb`: Request sanitize callback.
/// - `user_data`: Opaque pointer passed to `cb`.
/// - `free_fn`: Optional destructor for `user_data`.
///
/// # Safety
/// `name` must be a valid C string. `cb` must be a valid function pointer.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_register_llm_sanitize_request_guardrail(
    name: *const c_char,
    priority: i32,
    cb: NvAgentRtLlmRequestCb,
    user_data: *mut libc::c_void,
    free_fn: NvAgentRtFreeFn,
) -> NvAgentRtStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_llm_sanitize_request_fn(cb, user_data, free_fn);
    match core::nv_agentrt_register_llm_sanitize_request_guardrail(&name, priority, wrapped) {
        Ok(()) => NvAgentRtStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister an LLM request sanitization guardrail by name.
///
/// # Safety
/// `name` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_deregister_llm_sanitize_request_guardrail(
    name: *const c_char,
) -> NvAgentRtStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nv_agentrt_deregister_llm_sanitize_request_guardrail(&name) {
        Ok(_) => NvAgentRtStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

macro_rules! ffi_guardrail_json_api {
    ($(#[$reg_doc:meta])* $register_name:ident,
     $(#[$dereg_doc:meta])* $deregister_name:ident,
     $core_register:path, $core_deregister:path) => {
        $(#[$reg_doc])*
        #[no_mangle]
        pub unsafe extern "C" fn $register_name(
            name: *const c_char,
            priority: i32,
            cb: NvAgentRtJsonCb,
            user_data: *mut libc::c_void,
            free_fn: NvAgentRtFreeFn,
        ) -> NvAgentRtStatus {
            clear_last_error();
            let name = match c_str_to_string(name) {
                Ok(s) => s,
                Err(status) => return status,
            };
            let wrapped = wrap_json_fn(cb, user_data, free_fn);
            match $core_register(&name, priority, wrapped) {
                Ok(()) => NvAgentRtStatus::Ok,
                Err(e) => status_from_error(&e),
            }
        }

        $(#[$dereg_doc])*
        #[no_mangle]
        pub unsafe extern "C" fn $deregister_name(
            name: *const c_char,
        ) -> NvAgentRtStatus {
            clear_last_error();
            let name = match c_str_to_string(name) {
                Ok(s) => s,
                Err(status) => return status,
            };
            match $core_deregister(&name) {
                Ok(_) => NvAgentRtStatus::Ok,
                Err(e) => status_from_error(&e),
            }
        }
    };
}

ffi_guardrail_json_api!(
    /// Register an LLM response sanitization guardrail. The callback can inspect
    /// and modify the LLM response JSON after it is received.
    ///
    /// # Parameters
    /// - `name`: Unique guardrail name.
    /// - `priority`: Execution priority (lower runs first).
    /// - `cb`: JSON-to-JSON callback that receives response JSON and returns sanitized JSON.
    /// - `user_data`: Opaque pointer passed to `cb`.
    /// - `free_fn`: Optional destructor for `user_data`.
    ///
    /// # Safety
    /// `name` must be a valid C string. `cb` must be a valid function pointer.
    nv_agentrt_register_llm_sanitize_response_guardrail,
    /// Deregister an LLM response sanitization guardrail by name.
    ///
    /// # Safety
    /// `name` must be a valid C string.
    nv_agentrt_deregister_llm_sanitize_response_guardrail,
    core::nv_agentrt_register_llm_sanitize_response_guardrail,
    core::nv_agentrt_deregister_llm_sanitize_response_guardrail
);

/// Register an LLM conditional execution guardrail. The callback decides
/// whether an LLM call should proceed.
///
/// # Parameters
/// - `name`: Unique guardrail name.
/// - `priority`: Execution priority (lower runs first).
/// - `cb`: Conditional callback. Returns null to allow, or error message to reject.
/// - `user_data`: Opaque pointer passed to `cb`.
/// - `free_fn`: Optional destructor for `user_data`.
///
/// # Safety
/// `name` must be a valid C string. `cb` must be a valid function pointer.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_register_llm_conditional_execution_guardrail(
    name: *const c_char,
    priority: i32,
    cb: NvAgentRtLlmConditionalCb,
    user_data: *mut libc::c_void,
    free_fn: NvAgentRtFreeFn,
) -> NvAgentRtStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_llm_conditional_fn(cb, user_data, free_fn);
    match core::nv_agentrt_register_llm_conditional_execution_guardrail(&name, priority, wrapped) {
        Ok(()) => NvAgentRtStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister an LLM conditional execution guardrail by name.
///
/// # Safety
/// `name` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_deregister_llm_conditional_execution_guardrail(
    name: *const c_char,
) -> NvAgentRtStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nv_agentrt_deregister_llm_conditional_execution_guardrail(&name) {
        Ok(_) => NvAgentRtStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// LLM intercept registrations
// ---------------------------------------------------------------------------

/// Register an LLM request intercept. The callback can transform the request
/// before it reaches the LLM provider.
///
/// # Parameters
/// - `name`: Unique intercept name.
/// - `priority`: Execution priority (lower runs first).
/// - `break_chain`: If true, stop processing further intercepts after this one.
/// - `cb`: Request transform callback.
/// - `user_data`: Opaque pointer passed to `cb`.
/// - `free_fn`: Optional destructor for `user_data`.
///
/// # Safety
/// `name` must be a valid C string. `cb` must be a valid function pointer.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_register_llm_request_intercept(
    name: *const c_char,
    priority: i32,
    break_chain: bool,
    cb: NvAgentRtLlmRequestCb,
    user_data: *mut libc::c_void,
    free_fn: NvAgentRtFreeFn,
) -> NvAgentRtStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_llm_sanitize_request_fn(cb, user_data, free_fn);
    match core::nv_agentrt_register_llm_request_intercept(&name, priority, break_chain, wrapped) {
        Ok(()) => NvAgentRtStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister an LLM request intercept by name.
///
/// # Safety
/// `name` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_deregister_llm_request_intercept(
    name: *const c_char,
) -> NvAgentRtStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nv_agentrt_deregister_llm_request_intercept(&name) {
        Ok(_) => NvAgentRtStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Register an LLM response intercept. The callback can transform the response
/// JSON after it is received from the LLM provider.
///
/// # Parameters
/// - `name`: Unique intercept name.
/// - `priority`: Execution priority (lower runs first).
/// - `break_chain`: If true, stop processing further intercepts after this one.
/// - `cb`: JSON transform callback.
/// - `user_data`: Opaque pointer passed to `cb`.
/// - `free_fn`: Optional destructor for `user_data`.
///
/// # Safety
/// `name` must be a valid C string. `cb` must be a valid function pointer.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_register_llm_response_intercept(
    name: *const c_char,
    priority: i32,
    break_chain: bool,
    cb: NvAgentRtJsonCb,
    user_data: *mut libc::c_void,
    free_fn: NvAgentRtFreeFn,
) -> NvAgentRtStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_json_fn(cb, user_data, free_fn);
    match core::nv_agentrt_register_llm_response_intercept(&name, priority, break_chain, wrapped) {
        Ok(()) => NvAgentRtStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister an LLM response intercept by name.
///
/// # Safety
/// `name` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_deregister_llm_response_intercept(
    name: *const c_char,
) -> NvAgentRtStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nv_agentrt_deregister_llm_response_intercept(&name) {
        Ok(_) => NvAgentRtStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Register an LLM streaming response intercept. The callback transforms
/// individual SSE events as they arrive during a streaming LLM call.
///
/// # Parameters
/// - `name`: Unique intercept name.
/// - `priority`: Execution priority (lower runs first).
/// - `break_chain`: If true, stop processing further intercepts after this one.
/// - `cb`: SSE event transform callback (receives/returns JSON).
/// - `user_data`: Opaque pointer passed to `cb`.
/// - `free_fn`: Optional destructor for `user_data`.
///
/// # Safety
/// `name` must be a valid C string. `cb` must be a valid function pointer.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_register_llm_stream_response_intercept(
    name: *const c_char,
    priority: i32,
    break_chain: bool,
    cb: NvAgentRtSseInterceptCb,
    user_data: *mut libc::c_void,
    free_fn: NvAgentRtFreeFn,
) -> NvAgentRtStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_sse_intercept_fn(cb, user_data, free_fn);
    match core::nv_agentrt_register_llm_stream_response_intercept(
        &name,
        priority,
        break_chain,
        wrapped,
    ) {
        Ok(()) => NvAgentRtStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister an LLM streaming response intercept by name.
///
/// # Safety
/// `name` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_deregister_llm_stream_response_intercept(
    name: *const c_char,
) -> NvAgentRtStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nv_agentrt_deregister_llm_stream_response_intercept(&name) {
        Ok(_) => NvAgentRtStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Register an LLM execution intercept. When the condition callback returns true,
/// the execution callback replaces the default LLM call.
///
/// # Parameters
/// - `name`: Unique intercept name.
/// - `priority`: Execution priority (lower runs first).
/// - `cond_cb`: Condition callback.
/// - `cond_user_data`: Opaque pointer for the condition callback.
/// - `cond_free`: Optional destructor for `cond_user_data`.
/// - `exec_cb`: Execution callback.
/// - `exec_user_data`: Opaque pointer for the execution callback.
/// - `exec_free`: Optional destructor for `exec_user_data`.
///
/// # Safety
/// `name` must be a valid C string. Callback pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_register_llm_execution_intercept(
    name: *const c_char,
    priority: i32,
    cond_cb: NvAgentRtLlmExecConditionalCb,
    cond_user_data: *mut libc::c_void,
    cond_free: NvAgentRtFreeFn,
    exec_cb: NvAgentRtLlmExecCb,
    exec_user_data: *mut libc::c_void,
    exec_free: NvAgentRtFreeFn,
) -> NvAgentRtStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let cond = wrap_llm_exec_conditional_fn(cond_cb, cond_user_data, cond_free);
    let exec = wrap_llm_exec_fn(exec_cb, exec_user_data, exec_free);
    match core::nv_agentrt_register_llm_execution_intercept(&name, priority, cond, exec) {
        Ok(()) => NvAgentRtStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister an LLM execution intercept by name.
///
/// # Safety
/// `name` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_deregister_llm_execution_intercept(
    name: *const c_char,
) -> NvAgentRtStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nv_agentrt_deregister_llm_execution_intercept(&name) {
        Ok(_) => NvAgentRtStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Register an LLM streaming execution intercept. When the condition callback
/// returns true, the execution callback replaces the default streaming LLM call.
///
/// # Parameters
/// - `name`: Unique intercept name.
/// - `priority`: Execution priority (lower runs first).
/// - `cond_cb`: Condition callback.
/// - `cond_user_data`: Opaque pointer for the condition callback.
/// - `cond_free`: Optional destructor for `cond_user_data`.
/// - `exec_cb`: Execution callback.
/// - `exec_user_data`: Opaque pointer for the execution callback.
/// - `exec_free`: Optional destructor for `exec_user_data`.
///
/// # Safety
/// `name` must be a valid C string. Callback pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_register_llm_stream_execution_intercept(
    name: *const c_char,
    priority: i32,
    cond_cb: NvAgentRtLlmExecConditionalCb,
    cond_user_data: *mut libc::c_void,
    cond_free: NvAgentRtFreeFn,
    exec_cb: NvAgentRtLlmExecCb,
    exec_user_data: *mut libc::c_void,
    exec_free: NvAgentRtFreeFn,
) -> NvAgentRtStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let cond = wrap_llm_exec_conditional_fn(cond_cb, cond_user_data, cond_free);
    let exec = wrap_llm_stream_exec_fn(exec_cb, exec_user_data, exec_free);
    match core::nv_agentrt_register_llm_stream_execution_intercept(&name, priority, cond, exec) {
        Ok(()) => NvAgentRtStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister an LLM streaming execution intercept by name.
///
/// # Safety
/// `name` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_deregister_llm_stream_execution_intercept(
    name: *const c_char,
) -> NvAgentRtStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nv_agentrt_deregister_llm_stream_execution_intercept(&name) {
        Ok(_) => NvAgentRtStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// Subscriber registrations
// ---------------------------------------------------------------------------

/// Register an event subscriber. The callback is invoked for every lifecycle
/// event emitted by the runtime.
///
/// # Parameters
/// - `name`: Unique subscriber name.
/// - `cb`: Event callback. The `FfiEvent` is valid only during the call.
/// - `user_data`: Opaque pointer passed to `cb`.
/// - `free_fn`: Optional destructor for `user_data`.
///
/// # Safety
/// `name` must be a valid C string. `cb` must be a valid function pointer.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_register_subscriber(
    name: *const c_char,
    cb: NvAgentRtEventSubscriberCb,
    user_data: *mut libc::c_void,
    free_fn: NvAgentRtFreeFn,
) -> NvAgentRtStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let wrapped = wrap_event_subscriber(cb, user_data, free_fn);
    match core::nv_agentrt_register_subscriber(&name, wrapped) {
        Ok(()) => NvAgentRtStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

/// Deregister an event subscriber by name.
///
/// # Safety
/// `name` must be a valid C string.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_deregister_subscriber(name: *const c_char) -> NvAgentRtStatus {
    clear_last_error();
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    match core::nv_agentrt_deregister_subscriber(&name) {
        Ok(_) => NvAgentRtStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// Scope stack isolation
// ---------------------------------------------------------------------------

/// Create a new isolated scope stack with its own root scope.
///
/// Each scope stack is independent: scopes pushed on one do not appear on another.
/// Use `nv_agentrt_scope_stack_set_thread` to bind a stack to the current thread
/// before making other NVAgentRT API calls.
///
/// # Parameters
/// - `out`: On success, receives a heap-allocated `FfiScopeStack` that must be
///   freed with `nv_agentrt_scope_stack_free`.
///
/// # Safety
/// `out` must be a valid, non-null pointer.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_scope_stack_create(
    out: *mut *mut FfiScopeStack,
) -> NvAgentRtStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("out pointer is null");
        return NvAgentRtStatus::NullPointer;
    }
    let handle = core::create_scope_stack();
    unsafe { *out = Box::into_raw(Box::new(FfiScopeStack(handle))) };
    NvAgentRtStatus::Ok
}

/// Bind an isolated scope stack to the current OS thread.
///
/// After this call, all NVAgentRT scope operations on the current thread
/// (e.g. `nv_agentrt_push_scope`, `nv_agentrt_get_handle`) will use the
/// given scope stack. This is typically used from Go goroutines that have
/// called `runtime.LockOSThread()`.
///
/// The `FfiScopeStack` is **not** consumed — the caller retains ownership
/// and must still free it when done.
///
/// # Safety
/// `stack` must be a valid, non-null `FfiScopeStack` pointer.
#[no_mangle]
pub unsafe extern "C" fn nv_agentrt_scope_stack_set_thread(
    stack: *const FfiScopeStack,
) -> NvAgentRtStatus {
    clear_last_error();
    if stack.is_null() {
        set_last_error("stack pointer is null");
        return NvAgentRtStatus::NullPointer;
    }
    let handle = unsafe { &*stack }.0.clone();
    core::set_thread_scope_stack(handle);
    NvAgentRtStatus::Ok
}
