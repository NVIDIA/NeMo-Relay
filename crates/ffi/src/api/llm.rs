// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

// ---------------------------------------------------------------------------
// LLM lifecycle
// ---------------------------------------------------------------------------

/// Begin an LLM call, running pre-call guardrails and intercepts.
///
/// # Parameters
/// - `name`: Null-terminated LLM provider name.
/// - `native_json`: The request payload as a JSON C string representing an
///   `LLMRequest` (`{"headers": {...}, "content": {...}}`).
/// - `parent`: Optional parent scope handle, or null.
/// - `attributes`: Bitfield of LLM attributes.
/// - `data_json`: Optional JSON data, or null.
/// - `metadata_json`: Optional JSON metadata, or null.
/// - `model_name`: Optional LLM model identifier, or null.
/// - `out`: On success, receives a heap-allocated `FfiLLMHandle`.
///
/// # Safety
/// `name`, `native_json`, and `out` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_llm_call(
    name: *const c_char,
    native_json: *const c_char,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    model_name: *const c_char,
    out: *mut *mut FfiLLMHandle,
) -> NemoFlowStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("null pointer argument");
        return NemoFlowStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let native = match c_str_to_json(native_json) {
        Some(n) => n,
        None => return NemoFlowStatus::InvalidJson,
    };
    let request: LLMRequest = match serde_json::from_value(native) {
        Ok(r) => r,
        Err(_) => {
            set_last_error("failed to parse native_json as LLMRequest");
            return NemoFlowStatus::InvalidJson;
        }
    };
    let parent_ref = if parent.is_null() {
        None
    } else {
        Some(&unsafe { &*parent }.0)
    };
    let attrs = LLMAttributes::from_bits_truncate(attributes);
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NemoFlowStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NemoFlowStatus::InvalidJson,
    };
    let model_name_opt = if model_name.is_null() {
        None
    } else {
        match c_str_to_string(model_name) {
            Ok(s) => Some(s),
            Err(status) => return status,
        }
    };

    match core_llm_api::llm_call(
        &name,
        &request,
        parent_ref,
        attrs,
        data,
        metadata,
        model_name_opt,
        None,
    ) {
        Ok(h) => {
            unsafe { *out = Box::into_raw(Box::new(FfiLLMHandle(h))) };
            NemoFlowStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// End an LLM call, running post-call guardrails and intercepts.
///
/// # Parameters
/// - `handle`: The LLM handle from `nemo_flow_llm_call`.
/// - `response_json`: LLM response as a JSON C string.
/// - `data_json`: Optional JSON data, or null.
/// - `metadata_json`: Optional JSON metadata, or null.
///
/// # Safety
/// `handle` and `response_json` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_llm_call_end(
    handle: *const FfiLLMHandle,
    response_json: *const c_char,
    data_json: *const c_char,
    metadata_json: *const c_char,
) -> NemoFlowStatus {
    clear_last_error();
    if handle.is_null() {
        set_last_error("handle is null");
        return NemoFlowStatus::NullPointer;
    }
    let response = match c_str_to_json(response_json) {
        Some(r) => r,
        None => return NemoFlowStatus::InvalidJson,
    };
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NemoFlowStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NemoFlowStatus::InvalidJson,
    };

    match core_llm_api::llm_call_end(&unsafe { &*handle }.0, response, data, metadata, None) {
        Ok(()) => NemoFlowStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// Built-in codec constructors
// ---------------------------------------------------------------------------

/// Create a new OpenAI Chat Completions codec handle.
///
/// The returned handle implements both request codec (decode/encode) and
/// response codec (decode_response). Free with `nemo_flow_codec_free`.
///
/// # Safety
/// Caller must free the returned handle via `nemo_flow_codec_free`.
#[unsafe(no_mangle)]
pub extern "C" fn nemo_flow_openai_chat_codec_new() -> *mut FfiCodecHandle {
    Box::into_raw(Box::new(FfiCodecHandle {
        codec: Arc::new(nemo_flow::codec::openai_chat::OpenAIChatCodec),
        response_codec: Arc::new(nemo_flow::codec::openai_chat::OpenAIChatCodec),
    }))
}

/// Create a new OpenAI Responses API codec handle.
///
/// The returned handle implements both request codec (decode/encode) and
/// response codec (decode_response). Free with `nemo_flow_codec_free`.
///
/// # Safety
/// Caller must free the returned handle via `nemo_flow_codec_free`.
#[unsafe(no_mangle)]
pub extern "C" fn nemo_flow_openai_responses_codec_new() -> *mut FfiCodecHandle {
    Box::into_raw(Box::new(FfiCodecHandle {
        codec: Arc::new(nemo_flow::codec::openai_responses::OpenAIResponsesCodec),
        response_codec: Arc::new(nemo_flow::codec::openai_responses::OpenAIResponsesCodec),
    }))
}

/// Create a new Anthropic Messages API codec handle.
///
/// The returned handle implements both request codec (decode/encode) and
/// response codec (decode_response). Free with `nemo_flow_codec_free`.
///
/// # Safety
/// Caller must free the returned handle via `nemo_flow_codec_free`.
#[unsafe(no_mangle)]
pub extern "C" fn nemo_flow_anthropic_messages_codec_new() -> *mut FfiCodecHandle {
    Box::into_raw(Box::new(FfiCodecHandle {
        codec: Arc::new(nemo_flow::codec::anthropic::AnthropicMessagesCodec),
        response_codec: Arc::new(nemo_flow::codec::anthropic::AnthropicMessagesCodec),
    }))
}

/// Execute an LLM call end-to-end: run conditional-execution guardrails (on raw
/// request), then request intercepts, sanitize-request guardrails, execution
/// intercepts, the callback, and sanitize-response
/// guardrails. On rejection, only a standalone Mark event is emitted (no
/// Start/End pair) and `GuardrailRejected` is returned. Blocks the calling
/// thread until completion.
///
/// # Parameters
/// - `name`: Null-terminated LLM provider name.
/// - `native_json`: The request payload as a JSON C string representing an
///   `LLMRequest` (`{"headers": {...}, "content": {...}}`).
/// - `func`: C callback that performs the actual LLM call.
/// - `func_user_data`: Opaque pointer passed to `func`.
/// - `func_free`: Optional destructor for `func_user_data`.
/// - `parent`: Optional parent scope handle, or null.
/// - `attributes`: Bitfield of LLM attributes.
/// - `data_json`: Optional JSON data, or null.
/// - `metadata_json`: Optional JSON metadata, or null.
/// - `model_name`: Optional LLM model identifier, or null.
/// - `out`: On success, receives the response as a JSON C string. Caller must
///   free with `nemo_flow_string_free`.
///
/// # Safety
/// `name`, `native_json`, and `out` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_llm_call_execute(
    name: *const c_char,
    native_json: *const c_char,
    func: NemoFlowLlmExecCb,
    func_user_data: *mut libc::c_void,
    func_free: NemoFlowFreeFn,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    model_name: *const c_char,
    codec_decode: NemoFlowCodecDecodeFn,
    codec_encode: NemoFlowCodecEncodeFn,
    codec_user_data: *mut libc::c_void,
    codec_free_fn: NemoFlowFreeFn,
    response_codec: *const FfiCodecHandle,
    out: *mut *mut c_char,
) -> NemoFlowStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("null pointer argument");
        return NemoFlowStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let native = match c_str_to_json(native_json) {
        Some(n) => n,
        None => return NemoFlowStatus::InvalidJson,
    };
    let request: LLMRequest = match serde_json::from_value(native) {
        Ok(r) => r,
        Err(_) => {
            set_last_error("failed to parse native_json as LLMRequest");
            return NemoFlowStatus::InvalidJson;
        }
    };
    let parent_handle = if parent.is_null() {
        None
    } else {
        Some(unsafe { &*parent }.0.clone())
    };
    let attrs = LLMAttributes::from_bits_truncate(attributes);
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NemoFlowStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NemoFlowStatus::InvalidJson,
    };
    let model_name_opt = if model_name.is_null() {
        None
    } else {
        match c_str_to_string(model_name) {
            Ok(s) => Some(s),
            Err(status) => return status,
        }
    };
    let codec = match (codec_decode, codec_encode) {
        (Some(decode_cb), Some(encode_cb)) => Some(wrap_codec_fn(
            decode_cb,
            encode_cb,
            codec_user_data,
            codec_free_fn,
        )),
        _ => None,
    };
    let response_codec = if response_codec.is_null() {
        None
    } else {
        Some(unsafe { &*response_codec }.response_codec.clone())
    };

    let exec_fn = wrap_llm_exec_fn(func, func_user_data, func_free);
    let default_fn: LlmExecutionNextFn = Arc::new(move |request| exec_fn(request));

    let scope_stack = current_scope_stack();
    let result = tokio_runtime().block_on(TASK_SCOPE_STACK.scope(scope_stack, async {
        core_llm_api::llm_call_execute(
            &name,
            request,
            default_fn,
            parent_handle,
            attrs,
            data,
            metadata,
            model_name_opt,
            codec,
            response_codec,
        )
        .await
    }));

    match result {
        Ok(json) => {
            unsafe { *out = json_to_c_string(&json) };
            NemoFlowStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// Stream
// ---------------------------------------------------------------------------

/// Opaque stream handle for consuming LLM streaming responses chunk by chunk.
/// Use `nemo_flow_stream_next` to poll and `nemo_flow_stream_free` to release.
pub struct FfiStream {
    pub(crate) receiver:
        tokio::sync::Mutex<tokio::sync::mpsc::Receiver<FlowResult<serde_json::Value>>>,
}

/// Execute a streaming LLM call end-to-end. Conditional-execution guardrails
/// run first on the raw request. Returns a stream handle that can be polled
/// with `nemo_flow_stream_next`. Blocks until the stream is set up.
///
/// # Parameters
/// - `name`: Null-terminated LLM provider name.
/// - `native_json`: The request payload as a JSON C string representing an
///   `LLMRequest` (`{"headers": {...}, "content": {...}}`).
/// - `func`: C callback that performs the actual LLM call.
/// - `func_user_data`: Opaque pointer passed to `func`.
/// - `func_free`: Optional destructor for `func_user_data`.
/// - `collector`: Callback invoked with each intercepted chunk as a JSON string.
///   May be null, in which case chunks are not collected.
/// - `finalizer`: Callback invoked once when the stream is exhausted to produce
///   the aggregated response as a JSON C string. May be null, in which case the
///   finalizer returns `Json::Null`.
/// - `parent`: Optional parent scope handle, or null.
/// - `attributes`: Bitfield of LLM attributes.
/// - `data_json`: Optional JSON data, or null.
/// - `metadata_json`: Optional JSON metadata, or null.
/// - `model_name`: Optional LLM model identifier, or null.
/// - `out`: On success, receives a heap-allocated `FfiStream`.
///
/// # Safety
/// `name`, `native_json`, and `out` must be valid, non-null pointers. `collector`
/// and `finalizer` may be null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_llm_stream_call_execute(
    name: *const c_char,
    native_json: *const c_char,
    func: NemoFlowLlmExecCb,
    func_user_data: *mut libc::c_void,
    func_free: NemoFlowFreeFn,
    collector: Option<NemoFlowCollectorCb>,
    finalizer: Option<NemoFlowFinalizerCb>,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    model_name: *const c_char,
    codec_decode: NemoFlowCodecDecodeFn,
    codec_encode: NemoFlowCodecEncodeFn,
    codec_user_data: *mut libc::c_void,
    codec_free_fn: NemoFlowFreeFn,
    response_codec: *const FfiCodecHandle,
    out: *mut *mut FfiStream,
) -> NemoFlowStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("null pointer argument");
        return NemoFlowStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let native = match c_str_to_json(native_json) {
        Some(n) => n,
        None => return NemoFlowStatus::InvalidJson,
    };
    let request: LLMRequest = match serde_json::from_value(native) {
        Ok(r) => r,
        Err(_) => {
            set_last_error("failed to parse native_json as LLMRequest");
            return NemoFlowStatus::InvalidJson;
        }
    };
    let parent_handle = if parent.is_null() {
        None
    } else {
        Some(unsafe { &*parent }.0.clone())
    };
    let attrs = LLMAttributes::from_bits_truncate(attributes);
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NemoFlowStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NemoFlowStatus::InvalidJson,
    };
    let model_name_opt = if model_name.is_null() {
        None
    } else {
        match c_str_to_string(model_name) {
            Ok(s) => Some(s),
            Err(status) => return status,
        }
    };
    let codec = match (codec_decode, codec_encode) {
        (Some(decode_cb), Some(encode_cb)) => Some(wrap_codec_fn(
            decode_cb,
            encode_cb,
            codec_user_data,
            codec_free_fn,
        )),
        _ => None,
    };
    let response_codec = if response_codec.is_null() {
        None
    } else {
        Some(unsafe { &*response_codec }.response_codec.clone())
    };

    let exec_fn = wrap_llm_stream_exec_fn(func, func_user_data, func_free);
    let default_fn: LlmStreamExecutionNextFn = Arc::new(move |request| exec_fn(request));

    let wrapped_collector: Box<dyn FnMut(serde_json::Value) -> FlowResult<()> + Send> =
        match collector {
            Some(cb) => wrap_collector_fn(cb),
            None => Box::new(|_: serde_json::Value| Ok(())),
        };

    let wrapped_finalizer: Box<dyn FnOnce() -> serde_json::Value + Send> = match finalizer {
        Some(cb) => wrap_finalizer_fn(cb),
        None => Box::new(|| serde_json::Value::Null),
    };

    let scope_stack = current_scope_stack();
    let result = tokio_runtime().block_on(TASK_SCOPE_STACK.scope(scope_stack, async {
        core_llm_api::llm_stream_call_execute(
            &name,
            request,
            default_fn,
            wrapped_collector,
            wrapped_finalizer,
            parent_handle,
            attrs,
            data,
            metadata,
            model_name_opt,
            codec,
            response_codec,
        )
        .await
    }));

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
            NemoFlowStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// Poll the next chunk from a streaming LLM response. Blocks until a chunk is
/// available.
///
/// # Returns
/// - `1`: A chunk was written to `*out_chunk`. Caller must free with
///   `nemo_flow_string_free`.
/// - `0`: The stream is complete (no more chunks).
/// - `-1`: An error occurred. Call `nemo_flow_last_error` for details.
///
/// # Safety
/// `stream` and `out_chunk` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_stream_next(
    stream: *mut FfiStream,
    out_chunk: *mut *mut c_char,
) -> i32 {
    clear_last_error();
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
        Some(Ok(chunk)) => {
            unsafe { *out_chunk = json_to_c_string(&chunk) };
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
/// `nemo_flow_llm_stream_call_execute`, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_flow_stream_free(stream: *mut FfiStream) {
    if !stream.is_null() {
        drop(unsafe { Box::from_raw(stream) });
    }
}
