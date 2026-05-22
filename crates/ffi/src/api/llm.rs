// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    Arc, FfiCodecHandle, FfiLLMHandle, FfiScopeHandle, FlowResult, LlmAttributes,
    LlmExecutionNextFn, LlmRequest, LlmStreamExecutionNextFn, NemoRelayCodecDecodeFn,
    NemoRelayCodecEncodeFn, NemoRelayCollectorCb, NemoRelayFinalizerCb, NemoRelayFreeFn,
    NemoRelayLlmExecCb, NemoRelayStatus, TASK_SCOPE_STACK, c_char, c_str_to_json,
    c_str_to_opt_json, c_str_to_string, clear_last_error, core_llm_api, current_scope_stack,
    json_to_c_string, set_last_error, status_from_error, tokio_runtime,
    unix_micros_to_opt_timestamp, wrap_codec_fn, wrap_collector_fn, wrap_finalizer_fn,
    wrap_llm_exec_fn, wrap_llm_stream_exec_fn,
};
use tokio_stream::StreamExt;

// ---------------------------------------------------------------------------
// LLM lifecycle
// ---------------------------------------------------------------------------

/// Begin a manual LLM call lifecycle span.
///
/// This emits an LLM Start event after applying sanitize-request guardrails to
/// the observability payload. Request and execution intercepts only run through
/// `nemo_relay_llm_call_execute`.
///
/// # Parameters
/// - `name`: Null-terminated LLM provider name.
/// - `native_json`: The request payload as a JSON C string representing an
///   `LlmRequest` (`{"headers": {...}, "content": {...}}`). The request
///   becomes the start-event data after sanitize-request guardrails.
/// - `parent`: Optional parent scope handle, or null to use the current top of
///   stack.
/// - `attributes`: Bitfield of LLM attributes.
/// - `data_json`: Optional null-terminated JSON string stored on the LLM
///   handle, or null.
/// - `metadata_json`: Optional null-terminated JSON metadata string recorded
///   on the start event, or null.
/// - `model_name`: Optional null-terminated LLM model identifier recorded in
///   the LLM event category profile, or null.
/// - `timestamp_unix_micros`: Optional Unix microseconds timestamp for the
///   handle start time and start event, or null to use the current UTC time.
/// - `out`: On success, receives a heap-allocated `FfiLLMHandle` that must be
///   freed with `nemo_relay_llm_handle_free`.
///
/// # Errors
/// Returns `InvalidJson` for invalid JSON inputs and `InvalidArg` when
/// `timestamp_unix_micros` is outside the supported timestamp range.
///
/// # Safety
/// `name`, `native_json`, and `out` must be valid, non-null pointers. Optional
/// pointer arguments may be null; when non-null, they must be valid for reads
/// for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_llm_call(
    name: *const c_char,
    native_json: *const c_char,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    model_name: *const c_char,
    timestamp_unix_micros: *const i64,
    out: *mut *mut FfiLLMHandle,
) -> NemoRelayStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("null pointer argument");
        return NemoRelayStatus::NullPointer;
    }
    let name = match c_str_to_string(name) {
        Ok(s) => s,
        Err(status) => return status,
    };
    let native = match c_str_to_json(native_json) {
        Some(n) => n,
        None => return NemoRelayStatus::InvalidJson,
    };
    let request: LlmRequest = match serde_json::from_value(native) {
        Ok(r) => r,
        Err(_) => {
            set_last_error("failed to parse native_json as LlmRequest");
            return NemoRelayStatus::InvalidJson;
        }
    };
    let parent_ref = if parent.is_null() {
        None
    } else {
        Some(&unsafe { &*parent }.0)
    };
    let attrs = LlmAttributes::from_bits_truncate(attributes);
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NemoRelayStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NemoRelayStatus::InvalidJson,
    };
    let model_name_opt = if model_name.is_null() {
        None
    } else {
        match c_str_to_string(model_name) {
            Ok(s) => Some(s),
            Err(status) => return status,
        }
    };
    let timestamp = match unix_micros_to_opt_timestamp(timestamp_unix_micros) {
        Some(v) => v,
        None => return NemoRelayStatus::InvalidArg,
    };

    match core_llm_api::llm_call(
        core_llm_api::LlmCallParams::builder()
            .name(&name)
            .request(&request)
            .parent_opt(parent_ref)
            .attributes(attrs)
            .data_opt(data)
            .metadata_opt(metadata)
            .model_name_opt(model_name_opt)
            .timestamp_opt(timestamp)
            .build(),
    ) {
        Ok(h) => {
            unsafe { *out = Box::into_raw(Box::new(FfiLLMHandle(h))) };
            NemoRelayStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// End a manual LLM call lifecycle span.
///
/// This emits an LLM End event after applying sanitize-response guardrails to
/// the observability payload. Response intercepts only run through
/// `nemo_relay_llm_call_execute`.
///
/// # Parameters
/// - `handle`: The LLM handle from `nemo_relay_llm_call`.
/// - `response_json`: LLM response as a null-terminated JSON C string. This
///   response becomes the end-event data after sanitize-response guardrails
///   unless it sanitizes to JSON null.
/// - `data_json`: Optional null-terminated JSON data used when the sanitized
///   response is JSON null, or null.
/// - `metadata_json`: Optional null-terminated JSON metadata recorded on the
///   end event, or null.
/// - `timestamp_unix_micros`: Optional Unix microseconds timestamp for the end
///   event, or null to use the runtime default end timestamp.
///
/// # Errors
/// Returns `InvalidJson` for invalid JSON inputs and `InvalidArg` when
/// `timestamp_unix_micros` is outside the supported timestamp range.
///
/// # Safety
/// `handle` and `response_json` must be valid, non-null pointers. Optional
/// pointer arguments may be null; when non-null, they must be valid for reads
/// for the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_llm_call_end(
    handle: *const FfiLLMHandle,
    response_json: *const c_char,
    data_json: *const c_char,
    metadata_json: *const c_char,
    timestamp_unix_micros: *const i64,
) -> NemoRelayStatus {
    clear_last_error();
    if handle.is_null() {
        set_last_error("handle is null");
        return NemoRelayStatus::NullPointer;
    }
    let response = match c_str_to_json(response_json) {
        Some(r) => r,
        None => return NemoRelayStatus::InvalidJson,
    };
    let data = match c_str_to_opt_json(data_json) {
        Some(d) => d,
        None => return NemoRelayStatus::InvalidJson,
    };
    let metadata = match c_str_to_opt_json(metadata_json) {
        Some(m) => m,
        None => return NemoRelayStatus::InvalidJson,
    };
    let timestamp = match unix_micros_to_opt_timestamp(timestamp_unix_micros) {
        Some(v) => v,
        None => return NemoRelayStatus::InvalidArg,
    };

    match core_llm_api::llm_call_end(
        core_llm_api::LlmCallEndParams::builder()
            .handle(&unsafe { &*handle }.0)
            .response(response)
            .data_opt(data)
            .metadata_opt(metadata)
            .timestamp_opt(timestamp)
            .build(),
    ) {
        Ok(()) => NemoRelayStatus::Ok,
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// Built-in codec constructors
// ---------------------------------------------------------------------------

/// Create a new OpenAI Chat Completions codec handle.
///
/// The returned handle implements both request codec (decode/encode) and
/// response codec (decode_response). Free with `nemo_relay_codec_free`.
///
/// # Safety
/// Caller must free the returned handle via `nemo_relay_codec_free`.
#[unsafe(no_mangle)]
pub extern "C" fn nemo_relay_openai_chat_codec_new() -> *mut FfiCodecHandle {
    Box::into_raw(Box::new(FfiCodecHandle {
        codec: Arc::new(nemo_relay::codec::openai_chat::OpenAIChatCodec),
        response_codec: Arc::new(nemo_relay::codec::openai_chat::OpenAIChatCodec),
    }))
}

/// Create a new OpenAI Responses API codec handle.
///
/// The returned handle implements both request codec (decode/encode) and
/// response codec (decode_response). Free with `nemo_relay_codec_free`.
///
/// # Safety
/// Caller must free the returned handle via `nemo_relay_codec_free`.
#[unsafe(no_mangle)]
pub extern "C" fn nemo_relay_openai_responses_codec_new() -> *mut FfiCodecHandle {
    Box::into_raw(Box::new(FfiCodecHandle {
        codec: Arc::new(nemo_relay::codec::openai_responses::OpenAIResponsesCodec),
        response_codec: Arc::new(nemo_relay::codec::openai_responses::OpenAIResponsesCodec),
    }))
}

/// Create a new Anthropic Messages API codec handle.
///
/// The returned handle implements both request codec (decode/encode) and
/// response codec (decode_response). Free with `nemo_relay_codec_free`.
///
/// # Safety
/// Caller must free the returned handle via `nemo_relay_codec_free`.
#[unsafe(no_mangle)]
pub extern "C" fn nemo_relay_anthropic_messages_codec_new() -> *mut FfiCodecHandle {
    Box::into_raw(Box::new(FfiCodecHandle {
        codec: Arc::new(nemo_relay::codec::anthropic::AnthropicMessagesCodec),
        response_codec: Arc::new(nemo_relay::codec::anthropic::AnthropicMessagesCodec),
    }))
}

struct ParsedExecuteInputs {
    name: String,
    request: LlmRequest,
    parent_handle: Option<nemo_relay::api::scope::ScopeHandle>,
    attrs: LlmAttributes,
    data: Option<serde_json::Value>,
    metadata: Option<serde_json::Value>,
    model_name: Option<String>,
    codec: Option<Arc<dyn nemo_relay::codec::traits::LlmCodec>>,
    response_codec: Option<Arc<dyn nemo_relay::codec::traits::LlmResponseCodec>>,
}

struct RawExecuteInputs {
    name: *const c_char,
    native_json: *const c_char,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    model_name: *const c_char,
    codec_decode: NemoRelayCodecDecodeFn,
    codec_encode: NemoRelayCodecEncodeFn,
    codec_user_data: *mut libc::c_void,
    codec_free_fn: NemoRelayFreeFn,
    response_codec: *const FfiCodecHandle,
}

fn parse_llm_request(native_json: *const c_char) -> Result<LlmRequest, NemoRelayStatus> {
    let native = c_str_to_json(native_json).ok_or(NemoRelayStatus::InvalidJson)?;
    serde_json::from_value(native).map_err(|_| {
        set_last_error("failed to parse native_json as LlmRequest");
        NemoRelayStatus::InvalidJson
    })
}

fn parse_optional_model_name(model_name: *const c_char) -> Result<Option<String>, NemoRelayStatus> {
    if model_name.is_null() {
        Ok(None)
    } else {
        c_str_to_string(model_name).map(Some)
    }
}

fn parse_execute_inputs(raw: RawExecuteInputs) -> Result<ParsedExecuteInputs, NemoRelayStatus> {
    let name = c_str_to_string(raw.name)?;
    let request = parse_llm_request(raw.native_json)?;
    let parent_handle = if raw.parent.is_null() {
        None
    } else {
        Some(unsafe { &*raw.parent }.0.clone())
    };
    let attrs = LlmAttributes::from_bits_truncate(raw.attributes);
    let data = c_str_to_opt_json(raw.data_json).ok_or(NemoRelayStatus::InvalidJson)?;
    let metadata = c_str_to_opt_json(raw.metadata_json).ok_or(NemoRelayStatus::InvalidJson)?;
    let model_name = parse_optional_model_name(raw.model_name)?;
    let codec = match (raw.codec_decode, raw.codec_encode) {
        (Some(decode_cb), Some(encode_cb)) => Some(wrap_codec_fn(
            decode_cb,
            encode_cb,
            raw.codec_user_data,
            raw.codec_free_fn,
        )),
        (None, None) => None,
        _ => {
            set_last_error(
                "codec_decode and codec_encode must either both be provided or both be null",
            );
            return Err(NemoRelayStatus::InvalidArg);
        }
    };
    let response_codec = if raw.response_codec.is_null() {
        None
    } else {
        Some(unsafe { &*raw.response_codec }.response_codec.clone())
    };

    Ok(ParsedExecuteInputs {
        name,
        request,
        parent_handle,
        attrs,
        data,
        metadata,
        model_name,
        codec,
        response_codec,
    })
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
///   `LlmRequest` (`{"headers": {...}, "content": {...}}`).
/// - `func`: C callback that performs the actual LLM call.
/// - `func_user_data`: Opaque pointer passed to `func`.
/// - `func_free`: Optional destructor for `func_user_data`.
/// - `parent`: Optional parent scope handle, or null.
/// - `attributes`: Bitfield of LLM attributes.
/// - `data_json`: Optional JSON data, or null.
/// - `metadata_json`: Optional JSON metadata, or null.
/// - `model_name`: Optional LLM model identifier, or null.
/// - `out`: On success, receives the response as a JSON C string. Caller must
///   free with `nemo_relay_string_free`.
///
/// # Safety
/// `name`, `native_json`, and `out` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_llm_call_execute(
    name: *const c_char,
    native_json: *const c_char,
    func: NemoRelayLlmExecCb,
    func_user_data: *mut libc::c_void,
    func_free: NemoRelayFreeFn,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    model_name: *const c_char,
    codec_decode: NemoRelayCodecDecodeFn,
    codec_encode: NemoRelayCodecEncodeFn,
    codec_user_data: *mut libc::c_void,
    codec_free_fn: NemoRelayFreeFn,
    response_codec: *const FfiCodecHandle,
    out: *mut *mut c_char,
) -> NemoRelayStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("null pointer argument");
        return NemoRelayStatus::NullPointer;
    }
    let parsed = match parse_execute_inputs(RawExecuteInputs {
        name,
        native_json,
        parent,
        attributes,
        data_json,
        metadata_json,
        model_name,
        codec_decode,
        codec_encode,
        codec_user_data,
        codec_free_fn,
        response_codec,
    }) {
        Ok(parsed) => parsed,
        Err(status) => return status,
    };

    let exec_fn = wrap_llm_exec_fn(func, func_user_data, func_free);
    let default_fn: LlmExecutionNextFn = Arc::new(move |request| exec_fn(request));

    let scope_stack = current_scope_stack();
    let result = tokio_runtime().block_on(TASK_SCOPE_STACK.scope(scope_stack, async {
        core_llm_api::llm_call_execute(
            core_llm_api::LlmCallExecuteParams::builder()
                .name(parsed.name)
                .request(parsed.request)
                .func(default_fn)
                .parent_opt(parsed.parent_handle)
                .attributes(parsed.attrs)
                .data_opt(parsed.data)
                .metadata_opt(parsed.metadata)
                .model_name_opt(parsed.model_name)
                .codec_opt(parsed.codec)
                .response_codec_opt(parsed.response_codec)
                .build(),
        )
        .await
    }));

    match result {
        Ok(json) => {
            unsafe { *out = json_to_c_string(&json) };
            NemoRelayStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

// ---------------------------------------------------------------------------
// Stream
// ---------------------------------------------------------------------------

/// Opaque stream handle for consuming LLM streaming responses chunk by chunk.
/// Use `nemo_relay_stream_next` to poll and `nemo_relay_stream_free` to release.
pub struct FfiStream {
    pub(crate) receiver:
        tokio::sync::Mutex<tokio::sync::mpsc::Receiver<FlowResult<serde_json::Value>>>,
}

/// Execute a streaming LLM call end-to-end. Conditional-execution guardrails
/// run first on the raw request. Returns a stream handle that can be polled
/// with `nemo_relay_stream_next`. Blocks until the stream is set up.
///
/// # Parameters
/// - `name`: Null-terminated LLM provider name.
/// - `native_json`: The request payload as a JSON C string representing an
///   `LlmRequest` (`{"headers": {...}, "content": {...}}`).
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
pub unsafe extern "C" fn nemo_relay_llm_stream_call_execute(
    name: *const c_char,
    native_json: *const c_char,
    func: NemoRelayLlmExecCb,
    func_user_data: *mut libc::c_void,
    func_free: NemoRelayFreeFn,
    collector: Option<NemoRelayCollectorCb>,
    finalizer: Option<NemoRelayFinalizerCb>,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    model_name: *const c_char,
    codec_decode: NemoRelayCodecDecodeFn,
    codec_encode: NemoRelayCodecEncodeFn,
    codec_user_data: *mut libc::c_void,
    codec_free_fn: NemoRelayFreeFn,
    response_codec: *const FfiCodecHandle,
    out: *mut *mut FfiStream,
) -> NemoRelayStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("null pointer argument");
        return NemoRelayStatus::NullPointer;
    }
    let parsed = match parse_execute_inputs(RawExecuteInputs {
        name,
        native_json,
        parent,
        attributes,
        data_json,
        metadata_json,
        model_name,
        codec_decode,
        codec_encode,
        codec_user_data,
        codec_free_fn,
        response_codec,
    }) {
        Ok(parsed) => parsed,
        Err(status) => return status,
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
            core_llm_api::LlmStreamCallExecuteParams::builder()
                .name(parsed.name)
                .request(parsed.request)
                .func(default_fn)
                .collector(wrapped_collector)
                .finalizer(wrapped_finalizer)
                .parent_opt(parsed.parent_handle)
                .attributes(parsed.attrs)
                .data_opt(parsed.data)
                .metadata_opt(parsed.metadata)
                .model_name_opt(parsed.model_name)
                .codec_opt(parsed.codec)
                .response_codec_opt(parsed.response_codec)
                .build(),
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
            NemoRelayStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// Poll the next chunk from a streaming LLM response. Blocks until a chunk is
/// available.
///
/// # Returns
/// - `1`: A chunk was written to `*out_chunk`. Caller must free with
///   `nemo_relay_string_free`.
/// - `0`: The stream is complete (no more chunks).
/// - `-1`: An error occurred. Call `nemo_relay_last_error` for details.
///
/// # Safety
/// `stream` and `out_chunk` must be valid, non-null pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_stream_next(
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
/// `nemo_relay_llm_stream_call_execute`, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_stream_free(stream: *mut FfiStream) {
    if !stream.is_null() {
        drop(unsafe { Box::from_raw(stream) });
    }
}
