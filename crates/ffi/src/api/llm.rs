// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    Arc, FfiCodecHandle, FfiLLMHandle, FfiLLMRequest, FfiLlmSanitizeRequestCodec,
    FfiLlmSanitizeResponseCodec, FfiScopeHandle, FlowResult, LlmAttributes, LlmExecutionNextFn,
    LlmRequest, LlmStreamExecutionNextFn, NemoRelayCodecDecodeFn, NemoRelayCodecEncodeFn,
    NemoRelayFreeFn, NemoRelayLlmExecCb, NemoRelayStatus, c_char, c_str_to_json, c_str_to_opt_json,
    c_str_to_string, clear_last_error, core_llm_api, json_to_c_string, set_last_error,
    status_from_error, tokio_runtime, unix_micros_to_opt_timestamp, wrap_codec_fn,
    wrap_llm_exec_fn, wrap_llm_stream_exec_fn, wrap_stream_callbacks,
};
use nemo_relay_runtime::api::runtime::LlmFinalizer;
use std::panic::{AssertUnwindSafe, catch_unwind};
use tokio_stream::StreamExt;

/// Decode a request through a callback-scoped sanitizer codec capability.
///
/// The returned JSON string must be freed with `nemo_relay_string_free`.
///
/// # Safety
/// Both pointers must be non-null and valid only during the sanitizer callback.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_llm_sanitize_request_codec_decode(
    codec: *const FfiLlmSanitizeRequestCodec,
    request: *const FfiLLMRequest,
) -> *mut c_char {
    clear_last_error();
    if codec.is_null() || request.is_null() {
        set_last_error("null sanitizer request codec argument");
        return std::ptr::null_mut();
    }
    let result = catch_unwind(AssertUnwindSafe(
        || -> std::result::Result<*mut c_char, String> {
            let annotated = unsafe { &*codec }
                .0
                .decode(&unsafe { &*request }.0)
                .map_err(|error| error.to_string())?;
            let value = serde_json::to_value(annotated).map_err(|error| error.to_string())?;
            Ok(json_to_c_string(&value))
        },
    ));
    match result {
        Ok(Ok(value)) => value,
        Ok(Err(error)) => {
            set_last_error(&error);
            std::ptr::null_mut()
        }
        Err(_) => {
            set_last_error("sanitizer request codec decode panicked");
            std::ptr::null_mut()
        }
    }
}

/// Encode normalized request changes through a callback-scoped codec capability.
///
/// The returned request is owned by the caller and must be freed with
/// `nemo_relay_llm_request_free`. Returns null on failure.
///
/// # Safety
/// All pointers must be non-null and valid only during the sanitizer callback.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_llm_sanitize_request_codec_encode(
    codec: *const FfiLlmSanitizeRequestCodec,
    annotated_json: *const c_char,
    original: *const FfiLLMRequest,
) -> *mut FfiLLMRequest {
    clear_last_error();
    if codec.is_null() || annotated_json.is_null() || original.is_null() {
        set_last_error("null sanitizer request codec argument");
        return std::ptr::null_mut();
    }
    let result = catch_unwind(AssertUnwindSafe(
        || -> std::result::Result<*mut FfiLLMRequest, String> {
            let annotated = c_str_to_json(annotated_json)
                .and_then(|value| serde_json::from_value(value).ok())
                .ok_or_else(|| "invalid annotated request JSON".to_string())?;
            let request = unsafe { &*codec }
                .0
                .encode(&annotated, &unsafe { &*original }.0)
                .map_err(|error| error.to_string())?;
            Ok(Box::into_raw(Box::new(FfiLLMRequest(request))))
        },
    ));
    match result {
        Ok(Ok(value)) => value,
        Ok(Err(error)) => {
            set_last_error(&error);
            std::ptr::null_mut()
        }
        Err(_) => {
            set_last_error("sanitizer request codec encode panicked");
            std::ptr::null_mut()
        }
    }
}

/// Decode a response through a callback-scoped sanitizer codec capability.
///
/// The returned JSON string must be freed with `nemo_relay_string_free`.
///
/// # Safety
/// All pointers must be non-null and valid only during the sanitizer callback.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_llm_sanitize_response_codec_decode(
    codec: *const FfiLlmSanitizeResponseCodec,
    response_json: *const c_char,
) -> *mut c_char {
    clear_last_error();
    if codec.is_null() || response_json.is_null() {
        set_last_error("null sanitizer response codec argument");
        return std::ptr::null_mut();
    }
    let result = catch_unwind(AssertUnwindSafe(
        || -> std::result::Result<*mut c_char, String> {
            let response =
                c_str_to_json(response_json).ok_or_else(|| "invalid response JSON".to_string())?;
            let annotated = unsafe { &*codec }
                .0
                .decode_response(&response)
                .map_err(|error| error.to_string())?;
            let value = serde_json::to_value(annotated).map_err(|error| error.to_string())?;
            Ok(json_to_c_string(&value))
        },
    ));
    match result {
        Ok(Ok(value)) => value,
        Ok(Err(error)) => {
            set_last_error(&error);
            std::ptr::null_mut()
        }
        Err(_) => {
            set_last_error("sanitizer response codec decode panicked");
            std::ptr::null_mut()
        }
    }
}

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
    request_codec: *const FfiCodecHandle,
    annotated_request_json: *const c_char,
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

    let codec = if request_codec.is_null() {
        None
    } else {
        match unsafe { &*request_codec }.codec.clone() {
            Some(codec) => Some(codec),
            None => {
                set_last_error("codec has no request decoder");
                return NemoRelayStatus::InvalidArg;
            }
        }
    };
    let annotated = match super::codec::parse_annotation(annotated_request_json) {
        Ok(value) => value,
        Err(status) => return status,
    };
    match core_llm_api::llm_call(
        core_llm_api::LlmCallParams::builder()
            .name(&name)
            .request(&request)
            .codec_opt(codec)
            .annotated_request_opt(annotated.map(Arc::new))
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
    response_codec: *const FfiCodecHandle,
    annotated_response_json: *const c_char,
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

    let codec = if response_codec.is_null() {
        None
    } else {
        match unsafe { &*response_codec }.response_codec.clone() {
            Some(codec) => Some(codec),
            None => {
                set_last_error("codec has no response decoder");
                return NemoRelayStatus::InvalidArg;
            }
        }
    };
    let annotated = match super::codec::parse_annotation(annotated_response_json) {
        Ok(value) => value,
        Err(status) => return status,
    };
    match core_llm_api::llm_call_end(
        core_llm_api::LlmCallEndParams::builder()
            .handle(&unsafe { &*handle }.0)
            .response(response)
            .response_codec_opt(codec)
            .annotated_response_opt(annotated.map(Arc::new))
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
// Built-in codec constructor
// ---------------------------------------------------------------------------

struct ParsedExecuteInputs {
    name: String,
    request: LlmRequest,
    parent_handle: Option<nemo_relay_runtime::api::scope::ScopeHandle>,
    attrs: LlmAttributes,
    data: Option<serde_json::Value>,
    metadata: Option<serde_json::Value>,
    model_name: Option<String>,
    codec: Option<Arc<dyn nemo_relay_types::codec::traits::LlmCodec>>,
    response_codec: Option<Arc<dyn nemo_relay_types::codec::traits::LlmResponseCodec>>,
}

struct RawExecuteInputs {
    name: *const c_char,
    native_json: *const c_char,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    model_name: *const c_char,
    codec: Option<Arc<dyn nemo_relay_types::codec::traits::LlmCodec>>,
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

/// Parse LLM call inputs while keeping ownership of the request codec.
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
    let codec = raw.codec;
    if !raw.response_codec.is_null() && unsafe { &*raw.response_codec }.response_codec.is_none() {
        set_last_error("codec has no response decoder");
        return Err(NemoRelayStatus::InvalidArg);
    }
    let response_codec = if raw.response_codec.is_null() {
        None
    } else {
        unsafe { &*raw.response_codec }.response_codec.clone()
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
/// - `execution_context_json`: Optional LLM application context JSON; null omits it.
///   Limited to 16 KiB and 64 container levels; exposed only to execution intercepts.
/// - `skip_stages`: Bitwise OR of `NEMO_RELAY_SKIP_*` constants. Zero runs all
///   middleware stages. Unknown bits return `InvalidArg`. Event sanitization,
///   codecs, trace context, and completion still run.
///
/// Relay owns callback data when the call starts. It releases the data even if validation fails.
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
    request_codec: *const FfiCodecHandle,
    response_codec: *const FfiCodecHandle,
    skip_stages: u32,
    execution_context_json: *const c_char,
    cancellation: *const super::FfiCancellation,
    out: *mut *mut c_char,
) -> NemoRelayStatus {
    clear_last_error();
    let exec_fn = wrap_llm_exec_fn(func, func_user_data, func_free);
    let codec = match (codec_decode, codec_encode) {
        (Some(decode), Some(encode)) => Some(wrap_codec_fn(
            decode,
            encode,
            codec_user_data,
            codec_free_fn,
        )),
        (None, None) => {
            if let Some(free) = codec_free_fn {
                unsafe { free(codec_user_data) };
            }
            None
        }
        _ => {
            if let Some(free) = codec_free_fn {
                unsafe { free(codec_user_data) };
            }
            set_last_error(
                "codec_decode and codec_encode must either both be provided or both be null",
            );
            return NemoRelayStatus::InvalidArg;
        }
    };
    let codec = if request_codec.is_null() {
        codec
    } else {
        if codec.is_some() {
            set_last_error("supply a request codec handle or callbacks, not both");
            return NemoRelayStatus::InvalidArg;
        }
        match unsafe { &*request_codec }.codec.clone() {
            Some(codec) => Some(codec),
            None => {
                set_last_error("codec has no request decoder");
                return NemoRelayStatus::InvalidArg;
            }
        }
    };
    let execution_context = match parse_application_context(execution_context_json) {
        Ok(context) => context,
        Err(status) => return status,
    };
    let execution_options = match super::execution_options(skip_stages) {
        Ok(options) => options,
        Err(status) => return status,
    };
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
        codec,
        response_codec,
    }) {
        Ok(parsed) => parsed,
        Err(status) => return status,
    };

    let default_fn: LlmExecutionNextFn = Arc::new(move |request| exec_fn(request));

    let cancellation = unsafe { cancellation.as_ref() }.cloned();
    let signal = cancellation;
    let result = super::block_on(async move {
        super::cancellation::run(signal, async move {
            core_llm_api::llm_call_execute(
                core_llm_api::LlmCallExecuteParams::builder()
                    .name(parsed.name)
                    .request(parsed.request)
                    .func(default_fn)
                    .execution_options(execution_options)
                    .execution_context_opt(execution_context)
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
        })
        .await
    });

    let result = match result {
        Ok(super::cancellation::Outcome::Completed(value)) => Ok(value),
        Err(error) => Err(error),
        Ok(super::cancellation::Outcome::Cancelled) => {
            set_last_error("execution cancelled");
            return NemoRelayStatus::Cancelled;
        }
    };
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

#[derive(Clone)]
pub(crate) struct StreamCloseError {
    message: String,
    cause_ids: Vec<u64>,
}

impl From<nemo_relay_types::error::FlowError> for StreamCloseError {
    fn from(error: nemo_relay_types::error::FlowError) -> Self {
        Self {
            message: error.to_string(),
            cause_ids: crate::error::callback_cause_ids(&error),
        }
    }
}

/// Opaque stream handle for consuming LLM streaming responses chunk by chunk.
/// Use `nemo_relay_stream_next` to poll and `nemo_relay_stream_free` to release.
pub struct FfiStream {
    pub(crate) pending: std::sync::Mutex<Option<ForwarderParts>>,
    pub(crate) receiver:
        tokio::sync::Mutex<tokio::sync::mpsc::Receiver<FlowResult<serde_json::Value>>>,
    pub(crate) cancel: tokio::sync::watch::Sender<bool>,
    pub(crate) closed: tokio::sync::watch::Receiver<Option<Result<(), StreamCloseError>>>,
    pub(crate) callbacks: Arc<crate::callable::StreamCallbackState>,
    pub(crate) close_result: tokio::sync::Mutex<Option<Result<(), StreamCloseError>>>,
}

pub(crate) struct ForwarderParts {
    pub(crate) stream: nemo_relay_runtime::api::runtime::LlmJsonStream,
    pub(crate) tx: tokio::sync::mpsc::Sender<FlowResult<serde_json::Value>>,
    pub(crate) cancel_rx: tokio::sync::watch::Receiver<bool>,
    pub(crate) closed: tokio::sync::watch::Sender<Option<Result<(), StreamCloseError>>>,
    pub(crate) context: nemo_relay_runtime::api::runtime::MiddlewareContinuationContext,
}

impl FfiStream {
    fn cancel(&self) {
        self.cancel.send_replace(true);
        self.start();
    }

    fn start(&self) {
        let parts = self
            .pending
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        if let Some(parts) = parts {
            let context = parts.context;
            tokio_runtime().spawn(super::stream_source::STREAM_CALLBACK_STATE.scope(
                self.callbacks.clone(),
                async move {
                    context
                        .run(forward_stream_to_channel(
                            parts.stream,
                            parts.tx,
                            parts.cancel_rx,
                            parts.closed,
                        ))
                        .await
                },
            ));
        }
    }
}

impl Drop for FfiStream {
    fn drop(&mut self) {
        self.cancel();
    }
}

async fn forward_stream_to_channel(
    mut stream: nemo_relay_runtime::api::runtime::LlmJsonStream,
    tx: tokio::sync::mpsc::Sender<FlowResult<serde_json::Value>>,
    mut cancel: tokio::sync::watch::Receiver<bool>,
    closed: tokio::sync::watch::Sender<Option<Result<(), StreamCloseError>>>,
) {
    // Reserve before pulling so a cancel never consumes a chunk or error it cannot deliver.
    loop {
        let permit = tokio::select! {
            biased;
            _ = cancel.changed() => break,
            permit = tx.reserve() => match permit {
                Ok(permit) => permit,
                Err(_) => break,
            },
        };
        let item = tokio::select! {
            biased;
            _ = cancel.changed() => break,
            item = stream.next() => item,
        };
        let Some(item) = item else {
            break;
        };
        permit.send(item);
    }
    let result = stream.close().await.map_err(StreamCloseError::from);
    drop(stream);
    closed.send_replace(Some(result));
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
///   finalizer returns `Json::Null`. A null or non-JSON result marks the LLM end
///   event ERROR with a null response and is reported by `nemo_relay_stream_close`.
/// - `callback_user_data`: Opaque pointer passed to `collector` and `finalizer`.
/// - `callback_free`: Optional destructor for `callback_user_data`.
/// - `parent`: Optional parent scope handle, or null.
/// - `attributes`: Bitfield of LLM attributes.
/// - `data_json`: Optional JSON data, or null.
/// - `metadata_json`: Optional JSON metadata, or null.
/// - `model_name`: Optional LLM model identifier, or null.
/// - `out`: On success, receives a heap-allocated `FfiStream`.
///
/// - `execution_context_json`: Optional LLM application context JSON; null omits it.
///   Limited to 16 KiB and 64 container levels; exposed only to execution intercepts.
/// - `skip_stages`: Bitwise OR of `NEMO_RELAY_SKIP_*` constants. Zero runs all
///   middleware stages. Unknown bits return `InvalidArg`. Event sanitization,
///   codecs, trace context, and completion still run.
///
/// Relay owns callback data when the call starts. It releases the data even if validation fails.
///
/// # Safety
/// `name`, `native_json`, and `out` must be valid, non-null pointers. `collector`
/// and `finalizer` may be null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_llm_stream_call_execute(
    name: *const c_char,
    native_json: *const c_char,
    func: crate::callable::NemoRelayOptionalLlmExecCb,
    stream_func: crate::callable::NemoRelayOptionalLlmStreamExecCb,
    func_user_data: *mut libc::c_void,
    func_free: NemoRelayFreeFn,
    collector: crate::callable::NemoRelayOptionalCollectorCb,
    finalizer: crate::callable::NemoRelayOptionalFinalizerCb,
    callback_user_data: *mut libc::c_void,
    callback_free: NemoRelayFreeFn,
    parent: *const FfiScopeHandle,
    attributes: u32,
    data_json: *const c_char,
    metadata_json: *const c_char,
    model_name: *const c_char,
    codec_decode: NemoRelayCodecDecodeFn,
    codec_encode: NemoRelayCodecEncodeFn,
    codec_user_data: *mut libc::c_void,
    codec_free_fn: NemoRelayFreeFn,
    request_codec: *const FfiCodecHandle,
    response_codec: *const FfiCodecHandle,
    skip_stages: u32,
    execution_context_json: *const c_char,
    cancellation: *const super::FfiCancellation,
    out: *mut *mut FfiStream,
) -> NemoRelayStatus {
    clear_last_error();
    let (wrapped_collector, wrapped_finalizer, callbacks) =
        wrap_stream_callbacks(collector, finalizer, callback_user_data, callback_free);
    let default_fn: LlmStreamExecutionNextFn = match stream_func {
        Some(callback) => {
            crate::callable::wrap_llm_stream_source_fn(callback, func_user_data, func_free)
        }
        None => {
            let Some(func) = func else {
                if let Some(free) = func_free {
                    unsafe { free(func_user_data) };
                }
                if let Some(free) = codec_free_fn {
                    unsafe { free(codec_user_data) };
                }
                set_last_error("a stream source factory or single-value provider is required");
                return NemoRelayStatus::InvalidArg;
            };
            let exec = wrap_llm_stream_exec_fn(func, func_user_data, func_free);
            Arc::new(move |request| exec(request))
        }
    };
    let codec = match (codec_decode, codec_encode) {
        (Some(decode), Some(encode)) => Some(wrap_codec_fn(
            decode,
            encode,
            codec_user_data,
            codec_free_fn,
        )),
        (None, None) => {
            if let Some(free) = codec_free_fn {
                unsafe { free(codec_user_data) };
            }
            None
        }
        _ => {
            if let Some(free) = codec_free_fn {
                unsafe { free(codec_user_data) };
            }
            set_last_error(
                "codec_decode and codec_encode must either both be provided or both be null",
            );
            return NemoRelayStatus::InvalidArg;
        }
    };
    let codec = if request_codec.is_null() {
        codec
    } else {
        if codec.is_some() {
            set_last_error("supply a request codec handle or callbacks, not both");
            return NemoRelayStatus::InvalidArg;
        }
        match unsafe { &*request_codec }.codec.clone() {
            Some(codec) => Some(codec),
            None => {
                set_last_error("codec has no request decoder");
                return NemoRelayStatus::InvalidArg;
            }
        }
    };
    let execution_context = match parse_application_context(execution_context_json) {
        Ok(context) => context,
        Err(status) => return status,
    };
    let execution_options = match super::execution_options(skip_stages) {
        Ok(options) => options,
        Err(status) => return status,
    };
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
        codec,
        response_codec,
    }) {
        Ok(parsed) => parsed,
        Err(status) => return status,
    };

    let forwarding_context =
        nemo_relay_runtime::api::runtime::MiddlewareContinuationContext::capture();
    let cancellation = unsafe { cancellation.as_ref() }.cloned();
    let signal = cancellation;
    let result = super::block_on(super::stream_source::STREAM_CALLBACK_STATE.scope(
        callbacks.clone(),
        async move {
            super::cancellation::run(signal, async move {
                core_llm_api::llm_stream_call_execute(
                    core_llm_api::LlmStreamCallExecuteParams::builder()
                        .name(parsed.name)
                        .request(parsed.request)
                        .func(default_fn)
                        .execution_options(execution_options)
                        .execution_context_opt(execution_context)
                        .collector(wrapped_collector)
                        .finalizer(LlmFinalizer::Fallible(wrapped_finalizer))
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
            })
            .await
        },
    ));

    let result = match result {
        Ok(super::cancellation::Outcome::Completed(value)) => Ok(value),
        Err(error) => Err(error),
        Ok(super::cancellation::Outcome::Cancelled) => {
            set_last_error("execution cancelled");
            return NemoRelayStatus::Cancelled;
        }
    };
    match result {
        Ok(rust_stream) => {
            let (tx, rx) = tokio::sync::mpsc::channel(32);
            let (cancel, cancel_rx) = tokio::sync::watch::channel(false);
            let (closed, closed_rx) = tokio::sync::watch::channel(None);
            let ffi_stream = Box::new(FfiStream {
                pending: std::sync::Mutex::new(Some(ForwarderParts {
                    stream: rust_stream,
                    tx,
                    cancel_rx,
                    closed,
                    context: forwarding_context,
                })),
                receiver: tokio::sync::Mutex::new(rx),
                cancel,
                closed: closed_rx,
                callbacks,
                close_result: tokio::sync::Mutex::new(None),
            });
            unsafe { *out = Box::into_raw(ffi_stream) };
            NemoRelayStatus::Ok
        }
        Err(e) => status_from_error(&e),
    }
}

/// Stop a stream producer.
///
/// # Safety
/// `stream` must be a valid `FfiStream` pointer returned by
/// `nemo_relay_llm_stream_call_execute`, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_stream_cancel(stream: *mut FfiStream) {
    if stream.is_null() {
        return;
    }
    // SAFETY: the caller guarantees `stream` came from
    // `nemo_relay_llm_stream_call_execute` and has not been freed.
    unsafe { &*stream }.cancel();
}

/// Stop a stream producer and wait for cleanup to complete.
///
/// This operation is idempotent. It does not free `stream`; callers must still
/// release the handle with [`nemo_relay_stream_free`].
///
/// # Safety
/// `stream` must be a valid `FfiStream` pointer returned by
/// `nemo_relay_llm_stream_call_execute`, or null.
/// Call this outside owned callbacks. Managed-stream control from a callback is invalid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_stream_close(stream: *mut FfiStream) -> NemoRelayStatus {
    clear_last_error();
    if stream.is_null() {
        set_last_error("null pointer argument");
        return NemoRelayStatus::NullPointer;
    }
    let stream = unsafe { &*stream };
    stream.cancel();
    let result = match super::block_on(async { Ok(finish_close(stream).await) }) {
        Ok(result) => result,
        Err(error) => Err(StreamCloseError::from(error)),
    };
    match result {
        Ok(()) => NemoRelayStatus::Ok,
        Err(error) => {
            set_last_error(&error.message);
            unsafe {
                crate::error::nemo_relay_set_last_error_cause_ids(
                    error.cause_ids.as_ptr(),
                    error.cause_ids.len(),
                );
            }
            NemoRelayStatus::Internal
        }
    }
}

async fn finish_close(stream: &FfiStream) -> Result<(), StreamCloseError> {
    let mut close_result = stream.close_result.lock().await;
    if let Some(result) = close_result.as_ref() {
        return result.clone();
    }
    let cleanup = stream
        .closed
        .clone()
        .wait_for(Option::is_some)
        .await
        .map_err(|_| StreamCloseError {
            message: "stream close task ended early".into(),
            cause_ids: Vec::new(),
        })?
        .clone()
        .expect("close state checked above");
    let mut receiver = stream.receiver.lock().await;
    receiver.close();
    let mut undelivered = None;
    while let Ok(item) = receiver.try_recv() {
        if let Err(error) = item {
            undelivered.get_or_insert_with(|| StreamCloseError::from(error));
        }
    }
    let finalizer_error = stream.callbacks.take_failure().map(StreamCloseError::from);
    let result = combine_close_errors([undelivered, finalizer_error, cleanup.err()]);
    *close_result = Some(result.clone());
    result
}

fn combine_close_errors(errors: [Option<StreamCloseError>; 3]) -> Result<(), StreamCloseError> {
    let mut errors = errors.into_iter().flatten();
    let Some(first) = errors.next() else {
        return Ok(());
    };
    Err(errors.fold(first, |mut acc, next| {
        acc.message = format!(
            "{}; stream cleanup also failed: {}",
            acc.message, next.message
        );
        for id in next.cause_ids {
            if !acc.cause_ids.contains(&id) {
                acc.cause_ids.push(id);
            }
        }
        acc
    }))
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
/// Call this outside owned callbacks. Managed-stream control from a callback is invalid.
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
    stream.start();
    let result = super::block_on(async {
        let mut guard = stream.receiver.lock().await;
        Ok(guard.recv().await)
    });
    match result {
        Ok(None) => 0, // stream done
        Ok(Some(Ok(chunk))) => {
            unsafe { *out_chunk = json_to_c_string(&chunk) };
            1
        }
        Ok(Some(Err(e))) | Err(e) => {
            status_from_error(&e);
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

fn parse_application_context(
    pointer: *const c_char,
) -> Result<Option<nemo_relay_types::api::llm::ApplicationExecutionContext>, NemoRelayStatus> {
    use nemo_relay_types::api::llm::ApplicationExecutionContext;
    if pointer.is_null() {
        return Ok(None);
    }
    // The public entry point requires a readable NUL-terminated C string.
    let input = unsafe { std::ffi::CStr::from_ptr(pointer) }
        .to_str()
        .map_err(|_| {
            set_last_error("invalid execution context UTF-8");
            NemoRelayStatus::InvalidJson
        })?;
    context_input_budget(input.as_bytes()).map_err(|message| {
        set_last_error(message);
        NemoRelayStatus::InvalidArg
    })?;
    let value: serde_json::Value = serde_json::from_str(input).map_err(|_| {
        set_last_error("invalid execution context JSON");
        NemoRelayStatus::InvalidJson
    })?;
    ApplicationExecutionContext::try_from(value)
        .map(Some)
        .map_err(|error| {
            set_last_error(&error.to_string());
            NemoRelayStatus::InvalidArg
        })
}

fn context_input_budget(input: &[u8]) -> Result<(), &'static str> {
    let mut remaining_bytes = 16_384_usize;
    let mut depth = 0_usize;
    let mut quoted = false;
    let mut index = 0;
    while let Some(&byte) = input.get(index) {
        index += 1;
        let charge = if quoted {
            match byte {
                b'"' => {
                    quoted = false;
                    0
                }
                b'\\' => match input.get(index) {
                    Some(b'u') => {
                        index = index.saturating_add(5);
                        // One decoded byte is a safe lower bound, even for
                        // paired surrogates. Exact size is checked after parsing.
                        1
                    }
                    Some(b'/') => {
                        index += 1;
                        1
                    }
                    Some(_) => {
                        index += 1;
                        2
                    }
                    None => 0,
                },
                _ => 1,
            }
        } else {
            match byte {
                b' ' | b'\n' | b'\r' | b'\t' => 0,
                b'"' => {
                    quoted = true;
                    2
                }
                b'{' | b'[' => {
                    depth += 1;
                    if depth > 64 {
                        return Err("LLM execution context exceeds 64 container levels");
                    }
                    1
                }
                b'}' | b']' => {
                    depth = depth.saturating_sub(1);
                    1
                }
                b'-' | b'0'..=b'9' => {
                    // Number spelling can be longer than its normalized value.
                    // serde_json parses numbers without constructing a JSON string.
                    while input.get(index).is_some_and(|byte| {
                        matches!(byte, b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-')
                    }) {
                        index += 1;
                    }
                    1
                }
                _ => 1,
            }
        };
        remaining_bytes = remaining_bytes
            .checked_sub(charge)
            .ok_or("LLM execution context exceeds 16384 serialized bytes")?;
    }
    Ok(())
}
