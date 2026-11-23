// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Owned codecs and invocation-scoped optimization recording.

use super::*;
use crate::callable::{NemoRelayCodecCallbacks, codec_from_callbacks};
use crate::types::*;
use nemo_relay_types::Json;
use nemo_relay_types::error::FlowError;
use serde::de::DeserializeOwned;

pub(super) fn parse_annotation<T: DeserializeOwned>(
    value: *const c_char,
) -> Result<Option<T>, NemoRelayStatus> {
    if value.is_null() {
        return Ok(None);
    }
    let value = c_str_to_string(value)?;
    serde_json::from_str(&value).map(Some).map_err(|error| {
        set_last_error(&format!("invalid annotation JSON: {error}"));
        NemoRelayStatus::InvalidJson
    })
}

/// Create a request codec, response codec, or both. Built-in providers use null callbacks.
/// `NEMO_RELAY_CODEC_CUSTOM` requires at least one codec direction.
/// Request decode and encode must be provided together. Relay owns callback state
/// when the call starts, even if validation fails. It frees the state exactly once
/// after all native references are released. Unknown providers return null and set the last
/// error. Release a successful result with `nemo_relay_codec_free`.
///
/// # Safety
/// Non-null callbacks must point to a valid struct. Callback state must remain
/// valid and thread-safe until its destructor runs. Callback result strings
/// transfer ownership to Relay and must use the Relay string allocator contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_codec_new(
    provider: u32,
    callbacks: *const NemoRelayCodecCallbacks,
) -> *mut FfiCodecHandle {
    use nemo_relay_codecs::resolve::ProviderSurface;
    clear_last_error();
    let custom = if callbacks.is_null() {
        None
    } else {
        Some(codec_from_callbacks(unsafe { *callbacks }))
    };
    if provider == NEMO_RELAY_CODEC_CUSTOM {
        return match custom {
            Some(Ok(codec)) => Box::into_raw(Box::new(codec)),
            Some(Err(error)) => {
                status_from_error(&error);
                std::ptr::null_mut()
            }
            None => {
                set_last_error("custom codec requires callbacks");
                std::ptr::null_mut()
            }
        };
    }
    if custom.is_some() {
        set_last_error("built-in codecs do not accept callbacks");
        return std::ptr::null_mut();
    }
    let surface = match provider {
        NEMO_RELAY_CODEC_OPENAI_CHAT => ProviderSurface::OpenAIChat,
        NEMO_RELAY_CODEC_OPENAI_RESPONSES => ProviderSurface::OpenAIResponses,
        NEMO_RELAY_CODEC_ANTHROPIC_MESSAGES => ProviderSurface::AnthropicMessages,
        NEMO_RELAY_CODEC_GEMINI_GENERATE_CONTENT => ProviderSurface::GeminiGenerateContent,
        NEMO_RELAY_CODEC_OCI_GENAI => ProviderSurface::OCIGenAI,
        _ => {
            set_last_error("unknown codec provider");
            return std::ptr::null_mut();
        }
    };
    Box::into_raw(Box::new(FfiCodecHandle {
        codec: Some(nemo_relay_codecs::resolve::request_codec(surface)),
        response_codec: Some(nemo_relay_runtime::codec::response_codec(surface)),
    }))
}

fn codec_json(operation: impl FnOnce() -> FlowResult<Json>) -> *mut c_char {
    clear_last_error();
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(value)) => json_to_c_string(&value),
        Ok(Err(error)) => {
            status_from_error(&error);
            std::ptr::null_mut()
        }
        Err(_) => {
            set_last_error("codec operation panicked");
            std::ptr::null_mut()
        }
    }
}

/// Decode an owned codec's request to annotated request JSON. Free the result
/// with `nemo_relay_string_free`. Null indicates failure. Read `nemo_relay_last_error`.
///
/// # Safety
/// `codec` and `request` must be live, non-null handles for this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_codec_decode(
    codec: *const FfiCodecHandle,
    request: *const FfiLLMRequest,
) -> *mut c_char {
    codec_json(|| {
        let codec = unsafe { codec.as_ref() }
            .and_then(|c| c.codec.as_ref())
            .ok_or_else(|| FlowError::InvalidArgument("codec has no request decoder".into()))?;
        let request = unsafe { request.as_ref() }
            .ok_or_else(|| FlowError::InvalidArgument("request is null".into()))?;
        serde_json::to_value(codec.decode(&request.0)?)
            .map_err(|e| FlowError::Internal(e.to_string()))
    })
}

/// Merge annotated request JSON into an original request. Returns the complete
/// request JSON with headers and content. Free it with `nemo_relay_string_free`.
///
/// # Safety
/// Handles must be live and non-null; `annotated_json` must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_codec_encode(
    codec: *const FfiCodecHandle,
    annotated_json: *const c_char,
    original: *const FfiLLMRequest,
) -> *mut c_char {
    codec_json(|| {
        let codec = unsafe { codec.as_ref() }
            .and_then(|c| c.codec.as_ref())
            .ok_or_else(|| FlowError::InvalidArgument("codec has no request encoder".into()))?;
        let original = unsafe { original.as_ref() }
            .ok_or_else(|| FlowError::InvalidArgument("request is null".into()))?;
        let annotated = parse_annotation(annotated_json)
            .map_err(|_| FlowError::InvalidArgument("invalid request annotation".into()))?
            .ok_or_else(|| FlowError::InvalidArgument("request annotation is required".into()))?;
        serde_json::to_value(codec.encode(&annotated, &original.0)?)
            .map_err(|e| FlowError::Internal(e.to_string()))
    })
}

/// Decode a response into annotated response JSON. Free the result with
/// `nemo_relay_string_free`. Null indicates failure. Read `nemo_relay_last_error`.
///
/// # Safety
/// `codec` must be live and `response_json` must be a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_codec_decode_response(
    codec: *const FfiCodecHandle,
    response_json: *const c_char,
) -> *mut c_char {
    codec_json(|| {
        let codec = unsafe { codec.as_ref() }
            .and_then(|c| c.response_codec.as_ref())
            .ok_or_else(|| FlowError::InvalidArgument("codec has no response decoder".into()))?;
        let value = c_str_to_json(response_json)
            .ok_or_else(|| FlowError::InvalidArgument("invalid response JSON".into()))?;
        serde_json::to_value(codec.decode_response(&value)?)
            .map_err(|e| FlowError::Internal(e.to_string()))
    })
}

/// Retain the optimization recorder owned by a manual LLM call handle.
/// The returned recorder has an independent lifetime and must be freed with
/// `nemo_relay_llm_optimization_recorder_free`. Ending the LLM call closes
/// recording. Later records return `accepted = false`.
///
/// # Safety
/// `handle` must be a live LLM handle or null. `out` must be valid writable
/// pointer storage. A non-null output slot is cleared before validation.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_llm_handle_retain_optimization_recorder(
    handle: *const FfiLLMHandle,
    out: *mut *mut FfiLlmOptimizationRecorder,
) -> NemoRelayStatus {
    clear_last_error();
    if out.is_null() {
        set_last_error("optimization recorder output is null");
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *out = std::ptr::null_mut() };
    let Some(handle) = (unsafe { handle.as_ref() }) else {
        set_last_error("LLM handle is null");
        return NemoRelayStatus::NullPointer;
    };
    unsafe {
        *out = Box::into_raw(Box::new(FfiLlmOptimizationRecorder(
            handle.0.optimization_recorder.clone(),
        )));
    }
    NemoRelayStatus::Ok
}

/// Record optimization evidence on a borrowed or retained recorder.
/// `accepted` reports whether Relay accepted the record. Rejection does not fail execution.
///
/// # Safety
/// `recorder` must be live, either retained or borrowed from an active callback;
/// `contribution_json` is a valid C string and `accepted` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_llm_optimization_record(
    recorder: *const FfiLlmOptimizationRecorder,
    contribution_json: *const c_char,
    accepted: *mut bool,
) -> NemoRelayStatus {
    clear_last_error();
    if recorder.is_null() || accepted.is_null() {
        set_last_error("null optimization recorder or output");
        return NemoRelayStatus::NullPointer;
    }
    unsafe { *accepted = false };
    let contribution = match parse_annotation(contribution_json) {
        Ok(Some(value)) => value,
        Ok(None) => {
            set_last_error("contribution is required");
            return NemoRelayStatus::InvalidArg;
        }
        Err(status) => return status,
    };
    unsafe { *accepted = (&*recorder).0.record(contribution) };
    NemoRelayStatus::Ok
}

/// Retain a borrowed optimization recorder for deferred streaming evidence.
/// Recording ends with the LLM call. Later records return false.
/// Free the owned result with `nemo_relay_llm_optimization_recorder_free`.
/// # Safety
/// Recorder must be live during this call, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_llm_optimization_recorder_retain(
    recorder: *const FfiLlmOptimizationRecorder,
) -> *mut FfiLlmOptimizationRecorder {
    clear_last_error();
    match unsafe { recorder.as_ref() } {
        Some(recorder) => Box::into_raw(Box::new(FfiLlmOptimizationRecorder(recorder.0.clone()))),
        None => {
            set_last_error("no active optimization recorder");
            std::ptr::null_mut()
        }
    }
}
/// Release a retained recorder. Borrowed callback recorders must never be freed.
/// # Safety
/// The recorder must be an owned, live handle or null. No other call can use it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_llm_optimization_recorder_free(
    recorder: *mut FfiLlmOptimizationRecorder,
) {
    if !recorder.is_null() {
        drop(unsafe { Box::from_raw(recorder) });
    }
}

/// Clone a codec's owned references. The result has an independent lifetime.
/// # Safety
/// Codec must be live or null; release the result with `nemo_relay_codec_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nemo_relay_codec_clone(
    codec: *const FfiCodecHandle,
) -> *mut FfiCodecHandle {
    clear_last_error();
    match unsafe { codec.as_ref() } {
        Some(codec) => Box::into_raw(Box::new(FfiCodecHandle {
            codec: codec.codec.clone(),
            response_codec: codec.response_codec.clone(),
        })),
        None => {
            set_last_error("codec is closed");
            std::ptr::null_mut()
        }
    }
}
