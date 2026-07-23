// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::type_complexity)]
//! JavaScript callable wrappers for NeMo Relay callbacks.
//!
//! This module bridges JavaScript functions (received as NAPI `ThreadsafeFunction` values)
//! into the Rust closure signatures expected by the NeMo Relay core runtime. Each wrapper
//! handles serialization of arguments to/from JSON and manages cross-thread communication
//! between the Rust async runtime and the Node.js event loop.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use napi::threadsafe_function::{ErrorStrategy, ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi::{Env, JsFunction, JsUnknown, NapiRaw, NapiValue};
use nemo_relay::api::runtime::{
    ContextualLlmSanitizeRequestFn, ContextualLlmSanitizeResponseFn, EventSanitizeFn,
    EventSubscriberFn, LlmConditionalFn, LlmExecutionNextFn, LlmJsonStream, LlmRequestInterceptFn,
    LlmSanitizeContext, LlmSanitizeRequestFn, LlmSanitizeResponseFn, LlmStreamExecutionNextFn,
    ToolConditionalFn, ToolExecutionNextFn, ToolInterceptFn, ToolSanitizeFn,
};
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;
use tokio_stream::StreamExt;

use nemo_relay::api::event::{
    CategoryProfile, Event, EventCategory, EventSanitizeFields as CoreEventSanitizeFields,
    PendingMarkSpec,
};
use nemo_relay::api::llm::{LlmRequest, LlmRequestInterceptOutcome};
use nemo_relay::api::tool::ToolExecutionInterceptOutcome;
use nemo_relay::codec::optimization::LlmOptimizationContribution;
use nemo_relay::codec::request::AnnotatedLlmRequest;
use nemo_relay::codec::response::AnnotatedLlmResponse;
use nemo_relay::codec::traits::{LlmCodec, LlmResponseCodec};
use nemo_relay::error::{FlowError, Result};

use crate::callback_factory;
use crate::convert::{callback_json, record_callback_error};
use crate::promise_call::{JsonNextFn, JsonStreamNextFn, PromiseAwareFn};
use crate::types::{EventSanitizeFields, JsEvent, event_sanitize_fields_from_json};

/// JavaScript-facing pending mark DTO.
#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct JsPendingMarkSpec {
    name: String,
    #[serde(default)]
    category: Option<EventCategory>,
    #[serde(default)]
    category_profile: Option<CategoryProfile>,
    #[serde(default)]
    data: Option<Json>,
    #[serde(default)]
    metadata: Option<Json>,
}

impl From<JsPendingMarkSpec> for PendingMarkSpec {
    fn from(mark: JsPendingMarkSpec) -> Self {
        Self {
            name: mark.name,
            category: mark.category,
            category_profile: mark.category_profile,
            data: mark.data,
            metadata: mark.metadata,
        }
    }
}

impl From<PendingMarkSpec> for JsPendingMarkSpec {
    fn from(mark: PendingMarkSpec) -> Self {
        Self {
            name: mark.name,
            category: mark.category,
            category_profile: mark.category_profile,
            data: mark.data,
            metadata: mark.metadata,
        }
    }
}

/// Convert canonical pending marks to JavaScript-facing DTOs.
#[must_use]
pub(crate) fn js_pending_marks(marks: Vec<PendingMarkSpec>) -> Vec<JsPendingMarkSpec> {
    marks.into_iter().map(Into::into).collect()
}

#[derive(Deserialize)]
struct MiddlewareCallbackResult {
    ok: bool,
    #[serde(default)]
    value: Json,
    #[serde(default)]
    error: String,
}

/// Wrap a middleware callback so exceptions cross the N-API boundary as data.
///
/// A raw `ThreadsafeFunction::call_with_return_value` aborts the Node process when
/// a JavaScript callback throws. This wrapper preserves the callback signature and
/// returns a JSON envelope that the Rust middleware adapters can decode safely.
pub(crate) fn safe_middleware_callback(env: &Env, func: &JsFunction) -> napi::Result<JsFunction> {
    let factory: JsFunction = env.run_script(
        r#"((fn) => function __nemo_relay_middleware_wrapper(...args) {
  try {
    const value = fn(...args);
    return { ok: true, value: value === undefined ? null : value };
  } catch (error) {
    let message = 'JavaScript callback threw';
    try {
      message = String(error?.message ?? error);
    } catch {}
    return { ok: false, error: message };
  }
})"#,
    )?;
    let func_unknown = unsafe { JsUnknown::from_raw_unchecked(env.raw(), func.raw()) };
    let wrapper_unknown = factory.call(None, &[func_unknown])?;
    Ok(unsafe { wrapper_unknown.cast::<JsFunction>() })
}

/// Wrap a synchronous execution callback so failures cross the N-API boundary as data.
///
/// The return value is validated before NAPI-RS attempts to convert it to JSON. This
/// prevents unsupported values such as `BigInt` from reaching conversion paths that
/// abort the Node process.
pub(crate) fn safe_execution_callback(env: &Env, func: &JsFunction) -> napi::Result<JsFunction> {
    callback_factory::wrap_execution_callback(env, func)
}

pub(crate) fn unwrap_middleware_result(value: Json, error_prefix: &str) -> Result<Json> {
    let result: MiddlewareCallbackResult = serde_json::from_value(value).map_err(|error| {
        FlowError::Internal(format!(
            "{error_prefix}: invalid middleware callback result: {error}"
        ))
    })?;
    if result.ok {
        Ok(result.value)
    } else {
        Err(FlowError::Internal(format!(
            "{error_prefix}: {}",
            result.error
        )))
    }
}

fn recv_middleware_json_result(
    rx: std::sync::mpsc::Receiver<Json>,
    error_prefix: &str,
) -> Result<Json> {
    let value = rx
        .recv()
        .map_err(|error| FlowError::Internal(format!("{error_prefix}: {error}")))?;
    unwrap_middleware_result(value, error_prefix)
}

fn recv_middleware_json_or_value(
    rx: std::sync::mpsc::Receiver<Json>,
    error_prefix: &str,
    fallback: Json,
) -> Json {
    match recv_middleware_json_result(rx, error_prefix) {
        Ok(value) => value,
        Err(error) => {
            record_callback_error(error.to_string());
            fallback
        }
    }
}

fn recv_middleware_option_string_result(
    rx: std::sync::mpsc::Receiver<Json>,
    error_prefix: &str,
) -> Result<Option<String>> {
    match recv_middleware_json_result(rx, error_prefix)? {
        Json::Null => Ok(None),
        Json::String(value) => Ok(Some(value)),
        other => Err(FlowError::Internal(format!(
            "{error_prefix}: expected string or null, got {other:?}",
        ))),
    }
}

fn recv_json_or_null(rx: std::sync::mpsc::Receiver<Json>, error_prefix: &str) -> Json {
    rx.recv().unwrap_or_else(|e| {
        record_callback_error(format!("{error_prefix}: {e}"));
        Json::Null
    })
}

fn recv_json_result(rx: std::sync::mpsc::Receiver<Json>, error_prefix: &str) -> Result<Json> {
    rx.recv()
        .map_err(|e| FlowError::Internal(format!("{error_prefix}: {e}")))
}

fn recv_option_string_result(
    rx: std::sync::mpsc::Receiver<Json>,
    error_prefix: &str,
) -> Result<Option<String>> {
    match recv_json_result(rx, error_prefix)? {
        Json::Null => Ok(None),
        Json::String(value) => Ok(Some(value)),
        other => Err(FlowError::Internal(format!(
            "{error_prefix}: expected string or null, got {other:?}",
        ))),
    }
}

fn recv_llm_request_result(
    rx: std::sync::mpsc::Receiver<Json>,
    error_prefix: &str,
) -> Result<LlmRequest> {
    let result = recv_json_result(rx, error_prefix)?;
    serde_json::from_value(result).map_err(|e| {
        FlowError::Internal(format!(
            "{error_prefix}: failed to deserialize LlmRequest: {e}"
        ))
    })
}

/// Wrap a JS function `(name: string, args: object) => object` for tool sanitize/intercept.
pub fn wrap_js_tool_fn(
    func: ThreadsafeFunction<(String, Json), ErrorStrategy::Fatal>,
) -> ToolSanitizeFn {
    let func = Arc::new(func);
    Arc::new(move |name: &str, args: Json| {
        let func = func.clone();
        let name = name.to_string();
        let fallback = args.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        let status = func.call_with_return_value(
            (name, args),
            ThreadsafeFunctionCallMode::Blocking,
            move |val: Option<Json>| {
                let _ = tx.send(callback_json(val));
                Ok(())
            },
        );
        if status != napi::Status::Ok {
            record_callback_error(format!(
                "nemo_relay: failed to queue JS tool callback: {status:?}"
            ));
            return fallback;
        }
        // TODO: This closure returns Json (not Result<Json>), so we cannot propagate
        // errors through the type system. Log the error so failures are not silent.
        recv_middleware_json_or_value(rx, "nemo_relay: JS tool callback failed", fallback)
    })
}

/// Wrap a JS function `(name: string, args: object) => string | null` for tool conditional guardrails.
pub fn wrap_js_tool_conditional_fn(
    func: ThreadsafeFunction<(String, Json), ErrorStrategy::Fatal>,
) -> ToolConditionalFn {
    let func = Arc::new(func);
    Arc::new(move |name: &str, args: &Json| {
        let func = func.clone();
        let name = name.to_string();
        let args = args.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        let status = func.call_with_return_value(
            (name, args),
            ThreadsafeFunctionCallMode::Blocking,
            move |val: Option<Json>| {
                let _ = tx.send(callback_json(val));
                Ok(())
            },
        );
        if status != napi::Status::Ok {
            return Err(FlowError::Internal(format!(
                "failed to queue JS tool conditional callback: {status:?}",
            )));
        }
        recv_middleware_option_string_result(rx, "JS tool conditional callback failed")
    })
}

/// Wrap a JS function `(name: string, args: object) => object` for tool request intercepts.
pub fn wrap_js_tool_request_intercept_fn(
    func: ThreadsafeFunction<(String, Json), ErrorStrategy::Fatal>,
) -> ToolInterceptFn {
    let func = Arc::new(func);
    Arc::new(move |name: &str, args: Json| {
        let func = func.clone();
        let name = name.to_string();
        let (tx, rx) = std::sync::mpsc::channel();
        let status = func.call_with_return_value(
            (name, args),
            ThreadsafeFunctionCallMode::Blocking,
            move |val: Option<Json>| {
                let _ = tx.send(callback_json(val));
                Ok(())
            },
        );
        if status != napi::Status::Ok {
            return Err(FlowError::Internal(format!(
                "failed to queue JS tool callback: {status:?}",
            )));
        }
        recv_middleware_json_result(rx, "JS tool callback failed")
    })
}

/// Wrap a JS function `(args: object) => object` for tool execution (synchronous callbacks).
pub fn wrap_js_tool_exec_fn(
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Box<dyn Fn(Json) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>> + Send + Sync> {
    let func = Arc::new(func);
    Box::new(move |args: Json| {
        let func = func.clone();
        Box::pin(async move {
            let (tx, rx) = tokio::sync::oneshot::channel();
            let status = func.call_with_return_value(
                args,
                ThreadsafeFunctionCallMode::Blocking,
                move |val: Option<Json>| {
                    let _ = tx.send(callback_json(val));
                    Ok(())
                },
            );
            if status != napi::Status::Ok {
                return Err(FlowError::Internal(format!(
                    "failed to queue JS tool execution callback: {status:?}",
                )));
            }
            let result = rx.await.map_err(|e| FlowError::Internal(e.to_string()))?;
            unwrap_middleware_result(result, "JS tool execution callback failed")
        })
    })
}

/// Wrap a JS function for unified LLM request intercepts (3-arg signature).
///
/// The JS callback receives a single JSON object
/// `{ name: string, request: LlmRequest, annotated: AnnotatedLlmRequest | null }`
/// and must return `{ request, annotated?, pendingMarks?, optimizationContributions? }`.
/// When `annotated` is non-null, request content is read-only and provider-body
/// edits must be made through the returned annotation; headers remain writable.
pub fn wrap_js_llm_request_intercept_fn(
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> LlmRequestInterceptFn {
    let func = Arc::new(func);
    Arc::new(
        move |name: &str,
              request: LlmRequest,
              annotated: Option<AnnotatedLlmRequest>|
              -> Result<LlmRequestInterceptOutcome> {
            let func = func.clone();
            let req_json = serde_json::to_value(&request).unwrap_or(Json::Null);
            let annotated_json = annotated
                .as_ref()
                .map(|a| serde_json::to_value(a).unwrap_or(Json::Null))
                .unwrap_or(Json::Null);
            let arg = serde_json::json!({
                "name": name,
                "request": req_json,
                "annotated": annotated_json,
            });
            let (tx, rx) = std::sync::mpsc::channel();
            let status = func.call_with_return_value(
                arg,
                ThreadsafeFunctionCallMode::Blocking,
                move |val: Option<Json>| {
                    let _ = tx.send(callback_json(val));
                    Ok(())
                },
            );
            if status != napi::Status::Ok {
                return Err(FlowError::Internal(format!(
                    "failed to queue JS LLM request intercept callback: {status:?}",
                )));
            }
            let result =
                recv_middleware_json_result(rx, "JS LLM request intercept callback failed")?;

            #[derive(Deserialize)]
            #[serde(rename_all = "camelCase")]
            struct JsOutcome {
                request: LlmRequest,
                #[serde(default)]
                annotated: Option<AnnotatedLlmRequest>,
                #[serde(default)]
                pending_marks: Vec<JsPendingMarkSpec>,
                #[serde(default)]
                optimization_contributions: Vec<LlmOptimizationContribution>,
            }
            let outcome: JsOutcome = serde_json::from_value(result).map_err(|e| {
                FlowError::Internal(format!("invalid JS LLM request intercept outcome: {e}"))
            })?;
            Ok(LlmRequestInterceptOutcome {
                request: outcome.request,
                annotated_request: outcome.annotated,
                pending_marks: outcome.pending_marks.into_iter().map(Into::into).collect(),
                optimization_contributions: outcome.optimization_contributions,
            })
        },
    )
}

/// Wrap a JS function for LLM sanitize request: `(request: LlmRequest) => LlmRequest`.
/// Since ThreadsafeFunction requires serde-serializable args, we serialize the request as JSON.
pub fn wrap_js_llm_sanitize_request_fn(
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> LlmSanitizeRequestFn {
    let func = Arc::new(func);
    Arc::new(move |request: LlmRequest| {
        let func = func.clone();
        let req_json = serde_json::to_value(&request).unwrap_or(Json::Null);
        let (tx, rx) = std::sync::mpsc::channel();
        let status = func.call_with_return_value(
            req_json,
            ThreadsafeFunctionCallMode::Blocking,
            move |val: Option<Json>| {
                let _ = tx.send(callback_json(val));
                Ok(())
            },
        );
        if status != napi::Status::Ok {
            record_callback_error(format!(
                "nemo_relay: failed to queue JS LLM sanitize request callback: {status:?}"
            ));
            return request;
        }
        // TODO: This closure returns LlmRequest (not Result), so we cannot propagate
        // errors through the type system. Log the error so failures are not silent.
        let result = recv_middleware_json_or_value(
            rx,
            "nemo_relay: JS LLM sanitize request callback failed",
            serde_json::to_value(&request).unwrap_or(Json::Null),
        );
        serde_json::from_value(result).unwrap_or_else(|error| {
            record_callback_error(format!(
                "nemo_relay: JS LLM sanitize request callback failed: failed to deserialize LlmRequest: {error}"
            ));
            request
        })
    })
}

/// Wrap a JS function for LLM sanitize response: `(response: Json) => Json`.
pub fn wrap_js_llm_response_fn(
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> LlmSanitizeResponseFn {
    let func = Arc::new(func);
    Arc::new(move |response: Json| {
        let func = func.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        let status = func.call_with_return_value(
            response.clone(),
            ThreadsafeFunctionCallMode::Blocking,
            move |val: Option<Json>| {
                let _ = tx.send(callback_json(val));
                Ok(())
            },
        );
        if status != napi::Status::Ok {
            record_callback_error(format!(
                "nemo_relay: failed to queue JS LLM response callback: {status:?}"
            ));
            return response;
        }
        // TODO: This closure returns Json (not Result<Json>), so we cannot propagate
        // errors through the type system. Log the error and fall back to original response.
        recv_middleware_json_or_value(rx, "nemo_relay: JS LLM response callback failed", response)
    })
}

/// Wrap a JS function for contextual LLM request sanitization.
///
/// The callback receives `(request, context)`; returning `null` omits the event payload.
pub fn wrap_js_contextual_llm_sanitize_request_fn(
    func: ThreadsafeFunction<(Json, Json), ErrorStrategy::Fatal>,
) -> ContextualLlmSanitizeRequestFn {
    let func = Arc::new(func);
    Arc::new(move |request: LlmRequest, context: LlmSanitizeContext| {
        let context = serde_json::json!({
            "has_active_codec": context.has_active_codec,
            "codec_name": context.codec_name,
        });
        let request = serde_json::to_value(request).unwrap_or(Json::Null);
        let (tx, rx) = std::sync::mpsc::channel();
        if func.call_with_return_value(
            (request, context),
            ThreadsafeFunctionCallMode::Blocking,
            move |value: Option<Json>| {
                let _ = tx.send(callback_json(value));
                Ok(())
            },
        ) != napi::Status::Ok
        {
            return None;
        }
        recv_middleware_json_result(rx, "nemo_relay: JS contextual LLM request callback failed")
            .ok()
            .and_then(|value| (!value.is_null()).then_some(value))
            .and_then(|value| serde_json::from_value(value).ok())
    })
}

/// Wrap a JS function for contextual LLM response sanitization.
///
/// The callback receives `(response, context)`; returning `null` omits the event payload.
pub fn wrap_js_contextual_llm_sanitize_response_fn(
    func: ThreadsafeFunction<(Json, Json), ErrorStrategy::Fatal>,
) -> ContextualLlmSanitizeResponseFn {
    let func = Arc::new(func);
    Arc::new(move |response: Json, context: LlmSanitizeContext| {
        let context = serde_json::json!({
            "has_active_codec": context.has_active_codec,
            "codec_name": context.codec_name,
        });
        let (tx, rx) = std::sync::mpsc::channel();
        if func.call_with_return_value(
            (response, context),
            ThreadsafeFunctionCallMode::Blocking,
            move |value: Option<Json>| {
                let _ = tx.send(callback_json(value));
                Ok(())
            },
        ) != napi::Status::Ok
        {
            return None;
        }
        recv_middleware_json_result(rx, "nemo_relay: JS contextual LLM response callback failed")
            .ok()
            .and_then(|value| (!value.is_null()).then_some(value))
    })
}

/// Wrap a JS function for LLM conditional guardrails: `(request: object) => string | null`.
pub fn wrap_js_llm_conditional_fn(
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> LlmConditionalFn {
    let func = Arc::new(func);
    Arc::new(move |request: &LlmRequest| {
        let func = func.clone();
        let req_json = serde_json::to_value(request).unwrap_or(Json::Null);
        let (tx, rx) = std::sync::mpsc::channel();
        let status = func.call_with_return_value(
            req_json,
            ThreadsafeFunctionCallMode::Blocking,
            move |val: Option<Json>| {
                let _ = tx.send(callback_json(val));
                Ok(())
            },
        );
        if status != napi::Status::Ok {
            return Err(FlowError::Internal(format!(
                "failed to queue JS LLM conditional callback: {status:?}",
            )));
        }
        recv_middleware_option_string_result(rx, "JS LLM conditional callback failed")
    })
}

/// Wrap a JS function for LLM execution: `(request: object) => object`.
///
/// The JS callback receives the `LlmRequest` serialized as a plain JSON object
/// and returns the response as JSON.
pub fn wrap_js_llm_exec_fn(
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Box<dyn Fn(LlmRequest) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>> + Send + Sync> {
    let func = Arc::new(func);
    Box::new(move |request: LlmRequest| {
        let func = func.clone();
        let req_json = serde_json::to_value(&request).unwrap_or(Json::Null);
        Box::pin(async move {
            let (tx, rx) = tokio::sync::oneshot::channel();
            let status = func.call_with_return_value(
                req_json,
                ThreadsafeFunctionCallMode::Blocking,
                move |val: Option<Json>| {
                    let _ = tx.send(callback_json(val));
                    Ok(())
                },
            );
            if status != napi::Status::Ok {
                return Err(FlowError::Internal(format!(
                    "failed to queue JS LLM execution callback: {status:?}",
                )));
            }
            let result = rx.await.map_err(|e| FlowError::Internal(e.to_string()))?;
            unwrap_middleware_result(result, "JS LLM execution callback failed")
        })
    })
}

/// Wrap a JS function `(chunk: object) => void` as a collector callback.
///
/// The collector is called with each intercepted chunk during a streaming LLM response.
/// It is used to accumulate chunks on the JavaScript side for aggregation.
/// If the JS function throws, the error is currently swallowed and treated as
/// `Ok(())` because `ErrorStrategy::Fatal` aborts the process on JS exceptions.
/// For practical purposes, a non-throwing collector always returns `Ok(())`.
pub fn wrap_js_collector_fn(
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Box<dyn FnMut(Json) -> Result<()> + Send> {
    Box::new(move |chunk: Json| {
        let status = func.call(chunk, ThreadsafeFunctionCallMode::Blocking);
        if status == napi::Status::Ok {
            Ok(())
        } else {
            let message = format!("nemo_relay: failed to queue JS collector callback: {status:?}");
            record_callback_error(message.clone());
            Err(FlowError::Internal(message))
        }
    })
}

/// Wrap a JS function `() => object` as a finalizer callback.
///
/// The finalizer is called exactly once when the stream is exhausted.
/// It takes no arguments and must return a JSON value representing the
/// aggregated response.
pub fn wrap_js_finalizer_fn(
    func: ThreadsafeFunction<(), ErrorStrategy::Fatal>,
) -> Box<dyn FnOnce() -> Json + Send> {
    Box::new(move || {
        let (tx, rx) = std::sync::mpsc::channel();
        let status = func.call_with_return_value(
            (),
            ThreadsafeFunctionCallMode::Blocking,
            move |val: Option<Json>| {
                let _ = tx.send(callback_json(val));
                Ok(())
            },
        );
        if status != napi::Status::Ok {
            record_callback_error(format!(
                "nemo_relay: failed to queue JS finalizer callback: {status:?}"
            ));
            return Json::Null;
        }
        // TODO: This closure returns Json (not Result<Json>), so we cannot propagate
        // errors through the type system. Log the error so failures are not silent.
        recv_json_or_null(rx, "nemo_relay: JS finalizer callback failed")
    })
}

/// Wrap a JS function for event subscriber: `(event: JsEvent) => void`.
pub fn wrap_js_event_subscriber(
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> EventSubscriberFn {
    let func = Arc::new(func);
    Arc::new(move |event: &Event| {
        let event_json = match JsEvent::try_from_event(event) {
            Ok(event) => event.into_json(),
            Err(error) => {
                record_callback_error(format!(
                    "nemo_relay: failed to serialize JS event subscriber payload: {error}"
                ));
                return;
            }
        };
        let status = func.call(event_json, ThreadsafeFunctionCallMode::NonBlocking);
        if status != napi::Status::Ok {
            record_callback_error(format!(
                "nemo_relay: failed to queue JS event subscriber callback: {status:?}"
            ));
        }
    })
}

/// Wrap a JS event sanitizer: ``(event, fields) => fields``.
pub fn wrap_js_event_sanitize_fn(
    func: ThreadsafeFunction<(Json, Json), ErrorStrategy::Fatal>,
) -> EventSanitizeFn {
    let func = Arc::new(func);
    Arc::new(move |event: &Event, fields: CoreEventSanitizeFields| {
        let event_json = match JsEvent::try_from_event(event) {
            Ok(event) => event.into_json(),
            Err(error) => {
                record_callback_error(format!(
                    "nemo_relay: failed to serialize JS event sanitizer context: {error}"
                ));
                return CoreEventSanitizeFields::default();
            }
        };
        let js_fields = EventSanitizeFields {
            data: fields.data.clone(),
            category_profile: fields
                .category_profile
                .as_ref()
                .and_then(|value| serde_json::to_value(value).ok()),
            metadata: fields.metadata.clone(),
        };
        let (tx, rx) = std::sync::mpsc::channel();
        let status = func.call_with_return_value(
            (
                event_json,
                serde_json::to_value(js_fields).unwrap_or(Json::Null),
            ),
            ThreadsafeFunctionCallMode::Blocking,
            move |value: Option<Json>| {
                let _ = tx.send(callback_json(value));
                Ok(())
            },
        );
        if status != napi::Status::Ok {
            record_callback_error(format!(
                "nemo_relay: failed to queue JS event sanitizer callback: {status:?}"
            ));
            return CoreEventSanitizeFields::default();
        }
        let sanitized = (|| -> Result<_> {
            let result =
                recv_middleware_json_result(rx, "nemo_relay: JS event sanitizer callback failed")?;
            let result = event_sanitize_fields_from_json(result).map_err(|error| {
                FlowError::Internal(format!(
                    "nemo_relay: invalid JS event sanitizer result: {error}"
                ))
            })?;
            let category_profile = result
                .category_profile
                .map(serde_json::from_value)
                .transpose()
                .map_err(|error| {
                    FlowError::Internal(format!(
                        "nemo_relay: invalid JS event sanitizer result: {error}"
                    ))
                })?;
            Ok(CoreEventSanitizeFields {
                data: result.data,
                category_profile,
                metadata: result.metadata,
            })
        })();
        match sanitized {
            Ok(sanitized) => sanitized,
            Err(error) => {
                record_callback_error(error.to_string());
                CoreEventSanitizeFields::default()
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Codec wrappers
// ---------------------------------------------------------------------------

/// A NAPI-RS wrapper that implements the core [`LlmCodec`] trait by delegating
/// `decode` and `encode` to JavaScript functions via `ThreadsafeFunction`.
struct NapiCodec {
    decode: Arc<ThreadsafeFunction<Json, ErrorStrategy::Fatal>>,
    encode: Arc<ThreadsafeFunction<Json, ErrorStrategy::Fatal>>,
}

impl LlmCodec for NapiCodec {
    fn decode(&self, request: &LlmRequest) -> Result<AnnotatedLlmRequest> {
        let req_json = serde_json::to_value(request).unwrap_or(Json::Null);
        let (tx, rx) = std::sync::mpsc::channel();
        let status = self.decode.call_with_return_value(
            req_json,
            ThreadsafeFunctionCallMode::Blocking,
            move |val: Option<Json>| {
                let _ = tx.send(callback_json(val));
                Ok(())
            },
        );
        if status != napi::Status::Ok {
            return Err(FlowError::Internal(format!(
                "failed to queue JS codec decode callback: {status:?}",
            )));
        }
        let result = recv_json_result(rx, "JS codec decode callback failed")?;
        serde_json::from_value(result).map_err(|e| {
            FlowError::Internal(format!(
                "JS codec decode callback: failed to deserialize AnnotatedLlmRequest: {e}"
            ))
        })
    }

    fn encode(&self, annotated: &AnnotatedLlmRequest, original: &LlmRequest) -> Result<LlmRequest> {
        let annotated_json = serde_json::to_value(annotated).unwrap_or(Json::Null);
        let original_json = serde_json::to_value(original).unwrap_or(Json::Null);
        let arg = serde_json::json!({"annotated": annotated_json, "original": original_json});
        let (tx, rx) = std::sync::mpsc::channel();
        let status = self.encode.call_with_return_value(
            arg,
            ThreadsafeFunctionCallMode::Blocking,
            move |val: Option<Json>| {
                let _ = tx.send(callback_json(val));
                Ok(())
            },
        );
        if status != napi::Status::Ok {
            return Err(FlowError::Internal(format!(
                "failed to queue JS codec encode callback: {status:?}",
            )));
        }
        recv_llm_request_result(rx, "JS codec encode callback failed")
    }
}

/// Wrap two JS functions (decode, encode) into an `Arc<dyn LlmCodec>` suitable
/// for registration with the core codec registry.
pub fn wrap_js_codec(
    decode: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
    encode: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Arc<dyn LlmCodec> {
    Arc::new(NapiCodec {
        decode: Arc::new(decode),
        encode: Arc::new(encode),
    })
}

// ---------------------------------------------------------------------------
// Response codec wrapper
// ---------------------------------------------------------------------------

/// A NAPI-RS wrapper that implements the core [`LlmResponseCodec`] trait by
/// delegating `decode_response` to a JavaScript function via `ThreadsafeFunction`.
struct NapiResponseCodec {
    decode_response: Arc<ThreadsafeFunction<Json, ErrorStrategy::Fatal>>,
}

impl LlmResponseCodec for NapiResponseCodec {
    fn decode_response(&self, response: &Json) -> Result<AnnotatedLlmResponse> {
        let (tx, rx) = std::sync::mpsc::channel();
        let status = self.decode_response.call_with_return_value(
            response.clone(),
            ThreadsafeFunctionCallMode::Blocking,
            move |v: Option<Json>| {
                tx.send(callback_json(v)).ok();
                Ok(())
            },
        );
        if status != napi::Status::Ok {
            return Err(FlowError::Internal(format!(
                "decode_response call failed: {status:?}"
            )));
        }
        let result = rx
            .recv()
            .map_err(|_| FlowError::Internal("decode_response callback did not return".into()))?;
        serde_json::from_value(result).map_err(|e| {
            FlowError::Internal(format!(
                "decode_response returned invalid AnnotatedLlmResponse: {e}"
            ))
        })
    }
}

/// Wrap a JS decode_response function into an `Arc<dyn LlmResponseCodec>`.
pub fn wrap_js_response_codec(
    decode_response: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Arc<dyn LlmResponseCodec> {
    Arc::new(NapiResponseCodec {
        decode_response: Arc::new(decode_response),
    })
}

/// Wrap a JS function `(args, next) => { result, pendingMarks? }` for tool execution intercept.
///
/// The JS callback receives the tool arguments and a real `next(args)` function
/// that returns a Promise for the downstream result.
pub fn wrap_js_tool_exec_intercept_fn(
    func: Arc<PromiseAwareFn>,
) -> nemo_relay::api::runtime::ToolExecutionFn {
    Arc::new(move |_name: &str, args: Json, next: ToolExecutionNextFn| {
        let func = func.clone();
        let next_json: JsonNextFn = Arc::new(move |next_args| next(next_args));
        Box::pin(async move {
            let result = func.call_with_json_next(args, next_json).await?;
            #[derive(Deserialize)]
            #[serde(rename_all = "camelCase")]
            struct JsOutcome {
                result: Json,
                #[serde(default)]
                pending_marks: Vec<JsPendingMarkSpec>,
            }
            let outcome: JsOutcome = serde_json::from_value(result).map_err(|error| {
                FlowError::Internal(format!(
                    "invalid JS tool execution intercept outcome: {error}"
                ))
            })?;
            Ok(ToolExecutionInterceptOutcome {
                result: outcome.result,
                pending_marks: outcome.pending_marks.into_iter().map(Into::into).collect(),
            })
        })
    })
}

/// Wrap a JS function `(request, next) => result` for LLM execution intercept.
///
/// The JS callback receives the `LlmRequest` serialized as a plain JSON object
/// and a real `next(request)` function that returns a Promise for the downstream
/// result.
pub fn wrap_js_llm_exec_intercept_fn(
    func: Arc<PromiseAwareFn>,
) -> Arc<
    dyn Fn(
            &str,
            LlmRequest,
            LlmExecutionNextFn,
        ) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>>
        + Send
        + Sync,
> {
    Arc::new(
        move |_name: &str, request: LlmRequest, next: LlmExecutionNextFn| {
            let func = func.clone();
            let req_json = serde_json::to_value(&request).unwrap_or(Json::Null);
            let next_json: JsonNextFn = Arc::new(move |next_request_json| {
                let next = next.clone();
                Box::pin(async move {
                    let next_request: LlmRequest = serde_json::from_value(next_request_json)
                        .map_err(|e| {
                            FlowError::Internal(format!("invalid LlmRequest from JS next: {e}"))
                        })?;
                    next(next_request).await
                })
            });
            Box::pin(async move { func.call_with_json_next(req_json, next_json).await })
        },
    )
}

/// Wrap a JS function `(request, next) => result` for LLM stream execution intercept.
///
/// The JS callback receives the `LlmRequest` serialized as a plain JSON object
/// and a real `next(request)` function whose Promise resolves to an array of
/// downstream JSON chunks. Returning an array preserves streaming semantics;
/// returning any other JSON value produces a single-chunk stream.
pub fn wrap_js_llm_stream_exec_intercept_fn(
    func: Arc<PromiseAwareFn>,
) -> Arc<
    dyn Fn(
            &str,
            LlmRequest,
            LlmStreamExecutionNextFn,
        ) -> Pin<Box<dyn Future<Output = Result<LlmJsonStream>> + Send>>
        + Send
        + Sync,
> {
    Arc::new(
        move |_name: &str, request: LlmRequest, next: LlmStreamExecutionNextFn| {
            let func = func.clone();
            let req_json = serde_json::to_value(&request).unwrap_or(Json::Null);
            let next_stream: JsonStreamNextFn = Arc::new(move |next_request_json| {
                let next = next.clone();
                Box::pin(async move {
                    let next_request: LlmRequest = serde_json::from_value(next_request_json)
                        .map_err(|e| {
                            FlowError::Internal(format!("invalid LlmRequest from JS next: {e}"))
                        })?;
                    let mut stream = next(next_request).await?;
                    let mut chunks = Vec::new();
                    while let Some(item) = stream.next().await {
                        chunks.push(item?);
                    }
                    Ok(chunks)
                })
            });
            Box::pin(async move {
                let result = func.call_with_stream_next(req_json, next_stream).await?;
                let chunks = match result {
                    Json::Array(values) => values.into_iter().map(Ok).collect::<Vec<_>>(),
                    value => vec![Ok(value)],
                };
                let stream = tokio_stream::iter(chunks);
                Ok(LlmJsonStream::new(stream))
            })
        },
    )
}
