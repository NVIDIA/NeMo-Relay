// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::type_complexity)]
//! Wrappers that adapt JavaScript callback functions into Rust closures.
//!
//! Each wrapper takes a `js_sys::Function`, wraps it with `SendWrapper` (since
//! JS functions are not `Send`), and returns the closure shape expected by the
//! core runtime. Registry-stored callbacks return `Arc`-backed closures, while
//! one-shot or mutable callback shapes remain boxed.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use js_sys::Function;
#[cfg(target_arch = "wasm32")]
use send_wrapper::SendWrapper;
#[cfg(target_arch = "wasm32")]
use serde::Serialize;
use serde_json::Value as Json;
#[cfg(target_arch = "wasm32")]
use tokio_stream::StreamExt;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsValue;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::JsFuture;

use nemo_relay::api::event::Event;
#[cfg(target_arch = "wasm32")]
use nemo_relay::api::event::PendingMarkSpec;
use nemo_relay::api::llm::{LlmRequest, LlmRequestInterceptOutcome};
use nemo_relay::api::runtime::{
    EventSubscriberFn, LlmConditionalFn, LlmExecutionNextFn, LlmRequestInterceptFn,
    LlmSanitizeRequestFn, LlmSanitizeResponseFn, LlmStreamExecutionNextFn, ToolConditionalFn,
    ToolExecutionNextFn, ToolInterceptFn, ToolSanitizeFn,
};
use nemo_relay::codec::request::AnnotatedLlmRequest;
#[cfg(target_arch = "wasm32")]
use nemo_relay::codec::response::AnnotatedLlmResponse;
use nemo_relay::codec::traits::{LlmCodec, LlmResponseCodec};
use nemo_relay::error::{FlowError, Result};

#[cfg(target_arch = "wasm32")]
use crate::convert::record_callback_error;
#[cfg(target_arch = "wasm32")]
use crate::convert::{js_callback_to_json, js_to_json, json_to_js};
#[cfg(target_arch = "wasm32")]
use crate::types::WasmEvent;

/// Extract a human-readable error message from a `JsValue`.
///
/// Tries `.as_string()` first (for string errors), then falls back to debug format.
#[cfg(target_arch = "wasm32")]
fn js_error_message(e: &JsValue) -> String {
    e.as_string().unwrap_or_else(|| format!("{e:?}"))
}

#[cfg(target_arch = "wasm32")]
fn relay_error_from_js(e: &JsValue) -> FlowError {
    FlowError::Internal(js_error_message(e))
}

#[cfg(target_arch = "wasm32")]
fn relay_json_from_js(val: &JsValue) -> Result<Json> {
    js_callback_to_json(val).map_err(|e| relay_error_from_js(&e))
}

#[cfg(target_arch = "wasm32")]
fn log_callback_issue(context: &str, e: &JsValue) {
    let message = format!("{context}: {}", js_error_message(e));
    record_callback_error(message.clone());
    eprintln!("{message}");
}

#[cfg(target_arch = "wasm32")]
fn callback_json_or_fallback(
    result: std::result::Result<JsValue, JsValue>,
    conversion_context: &str,
    throw_context: &str,
    fallback: Json,
) -> Json {
    match result {
        Ok(value) => js_callback_to_json(&value).unwrap_or_else(|e| {
            log_callback_issue(conversion_context, &e);
            fallback.clone()
        }),
        Err(e) => {
            log_callback_issue(throw_context, &e);
            fallback
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn js_value_to_llm_request(next_val: &JsValue) -> std::result::Result<LlmRequest, JsValue> {
    let next_json = js_to_json(next_val).map_err(|e| JsValue::from_str(&js_error_message(&e)))?;
    serde_json::from_value(next_json)
        .map_err(|e| JsValue::from_str(&format!("invalid LlmRequest from JS next: {e}")))
}

#[cfg(target_arch = "wasm32")]
async fn collect_stream_chunks(
    next: LlmStreamExecutionNextFn,
    next_request: LlmRequest,
) -> std::result::Result<Json, JsValue> {
    let mut stream = next(next_request)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let mut chunks = Vec::new();
    while let Some(item) = stream.next().await {
        match item.map_err(|e| JsValue::from_str(&e.to_string()))? {
            Json::Array(values) => chunks.extend(values),
            value => chunks.push(value),
        }
    }
    Ok(Json::Array(chunks))
}

#[cfg(target_arch = "wasm32")]
async fn resolve_js_value(result: std::result::Result<JsValue, JsValue>) -> Result<Json> {
    match result {
        Ok(val) => {
            if let Some(promise) = val.dyn_ref::<js_sys::Promise>() {
                let resolved = JsFuture::from(promise.clone())
                    .await
                    .map_err(|e| relay_error_from_js(&e))?;
                relay_json_from_js(&resolved)
            } else {
                relay_json_from_js(&val)
            }
        }
        Err(e) => Err(relay_error_from_js(&e)),
    }
}

#[cfg(target_arch = "wasm32")]
fn json_to_result_stream(
    value: Json,
) -> Pin<Box<dyn tokio_stream::Stream<Item = Result<Json>> + Send>> {
    let chunks = match value {
        Json::Array(values) => values.into_iter().map(Ok).collect::<Vec<_>>(),
        value => vec![Ok(value)],
    };
    Box::pin(tokio_stream::iter(chunks))
}

#[cfg(not(target_arch = "wasm32"))]
fn wasm_only_error() -> FlowError {
    FlowError::Internal(
        "WebAssembly callback wrappers are only supported on wasm32 targets".to_string(),
    )
}

/// Wrap a JS function `(name, args) => result` for tool sanitize/intercept.
#[cfg(not(target_arch = "wasm32"))]
pub fn wrap_js_tool_fn(_func: Function) -> ToolSanitizeFn {
    Arc::new(move |_name: &str, _args: Json| Json::Null)
}

#[cfg(target_arch = "wasm32")]
pub fn wrap_js_tool_fn(func: Function) -> ToolSanitizeFn {
    let func = SendWrapper::new(func);
    Arc::new(move |name: &str, args: Json| {
        let js_name = JsValue::from_str(name);
        let js_args = json_to_js(&args);
        // TODO: This closure returns Json (not Result<Json>), so we cannot propagate
        // errors through the type system. Log errors so failures are not silent.
        callback_json_or_fallback(
            func.call2(&JsValue::NULL, &js_name, &js_args),
            "nemo_relay: JS tool callback result conversion failed",
            "nemo_relay: JS tool callback threw",
            Json::Null,
        )
    })
}

/// Wrap a JS function `(name, args) => string | null` for tool conditional guardrails.
#[cfg(not(target_arch = "wasm32"))]
pub fn wrap_js_tool_conditional_fn(_func: Function) -> ToolConditionalFn {
    Arc::new(move |_name: &str, _args: &Json| Ok(None))
}

#[cfg(target_arch = "wasm32")]
pub fn wrap_js_tool_conditional_fn(func: Function) -> ToolConditionalFn {
    let func = SendWrapper::new(func);
    Arc::new(move |name: &str, args: &Json| {
        let js_name = JsValue::from_str(name);
        let js_args = json_to_js(args);
        let result = func
            .call2(&JsValue::NULL, &js_name, &js_args)
            .map_err(|e| FlowError::Internal(js_error_message(&e)))?;

        if result.is_null() || result.is_undefined() {
            Ok(None)
        } else {
            result.as_string().map(Some).ok_or_else(|| {
                FlowError::Internal(
                    "JS tool conditional callback returned unexpected type (expected string or null)"
                        .to_string(),
                )
            })
        }
    })
}

/// Wrap a JS function `(name, args) => result` for fallible tool request intercepts.
#[cfg(not(target_arch = "wasm32"))]
pub fn wrap_js_tool_request_intercept_fn(_func: Function) -> ToolInterceptFn {
    Arc::new(move |_name: &str, args: Json| Ok(args))
}

#[cfg(target_arch = "wasm32")]
pub fn wrap_js_tool_request_intercept_fn(func: Function) -> ToolInterceptFn {
    let func = SendWrapper::new(func);
    Arc::new(move |name: &str, args: Json| {
        let js_name = JsValue::from_str(name);
        let js_args = json_to_js(&args);
        let result = func
            .call2(&JsValue::NULL, &js_name, &js_args)
            .map_err(|e| relay_error_from_js(&e))?;
        relay_json_from_js(&result)
    })
}

/// Wrap a JS function `(args) => result | Promise<result>` for tool execution.
#[cfg(not(target_arch = "wasm32"))]
pub fn wrap_js_tool_exec_fn(
    _func: Function,
) -> Box<dyn Fn(Json) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>> + Send + Sync> {
    Box::new(move |_args: Json| Box::pin(async move { Err(wasm_only_error()) }))
}

#[cfg(target_arch = "wasm32")]
pub fn wrap_js_tool_exec_fn(
    func: Function,
) -> Box<dyn Fn(Json) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>> + Send + Sync> {
    let func = SendWrapper::new(func);
    Box::new(move |args: Json| {
        let js_args = json_to_js(&args);
        let result = func.call1(&JsValue::NULL, &js_args);
        Box::pin(SendWrapper::new(async move {
            match result {
                Ok(val) => {
                    // Check if it's a Promise
                    if let Some(promise) = val.dyn_ref::<js_sys::Promise>() {
                        match JsFuture::from(promise.clone()).await {
                            Ok(resolved) => relay_json_from_js(&resolved),
                            Err(e) => Err(relay_error_from_js(&e)),
                        }
                    } else {
                        relay_json_from_js(&val)
                    }
                }
                Err(e) => Err(relay_error_from_js(&e)),
            }
        }))
    })
}

/// Wrap a JS function for unified LLM request intercepts.
///
/// Supports both `(name, request, annotated) => { request, annotated }` and
/// `({ name, request, annotated }) => { request, annotated }`.
#[cfg(not(target_arch = "wasm32"))]
pub fn wrap_js_llm_request_intercept_fn(_func: Function) -> LlmRequestInterceptFn {
    Arc::new(
        move |_name: &str, request: LlmRequest, annotated: Option<AnnotatedLlmRequest>| {
            Ok(LlmRequestInterceptOutcome::new(request, annotated))
        },
    )
}

#[cfg(target_arch = "wasm32")]
pub fn wrap_js_llm_request_intercept_fn(func: Function) -> LlmRequestInterceptFn {
    let func = SendWrapper::new(func);
    Arc::new(
        move |name: &str,
              request: LlmRequest,
              annotated: Option<AnnotatedLlmRequest>|
              -> Result<LlmRequestInterceptOutcome> {
            let req_json = serde_json::to_value(&request).unwrap_or(Json::Null);
            let js_name = JsValue::from_str(name);
            let js_req = json_to_js(&req_json);
            let js_annotated = match &annotated {
                Some(a) => {
                    let a_json = serde_json::to_value(a).unwrap_or(Json::Null);
                    json_to_js(&a_json)
                }
                None => JsValue::NULL,
            };
            let result = if func.length() <= 1 {
                let payload = js_sys::Object::new();
                js_sys::Reflect::set(&payload, &JsValue::from_str("name"), &js_name)
                    .map_err(|e| FlowError::Internal(js_error_message(&e)))?;
                js_sys::Reflect::set(&payload, &JsValue::from_str("request"), &js_req)
                    .map_err(|e| FlowError::Internal(js_error_message(&e)))?;
                js_sys::Reflect::set(&payload, &JsValue::from_str("annotated"), &js_annotated)
                    .map_err(|e| FlowError::Internal(js_error_message(&e)))?;
                func.call1(&JsValue::NULL, &payload.into())
                    .map_err(|e| FlowError::Internal(js_error_message(&e)))?
            } else {
                func.call3(&JsValue::NULL, &js_name, &js_req, &js_annotated)
                    .map_err(|e| FlowError::Internal(js_error_message(&e)))?
            };

            // Extract "request" property from result
            let js_new_req =
                js_sys::Reflect::get(&result, &JsValue::from_str("request")).map_err(|e| {
                    FlowError::Internal(format!(
                        "failed to get 'request' from intercept result: {}",
                        js_error_message(&e)
                    ))
                })?;
            let new_req_json =
                js_callback_to_json(&js_new_req).map_err(|e| relay_error_from_js(&e))?;
            let new_request: LlmRequest = serde_json::from_value(new_req_json).map_err(|e| {
                FlowError::Internal(format!("failed to deserialize LlmRequest: {e}"))
            })?;

            // Extract "annotated" property from result
            let js_new_annotated = js_sys::Reflect::get(&result, &JsValue::from_str("annotated"))
                .map_err(|e| {
                FlowError::Internal(format!(
                    "failed to get 'annotated' from intercept result: {}",
                    js_error_message(&e)
                ))
            })?;
            let new_annotated = if js_new_annotated.is_null() || js_new_annotated.is_undefined() {
                None
            } else {
                let ann_json = js_to_json(&js_new_annotated)
                    .map_err(|e| FlowError::Internal(js_error_message(&e)))?;
                Some(
                    serde_json::from_value::<AnnotatedLlmRequest>(ann_json).map_err(|e| {
                        FlowError::Internal(format!(
                            "failed to deserialize AnnotatedLlmRequest: {e}"
                        ))
                    })?,
                )
            };

            let js_pending_marks =
                js_sys::Reflect::get(&result, &JsValue::from_str("pendingMarks"))
                    .map_err(|e| FlowError::Internal(js_error_message(&e)))?;
            let pending_marks = if js_pending_marks.is_null() || js_pending_marks.is_undefined() {
                Vec::new()
            } else {
                let marks_json = js_to_json(&js_pending_marks)
                    .map_err(|e| FlowError::Internal(js_error_message(&e)))?;
                serde_json::from_value::<Vec<PendingMarkSpec>>(marks_json).map_err(|e| {
                    FlowError::Internal(format!("failed to deserialize pendingMarks: {e}"))
                })?
            };

            Ok(LlmRequestInterceptOutcome {
                request: new_request,
                annotated_request: new_annotated,
                pending_marks,
            })
        },
    )
}

/// Wrap a JS function for LLM sanitize request: `(request) => request`.
#[cfg(not(target_arch = "wasm32"))]
pub fn wrap_js_llm_sanitize_request_fn(_func: Function) -> LlmSanitizeRequestFn {
    Arc::new(move |request: LlmRequest| request)
}

#[cfg(target_arch = "wasm32")]
pub fn wrap_js_llm_sanitize_request_fn(func: Function) -> LlmSanitizeRequestFn {
    let func = SendWrapper::new(func);
    Arc::new(move |request: LlmRequest| {
        let req_json = serde_json::to_value(&request).unwrap_or(Json::Null);
        let js_req = json_to_js(&req_json);
        // TODO: This closure returns LlmRequest (not Result), so we cannot propagate
        // errors through the type system. Log errors so failures are not silent.
        let result_json = callback_json_or_fallback(
            func.call1(&JsValue::NULL, &js_req),
            "nemo_relay: JS LLM sanitize request result conversion failed",
            "nemo_relay: JS LLM sanitize request callback threw",
            Json::Null,
        );
        serde_json::from_value(result_json).unwrap_or(request)
    })
}

/// Wrap a JS function for LLM conditional guardrails: `(request) => string | null`.
#[cfg(not(target_arch = "wasm32"))]
pub fn wrap_js_llm_conditional_fn(_func: Function) -> LlmConditionalFn {
    Arc::new(move |_request: &LlmRequest| Ok(None))
}

#[cfg(target_arch = "wasm32")]
pub fn wrap_js_llm_conditional_fn(func: Function) -> LlmConditionalFn {
    let func = SendWrapper::new(func);
    Arc::new(move |request: &LlmRequest| {
        let req_json = serde_json::to_value(request).unwrap_or(Json::Null);
        let js_req = json_to_js(&req_json);
        let result = func
            .call1(&JsValue::NULL, &js_req)
            .map_err(|e| relay_error_from_js(&e))?;

        if result.is_null() || result.is_undefined() {
            Ok(None)
        } else {
            result.as_string().map(Some).ok_or_else(|| {
                FlowError::Internal(
                    "JS LLM conditional callback returned unexpected type (expected string or null)"
                        .to_string(),
                )
            })
        }
    })
}

/// Wrap a JS function for LLM execution: `(request) => result | Promise<result>`.
///
/// The `LlmRequest` is serialized to JSON before passing to JS.
#[cfg(not(target_arch = "wasm32"))]
pub fn wrap_js_llm_exec_fn(
    _func: Function,
) -> Box<dyn Fn(LlmRequest) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>> + Send + Sync> {
    Box::new(move |_request: LlmRequest| Box::pin(async move { Err(wasm_only_error()) }))
}

#[cfg(target_arch = "wasm32")]
pub fn wrap_js_llm_exec_fn(
    func: Function,
) -> Box<dyn Fn(LlmRequest) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>> + Send + Sync> {
    let func = SendWrapper::new(func);
    Box::new(move |request: LlmRequest| {
        let req_json = serde_json::to_value(&request).unwrap_or(Json::Null);
        let js_val = json_to_js(&req_json);
        let result = func.call1(&JsValue::NULL, &js_val);
        Box::pin(SendWrapper::new(async move {
            match result {
                Ok(val) => {
                    if let Some(promise) = val.dyn_ref::<js_sys::Promise>() {
                        match JsFuture::from(promise.clone()).await {
                            Ok(resolved) => relay_json_from_js(&resolved),
                            Err(e) => Err(relay_error_from_js(&e)),
                        }
                    } else {
                        relay_json_from_js(&val)
                    }
                }
                Err(e) => Err(relay_error_from_js(&e)),
            }
        }))
    })
}

/// Wrap a JS function `(chunk) => void` as a collector callback.
///
/// The collector is called with each intercepted Json chunk during a streaming LLM response.
/// It is used to accumulate chunks on the JavaScript side for aggregation.
/// If the JS function throws, the exception is converted to a `FlowError::Internal`
/// and returned as `Err`, which terminates the stream.
#[cfg(not(target_arch = "wasm32"))]
pub fn wrap_js_collector_fn(_func: Function) -> Box<dyn FnMut(Json) -> Result<()> + Send> {
    Box::new(move |_chunk: Json| Ok(()))
}

#[cfg(target_arch = "wasm32")]
pub fn wrap_js_collector_fn(func: Function) -> Box<dyn FnMut(Json) -> Result<()> + Send> {
    let func = SendWrapper::new(func);
    Box::new(move |chunk: Json| {
        let js_chunk = json_to_js(&chunk);
        match func.call1(&JsValue::NULL, &js_chunk) {
            Ok(_) => Ok(()),
            Err(e) => {
                let msg = e
                    .as_string()
                    .unwrap_or_else(|| "JS collector threw an exception".to_string());
                record_callback_error(format!("nemo_relay: {msg}"));
                Err(FlowError::Internal(msg))
            }
        }
    })
}

/// Wrap a JS function `() => object` as a finalizer callback.
///
/// The finalizer is called exactly once when the stream is exhausted.
/// It takes no arguments and must return a JSON value representing the
/// aggregated response.
#[cfg(not(target_arch = "wasm32"))]
pub fn wrap_js_finalizer_fn(_func: Function) -> Box<dyn FnOnce() -> Json + Send> {
    Box::new(move || Json::Null)
}

#[cfg(target_arch = "wasm32")]
pub fn wrap_js_finalizer_fn(func: Function) -> Box<dyn FnOnce() -> Json + Send> {
    let func = SendWrapper::new(func);
    Box::new(move || {
        // TODO: This closure returns Json (not Result<Json>), so we cannot propagate
        // errors through the type system. Log errors so failures are not silent.
        callback_json_or_fallback(
            func.call0(&JsValue::NULL),
            "nemo_relay: JS finalizer result conversion failed",
            "nemo_relay: JS finalizer callback threw",
            Json::Null,
        )
    })
}

/// Wrap a JS function for event subscriber: `(event) => void`.
#[cfg(not(target_arch = "wasm32"))]
pub fn wrap_js_event_subscriber(_func: Function) -> EventSubscriberFn {
    std::sync::Arc::new(move |_event: &Event| {})
}

#[cfg(target_arch = "wasm32")]
pub fn wrap_js_event_subscriber(func: Function) -> EventSubscriberFn {
    let func = SendWrapper::new(func);
    std::sync::Arc::new(move |event: &Event| {
        let wasm_event = match WasmEvent::try_from_event(event) {
            Ok(event) => event,
            Err(error) => {
                record_callback_error(format!(
                    "nemo_relay: failed to serialize JS event subscriber payload: {error}"
                ));
                return;
            }
        };
        let js_event = wasm_event
            .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
            .unwrap_or(JsValue::NULL);
        if let Err(e) = func.call1(&JsValue::NULL, &js_event) {
            record_callback_error(format!(
                "nemo_relay: JS event subscriber callback threw: {}",
                js_error_message(&e)
            ));
            eprintln!(
                "nemo_relay: JS event subscriber callback threw: {}",
                js_error_message(&e)
            );
        }
    })
}

/// Wrap a JS function `(args, next) => result | Promise<result>` for tool execution intercept.
///
/// The `next` parameter passed to JS is a reusable function `(args) => Promise<result>`
/// that invokes the next layer in the middleware chain. It can be called multiple times
/// to support retry patterns.
#[cfg(not(target_arch = "wasm32"))]
pub fn wrap_js_tool_exec_intercept_fn(
    _func: Function,
) -> Arc<
    dyn Fn(&str, Json, ToolExecutionNextFn) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>>
        + Send
        + Sync,
> {
    Arc::new(move |_name: &str, args: Json, next: ToolExecutionNextFn| next(args))
}

#[cfg(target_arch = "wasm32")]
pub fn wrap_js_tool_exec_intercept_fn(
    func: Function,
) -> Arc<
    dyn Fn(&str, Json, ToolExecutionNextFn) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>>
        + Send
        + Sync,
> {
    let func = SendWrapper::new(func);
    Arc::new(move |_name: &str, args: Json, next: ToolExecutionNextFn| {
        let js_args = json_to_js(&args);
        let next_clone = next.clone();
        let js_next = wasm_bindgen::closure::Closure::<dyn Fn(JsValue) -> JsValue>::new(
            move |next_args: JsValue| -> JsValue {
                let args_json = js_to_json(&next_args).unwrap_or(Json::Null);
                let next = next_clone.clone();
                let future = next(args_json);
                wasm_bindgen_futures::future_to_promise(async move {
                    let result = future
                        .await
                        .map_err(|e| JsValue::from_str(&e.to_string()))?;
                    Ok(json_to_js(&result))
                })
                .into()
            },
        );
        let js_next_val = js_next.as_ref().clone();
        let result = func.call2(&JsValue::NULL, &js_args, &js_next_val);
        Box::pin(SendWrapper::new(async move {
            let _closure_guard = js_next; // prevent drop until future completes
            match result {
                Ok(val) => {
                    if let Some(promise) = val.dyn_ref::<js_sys::Promise>() {
                        match JsFuture::from(promise.clone()).await {
                            Ok(resolved) => relay_json_from_js(&resolved),
                            Err(e) => Err(relay_error_from_js(&e)),
                        }
                    } else {
                        relay_json_from_js(&val)
                    }
                }
                Err(e) => Err(relay_error_from_js(&e)),
            }
        }))
    })
}

/// Wrap a JS function `(request, next) => result | Promise<result>` for LLM execution intercept.
///
/// The `next` parameter passed to JS is a reusable function `(request) => Promise<result>`
/// that invokes the next layer in the middleware chain. It can be called multiple times
/// to support retry patterns. The `LlmRequest` is serialized to JSON before passing to
/// JS; when JS calls `next`, the argument is deserialized back.
#[cfg(not(target_arch = "wasm32"))]
pub fn wrap_js_llm_exec_intercept_fn(
    _func: Function,
) -> Arc<
    dyn Fn(
            &str,
            LlmRequest,
            LlmExecutionNextFn,
        ) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>>
        + Send
        + Sync,
> {
    Arc::new(move |_name: &str, request: LlmRequest, next: LlmExecutionNextFn| next(request))
}

#[cfg(target_arch = "wasm32")]
pub fn wrap_js_llm_exec_intercept_fn(
    func: Function,
) -> Arc<
    dyn Fn(
            &str,
            LlmRequest,
            LlmExecutionNextFn,
        ) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>>
        + Send
        + Sync,
> {
    let func = SendWrapper::new(func);
    Arc::new(
        move |_name: &str, request: LlmRequest, next: LlmExecutionNextFn| {
            let req_json = serde_json::to_value(&request).unwrap_or(Json::Null);
            let js_request = json_to_js(&req_json);
            let next_clone = next.clone();
            let js_next = wasm_bindgen::closure::Closure::<dyn Fn(JsValue) -> JsValue>::new(
                move |next_val: JsValue| -> JsValue {
                    let next_json = js_to_json(&next_val).unwrap_or(Json::Null);
                    let next_request: LlmRequest =
                        serde_json::from_value(next_json).unwrap_or(request.clone());
                    let next = next_clone.clone();
                    let future = next(next_request);
                    wasm_bindgen_futures::future_to_promise(async move {
                        let result = future
                            .await
                            .map_err(|e| JsValue::from_str(&e.to_string()))?;
                        Ok(json_to_js(&result))
                    })
                    .into()
                },
            );
            let js_next_val = js_next.as_ref().clone();
            let result = func.call2(&JsValue::NULL, &js_request, &js_next_val);
            Box::pin(SendWrapper::new(async move {
                let _closure_guard = js_next; // prevent drop until future completes
                match result {
                    Ok(val) => {
                        if let Some(promise) = val.dyn_ref::<js_sys::Promise>() {
                            match JsFuture::from(promise.clone()).await {
                                Ok(resolved) => js_to_json(&resolved)
                                    .map_err(|e| FlowError::Internal(js_error_message(&e))),
                                Err(e) => Err(FlowError::Internal(js_error_message(&e))),
                            }
                        } else {
                            js_to_json(&val).map_err(|e| FlowError::Internal(js_error_message(&e)))
                        }
                    }
                    Err(e) => Err(FlowError::Internal(js_error_message(&e))),
                }
            }))
        },
    )
}

/// Wrap a JS function `(request, next) => result | Promise<result>` for LLM stream execution intercept.
///
/// The JS callback receives the `LlmRequest` serialized as a plain JSON object
/// and a real `next(request)` function. That `next(...)` callback resolves to
/// an array of downstream chunks, matching the Node binding's composition
/// model. Returning an array preserves downstream stream composition; returning
/// any other JSON value produces a single-chunk stream.
#[cfg(not(target_arch = "wasm32"))]
pub fn wrap_js_llm_stream_exec_intercept_fn(
    _func: Function,
) -> Arc<
    dyn Fn(
            &str,
            LlmRequest,
            LlmStreamExecutionNextFn,
        ) -> Pin<
            Box<
                dyn Future<
                        Output = Result<
                            Pin<Box<dyn tokio_stream::Stream<Item = Result<Json>> + Send>>,
                        >,
                    > + Send,
            >,
        > + Send
        + Sync,
> {
    Arc::new(move |_name: &str, request: LlmRequest, next: LlmStreamExecutionNextFn| next(request))
}

#[cfg(target_arch = "wasm32")]
/// Wrap a JS function `(request, next) => result | Promise<result>` for LLM
/// stream execution intercept on `wasm32`.
///
/// The bridge exposes `next(request)` as a JavaScript function that resolves to
/// a flattened chunk array. The wrapper then accepts either a plain JSON value
/// or a promise from the intercept and rehydrates the resolved value into a
/// Rust stream.
pub fn wrap_js_llm_stream_exec_intercept_fn(
    func: Function,
) -> Arc<
    dyn Fn(
            &str,
            LlmRequest,
            LlmStreamExecutionNextFn,
        ) -> Pin<
            Box<
                dyn Future<
                        Output = Result<
                            Pin<Box<dyn tokio_stream::Stream<Item = Result<Json>> + Send>>,
                        >,
                    > + Send,
            >,
        > + Send
        + Sync,
> {
    let func = SendWrapper::new(func);
    Arc::new(
        move |_name: &str, request: LlmRequest, next: LlmStreamExecutionNextFn| {
            let req_json = serde_json::to_value(&request).unwrap_or(Json::Null);
            let js_request = json_to_js(&req_json);
            let next_clone = next.clone();
            let js_next = wasm_bindgen::closure::Closure::<dyn Fn(JsValue) -> JsValue>::new(
                move |next_val: JsValue| -> JsValue {
                    let next_clone = next_clone.clone();
                    wasm_bindgen_futures::future_to_promise(async move {
                        let next_request = js_value_to_llm_request(&next_val)?;
                        let chunks = collect_stream_chunks(next_clone, next_request).await?;
                        Ok(json_to_js(&chunks))
                    })
                    .into()
                },
            );
            let js_next_val = js_next.as_ref().clone();
            let result = func.call2(&JsValue::NULL, &js_request, &js_next_val);
            Box::pin(SendWrapper::new(async move {
                let _closure_guard = js_next; // prevent drop until future completes
                let val = resolve_js_value(result).await?;
                Ok(json_to_result_stream(val))
            }))
        },
    )
}

// ---------------------------------------------------------------------------
// Codec wrappers
// ---------------------------------------------------------------------------

/// WebAssembly implementation of `LlmCodec` backed by two JS functions (decode + encode).
///
/// # Safety
///
/// `SendWrapper` is used because JS functions are not `Send`. This is safe in
/// WebAssembly because the runtime is single-threaded. The pattern matches all other
/// JS-function wrappers in this file.
#[cfg(target_arch = "wasm32")]
struct WasmCodec {
    decode_fn: SendWrapper<Function>,
    encode_fn: SendWrapper<Function>,
}

// SAFETY: WebAssembly is single-threaded; SendWrapper guarantees these are only accessed
// from the thread that created them.
#[cfg(target_arch = "wasm32")]
unsafe impl Send for WasmCodec {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for WasmCodec {}

#[cfg(target_arch = "wasm32")]
impl LlmCodec for WasmCodec {
    fn decode(&self, request: &LlmRequest) -> Result<AnnotatedLlmRequest> {
        let req_json = serde_json::to_value(request).unwrap_or(Json::Null);
        let js_req = json_to_js(&req_json);
        let result = self
            .decode_fn
            .call1(&JsValue::NULL, &js_req)
            .map_err(|e| FlowError::Internal(js_error_message(&e)))?;
        let result_json =
            js_to_json(&result).map_err(|e| FlowError::Internal(js_error_message(&e)))?;
        serde_json::from_value(result_json).map_err(|e| {
            FlowError::Internal(format!("failed to deserialize AnnotatedLlmRequest: {e}"))
        })
    }

    fn encode(&self, annotated: &AnnotatedLlmRequest, original: &LlmRequest) -> Result<LlmRequest> {
        let annotated_json = serde_json::to_value(annotated).unwrap_or(Json::Null);
        let js_annotated = json_to_js(&annotated_json);
        let original_json = serde_json::to_value(original).unwrap_or(Json::Null);
        let js_original = json_to_js(&original_json);
        let result = self
            .encode_fn
            .call2(&JsValue::NULL, &js_annotated, &js_original)
            .map_err(|e| FlowError::Internal(js_error_message(&e)))?;
        let result_json =
            js_to_json(&result).map_err(|e| FlowError::Internal(js_error_message(&e)))?;
        serde_json::from_value(result_json)
            .map_err(|e| FlowError::Internal(format!("failed to deserialize LlmRequest: {e}")))
    }
}

/// Wrap two JS functions `(request) => annotated` and `(annotated, original) => request`
/// into an `Arc<dyn LlmCodec>`.
#[cfg(not(target_arch = "wasm32"))]
pub fn wrap_js_codec(_decode_fn: Function, _encode_fn: Function) -> Arc<dyn LlmCodec> {
    struct UnsupportedCodec;

    impl LlmCodec for UnsupportedCodec {
        fn decode(&self, _request: &LlmRequest) -> Result<AnnotatedLlmRequest> {
            Err(wasm_only_error())
        }

        fn encode(
            &self,
            _annotated: &AnnotatedLlmRequest,
            _original: &LlmRequest,
        ) -> Result<LlmRequest> {
            Err(wasm_only_error())
        }
    }

    Arc::new(UnsupportedCodec)
}

#[cfg(target_arch = "wasm32")]
pub fn wrap_js_codec(decode_fn: Function, encode_fn: Function) -> Arc<dyn LlmCodec> {
    Arc::new(WasmCodec {
        decode_fn: SendWrapper::new(decode_fn),
        encode_fn: SendWrapper::new(encode_fn),
    })
}

// ---------------------------------------------------------------------------
// Response codec wrapper
// ---------------------------------------------------------------------------

/// Wraps a JS function implementing `(response: JsValue) => JsValue` into an
/// `Arc<dyn LlmResponseCodec>`.
///
/// # Safety
///
/// `SendWrapper` is used because JS functions are not `Send`. This is safe in
/// WebAssembly because the runtime is single-threaded.
#[cfg(target_arch = "wasm32")]
struct WasmResponseCodec {
    decode_response_fn: SendWrapper<Function>,
}

#[cfg(target_arch = "wasm32")]
unsafe impl Send for WasmResponseCodec {}
#[cfg(target_arch = "wasm32")]
unsafe impl Sync for WasmResponseCodec {}

#[cfg(target_arch = "wasm32")]
impl LlmResponseCodec for WasmResponseCodec {
    fn decode_response(&self, response: &Json) -> Result<AnnotatedLlmResponse> {
        let js_resp = json_to_js(response);
        let result = self
            .decode_response_fn
            .call1(&JsValue::NULL, &js_resp)
            .map_err(|e| FlowError::Internal(format!("decode_response() failed: {e:?}")))?;
        let result_json = js_callback_to_json(&result).map_err(|e| {
            FlowError::Internal(format!("decode_response() returned invalid JSON: {e:?}"))
        })?;
        serde_json::from_value(result_json).map_err(|e| {
            FlowError::Internal(format!("decode_response() returned unexpected type: {e}"))
        })
    }
}

/// Wrap a JS function into an `Arc<dyn LlmResponseCodec>`.
#[cfg(not(target_arch = "wasm32"))]
pub fn wrap_js_response_codec(_decode_response_fn: Function) -> Arc<dyn LlmResponseCodec> {
    panic!("wrap_js_response_codec is only available on wasm32")
}

/// Wrap a JS function into an `Arc<dyn LlmResponseCodec>`.
#[cfg(target_arch = "wasm32")]
pub fn wrap_js_response_codec(decode_response_fn: Function) -> Arc<dyn LlmResponseCodec> {
    Arc::new(WasmResponseCodec {
        decode_response_fn: SendWrapper::new(decode_response_fn),
    })
}

/// Wrap a JS function for LLM sanitize response: `(response) => response`.
///
/// Takes a `Json` value, passes it to JS, and deserializes the result back.
#[cfg(not(target_arch = "wasm32"))]
pub fn wrap_js_llm_response_fn(func: Function) -> LlmSanitizeResponseFn {
    let _ = func;
    Arc::new(move |response: Json| response)
}

#[cfg(target_arch = "wasm32")]
pub fn wrap_js_llm_response_fn(func: Function) -> LlmSanitizeResponseFn {
    let func = SendWrapper::new(func);
    Arc::new(move |response: Json| {
        let js_resp = json_to_js(&response);
        // TODO: This closure returns Json (not Result<Json>), so we cannot propagate
        // errors through the type system. Log errors and fall back to original response.
        callback_json_or_fallback(
            func.call1(&JsValue::NULL, &js_resp),
            "nemo_relay: JS LLM response callback result conversion failed",
            "nemo_relay: JS LLM response callback threw",
            response,
        )
    })
}

#[cfg(test)]
#[path = "../tests/coverage/callable_tests.rs"]
mod tests;
