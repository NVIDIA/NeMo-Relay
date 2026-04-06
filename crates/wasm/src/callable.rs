// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::type_complexity)]
//! Wrappers that adapt JavaScript callback functions into Rust closures.
//!
//! Each wrapper takes a `js_sys::Function`, wraps it with `SendWrapper` (since
//! JS functions are not `Send`), and returns a boxed closure matching the
//! signature expected by the core runtime for guardrails, intercepts,
//! execution functions, and event subscribers.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use js_sys::Function;
use send_wrapper::SendWrapper;
use serde::Serialize;
use serde_json::Value as Json;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use nvidia_nat_nexus_core::types::LLMRequest;
use nvidia_nat_nexus_core::{
    LlmExecutionNextFn, LlmStreamExecutionNextFn, NexusError, Result, ToolExecutionNextFn,
};

use crate::convert::record_callback_error;
use crate::convert::{js_to_json, json_to_js};
use crate::types::WasmEvent;

/// Extract a human-readable error message from a `JsValue`.
///
/// Tries `.as_string()` first (for string errors), then falls back to debug format.
fn js_error_message(e: &JsValue) -> String {
    e.as_string().unwrap_or_else(|| format!("{e:?}"))
}

/// Wrap a JS function `(name, args) => result` for tool sanitize/intercept.
pub fn wrap_js_tool_fn(func: Function) -> Box<dyn Fn(&str, Json) -> Json + Send + Sync> {
    let func = SendWrapper::new(func);
    Box::new(move |name: &str, args: Json| {
        let js_name = JsValue::from_str(name);
        let js_args = json_to_js(&args);
        match func.call2(&JsValue::NULL, &js_name, &js_args) {
            // TODO: This closure returns Json (not Result<Json>), so we cannot propagate
            // errors through the type system. Log errors so failures are not silent.
            Ok(result) => js_to_json(&result).unwrap_or_else(|e| {
                record_callback_error(format!(
                    "nat_nexus: JS tool callback result conversion failed: {}",
                    js_error_message(&e)
                ));
                eprintln!(
                    "nat_nexus: JS tool callback result conversion failed: {}",
                    js_error_message(&e)
                );
                Json::Null
            }),
            Err(e) => {
                record_callback_error(format!(
                    "nat_nexus: JS tool callback threw: {}",
                    js_error_message(&e)
                ));
                eprintln!(
                    "nat_nexus: JS tool callback threw: {}",
                    js_error_message(&e)
                );
                Json::Null
            }
        }
    })
}

/// Wrap a JS function `(name, args) => string | null` for tool conditional.
pub fn wrap_js_tool_conditional_fn(
    func: Function,
) -> Box<dyn Fn(&str, &Json) -> Option<String> + Send + Sync> {
    let func = SendWrapper::new(func);
    Box::new(move |name: &str, args: &Json| {
        let js_name = JsValue::from_str(name);
        let js_args = json_to_js(args);
        match func.call2(&JsValue::NULL, &js_name, &js_args) {
            Ok(result) => {
                if result.is_null() || result.is_undefined() {
                    None
                } else {
                    result.as_string()
                }
            }
            // TODO: This closure returns Option<String> (not Result), so we cannot propagate
            // errors through the type system. Log the error so failures are not silent.
            Err(e) => {
                record_callback_error(format!(
                    "nat_nexus: JS tool conditional callback threw: {}",
                    js_error_message(&e)
                ));
                eprintln!(
                    "nat_nexus: JS tool conditional callback threw: {}",
                    js_error_message(&e)
                );
                None
            }
        }
    })
}

/// Wrap a JS function `(args) => result | Promise<result>` for tool execution.
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
                            Ok(resolved) => js_to_json(&resolved)
                                .map_err(|e| NexusError::Internal(js_error_message(&e))),
                            Err(e) => Err(NexusError::Internal(js_error_message(&e))),
                        }
                    } else {
                        js_to_json(&val).map_err(|e| NexusError::Internal(js_error_message(&e)))
                    }
                }
                Err(e) => Err(NexusError::Internal(js_error_message(&e))),
            }
        }))
    })
}

/// Wrap a JS function `(request) => request` for LLM request intercept.
///
/// Takes an `LLMRequest`, passes it to JS as a JSON object, and
/// deserializes the result back into an `LLMRequest`.
pub fn wrap_js_llm_request_intercept_fn(
    func: Function,
) -> Box<dyn Fn(&str, LLMRequest) -> LLMRequest + Send + Sync> {
    let func = SendWrapper::new(func);
    Box::new(move |_name: &str, request: LLMRequest| {
        let req_json = serde_json::to_value(&request).unwrap_or(Json::Null);
        let js_req = json_to_js(&req_json);
        match func.call1(&JsValue::NULL, &js_req) {
            // TODO: This closure returns LLMRequest (not Result), so we cannot propagate
            // errors through the type system. Log errors so failures are not silent.
            Ok(result) => {
                let result_json = js_to_json(&result).unwrap_or_else(|e| {
                    record_callback_error(format!(
                        "nat_nexus: JS LLM request intercept result conversion failed: {}",
                        js_error_message(&e)
                    ));
                    eprintln!(
                        "nat_nexus: JS LLM request intercept result conversion failed: {}",
                        js_error_message(&e)
                    );
                    Json::Null
                });
                serde_json::from_value(result_json).unwrap_or(request)
            }
            Err(e) => {
                record_callback_error(format!(
                    "nat_nexus: JS LLM request intercept callback threw: {}",
                    js_error_message(&e)
                ));
                eprintln!(
                    "nat_nexus: JS LLM request intercept callback threw: {}",
                    js_error_message(&e)
                );
                request
            }
        }
    })
}

/// Wrap a JS function for LLM sanitize request: `(request) => request`.
pub fn wrap_js_llm_sanitize_request_fn(
    func: Function,
) -> Box<dyn Fn(LLMRequest) -> LLMRequest + Send + Sync> {
    let func = SendWrapper::new(func);
    Box::new(move |request: LLMRequest| {
        let req_json = serde_json::to_value(&request).unwrap_or(Json::Null);
        let js_req = json_to_js(&req_json);
        match func.call1(&JsValue::NULL, &js_req) {
            // TODO: This closure returns LLMRequest (not Result), so we cannot propagate
            // errors through the type system. Log errors so failures are not silent.
            Ok(result) => {
                let result_json = js_to_json(&result).unwrap_or_else(|e| {
                    record_callback_error(format!(
                        "nat_nexus: JS LLM sanitize request result conversion failed: {}",
                        js_error_message(&e)
                    ));
                    eprintln!(
                        "nat_nexus: JS LLM sanitize request result conversion failed: {}",
                        js_error_message(&e)
                    );
                    Json::Null
                });
                serde_json::from_value(result_json).unwrap_or(request)
            }
            Err(e) => {
                record_callback_error(format!(
                    "nat_nexus: JS LLM sanitize request callback threw: {}",
                    js_error_message(&e)
                ));
                eprintln!(
                    "nat_nexus: JS LLM sanitize request callback threw: {}",
                    js_error_message(&e)
                );
                request
            }
        }
    })
}

/// Wrap a JS function for LLM conditional: `(request) => string | null`.
pub fn wrap_js_llm_conditional_fn(
    func: Function,
) -> Box<dyn Fn(&LLMRequest) -> Option<String> + Send + Sync> {
    let func = SendWrapper::new(func);
    Box::new(move |request: &LLMRequest| {
        let req_json = serde_json::to_value(request).unwrap_or(Json::Null);
        let js_req = json_to_js(&req_json);
        match func.call1(&JsValue::NULL, &js_req) {
            Ok(result) => {
                if result.is_null() || result.is_undefined() {
                    None
                } else {
                    result.as_string()
                }
            }
            // TODO: This closure returns Option<String> (not Result), so we cannot propagate
            // errors through the type system. Log the error so failures are not silent.
            Err(e) => {
                record_callback_error(format!(
                    "nat_nexus: JS LLM conditional callback threw: {}",
                    js_error_message(&e)
                ));
                eprintln!(
                    "nat_nexus: JS LLM conditional callback threw: {}",
                    js_error_message(&e)
                );
                None
            }
        }
    })
}

/// Wrap a JS function for LLM execution: `(request) => result | Promise<result>`.
///
/// The `LLMRequest` is serialized to JSON before passing to JS.
pub fn wrap_js_llm_exec_fn(
    func: Function,
) -> Box<dyn Fn(LLMRequest) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>> + Send + Sync> {
    let func = SendWrapper::new(func);
    Box::new(move |request: LLMRequest| {
        let req_json = serde_json::to_value(&request).unwrap_or(Json::Null);
        let js_val = json_to_js(&req_json);
        let result = func.call1(&JsValue::NULL, &js_val);
        Box::pin(SendWrapper::new(async move {
            match result {
                Ok(val) => {
                    if let Some(promise) = val.dyn_ref::<js_sys::Promise>() {
                        match JsFuture::from(promise.clone()).await {
                            Ok(resolved) => js_to_json(&resolved)
                                .map_err(|e| NexusError::Internal(js_error_message(&e))),
                            Err(e) => Err(NexusError::Internal(js_error_message(&e))),
                        }
                    } else {
                        js_to_json(&val).map_err(|e| NexusError::Internal(js_error_message(&e)))
                    }
                }
                Err(e) => Err(NexusError::Internal(js_error_message(&e))),
            }
        }))
    })
}

/// Wrap a JS function `(chunk) => void` as a collector callback.
///
/// The collector is called with each intercepted Json chunk during a streaming LLM response.
/// It is used to accumulate chunks on the JavaScript side for aggregation.
/// If the JS function throws, the exception is converted to a `NexusError::Internal`
/// and returned as `Err`, which terminates the stream.
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
                record_callback_error(format!("nat_nexus: {msg}"));
                Err(NexusError::Internal(msg))
            }
        }
    })
}

/// Wrap a JS function `() => object` as a finalizer callback.
///
/// The finalizer is called exactly once when the stream is exhausted.
/// It takes no arguments and must return a JSON value representing the
/// aggregated response.
pub fn wrap_js_finalizer_fn(func: Function) -> Box<dyn FnOnce() -> Json + Send> {
    let func = SendWrapper::new(func);
    Box::new(move || match func.call0(&JsValue::NULL) {
        // TODO: This closure returns Json (not Result<Json>), so we cannot propagate
        // errors through the type system. Log errors so failures are not silent.
        Ok(result) => js_to_json(&result).unwrap_or_else(|e| {
            record_callback_error(format!(
                "nat_nexus: JS finalizer result conversion failed: {}",
                js_error_message(&e)
            ));
            eprintln!(
                "nat_nexus: JS finalizer result conversion failed: {}",
                js_error_message(&e)
            );
            Json::Null
        }),
        Err(e) => {
            record_callback_error(format!(
                "nat_nexus: JS finalizer callback threw: {}",
                js_error_message(&e)
            ));
            eprintln!(
                "nat_nexus: JS finalizer callback threw: {}",
                js_error_message(&e)
            );
            Json::Null
        }
    })
}

/// Wrap a JS function for event subscriber: `(event) => void`.
pub fn wrap_js_event_subscriber(func: Function) -> nvidia_nat_nexus_core::EventSubscriberFn {
    let func = SendWrapper::new(func);
    std::sync::Arc::new(move |event: &nvidia_nat_nexus_core::Event| {
        let wasm_event = WasmEvent::from(event);
        let js_event = wasm_event
            .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
            .unwrap_or(JsValue::NULL);
        if let Err(e) = func.call1(&JsValue::NULL, &js_event) {
            record_callback_error(format!(
                "nat_nexus: JS event subscriber callback threw: {}",
                js_error_message(&e)
            ));
            eprintln!(
                "nat_nexus: JS event subscriber callback threw: {}",
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
                            Ok(resolved) => js_to_json(&resolved)
                                .map_err(|e| NexusError::Internal(js_error_message(&e))),
                            Err(e) => Err(NexusError::Internal(js_error_message(&e))),
                        }
                    } else {
                        js_to_json(&val).map_err(|e| NexusError::Internal(js_error_message(&e)))
                    }
                }
                Err(e) => Err(NexusError::Internal(js_error_message(&e))),
            }
        }))
    })
}

/// Wrap a JS function `(request, next) => result | Promise<result>` for LLM execution intercept.
///
/// The `next` parameter passed to JS is a reusable function `(request) => Promise<result>`
/// that invokes the next layer in the middleware chain. It can be called multiple times
/// to support retry patterns. The `LLMRequest` is serialized to JSON before passing to
/// JS; when JS calls `next`, the argument is deserialized back.
pub fn wrap_js_llm_exec_intercept_fn(
    func: Function,
) -> Arc<
    dyn Fn(
            &str,
            LLMRequest,
            LlmExecutionNextFn,
        ) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>>
        + Send
        + Sync,
> {
    let func = SendWrapper::new(func);
    Arc::new(
        move |_name: &str, request: LLMRequest, next: LlmExecutionNextFn| {
            let req_json = serde_json::to_value(&request).unwrap_or(Json::Null);
            let js_request = json_to_js(&req_json);
            let next_clone = next.clone();
            let js_next = wasm_bindgen::closure::Closure::<dyn Fn(JsValue) -> JsValue>::new(
                move |next_val: JsValue| -> JsValue {
                    let next_json = js_to_json(&next_val).unwrap_or(Json::Null);
                    let next_request: LLMRequest =
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
                                    .map_err(|e| NexusError::Internal(js_error_message(&e))),
                                Err(e) => Err(NexusError::Internal(js_error_message(&e))),
                            }
                        } else {
                            js_to_json(&val).map_err(|e| NexusError::Internal(js_error_message(&e)))
                        }
                    }
                    Err(e) => Err(NexusError::Internal(js_error_message(&e))),
                }
            }))
        },
    )
}

/// Wrap a JS function `(request, next) => result | Promise<result>` for LLM stream execution intercept.
///
/// The intercept callable produces a single JSON result which is wrapped into a
/// single-item stream internally. The `LLMRequest` is serialized to JSON before
/// passing to JS.
pub fn wrap_js_llm_stream_exec_intercept_fn(
    func: Function,
) -> Arc<
    dyn Fn(
            &str,
            LLMRequest,
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
        move |_name: &str, request: LLMRequest, _next: LlmStreamExecutionNextFn| {
            // For stream execution intercepts, we ignore `next` and produce a single-item stream
            // from the JS function's result, matching the existing WASM stream execution pattern.
            let req_json = serde_json::to_value(&request).unwrap_or(Json::Null);
            let js_val = json_to_js(&req_json);
            let result = func.call1(&JsValue::NULL, &js_val);
            Box::pin(SendWrapper::new(async move {
                let val = match result {
                    Ok(val) => {
                        if let Some(promise) = val.dyn_ref::<js_sys::Promise>() {
                            match JsFuture::from(promise.clone()).await {
                                Ok(resolved) => js_to_json(&resolved)
                                    .map_err(|e| NexusError::Internal(js_error_message(&e)))?,
                                Err(e) => return Err(NexusError::Internal(js_error_message(&e))),
                            }
                        } else {
                            js_to_json(&val)
                                .map_err(|e| NexusError::Internal(js_error_message(&e)))?
                        }
                    }
                    Err(e) => return Err(NexusError::Internal(js_error_message(&e))),
                };
                let stream = tokio_stream::once(Ok(val));
                Ok(Box::pin(stream)
                    as Pin<
                        Box<dyn tokio_stream::Stream<Item = Result<Json>> + Send>,
                    >)
            }))
        },
    )
}

/// Wrap a JS function for LLM sanitize response: `(response) => response`.
///
/// Takes a `Json` value, passes it to JS, and deserializes the result back.
pub fn wrap_js_llm_response_fn(func: Function) -> Box<dyn Fn(Json) -> Json + Send + Sync> {
    let func = SendWrapper::new(func);
    Box::new(move |response: Json| {
        let js_resp = json_to_js(&response);
        match func.call1(&JsValue::NULL, &js_resp) {
            // TODO: This closure returns Json (not Result<Json>), so we cannot propagate
            // errors through the type system. Log errors and fall back to original response.
            Ok(result) => js_to_json(&result).unwrap_or_else(|e| {
                record_callback_error(format!(
                    "nat_nexus: JS LLM response callback result conversion failed: {}",
                    js_error_message(&e)
                ));
                eprintln!(
                    "nat_nexus: JS LLM response callback result conversion failed: {}",
                    js_error_message(&e)
                );
                response.clone()
            }),
            Err(e) => {
                record_callback_error(format!(
                    "nat_nexus: JS LLM response callback threw: {}",
                    js_error_message(&e)
                ));
                eprintln!(
                    "nat_nexus: JS LLM response callback threw: {}",
                    js_error_message(&e)
                );
                response
            }
        }
    })
}
