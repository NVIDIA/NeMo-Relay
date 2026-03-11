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

use nvagentrt_core::types::{LLMRequest, LLMResponse};
use nvagentrt_core::{
    AgentRtError, LlmExecutionNextFn, LlmStreamExecutionNextFn, Result, ToolExecutionNextFn,
};

use crate::convert::{js_to_json, json_to_js};
use crate::types::WasmEvent;

/// Wrap a JS function `(name, args) => result` for tool sanitize/intercept.
pub fn wrap_js_tool_fn(func: Function) -> Box<dyn Fn(&str, Json) -> Json + Send + Sync> {
    let func = SendWrapper::new(func);
    Box::new(move |name: &str, args: Json| {
        let js_name = JsValue::from_str(name);
        let js_args = json_to_js(&args);
        match func.call2(&JsValue::NULL, &js_name, &js_args) {
            Ok(result) => js_to_json(&result).unwrap_or(Json::Null),
            Err(_) => Json::Null,
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
            Err(_) => None,
        }
    })
}

/// Wrap a JS function `(name, args) => boolean` for tool exec conditional.
pub fn wrap_js_tool_exec_conditional_fn(
    func: Function,
) -> Box<dyn Fn(&str, &Json) -> bool + Send + Sync> {
    let func = SendWrapper::new(func);
    Box::new(move |name: &str, args: &Json| {
        let js_name = JsValue::from_str(name);
        let js_args = json_to_js(args);
        match func.call2(&JsValue::NULL, &js_name, &js_args) {
            Ok(result) => result.as_bool().unwrap_or(false),
            Err(_) => false,
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
                                .map_err(|e| AgentRtError::Internal(format!("{e:?}"))),
                            Err(e) => Err(AgentRtError::Internal(format!("{e:?}"))),
                        }
                    } else {
                        js_to_json(&val).map_err(|e| AgentRtError::Internal(format!("{e:?}")))
                    }
                }
                Err(e) => Err(AgentRtError::Internal(format!("{e:?}"))),
            }
        }))
    })
}

/// Wrap a JS function `(value) => value` for JSON transform.
pub fn wrap_js_json_fn(func: Function) -> Box<dyn Fn(Json) -> Json + Send + Sync> {
    let func = SendWrapper::new(func);
    Box::new(move |value: Json| {
        let js_val = json_to_js(&value);
        match func.call1(&JsValue::NULL, &js_val) {
            Ok(result) => js_to_json(&result).unwrap_or(Json::Null),
            Err(_) => Json::Null,
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
            Ok(result) => {
                let result_json = js_to_json(&result).unwrap_or(Json::Null);
                serde_json::from_value(result_json).unwrap_or(request)
            }
            Err(_) => request,
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
            Err(_) => None,
        }
    })
}

/// Wrap a JS function for LLM exec conditional: `(native) => boolean`.
pub fn wrap_js_llm_exec_conditional_fn(func: Function) -> Box<dyn Fn(&Json) -> bool + Send + Sync> {
    let func = SendWrapper::new(func);
    Box::new(move |native: &Json| {
        let js_val = json_to_js(native);
        match func.call1(&JsValue::NULL, &js_val) {
            Ok(result) => result.as_bool().unwrap_or(false),
            Err(_) => false,
        }
    })
}

/// Wrap a JS function for LLM execution: `(native) => result | Promise<result>`.
pub fn wrap_js_llm_exec_fn(
    func: Function,
) -> Box<dyn Fn(Json) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>> + Send + Sync> {
    let func = SendWrapper::new(func);
    Box::new(move |native: Json| {
        let js_val = json_to_js(&native);
        let result = func.call1(&JsValue::NULL, &js_val);
        Box::pin(SendWrapper::new(async move {
            match result {
                Ok(val) => {
                    if let Some(promise) = val.dyn_ref::<js_sys::Promise>() {
                        match JsFuture::from(promise.clone()).await {
                            Ok(resolved) => js_to_json(&resolved)
                                .map_err(|e| AgentRtError::Internal(format!("{e:?}"))),
                            Err(e) => Err(AgentRtError::Internal(format!("{e:?}"))),
                        }
                    } else {
                        js_to_json(&val).map_err(|e| AgentRtError::Internal(format!("{e:?}")))
                    }
                }
                Err(e) => Err(AgentRtError::Internal(format!("{e:?}"))),
            }
        }))
    })
}

/// Wrap a JS function `(chunk) => void` as a collector callback.
///
/// The collector is called with each intercepted Json chunk during a streaming LLM response.
/// It is used to accumulate chunks on the JavaScript side for aggregation.
pub fn wrap_js_collector_fn(func: Function) -> Box<dyn FnMut(Json) + Send> {
    let func = SendWrapper::new(func);
    Box::new(move |chunk: Json| {
        let js_chunk = json_to_js(&chunk);
        let _ = func.call1(&JsValue::NULL, &js_chunk);
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
        Ok(result) => js_to_json(&result).unwrap_or(Json::Null),
        Err(_) => Json::Null,
    })
}

/// Wrap a JS function for stream response intercept: `(chunk) => chunk` (Json).
pub fn wrap_js_stream_response_intercept_fn(
    func: Function,
) -> Box<dyn Fn(Json) -> Json + Send + Sync> {
    let func = SendWrapper::new(func);
    Box::new(move |chunk: Json| {
        let js_chunk = json_to_js(&chunk);
        match func.call1(&JsValue::NULL, &js_chunk) {
            Ok(result) => js_to_json(&result).unwrap_or(chunk),
            Err(_) => chunk,
        }
    })
}

/// Wrap a JS function for event subscriber: `(event) => void`.
pub fn wrap_js_event_subscriber(
    func: Function,
) -> Box<dyn Fn(&nvagentrt_core::Event) + Send + Sync> {
    let func = SendWrapper::new(func);
    Box::new(move |event: &nvagentrt_core::Event| {
        let wasm_event = WasmEvent::from(event);
        let js_event = wasm_event
            .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
            .unwrap_or(JsValue::NULL);
        let _ = func.call1(&JsValue::NULL, &js_event);
    })
}

/// Wrap a JS function `(args, next) => result | Promise<result>` for tool execution intercept.
///
/// The `next` parameter passed to JS is a one-shot function `(args) => Promise<result>`
/// that invokes the next layer in the middleware chain.
pub fn wrap_js_tool_exec_intercept_fn(
    func: Function,
) -> Arc<
    dyn Fn(Json, ToolExecutionNextFn) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>>
        + Send
        + Sync,
> {
    let func = SendWrapper::new(func);
    Arc::new(move |args: Json, next: ToolExecutionNextFn| {
        let js_args = json_to_js(&args);
        let js_next =
            wasm_bindgen::closure::Closure::once_into_js(move |next_args: JsValue| -> JsValue {
                let args_json = js_to_json(&next_args).unwrap_or(Json::Null);
                let future = next(args_json);
                wasm_bindgen_futures::future_to_promise(async move {
                    let result = future
                        .await
                        .map_err(|e| JsValue::from_str(&e.to_string()))?;
                    Ok(json_to_js(&result))
                })
                .into()
            });
        let result = func.call2(&JsValue::NULL, &js_args, &js_next);
        Box::pin(SendWrapper::new(async move {
            match result {
                Ok(val) => {
                    if let Some(promise) = val.dyn_ref::<js_sys::Promise>() {
                        match JsFuture::from(promise.clone()).await {
                            Ok(resolved) => js_to_json(&resolved)
                                .map_err(|e| AgentRtError::Internal(format!("{e:?}"))),
                            Err(e) => Err(AgentRtError::Internal(format!("{e:?}"))),
                        }
                    } else {
                        js_to_json(&val).map_err(|e| AgentRtError::Internal(format!("{e:?}")))
                    }
                }
                Err(e) => Err(AgentRtError::Internal(format!("{e:?}"))),
            }
        }))
    })
}

/// Wrap a JS function `(native, next) => result | Promise<result>` for LLM execution intercept.
///
/// The `next` parameter passed to JS is a one-shot function `(native) => Promise<result>`
/// that invokes the next layer in the middleware chain.
pub fn wrap_js_llm_exec_intercept_fn(
    func: Function,
) -> Arc<
    dyn Fn(Json, LlmExecutionNextFn) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>>
        + Send
        + Sync,
> {
    let func = SendWrapper::new(func);
    Arc::new(move |native: Json, next: LlmExecutionNextFn| {
        let js_native = json_to_js(&native);
        let js_next =
            wasm_bindgen::closure::Closure::once_into_js(move |next_val: JsValue| -> JsValue {
                let native_json = js_to_json(&next_val).unwrap_or(Json::Null);
                let future = next(native_json);
                wasm_bindgen_futures::future_to_promise(async move {
                    let result = future
                        .await
                        .map_err(|e| JsValue::from_str(&e.to_string()))?;
                    Ok(json_to_js(&result))
                })
                .into()
            });
        let result = func.call2(&JsValue::NULL, &js_native, &js_next);
        Box::pin(SendWrapper::new(async move {
            match result {
                Ok(val) => {
                    if let Some(promise) = val.dyn_ref::<js_sys::Promise>() {
                        match JsFuture::from(promise.clone()).await {
                            Ok(resolved) => js_to_json(&resolved)
                                .map_err(|e| AgentRtError::Internal(format!("{e:?}"))),
                            Err(e) => Err(AgentRtError::Internal(format!("{e:?}"))),
                        }
                    } else {
                        js_to_json(&val).map_err(|e| AgentRtError::Internal(format!("{e:?}")))
                    }
                }
                Err(e) => Err(AgentRtError::Internal(format!("{e:?}"))),
            }
        }))
    })
}

/// Wrap a JS function `(native, next) => result | Promise<result>` for LLM stream execution intercept.
///
/// The intercept callable produces a single JSON result which is wrapped into a
/// single-item stream internally.
pub fn wrap_js_llm_stream_exec_intercept_fn(
    func: Function,
) -> Arc<
    dyn Fn(
            Json,
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
    Arc::new(move |native: Json, _next: LlmStreamExecutionNextFn| {
        // For stream execution intercepts, we ignore `next` and produce a single-item stream
        // from the JS function's result, matching the existing WASM stream execution pattern.
        let js_val = json_to_js(&native);
        let result = func.call1(&JsValue::NULL, &js_val);
        Box::pin(SendWrapper::new(async move {
            let val = match result {
                Ok(val) => {
                    if let Some(promise) = val.dyn_ref::<js_sys::Promise>() {
                        match JsFuture::from(promise.clone()).await {
                            Ok(resolved) => js_to_json(&resolved)
                                .map_err(|e| AgentRtError::Internal(format!("{e:?}")))?,
                            Err(e) => return Err(AgentRtError::Internal(format!("{e:?}"))),
                        }
                    } else {
                        js_to_json(&val).map_err(|e| AgentRtError::Internal(format!("{e:?}")))?
                    }
                }
                Err(e) => return Err(AgentRtError::Internal(format!("{e:?}"))),
            };
            let stream = tokio_stream::once(Ok(val));
            Ok(Box::pin(stream)
                as Pin<
                    Box<dyn tokio_stream::Stream<Item = Result<Json>> + Send>,
                >)
        }))
    })
}

/// Wrap a JS function for LLM response intercept: `(response) => response`.
///
/// Takes an `LLMResponse`, passes it to JS as a JSON object, and
/// deserializes the result back into an `LLMResponse`.
pub fn wrap_js_llm_response_fn(
    func: Function,
) -> Box<dyn Fn(LLMResponse) -> LLMResponse + Send + Sync> {
    let func = SendWrapper::new(func);
    Box::new(move |response: LLMResponse| {
        let resp_json = serde_json::to_value(&response).unwrap_or(Json::Null);
        let js_resp = json_to_js(&resp_json);
        match func.call1(&JsValue::NULL, &js_resp) {
            Ok(result) => {
                let result_json = js_to_json(&result).unwrap_or(Json::Null);
                serde_json::from_value(result_json).unwrap_or(response)
            }
            Err(_) => response,
        }
    })
}
