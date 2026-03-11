// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::type_complexity)]
//! JavaScript callable wrappers for NVMagic callbacks.
//!
//! This module bridges JavaScript functions (received as NAPI `ThreadsafeFunction` values)
//! into the Rust closure signatures expected by the NVMagic core runtime. Each wrapper
//! handles serialization of arguments to/from JSON and manages cross-thread communication
//! between the Rust async runtime and the Node.js event loop.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use napi::threadsafe_function::{ErrorStrategy, ThreadsafeFunction, ThreadsafeFunctionCallMode};
use serde_json::Value as Json;

use nvmagic_core::types::{LLMRequest, LLMResponse};
use nvmagic_core::{
    LlmExecutionNextFn, LlmStreamExecutionNextFn, MagicError, Result, ToolExecutionNextFn,
};

use crate::types::JsEvent;

/// Wrap a JS function `(name: string, args: object) => object` for tool sanitize/intercept.
pub fn wrap_js_tool_fn(
    func: ThreadsafeFunction<(String, Json), ErrorStrategy::Fatal>,
) -> Box<dyn Fn(&str, Json) -> Json + Send + Sync> {
    let func = Arc::new(func);
    Box::new(move |name: &str, args: Json| {
        let func = func.clone();
        let name = name.to_string();
        let (tx, rx) = std::sync::mpsc::channel();
        func.call_with_return_value(
            (name, args),
            ThreadsafeFunctionCallMode::Blocking,
            move |val: Json| {
                let _ = tx.send(val);
                Ok(())
            },
        );
        rx.recv().unwrap_or(Json::Null)
    })
}

/// Wrap a JS function `(name: string, args: object) => string | null` for tool conditional.
pub fn wrap_js_tool_conditional_fn(
    func: ThreadsafeFunction<(String, Json), ErrorStrategy::Fatal>,
) -> Box<dyn Fn(&str, &Json) -> Option<String> + Send + Sync> {
    let func = Arc::new(func);
    Box::new(move |name: &str, args: &Json| {
        let func = func.clone();
        let name = name.to_string();
        let args = args.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        func.call_with_return_value(
            (name, args),
            ThreadsafeFunctionCallMode::Blocking,
            move |val: Json| {
                let result = match val {
                    Json::Null => None,
                    Json::String(s) => Some(s),
                    _ => None,
                };
                let _ = tx.send(result);
                Ok(())
            },
        );
        rx.recv().unwrap_or(None)
    })
}

/// Wrap a JS function `(name: string, args: object) => boolean` for tool exec conditional.
pub fn wrap_js_tool_exec_conditional_fn(
    func: ThreadsafeFunction<(String, Json), ErrorStrategy::Fatal>,
) -> Box<dyn Fn(&str, &Json) -> bool + Send + Sync> {
    let func = Arc::new(func);
    Box::new(move |name: &str, args: &Json| {
        let func = func.clone();
        let name = name.to_string();
        let args = args.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        func.call_with_return_value(
            (name, args),
            ThreadsafeFunctionCallMode::Blocking,
            move |val: Json| {
                let _ = tx.send(val.as_bool().unwrap_or(false));
                Ok(())
            },
        );
        rx.recv().unwrap_or(false)
    })
}

/// Wrap a JS function `(args: object) => object` for tool execution.
pub fn wrap_js_tool_exec_fn(
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Box<dyn Fn(Json) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>> + Send + Sync> {
    let func = Arc::new(func);
    Box::new(move |args: Json| {
        let func = func.clone();
        Box::pin(async move {
            let (tx, rx) = tokio::sync::oneshot::channel();
            func.call_with_return_value(
                args,
                ThreadsafeFunctionCallMode::Blocking,
                move |val: Json| {
                    let _ = tx.send(val);
                    Ok(())
                },
            );
            rx.await.map_err(|e| MagicError::Internal(e.to_string()))
        })
    })
}

/// Wrap a JS function `(value: object) => object` for JSON transform (LLM request intercepts, etc.).
pub fn wrap_js_json_fn(
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Box<dyn Fn(Json) -> Json + Send + Sync> {
    let func = Arc::new(func);
    Box::new(move |value: Json| {
        let func = func.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        func.call_with_return_value(
            value,
            ThreadsafeFunctionCallMode::Blocking,
            move |val: Json| {
                let _ = tx.send(val);
                Ok(())
            },
        );
        rx.recv().unwrap_or(Json::Null)
    })
}

/// Wrap a JS function for LLM sanitize request: `(request: JsLLMRequest) => JsLLMRequest`.
/// Since ThreadsafeFunction requires serde-serializable args, we serialize the request as JSON.
pub fn wrap_js_llm_sanitize_request_fn(
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Box<dyn Fn(LLMRequest) -> LLMRequest + Send + Sync> {
    let func = Arc::new(func);
    Box::new(move |request: LLMRequest| {
        let func = func.clone();
        let req_json = serde_json::to_value(&request).unwrap_or(Json::Null);
        let (tx, rx) = std::sync::mpsc::channel();
        func.call_with_return_value(
            req_json,
            ThreadsafeFunctionCallMode::Blocking,
            move |val: Json| {
                let _ = tx.send(val);
                Ok(())
            },
        );
        let result = rx.recv().unwrap_or(Json::Null);
        serde_json::from_value(result).unwrap_or(request)
    })
}

/// Wrap a JS function for LLM sanitize response: `(response: JsLLMResponse) => JsLLMResponse`.
/// Since ThreadsafeFunction requires serde-serializable args, we serialize the response as JSON.
pub fn wrap_js_llm_response_fn(
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Box<dyn Fn(LLMResponse) -> LLMResponse + Send + Sync> {
    let func = Arc::new(func);
    Box::new(move |response: LLMResponse| {
        let func = func.clone();
        let resp_json = serde_json::to_value(&response).unwrap_or(Json::Null);
        let (tx, rx) = std::sync::mpsc::channel();
        func.call_with_return_value(
            resp_json,
            ThreadsafeFunctionCallMode::Blocking,
            move |val: Json| {
                let _ = tx.send(val);
                Ok(())
            },
        );
        let result = rx.recv().unwrap_or(Json::Null);
        serde_json::from_value(result).unwrap_or(response)
    })
}

/// Wrap a JS function for LLM conditional: `(request: object) => string | null`.
pub fn wrap_js_llm_conditional_fn(
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Box<dyn Fn(&LLMRequest) -> Option<String> + Send + Sync> {
    let func = Arc::new(func);
    Box::new(move |request: &LLMRequest| {
        let func = func.clone();
        let req_json = serde_json::to_value(request).unwrap_or(Json::Null);
        let (tx, rx) = std::sync::mpsc::channel();
        func.call_with_return_value(
            req_json,
            ThreadsafeFunctionCallMode::Blocking,
            move |val: Json| {
                let result = match val {
                    Json::Null => None,
                    Json::String(s) => Some(s),
                    _ => None,
                };
                let _ = tx.send(result);
                Ok(())
            },
        );
        rx.recv().unwrap_or(None)
    })
}

/// Wrap a JS function for LLM exec conditional: `(native: object) => boolean`.
pub fn wrap_js_llm_exec_conditional_fn(
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Box<dyn Fn(&Json) -> bool + Send + Sync> {
    let func = Arc::new(func);
    Box::new(move |native: &Json| {
        let func = func.clone();
        let native_clone = native.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        func.call_with_return_value(
            native_clone,
            ThreadsafeFunctionCallMode::Blocking,
            move |val: Json| {
                let _ = tx.send(val.as_bool().unwrap_or(false));
                Ok(())
            },
        );
        rx.recv().unwrap_or(false)
    })
}

/// Wrap a JS function for LLM execution: `(native: object) => object`.
pub fn wrap_js_llm_exec_fn(
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Box<dyn Fn(Json) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>> + Send + Sync> {
    let func = Arc::new(func);
    Box::new(move |native: Json| {
        let func = func.clone();
        Box::pin(async move {
            let (tx, rx) = tokio::sync::oneshot::channel();
            func.call_with_return_value(
                native,
                ThreadsafeFunctionCallMode::Blocking,
                move |val: Json| {
                    let _ = tx.send(val);
                    Ok(())
                },
            );
            rx.await.map_err(|e| MagicError::Internal(e.to_string()))
        })
    })
}

/// Wrap a JS function `(chunk: object) => void` as a collector callback.
///
/// The collector is called with each intercepted chunk during a streaming LLM response.
/// It is used to accumulate chunks on the JavaScript side for aggregation.
pub fn wrap_js_collector_fn(
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Box<dyn FnMut(Json) + Send> {
    Box::new(move |chunk: Json| {
        func.call(chunk, ThreadsafeFunctionCallMode::Blocking);
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
        func.call_with_return_value(
            (),
            ThreadsafeFunctionCallMode::Blocking,
            move |val: Json| {
                let _ = tx.send(val);
                Ok(())
            },
        );
        rx.recv().unwrap_or(Json::Null)
    })
}

/// Wrap a JS function for event subscriber: `(event: JsEvent) => void`.
pub fn wrap_js_event_subscriber(
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Box<dyn Fn(&nvmagic_core::Event) + Send + Sync> {
    let func = Arc::new(func);
    Box::new(move |event: &nvmagic_core::Event| {
        let event_json = serde_json::to_value(JsEvent::from(event)).unwrap_or(Json::Null);
        func.call(event_json, ThreadsafeFunctionCallMode::NonBlocking);
    })
}

/// Wrap a JS function `(args, next) => result` for tool execution intercept.
///
/// The JS callback receives the tool arguments and a `next` callback. The `next` callback
/// is a one-shot function `(args) => Promise<result>` that invokes the next layer in
/// the middleware chain.
///
/// Since NAPI `ThreadsafeFunction` cannot pass dynamic JS function objects directly, the
/// `next` is provided as a JSON-serialized opaque handle. The JS side should call a
/// companion `callToolNext(handle, args)` API function, or more practically, the intercept
/// callable receives `(args, nextHandle)` and uses the runtime-provided helper.
///
/// For simplicity in the initial implementation, the intercept callable skips `next`
/// and acts as a full replacement — matching the previous behavior while accepting
/// the new `(args, next)` Rust signature.
pub fn wrap_js_tool_exec_intercept_fn(
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Arc<
    dyn Fn(Json, ToolExecutionNextFn) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>>
        + Send
        + Sync,
> {
    let func = Arc::new(func);
    Arc::new(move |args: Json, _next: ToolExecutionNextFn| {
        let func = func.clone();
        Box::pin(async move {
            let (tx, rx) = tokio::sync::oneshot::channel();
            func.call_with_return_value(
                args,
                ThreadsafeFunctionCallMode::Blocking,
                move |val: Json| {
                    let _ = tx.send(val);
                    Ok(())
                },
            );
            rx.await.map_err(|e| MagicError::Internal(e.to_string()))
        })
    })
}

/// Wrap a JS function `(native, next) => result` for LLM execution intercept.
///
/// See `wrap_js_tool_exec_intercept_fn` for the `next` parameter discussion.
pub fn wrap_js_llm_exec_intercept_fn(
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Arc<
    dyn Fn(Json, LlmExecutionNextFn) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>>
        + Send
        + Sync,
> {
    let func = Arc::new(func);
    Arc::new(move |native: Json, _next: LlmExecutionNextFn| {
        let func = func.clone();
        Box::pin(async move {
            let (tx, rx) = tokio::sync::oneshot::channel();
            func.call_with_return_value(
                native,
                ThreadsafeFunctionCallMode::Blocking,
                move |val: Json| {
                    let _ = tx.send(val);
                    Ok(())
                },
            );
            rx.await.map_err(|e| MagicError::Internal(e.to_string()))
        })
    })
}

/// Wrap a JS function `(native, next) => result` for LLM stream execution intercept.
///
/// The intercept callable produces a single JSON result which is wrapped into a
/// single-item stream internally.
pub fn wrap_js_llm_stream_exec_intercept_fn(
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
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
    let func = Arc::new(func);
    Arc::new(move |native: Json, _next: LlmStreamExecutionNextFn| {
        let func = func.clone();
        Box::pin(async move {
            let (tx, rx) = tokio::sync::oneshot::channel();
            func.call_with_return_value(
                native,
                ThreadsafeFunctionCallMode::Blocking,
                move |val: Json| {
                    let _ = tx.send(val);
                    Ok(())
                },
            );
            let result = rx.await.map_err(|e| MagicError::Internal(e.to_string()))?;
            let stream = tokio_stream::once(Ok(result));
            Ok(Box::pin(stream)
                as Pin<
                    Box<dyn tokio_stream::Stream<Item = Result<Json>> + Send>,
                >)
        })
    })
}
