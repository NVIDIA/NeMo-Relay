// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::type_complexity)]
//! JavaScript callable wrappers for Nexus callbacks.
//!
//! This module bridges JavaScript functions (received as NAPI `ThreadsafeFunction` values)
//! into the Rust closure signatures expected by the Nexus core runtime. Each wrapper
//! handles serialization of arguments to/from JSON and manages cross-thread communication
//! between the Rust async runtime and the Node.js event loop.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use napi::threadsafe_function::{ErrorStrategy, ThreadsafeFunction, ThreadsafeFunctionCallMode};
use serde_json::Value as Json;

use nvidia_nat_nexus_core::types::LLMRequest;
use nvidia_nat_nexus_core::{
    LlmExecutionNextFn, LlmStreamExecutionNextFn, NexusError, Result, ToolExecutionNextFn,
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
        // TODO: This closure returns Json (not Result<Json>), so we cannot propagate
        // errors through the type system. Log the error so failures are not silent.
        rx.recv().unwrap_or_else(|e| {
            eprintln!("nat_nexus: JS tool callback failed: {e}");
            Json::Null
        })
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
        // TODO: This closure returns Option<String> (not Result), so we cannot propagate
        // errors through the type system. Log the error so failures are not silent.
        rx.recv().unwrap_or_else(|e| {
            eprintln!("nat_nexus: JS tool conditional callback failed: {e}");
            None
        })
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
            func.call_with_return_value(
                args,
                ThreadsafeFunctionCallMode::Blocking,
                move |val: Json| {
                    let _ = tx.send(val);
                    Ok(())
                },
            );
            rx.await.map_err(|e| NexusError::Internal(e.to_string()))
        })
    })
}

/// Wrap a JS function `(request: object) => object` for LLM request intercepts.
///
/// The JS callback receives the `LLMRequest` serialized as a plain JSON object
/// (`{ headers, content }`) and must return the same shape. The returned JSON is
/// deserialized back into an `LLMRequest`.
pub fn wrap_js_llm_request_intercept_fn(
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
        // TODO: This closure returns LLMRequest (not Result), so we cannot propagate
        // errors through the type system. Log the error so failures are not silent.
        let result = rx.recv().unwrap_or_else(|e| {
            eprintln!("nat_nexus: JS LLM request intercept callback failed: {e}");
            Json::Null
        });
        serde_json::from_value(result).unwrap_or(request)
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
        // TODO: This closure returns LLMRequest (not Result), so we cannot propagate
        // errors through the type system. Log the error so failures are not silent.
        let result = rx.recv().unwrap_or_else(|e| {
            eprintln!("nat_nexus: JS LLM sanitize request callback failed: {e}");
            Json::Null
        });
        serde_json::from_value(result).unwrap_or(request)
    })
}

/// Wrap a JS function for LLM sanitize response: `(response: Json) => Json`.
pub fn wrap_js_llm_response_fn(
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Box<dyn Fn(Json) -> Json + Send + Sync> {
    let func = Arc::new(func);
    Box::new(move |response: Json| {
        let func = func.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        func.call_with_return_value(
            response.clone(),
            ThreadsafeFunctionCallMode::Blocking,
            move |val: Json| {
                let _ = tx.send(val);
                Ok(())
            },
        );
        // TODO: This closure returns Json (not Result<Json>), so we cannot propagate
        // errors through the type system. Log the error and fall back to original response.
        rx.recv().unwrap_or_else(|e| {
            eprintln!("nat_nexus: JS LLM response callback failed: {e}");
            response
        })
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
        // TODO: This closure returns Option<String> (not Result), so we cannot propagate
        // errors through the type system. Log the error so failures are not silent.
        rx.recv().unwrap_or_else(|e| {
            eprintln!("nat_nexus: JS LLM conditional callback failed: {e}");
            None
        })
    })
}

/// Wrap a JS function for LLM execution: `(request: object) => object`.
///
/// The JS callback receives the `LLMRequest` serialized as a plain JSON object
/// and returns the response as JSON.
pub fn wrap_js_llm_exec_fn(
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Box<dyn Fn(LLMRequest) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>> + Send + Sync> {
    let func = Arc::new(func);
    Box::new(move |request: LLMRequest| {
        let func = func.clone();
        let req_json = serde_json::to_value(&request).unwrap_or(Json::Null);
        Box::pin(async move {
            let (tx, rx) = tokio::sync::oneshot::channel();
            func.call_with_return_value(
                req_json,
                ThreadsafeFunctionCallMode::Blocking,
                move |val: Json| {
                    let _ = tx.send(val);
                    Ok(())
                },
            );
            rx.await.map_err(|e| NexusError::Internal(e.to_string()))
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
        func.call(chunk, ThreadsafeFunctionCallMode::Blocking);
        Ok(())
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
        // TODO: This closure returns Json (not Result<Json>), so we cannot propagate
        // errors through the type system. Log the error so failures are not silent.
        rx.recv().unwrap_or_else(|e| {
            eprintln!("nat_nexus: JS finalizer callback failed: {e}");
            Json::Null
        })
    })
}

/// Wrap a JS function for event subscriber: `(event: JsEvent) => void`.
pub fn wrap_js_event_subscriber(
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Box<dyn Fn(&nvidia_nat_nexus_core::Event) + Send + Sync> {
    let func = Arc::new(func);
    Box::new(move |event: &nvidia_nat_nexus_core::Event| {
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
/// and acts as a full replacement -- matching the previous behavior while accepting
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
            rx.await.map_err(|e| NexusError::Internal(e.to_string()))
        })
    })
}

/// Wrap a JS function `(request, next) => result` for LLM execution intercept.
///
/// See `wrap_js_tool_exec_intercept_fn` for the `next` parameter discussion.
/// The JS callback receives the `LLMRequest` serialized as a plain JSON object.
pub fn wrap_js_llm_exec_intercept_fn(
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Arc<
    dyn Fn(LLMRequest, LlmExecutionNextFn) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>>
        + Send
        + Sync,
> {
    let func = Arc::new(func);
    Arc::new(move |request: LLMRequest, _next: LlmExecutionNextFn| {
        let func = func.clone();
        let req_json = serde_json::to_value(&request).unwrap_or(Json::Null);
        Box::pin(async move {
            let (tx, rx) = tokio::sync::oneshot::channel();
            func.call_with_return_value(
                req_json,
                ThreadsafeFunctionCallMode::Blocking,
                move |val: Json| {
                    let _ = tx.send(val);
                    Ok(())
                },
            );
            rx.await.map_err(|e| NexusError::Internal(e.to_string()))
        })
    })
}

/// Wrap a JS function `(request, next) => result` for LLM stream execution intercept.
///
/// The intercept callable produces a single JSON result which is wrapped into a
/// single-item stream internally. The JS callback receives the `LLMRequest`
/// serialized as a plain JSON object.
pub fn wrap_js_llm_stream_exec_intercept_fn(
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Arc<
    dyn Fn(
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
    let func = Arc::new(func);
    Arc::new(
        move |request: LLMRequest, _next: LlmStreamExecutionNextFn| {
            let func = func.clone();
            let req_json = serde_json::to_value(&request).unwrap_or(Json::Null);
            Box::pin(async move {
                let (tx, rx) = tokio::sync::oneshot::channel();
                func.call_with_return_value(
                    req_json,
                    ThreadsafeFunctionCallMode::Blocking,
                    move |val: Json| {
                        let _ = tx.send(val);
                        Ok(())
                    },
                );
                let result = rx.await.map_err(|e| NexusError::Internal(e.to_string()))?;
                let stream = tokio_stream::once(Ok(result));
                Ok(Box::pin(stream)
                    as Pin<
                        Box<dyn tokio_stream::Stream<Item = Result<Json>> + Send>,
                    >)
            })
        },
    )
}
