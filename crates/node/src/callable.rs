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
use tokio_stream::StreamExt;

use nvidia_nat_nexus_core::types::LLMRequest;
use nvidia_nat_nexus_core::{
    LlmExecutionNextFn, LlmStreamExecutionNextFn, NexusError, Result, ToolExecutionNextFn,
};

use crate::promise_call::{JsonNextFn, JsonStreamNextFn, PromiseAwareFn};
use crate::types::JsEvent;

fn recv_json_or_null(rx: std::sync::mpsc::Receiver<Json>, error_prefix: &str) -> Json {
    rx.recv().unwrap_or_else(|e| {
        eprintln!("{error_prefix}: {e}");
        Json::Null
    })
}

fn recv_json_or_value(
    rx: std::sync::mpsc::Receiver<Json>,
    error_prefix: &str,
    fallback: Json,
) -> Json {
    rx.recv().unwrap_or_else(|e| {
        eprintln!("{error_prefix}: {e}");
        fallback
    })
}

fn recv_option_string_or_none(
    rx: std::sync::mpsc::Receiver<Option<String>>,
    error_prefix: &str,
) -> Option<String> {
    rx.recv().unwrap_or_else(|e| {
        eprintln!("{error_prefix}: {e}");
        None
    })
}

fn recv_llm_request_or_value(
    rx: std::sync::mpsc::Receiver<Json>,
    error_prefix: &str,
    fallback: LLMRequest,
) -> LLMRequest {
    let result = recv_json_or_null(rx, error_prefix);
    serde_json::from_value(result).unwrap_or(fallback)
}

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
        recv_json_or_null(rx, "nat_nexus: JS tool callback failed")
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
        recv_option_string_or_none(rx, "nat_nexus: JS tool conditional callback failed")
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
) -> Box<dyn Fn(&str, LLMRequest) -> LLMRequest + Send + Sync> {
    let func = Arc::new(func);
    Box::new(move |_name: &str, request: LLMRequest| {
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
        recv_llm_request_or_value(
            rx,
            "nat_nexus: JS LLM request intercept callback failed",
            request,
        )
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
        recv_llm_request_or_value(
            rx,
            "nat_nexus: JS LLM sanitize request callback failed",
            request,
        )
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
        recv_json_or_value(rx, "nat_nexus: JS LLM response callback failed", response)
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
        recv_option_string_or_none(rx, "nat_nexus: JS LLM conditional callback failed")
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
        recv_json_or_null(rx, "nat_nexus: JS finalizer callback failed")
    })
}

/// Wrap a JS function for event subscriber: `(event: JsEvent) => void`.
pub fn wrap_js_event_subscriber(
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> nvidia_nat_nexus_core::EventSubscriberFn {
    let func = Arc::new(func);
    Arc::new(move |event: &nvidia_nat_nexus_core::Event| {
        let event_json = serde_json::to_value(JsEvent::from(event)).unwrap_or(Json::Null);
        func.call(event_json, ThreadsafeFunctionCallMode::NonBlocking);
    })
}

#[cfg(test)]
#[path = "callable_coverage_tests.rs"]
mod coverage_tests;

/// Wrap a JS function `(args, next) => result` for tool execution intercept.
///
/// The JS callback receives the tool arguments and a real `next(args)` function
/// that returns a Promise for the downstream result.
pub fn wrap_js_tool_exec_intercept_fn(
    func: Arc<PromiseAwareFn>,
) -> Arc<
    dyn Fn(&str, Json, ToolExecutionNextFn) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>>
        + Send
        + Sync,
> {
    Arc::new(move |_name: &str, args: Json, next: ToolExecutionNextFn| {
        let func = func.clone();
        let next_json: JsonNextFn = Arc::new(move |next_args| next(next_args));
        Box::pin(async move { func.call_with_json_next(args, next_json).await })
    })
}

/// Wrap a JS function `(request, next) => result` for LLM execution intercept.
///
/// The JS callback receives the `LLMRequest` serialized as a plain JSON object
/// and a real `next(request)` function that returns a Promise for the downstream
/// result.
pub fn wrap_js_llm_exec_intercept_fn(
    func: Arc<PromiseAwareFn>,
) -> Arc<
    dyn Fn(
            &str,
            LLMRequest,
            LlmExecutionNextFn,
        ) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>>
        + Send
        + Sync,
> {
    Arc::new(
        move |_name: &str, request: LLMRequest, next: LlmExecutionNextFn| {
            let func = func.clone();
            let req_json = serde_json::to_value(&request).unwrap_or(Json::Null);
            let next_json: JsonNextFn = Arc::new(move |next_request_json| {
                let next = next.clone();
                Box::pin(async move {
                    let next_request: LLMRequest = serde_json::from_value(next_request_json)
                        .map_err(|e| {
                            NexusError::Internal(format!("invalid LLMRequest from JS next: {e}"))
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
/// The JS callback receives the `LLMRequest` serialized as a plain JSON object
/// and a real `next(request)` function whose Promise resolves to an array of
/// downstream JSON chunks. Returning an array preserves streaming semantics;
/// returning any other JSON value produces a single-chunk stream.
pub fn wrap_js_llm_stream_exec_intercept_fn(
    func: Arc<PromiseAwareFn>,
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
    Arc::new(
        move |_name: &str, request: LLMRequest, next: LlmStreamExecutionNextFn| {
            let func = func.clone();
            let req_json = serde_json::to_value(&request).unwrap_or(Json::Null);
            let next_stream: JsonStreamNextFn = Arc::new(move |next_request_json| {
                let next = next.clone();
                Box::pin(async move {
                    let next_request: LLMRequest = serde_json::from_value(next_request_json)
                        .map_err(|e| {
                            NexusError::Internal(format!("invalid LLMRequest from JS next: {e}"))
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
                Ok(Box::pin(stream)
                    as Pin<
                        Box<dyn tokio_stream::Stream<Item = Result<Json>> + Send>,
                    >)
            })
        },
    )
}
