// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::type_complexity)]
//! JavaScript callable wrappers for NVAgentRT callbacks.
//!
//! This module bridges JavaScript functions (received as NAPI `ThreadsafeFunction` values)
//! into the Rust closure signatures expected by the NVAgentRT core runtime. Each wrapper
//! handles serialization of arguments to/from JSON and manages cross-thread communication
//! between the Rust async runtime and the Node.js event loop.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use napi::threadsafe_function::{ErrorStrategy, ThreadsafeFunction, ThreadsafeFunctionCallMode};
use serde_json::Value as Json;

use nvagentrt_core::types::{LLMRequest, SseEvent};
use nvagentrt_core::{AgentRtError, Result};

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
            rx.await.map_err(|e| AgentRtError::Internal(e.to_string()))
        })
    })
}

/// Wrap a JS function `(value: object) => object` for JSON transform (LLM sanitize response, etc.).
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

/// Wrap a JS function for LLM exec conditional: `(request: object) => boolean`.
pub fn wrap_js_llm_exec_conditional_fn(
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Box<dyn Fn(&LLMRequest) -> bool + Send + Sync> {
    let func = Arc::new(func);
    Box::new(move |request: &LLMRequest| {
        let func = func.clone();
        let req_json = serde_json::to_value(request).unwrap_or(Json::Null);
        let (tx, rx) = std::sync::mpsc::channel();
        func.call_with_return_value(
            req_json,
            ThreadsafeFunctionCallMode::Blocking,
            move |val: Json| {
                let _ = tx.send(val.as_bool().unwrap_or(false));
                Ok(())
            },
        );
        rx.recv().unwrap_or(false)
    })
}

/// Wrap a JS function for LLM execution: `(request: object) => object`.
pub fn wrap_js_llm_exec_fn(
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Box<dyn Fn(LLMRequest) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>> + Send + Sync> {
    let func = Arc::new(func);
    Box::new(move |request: LLMRequest| {
        let func = func.clone();
        Box::pin(async move {
            let req_json = serde_json::to_value(&request)
                .map_err(|e| AgentRtError::Internal(e.to_string()))?;
            let (tx, rx) = tokio::sync::oneshot::channel();
            func.call_with_return_value(
                req_json,
                ThreadsafeFunctionCallMode::Blocking,
                move |val: Json| {
                    let _ = tx.send(val);
                    Ok(())
                },
            );
            rx.await.map_err(|e| AgentRtError::Internal(e.to_string()))
        })
    })
}

/// Wrap a JS function for SSE intercept: `(event: JsSseEvent) => JsSseEvent`.
pub fn wrap_js_sse_intercept_fn(
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Box<dyn Fn(SseEvent) -> SseEvent + Send + Sync> {
    let func = Arc::new(func);
    Box::new(move |event: SseEvent| {
        let func = func.clone();
        let event_json = serde_json::to_value(&event).unwrap_or(Json::Null);
        let (tx, rx) = std::sync::mpsc::channel();
        func.call_with_return_value(
            event_json,
            ThreadsafeFunctionCallMode::Blocking,
            move |val: Json| {
                let _ = tx.send(val);
                Ok(())
            },
        );
        let result = rx.recv().unwrap_or(Json::Null);
        serde_json::from_value(result).unwrap_or(event)
    })
}

/// Wrap a JS function for event subscriber: `(event: JsEvent) => void`.
pub fn wrap_js_event_subscriber(
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Box<dyn Fn(&nvagentrt_core::Event) + Send + Sync> {
    let func = Arc::new(func);
    Box::new(move |event: &nvagentrt_core::Event| {
        let event_json = serde_json::to_value(JsEvent::from(event)).unwrap_or(Json::Null);
        func.call(event_json, ThreadsafeFunctionCallMode::NonBlocking);
    })
}
