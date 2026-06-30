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
use nemo_relay::api::runtime::{
    EventSubscriberFn, LlmConditionalFn, LlmExecutionNextFn, LlmRequestInterceptFn,
    LlmSanitizeRequestFn, LlmSanitizeResponseFn, LlmStreamExecutionNextFn, ToolConditionalFn,
    ToolExecutionNextFn, ToolInterceptFn, ToolSanitizeFn,
};
use serde::Deserialize;
use serde_json::Value as Json;
use tokio_stream::StreamExt;

use nemo_relay::api::event::{Event, PendingMarkSpec};
use nemo_relay::api::llm::{LlmRequest, LlmRequestInterceptOutcome};
use nemo_relay::codec::request::AnnotatedLlmRequest;
use nemo_relay::codec::response::AnnotatedLlmResponse;
use nemo_relay::codec::traits::{LlmCodec, LlmResponseCodec};
use nemo_relay::error::{FlowError, Result};

use crate::convert::{callback_json, record_callback_error};
use crate::promise_call::{JsonNextFn, JsonStreamNextFn, PromiseAwareFn};
use crate::types::JsEvent;

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

fn recv_json_or_value(
    rx: std::sync::mpsc::Receiver<Json>,
    error_prefix: &str,
    fallback: Json,
) -> Json {
    rx.recv().unwrap_or_else(|e| {
        record_callback_error(format!("{error_prefix}: {e}"));
        fallback
    })
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

fn recv_llm_request_or_value(
    rx: std::sync::mpsc::Receiver<Json>,
    error_prefix: &str,
    fallback: LlmRequest,
) -> LlmRequest {
    let result = recv_json_or_null(rx, error_prefix);
    serde_json::from_value(result).unwrap_or_else(|e| {
        record_callback_error(format!(
            "{error_prefix}: failed to deserialize LlmRequest: {e}"
        ));
        fallback
    })
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
            return Json::Null;
        }
        // TODO: This closure returns Json (not Result<Json>), so we cannot propagate
        // errors through the type system. Log the error so failures are not silent.
        recv_json_or_null(rx, "nemo_relay: JS tool callback failed")
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
        recv_option_string_result(rx, "JS tool conditional callback failed")
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
        recv_json_result(rx, "JS tool callback failed")
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
            rx.await.map_err(|e| FlowError::Internal(e.to_string()))
        })
    })
}

/// Wrap a JS function for unified LLM request intercepts (3-arg signature).
///
/// The JS callback receives a single JSON object
/// `{ name: string, request: LlmRequest, annotated: AnnotatedLlmRequest | null }`
/// and must return `{ request, annotated?, pendingMarks? }`.
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
            let result = recv_json_result(rx, "JS LLM request intercept callback failed")?;

            #[derive(Deserialize)]
            #[serde(rename_all = "camelCase")]
            struct JsOutcome {
                request: LlmRequest,
                #[serde(default)]
                annotated: Option<AnnotatedLlmRequest>,
                #[serde(default)]
                pending_marks: Vec<PendingMarkSpec>,
            }
            let outcome: JsOutcome = serde_json::from_value(result).map_err(|e| {
                FlowError::Internal(format!("invalid JS LLM request intercept outcome: {e}"))
            })?;
            Ok(LlmRequestInterceptOutcome {
                request: outcome.request,
                annotated_request: outcome.annotated,
                pending_marks: outcome.pending_marks,
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
        recv_llm_request_or_value(
            rx,
            "nemo_relay: JS LLM sanitize request callback failed",
            request,
        )
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
        recv_json_or_value(rx, "nemo_relay: JS LLM response callback failed", response)
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
        recv_option_string_result(rx, "JS LLM conditional callback failed")
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
            rx.await.map_err(|e| FlowError::Internal(e.to_string()))
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
                Ok(Box::pin(stream)
                    as Pin<
                        Box<dyn tokio_stream::Stream<Item = Result<Json>> + Send>,
                    >)
            })
        },
    )
}
