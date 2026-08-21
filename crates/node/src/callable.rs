// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::type_complexity)]
//! JavaScript callable wrappers for NeMo Relay callbacks.
//!
//! This module bridges JavaScript functions (received as NAPI `ThreadsafeFunction` values)
//! into the Rust closure signatures expected by the NeMo Relay core runtime. Each wrapper
//! handles serialization of arguments to/from JSON and manages cross-thread communication
//! between the Rust async runtime and the Node.js event loop.

use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use napi::bindgen_prelude::ToNapiValue;
use napi::threadsafe_function::{
    ErrorStrategy, ThreadSafeCallContext, ThreadsafeFunction, ThreadsafeFunctionCallMode,
};
use napi::{Env, JsFunction, JsObject, JsUnknown, NapiRaw, NapiValue};
use napi_derive::napi;
use nemo_relay::api::runtime::{
    EventMetadataInjectorFn, EventSanitizeFn, EventSubscriberFn, LlmCodecIdentity,
    LlmConditionalFn, LlmExecutionNextFn, LlmJsonStream, LlmRequestInterceptFn,
    LlmSanitizeRequestContext, LlmSanitizeRequestFn, LlmSanitizeResponseContext,
    LlmSanitizeResponseFn, LlmStreamExecutionNextFn, ToolConditionalFn, ToolExecutionNextFn,
    ToolInterceptFn, ToolSanitizeFn,
};
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;

use nemo_relay::api::event::{
    CategoryProfile, DataSchema, Event, EventCategory,
    EventSanitizeFields as CoreEventSanitizeFields, LogSeverity, PendingMarkSpec,
};
use nemo_relay::api::llm::{LlmRequest, LlmRequestInterceptOutcome};
use nemo_relay::api::tool::{ToolExecutionInterceptOutcome, ToolExecutionResult};
use nemo_relay::codec::optimization::LlmOptimizationContribution;
use nemo_relay::codec::request::AnnotatedLlmRequest;
use nemo_relay::codec::response::AnnotatedLlmResponse;
use nemo_relay::codec::traits::{LlmCodec, LlmResponseCodec};
use nemo_relay::error::{FlowError, Result};

use crate::callback_factory;
use crate::convert::{callback_json, record_callback_error, to_napi_err};
use crate::promise_call::{JsonNextFn, JsonStreamNextFn, PromiseAwareFn};
use crate::types::{EventSanitizeFields, JsEvent, event_sanitize_fields_from_json};

#[derive(Default)]
struct JsSubscriberCallbackState {
    next_id: u64,
    pending: BTreeSet<u64>,
}

fn js_subscriber_callbacks() -> &'static (Mutex<JsSubscriberCallbackState>, Condvar) {
    static CALLBACKS: OnceLock<(Mutex<JsSubscriberCallbackState>, Condvar)> = OnceLock::new();
    CALLBACKS.get_or_init(Default::default)
}

fn reserve_js_subscriber_callback() -> u64 {
    let (state, _) = js_subscriber_callbacks();
    let mut state = state.lock().unwrap();
    state.next_id += 1;
    let id = state.next_id;
    state.pending.insert(id);
    id
}

fn complete_js_subscriber_callback(id: u64) {
    let (state, completed) = js_subscriber_callbacks();
    let mut state = state.lock().unwrap();
    state.pending.remove(&id);
    completed.notify_all();
}

pub(crate) fn flush_js_subscriber_callbacks() -> Result<()> {
    let (state, completed) = js_subscriber_callbacks();
    let mut state = state
        .lock()
        .map_err(|error| FlowError::Internal(error.to_string()))?;
    let watermark = state.next_id;
    while state.pending.range(..=watermark).next().is_some() {
        state = completed
            .wait(state)
            .map_err(|error| FlowError::Internal(error.to_string()))?;
    }
    Ok(())
}

/// Structured codec identity delivered to JavaScript LLM sanitizers.
#[napi(object)]
#[derive(Clone)]
pub(crate) struct JsLlmCodecIdentity {
    pub kind: String,
    pub id: Option<String>,
}

/// Structured per-call request context delivered to JavaScript LLM sanitizers.
#[derive(Clone)]
pub(crate) struct JsLlmSanitizeRequestContext {
    pub codec: JsLlmCodecIdentity,
    resolved: Option<Arc<dyn LlmCodec>>,
}

/// Structured per-call response context delivered to JavaScript LLM sanitizers.
#[derive(Clone)]
pub(crate) struct JsLlmSanitizeResponseContext {
    pub codec: JsLlmCodecIdentity,
    resolved: Option<Arc<dyn LlmResponseCodec>>,
}

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
    data_schema: Option<DataSchema>,
    #[serde(default)]
    metadata: Option<Json>,
    #[serde(default)]
    severity: Option<LogSeverity>,
}

impl From<JsPendingMarkSpec> for PendingMarkSpec {
    fn from(mark: JsPendingMarkSpec) -> Self {
        Self {
            name: mark.name,
            category: mark.category,
            category_profile: mark.category_profile,
            data: mark.data,
            data_schema: mark.data_schema,
            metadata: mark.metadata,
            severity: mark.severity,
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
            data_schema: mark.data_schema,
            metadata: mark.metadata,
            severity: mark.severity,
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
    #[serde(default, rename = "exceptionType")]
    exception_type: String,
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

/// Wrap a conditional registration gate so throws and invalid return values
/// cross N-API as a JSON-compatible error envelope.
pub(crate) fn safe_conditional_gate_callback(
    env: &Env,
    func: &JsFunction,
) -> napi::Result<JsFunction> {
    let factory: JsFunction = env.run_script(
        r#"((fn) => function __nemo_relay_conditional_gate_wrapper(...args) {
  try {
    const returned = fn(...args);
    const value = returned === undefined ? null : returned;
    if (value === null || typeof value === 'string') {
      return { ok: true, value };
    }
    return { ok: false, error: 'conditional middleware guardrail must return a string or null' };
  } catch (error) {
    let message = 'JavaScript callback threw';
    try {
      message = String(error?.message ?? error);
    } catch {}
    let exceptionType = 'Error';
    try {
      const name = String(error?.name ?? '').trim();
      if (name) exceptionType = name;
    } catch {}
    return { ok: false, error: message, exceptionType };
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
    } else if result.exception_type.is_empty() {
        Err(FlowError::Internal(format!(
            "{error_prefix}: {}",
            result.error
        )))
    } else {
        Err(FlowError::CallbackException {
            message: format!("{error_prefix}: {}", result.error),
            exception_type: result.exception_type,
        })
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

async fn await_middleware_json_result(
    rx: tokio::sync::oneshot::Receiver<Json>,
    error_prefix: &str,
) -> Result<Json> {
    let value = rx
        .await
        .map_err(|error| FlowError::Internal(format!("{error_prefix}: {error}")))?;
    unwrap_middleware_result(value, error_prefix)
}

async fn await_middleware_json_or_value(
    rx: tokio::sync::oneshot::Receiver<Json>,
    error_prefix: &str,
    fallback: Json,
) -> Json {
    match await_middleware_json_result(rx, error_prefix).await {
        Ok(value) => value,
        Err(error) => {
            record_callback_error(error.to_string());
            fallback
        }
    }
}

async fn await_middleware_option_string_result(
    rx: tokio::sync::oneshot::Receiver<Json>,
    error_prefix: &str,
) -> Result<Option<String>> {
    match await_middleware_json_result(rx, error_prefix).await? {
        Json::Null => Ok(None),
        Json::String(value) => Ok(Some(value)),
        other => Err(FlowError::Internal(format!(
            "{error_prefix}: expected string or null, got {other:?}",
        ))),
    }
}

/// Wrap a Promise-aware JS `(name, args) => string | null` tool guardrail.
pub fn wrap_js_tool_conditional_promise_fn(func: Arc<PromiseAwareFn>) -> ToolConditionalFn {
    Arc::new(move |name: String, args: Json| {
        let func = func.clone();
        Box::pin(async move {
            let value = func
                .call_spread(vec![Json::String(name), args])
                .await
                .inspect_err(|error| record_callback_error(error.to_string()))?;
            match value {
                Json::Null => Ok(None),
                Json::String(reason) => Ok(Some(reason)),
                other => {
                    let error = FlowError::Internal(format!(
                        "JS tool conditional callback failed: expected string or null, got {other:?}"
                    ));
                    record_callback_error(error.to_string());
                    Err(error)
                }
            }
        })
    })
}

/// Wrap a Promise-aware JS `(name, args) => Json` tool request intercept.
pub fn wrap_js_tool_request_intercept_promise_fn(func: Arc<PromiseAwareFn>) -> ToolInterceptFn {
    Arc::new(move |name: String, args: Json| {
        let func = func.clone();
        Box::pin(async move {
            func.call_spread(vec![Json::String(name), args])
                .await
                .inspect_err(|error| record_callback_error(error.to_string()))
        })
    })
}

/// Wrap a Promise-aware JS tool sanitizer.
pub fn wrap_js_tool_sanitize_promise_fn(func: Arc<PromiseAwareFn>) -> ToolSanitizeFn {
    Arc::new(move |name: String, value: Json| {
        let func = func.clone();
        let publication = nemo_relay::api::runtime::subscriber_dispatcher::in_dispatcher_callback();
        Box::pin(async move {
            let args = vec![Json::String(name), value];
            let result = if publication {
                func.call_spread_for_publication(args).await
            } else {
                func.call_spread(args).await
            };
            result.inspect_err(|error| {
                record_callback_error(error.to_string());
            })
        })
    })
}

/// Wrap a Promise-aware JS LLM request sanitizer.
pub fn wrap_js_llm_sanitize_request_promise_fn(func: Arc<PromiseAwareFn>) -> LlmSanitizeRequestFn {
    Arc::new(
        move |request: LlmRequest, context: LlmSanitizeRequestContext| {
            let func = func.clone();
            let publication =
                nemo_relay::api::runtime::subscriber_dispatcher::in_dispatcher_callback();
            Box::pin(async move {
                let request = serde_json::to_value(request).map_err(|error| {
                    let error = FlowError::Internal(format!(
                        "failed to serialize JS LLM sanitize request: {error}"
                    ));
                    record_callback_error(error.to_string());
                    error
                })?;
                let context = js_llm_sanitize_request_context(&context);
                let build_args: crate::promise_call::Arg0Builder = Box::new(move |env| {
                    let mut args = env.create_array_with_length(2)?;
                    let request = unsafe {
                        JsUnknown::from_raw_unchecked(
                            env.raw(),
                            Json::to_napi_value(env.raw(), request)?,
                        )
                    };
                    args.set_element(0, request)?;
                    args.set_element(1, js_llm_sanitize_request_context_to_napi(env, context)?)?;
                    Ok(js_object_to_unknown(env, args))
                });
                let value = if publication {
                    func.call_spread_with_arg0_for_publication(build_args).await
                } else {
                    func.call_spread_with_arg0(build_args).await
                }
                .inspect_err(|error| {
                    record_callback_error(error.to_string());
                })?;
                if value.is_null() {
                    Ok(None)
                } else {
                    serde_json::from_value(value)
                        .map(Some)
                        .map_err(|error| {
                            let error = FlowError::Internal(format!(
                                "JS LLM sanitize request callback failed: failed to deserialize LlmRequest: {error}"
                            ));
                            record_callback_error(error.to_string());
                            error
                        })
                }
            })
        },
    )
}

/// Wrap a Promise-aware JS LLM response sanitizer.
pub fn wrap_js_llm_sanitize_response_promise_fn(
    func: Arc<PromiseAwareFn>,
) -> LlmSanitizeResponseFn {
    Arc::new(move |response: Json, context: LlmSanitizeResponseContext| {
        let func = func.clone();
        let publication = nemo_relay::api::runtime::subscriber_dispatcher::in_dispatcher_callback();
        Box::pin(async move {
            let context = js_llm_sanitize_response_context(&context);
            let build_args: crate::promise_call::Arg0Builder = Box::new(move |env| {
                let mut args = env.create_array_with_length(2)?;
                let response = unsafe {
                    JsUnknown::from_raw_unchecked(
                        env.raw(),
                        Json::to_napi_value(env.raw(), response)?,
                    )
                };
                args.set_element(0, response)?;
                args.set_element(1, js_llm_sanitize_response_context_to_napi(env, context)?)?;
                Ok(js_object_to_unknown(env, args))
            });
            let value = if publication {
                func.call_spread_with_arg0_for_publication(build_args).await
            } else {
                func.call_spread_with_arg0(build_args).await
            }
            .inspect_err(|error| {
                record_callback_error(error.to_string());
            })?;
            Ok((!value.is_null()).then_some(value))
        })
    })
}

/// Wrap a Promise-aware JS `(request) => string | null` LLM guardrail.
pub fn wrap_js_llm_conditional_promise_fn(func: Arc<PromiseAwareFn>) -> LlmConditionalFn {
    Arc::new(move |request: LlmRequest| {
        let func = func.clone();
        Box::pin(async move {
            let request = serde_json::to_value(request).map_err(|error| {
                let error = FlowError::Internal(format!(
                    "failed to serialize JS LLM conditional request: {error}"
                ));
                record_callback_error(error.to_string());
                error
            })?;
            let value = func
                .call(request)
                .await
                .inspect_err(|error| record_callback_error(error.to_string()))?;
            match value {
                Json::Null => Ok(None),
                Json::String(reason) => Ok(Some(reason)),
                other => {
                    let error = FlowError::Internal(format!(
                        "JS LLM conditional callback failed: expected string or null, got {other:?}"
                    ));
                    record_callback_error(error.to_string());
                    Err(error)
                }
            }
        })
    })
}

/// Wrap a Promise-aware JS LLM request intercept.
pub fn wrap_js_llm_request_intercept_promise_fn(
    func: Arc<PromiseAwareFn>,
) -> LlmRequestInterceptFn {
    Arc::new(
        move |name: String, request: LlmRequest, annotated: Option<AnnotatedLlmRequest>| {
            let func = func.clone();
            Box::pin(async move {
                let request = serde_json::to_value(request).map_err(|error| {
                    let error = FlowError::Internal(format!(
                        "failed to serialize JS LLM request intercept request: {error}"
                    ));
                    record_callback_error(error.to_string());
                    error
                })?;
                let annotated = serde_json::to_value(annotated).map_err(|error| {
                    let error = FlowError::Internal(format!(
                        "failed to serialize JS LLM request intercept annotation: {error}"
                    ));
                    record_callback_error(error.to_string());
                    error
                })?;
                let value = serde_json::json!({
                    "name": name,
                    "request": request,
                    "annotated": annotated,
                });
                let value = func.call(value).await.inspect_err(|error| {
                    record_callback_error(error.to_string());
                })?;
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
                let outcome: JsOutcome = serde_json::from_value(value).map_err(|error| {
                    let error = FlowError::Internal(format!(
                        "invalid JS LLM request intercept outcome: {error}"
                    ));
                    record_callback_error(error.to_string());
                    error
                })?;
                Ok(LlmRequestInterceptOutcome {
                    request: outcome.request,
                    annotated_request: outcome.annotated,
                    pending_marks: outcome.pending_marks.into_iter().map(Into::into).collect(),
                    optimization_contributions: outcome.optimization_contributions,
                })
            })
        },
    )
}

/// Wrap a Promise-aware JS event sanitizer.
///
/// All lifecycle publication invokes these callbacks from Relay's serial
/// dispatcher. The invocation context is also used by queued tool and LLM
/// observability sanitizers so a flush cannot wait on its own publication.
pub fn wrap_js_event_sanitize_promise_fn(func: Arc<PromiseAwareFn>) -> EventSanitizeFn {
    Arc::new(move |event: Arc<Event>, fields: CoreEventSanitizeFields| {
        let func = func.clone();
        Box::pin(async move {
            let event_json = JsEvent::try_from_event(&event)
                .map(JsEvent::into_json)
                .map_err(|error| {
                    let error = FlowError::Internal(format!(
                        "failed to serialize JS event sanitizer context: {error}"
                    ));
                    record_callback_error(error.to_string());
                    error
                })?;
            let js_fields = EventSanitizeFields {
                data: fields.data,
                category_profile: fields
                    .category_profile
                    .as_ref()
                    .map(serde_json::to_value)
                    .transpose()
                    .map_err(|error| {
                        let error = FlowError::Internal(format!(
                            "failed to serialize JS event sanitizer category profile: {error}"
                        ));
                        record_callback_error(error.to_string());
                        error
                    })?,
                metadata: fields.metadata,
            };
            let args = vec![
                event_json,
                serde_json::to_value(js_fields).map_err(|error| {
                    let error = FlowError::Internal(format!(
                        "failed to serialize JS event sanitizer fields: {error}"
                    ));
                    record_callback_error(error.to_string());
                    error
                })?,
            ];
            let publication =
                nemo_relay::api::runtime::subscriber_dispatcher::in_dispatcher_callback();
            let value = if publication {
                func.call_spread_for_publication(args).await
            } else {
                func.call_spread(args).await
            }
            .inspect_err(|error| {
                // Scope and mark publication happens on the dispatcher
                // thread. The core clears the governed observability fields while
                // making the binding-visible failure available to Node.
                record_callback_error(error.to_string());
            })?;
            let fields = event_sanitize_fields_from_json(value).map_err(|error| {
                let error =
                    FlowError::Internal(format!("invalid JS event sanitizer result: {error}"));
                record_callback_error(error.to_string());
                error
            })?;
            let category_profile = fields
                .category_profile
                .map(serde_json::from_value)
                .transpose()
                .map_err(|error| {
                    let error =
                        FlowError::Internal(format!("invalid JS event sanitizer result: {error}"));
                    record_callback_error(error.to_string());
                    error
                })?;
            Ok(CoreEventSanitizeFields {
                data: fields.data,
                category_profile,
                metadata: fields.metadata,
            })
        })
    })
}

/// Wrap a Promise-aware JavaScript event metadata injector.
pub fn wrap_js_event_metadata_injector_promise_fn(
    func: Arc<PromiseAwareFn>,
) -> EventMetadataInjectorFn {
    Arc::new(move |event: Arc<Event>| {
        let func = func.clone();
        Box::pin(async move {
            let event_json = JsEvent::try_from_event(&event)
                .map(JsEvent::into_json)
                .map_err(|error| {
                    let error = FlowError::Internal(format!(
                        "failed to serialize JavaScript event metadata injector context: {error}"
                    ));
                    record_callback_error(error.to_string());
                    error
                })?;
            let publication =
                nemo_relay::api::runtime::subscriber_dispatcher::in_dispatcher_callback();
            let value = if publication {
                func.call_spread_for_publication(vec![event_json]).await
            } else {
                func.call(event_json).await
            }
            .inspect_err(|error| record_callback_error(error.to_string()))?;
            serde_json::from_value::<BTreeMap<String, Json>>(value).map_err(|error| {
                let error = FlowError::Internal(format!(
                    "invalid JavaScript event metadata injector result: {error}"
                ));
                record_callback_error(error.to_string());
                error
            })
        })
    })
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
    Arc::new(move |name: String, args: Json| {
        let func = func.clone();
        Box::pin(async move {
            let (tx, rx) = tokio::sync::oneshot::channel();
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
                    "failed to queue JS tool callback: {status:?}"
                )));
            }
            await_middleware_json_result(rx, "nemo_relay: JS tool callback failed").await
        })
    })
}

/// Wrap a JS function `(name: string, args: object) => string | null` for tool conditional guardrails.
pub fn wrap_js_tool_conditional_fn(
    func: ThreadsafeFunction<(String, Json), ErrorStrategy::Fatal>,
) -> ToolConditionalFn {
    let func = Arc::new(func);
    Arc::new(move |name: String, args: Json| {
        let func = func.clone();
        Box::pin(async move {
            let (tx, rx) = tokio::sync::oneshot::channel();
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
            await_middleware_option_string_result(rx, "JS tool conditional callback failed").await
        })
    })
}

/// Wrap a JS function `(name: string, args: object) => object` for tool request intercepts.
pub fn wrap_js_tool_request_intercept_fn(
    func: ThreadsafeFunction<(String, Json), ErrorStrategy::Fatal>,
) -> ToolInterceptFn {
    let func = Arc::new(func);
    Arc::new(move |name: String, args: Json| {
        let func = func.clone();
        Box::pin(async move {
            let (tx, rx) = tokio::sync::oneshot::channel();
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
            await_middleware_json_result(rx, "JS tool callback failed").await
        })
    })
}

fn parse_tool_execution_result(value: Json) -> Result<ToolExecutionResult> {
    serde_json::from_value(value).map_err(|error| {
        FlowError::Internal(format!(
            "tool execution callback must return ToolExecutionResult: {error}"
        ))
    })
}

/// Wrap a JS function `(args: object) => ToolExecutionResult` for tool execution.
pub fn wrap_js_tool_exec_fn(
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> Box<
    dyn Fn(Json) -> Pin<Box<dyn Future<Output = Result<ToolExecutionResult>> + Send>> + Send + Sync,
> {
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
            parse_tool_execution_result(unwrap_middleware_result(
                result,
                "JS tool execution callback failed",
            )?)
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
        move |name: String, request: LlmRequest, annotated: Option<AnnotatedLlmRequest>| {
            let func = func.clone();
            Box::pin(async move {
                let req_json = serde_json::to_value(&request).map_err(|error| {
                    let error = FlowError::Internal(format!(
                        "failed to serialize JS LLM request intercept request: {error}"
                    ));
                    record_callback_error(error.to_string());
                    error
                })?;
                let annotated_json = serde_json::to_value(annotated).map_err(|error| {
                    let error = FlowError::Internal(format!(
                        "failed to serialize JS LLM request intercept annotation: {error}"
                    ));
                    record_callback_error(error.to_string());
                    error
                })?;
                let arg = serde_json::json!({
                    "name": name,
                    "request": req_json,
                    "annotated": annotated_json,
                });
                let (tx, rx) = tokio::sync::oneshot::channel();
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
                    await_middleware_json_result(rx, "JS LLM request intercept callback failed")
                        .await?;

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
            })
        },
    )
}

/// Wrap a JS function for LLM request sanitization. The callback receives
/// `(request, context)`.
pub fn wrap_js_llm_sanitize_request_fn(
    func: ThreadsafeFunction<(Json, JsLlmSanitizeRequestContext), ErrorStrategy::Fatal>,
) -> LlmSanitizeRequestFn {
    let func = Arc::new(func);
    Arc::new(
        move |request: LlmRequest, context: LlmSanitizeRequestContext| {
            let func = func.clone();
            Box::pin(async move {
                let context = js_llm_sanitize_request_context(&context);
                let request = serde_json::to_value(request).map_err(|error| {
                    let error = FlowError::Internal(format!(
                        "failed to serialize JS LLM sanitize request: {error}"
                    ));
                    record_callback_error(error.to_string());
                    error
                })?;
                let (tx, rx) = tokio::sync::oneshot::channel();
                if func.call_with_return_value(
                    (request, context),
                    ThreadsafeFunctionCallMode::Blocking,
                    move |value: Option<Json>| {
                        let _ = tx.send(callback_json(value));
                        Ok(())
                    },
                ) != napi::Status::Ok
                {
                    record_callback_error(
                        "nemo_relay: failed to queue JS LLM sanitize request callback",
                    );
                    return Err(FlowError::Internal(
                        "failed to queue JS LLM sanitize request callback".into(),
                    ));
                }
                let value = await_middleware_json_result(
                    rx,
                    "nemo_relay: JS LLM request sanitizer callback failed",
                )
                .await
                .inspect_err(|error| record_callback_error(error.to_string()))?;
                if value.is_null() {
                    return Ok(None);
                }
                serde_json::from_value(value)
                .map(Some)
                .map_err(|error| FlowError::Internal(format!(
                    "JS LLM sanitize request callback failed: failed to deserialize LlmRequest: {error}"
                )))
                .inspect_err(|error| record_callback_error(error.to_string()))
            })
        },
    )
}

/// Wrap a JS function for LLM response sanitization. The callback receives
/// `(response, context)`; returning `null` omits the event payload.
pub fn wrap_js_llm_sanitize_response_fn(
    func: ThreadsafeFunction<(Json, JsLlmSanitizeResponseContext), ErrorStrategy::Fatal>,
) -> LlmSanitizeResponseFn {
    let func = Arc::new(func);
    Arc::new(move |response: Json, context: LlmSanitizeResponseContext| {
        let func = func.clone();
        Box::pin(async move {
            let context = js_llm_sanitize_response_context(&context);
            let (tx, rx) = tokio::sync::oneshot::channel();
            if func.call_with_return_value(
                (response, context),
                ThreadsafeFunctionCallMode::Blocking,
                move |value: Option<Json>| {
                    let _ = tx.send(callback_json(value));
                    Ok(())
                },
            ) != napi::Status::Ok
            {
                record_callback_error(
                    "nemo_relay: failed to queue JS LLM sanitize response callback",
                );
                return Err(FlowError::Internal(
                    "failed to queue JS LLM sanitize response callback".into(),
                ));
            }
            let value = await_middleware_json_result(
                rx,
                "nemo_relay: JS LLM response sanitizer callback failed",
            )
            .await
            .inspect_err(|error| record_callback_error(error.to_string()))?;
            Ok((!value.is_null()).then_some(value))
        })
    })
}

fn js_llm_sanitize_request_context(
    context: &LlmSanitizeRequestContext,
) -> JsLlmSanitizeRequestContext {
    JsLlmSanitizeRequestContext {
        codec: js_codec_identity(context.codec()),
        resolved: context.resolve_codec(),
    }
}

fn js_codec_identity(identity: &LlmCodecIdentity) -> JsLlmCodecIdentity {
    match identity {
        LlmCodecIdentity::None => JsLlmCodecIdentity {
            kind: "none".into(),
            id: None,
        },
        LlmCodecIdentity::BuiltIn(codec) => JsLlmCodecIdentity {
            kind: "builtin".into(),
            id: Some(codec.id().into()),
        },
        LlmCodecIdentity::Runtime(id) => JsLlmCodecIdentity {
            kind: "runtime".into(),
            id: Some(id.clone()),
        },
        LlmCodecIdentity::Opaque => JsLlmCodecIdentity {
            kind: "opaque".into(),
            id: None,
        },
    }
}

fn js_llm_sanitize_response_context(
    context: &LlmSanitizeResponseContext,
) -> JsLlmSanitizeResponseContext {
    JsLlmSanitizeResponseContext {
        codec: js_codec_identity(context.codec()),
        resolved: context.resolve_codec(),
    }
}

fn js_object_to_unknown(env: &Env, object: JsObject) -> JsUnknown {
    unsafe { JsUnknown::from_raw_unchecked(env.raw(), object.raw()) }
}

fn request_codec_object(env: &Env, codec: Arc<dyn LlmCodec>) -> napi::Result<JsObject> {
    let mut object = env.create_object()?;
    let decode_codec = codec.clone();
    let decode = env.create_function_from_closure("decode", move |ctx| {
        let request = ctx.get::<Json>(0)?;
        let request = serde_json::from_value(request)
            .map_err(|error| napi::Error::from_reason(format!("invalid LlmRequest: {error}")))?;
        serde_json::to_value(decode_codec.decode(&request).map_err(to_napi_err)?)
            .map_err(|error| napi::Error::from_reason(error.to_string()))
    })?;
    let encode_codec = codec;
    let encode = env.create_function_from_closure("encode", move |ctx| {
        let annotated = ctx.get::<Json>(0)?;
        let original = ctx.get::<Json>(1)?;
        let annotated = serde_json::from_value(annotated).map_err(|error| {
            napi::Error::from_reason(format!("invalid AnnotatedLlmRequest: {error}"))
        })?;
        let original = serde_json::from_value(original)
            .map_err(|error| napi::Error::from_reason(format!("invalid LlmRequest: {error}")))?;
        serde_json::to_value(
            encode_codec
                .encode(&annotated, &original)
                .map_err(to_napi_err)?,
        )
        .map_err(|error| napi::Error::from_reason(error.to_string()))
    })?;
    object.set_named_property("decode", decode)?;
    object.set_named_property("encode", encode)?;
    Ok(object)
}

fn response_codec_object(env: &Env, codec: Arc<dyn LlmResponseCodec>) -> napi::Result<JsObject> {
    let mut object = env.create_object()?;
    let decode = env.create_function_from_closure("decodeResponse", move |ctx| {
        let response = ctx.get::<Json>(0)?;
        serde_json::to_value(codec.decode_response(&response).map_err(to_napi_err)?)
            .map_err(|error| napi::Error::from_reason(error.to_string()))
    })?;
    object.set_named_property("decodeResponse", decode)?;
    Ok(object)
}

/// Convert a request sanitizer context into the JavaScript object passed to a callback.
pub(crate) fn js_llm_sanitize_request_context_to_napi(
    env: &Env,
    context: JsLlmSanitizeRequestContext,
) -> napi::Result<JsUnknown> {
    let mut object = env.create_object()?;
    let codec = unsafe {
        JsUnknown::from_raw_unchecked(
            env.raw(),
            JsLlmCodecIdentity::to_napi_value(env.raw(), context.codec)?,
        )
    };
    object.set_named_property("codec", codec)?;

    let resolved = context.resolved;
    let resolve_codec =
        env.create_function_from_closure("resolveCodec", move |ctx| match resolved.clone() {
            Some(codec) => Ok(js_object_to_unknown(
                ctx.env,
                request_codec_object(ctx.env, codec)?,
            )),
            None => ctx
                .env
                .get_null()
                .map(|value| unsafe { JsUnknown::from_raw_unchecked(ctx.env.raw(), value.raw()) }),
        })?;
    object.set_named_property("resolveCodec", resolve_codec)?;
    Ok(js_object_to_unknown(env, object))
}

/// Convert a response sanitizer context into the JavaScript object passed to a callback.
pub(crate) fn js_llm_sanitize_response_context_to_napi(
    env: &Env,
    context: JsLlmSanitizeResponseContext,
) -> napi::Result<JsUnknown> {
    let mut object = env.create_object()?;
    let codec = unsafe {
        JsUnknown::from_raw_unchecked(
            env.raw(),
            JsLlmCodecIdentity::to_napi_value(env.raw(), context.codec)?,
        )
    };
    object.set_named_property("codec", codec)?;

    let resolved = context.resolved;
    let resolve_codec =
        env.create_function_from_closure("resolveCodec", move |ctx| match resolved.clone() {
            Some(codec) => Ok(js_object_to_unknown(
                ctx.env,
                response_codec_object(ctx.env, codec)?,
            )),
            None => ctx
                .env
                .get_null()
                .map(|value| unsafe { JsUnknown::from_raw_unchecked(ctx.env.raw(), value.raw()) }),
        })?;
    object.set_named_property("resolveCodec", resolve_codec)?;
    Ok(js_object_to_unknown(env, object))
}

/// Wrap a JS function for LLM conditional guardrails: `(request: object) => string | null`.
pub fn wrap_js_llm_conditional_fn(
    func: ThreadsafeFunction<Json, ErrorStrategy::Fatal>,
) -> LlmConditionalFn {
    let func = Arc::new(func);
    Arc::new(move |request: LlmRequest| {
        let func = func.clone();
        Box::pin(async move {
            let req_json = serde_json::to_value(request).map_err(|error| {
                let error = FlowError::Internal(format!(
                    "failed to serialize JS LLM conditional request: {error}"
                ));
                record_callback_error(error.to_string());
                error
            })?;
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
                    "failed to queue JS LLM conditional callback: {status:?}",
                )));
            }
            await_middleware_option_string_result(rx, "JS LLM conditional callback failed").await
        })
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

struct JsSubscriberCallbackCall {
    event: Json,
    callback_id: u64,
}

fn safe_subscriber_callback(env: &Env, func: &JsFunction) -> napi::Result<JsFunction> {
    let factory: JsFunction = env.run_script(
        r#"((fn) => function __nemo_relay_subscriber_wrapper(error, event, complete) {
  const messageFor = (error) => {
    try {
      return String(error?.message ?? error);
    } catch {
      return 'JavaScript callback failed';
    }
  };
  if (error != null) {
    if (typeof complete === 'function') complete(messageFor(error));
    return;
  }
  Promise.resolve()
    .then(() => fn(event))
    .then(() => complete(), (error) => complete(messageFor(error)));
})"#,
    )?;
    let func_unknown = unsafe { JsUnknown::from_raw_unchecked(env.raw(), func.raw()) };
    let wrapper_unknown = factory.call(None, &[func_unknown])?;
    Ok(unsafe { wrapper_unknown.cast::<JsFunction>() })
}

/// Wrap a JS function for event subscriber: `(event: JsEvent) => void | Promise<void>`.
pub fn wrap_js_event_subscriber(
    env: &Env,
    name: String,
    callback: JsFunction,
) -> napi::Result<EventSubscriberFn> {
    let callback = safe_subscriber_callback(env, &callback)?;
    let queue_error_name = name.clone();
    let mut func = create_js_event_subscriber_function(&callback, name)?;
    func.unref(env)?;
    let func = Arc::new(func);
    Ok(Arc::new(move |event: &Event| {
        let event_json = match JsEvent::try_from_event(event) {
            Ok(event) => event.into_json(),
            Err(error) => {
                record_callback_error(format!(
                    "nemo_relay: failed to serialize JS event subscriber '{queue_error_name}' payload: {error}"
                ));
                return;
            }
        };
        let callback_id = reserve_js_subscriber_callback();
        let status = func.call(
            Ok(JsSubscriberCallbackCall {
                event: event_json,
                callback_id,
            }),
            ThreadsafeFunctionCallMode::NonBlocking,
        );
        if status != napi::Status::Ok {
            record_callback_error(format!(
                "nemo_relay: failed to queue JS event subscriber '{queue_error_name}' callback: {status:?}"
            ));
            complete_js_subscriber_callback(callback_id);
        }
    }))
}

fn create_js_event_subscriber_function(
    callback: &JsFunction,
    name: String,
) -> napi::Result<ThreadsafeFunction<JsSubscriberCallbackCall, ErrorStrategy::CalleeHandled>> {
    callback.create_threadsafe_function::<
        JsSubscriberCallbackCall,
        JsUnknown,
        _,
        ErrorStrategy::CalleeHandled,
    >(0, move |ctx: ThreadSafeCallContext<JsSubscriberCallbackCall>| {
        let JsSubscriberCallbackCall { event, callback_id } = ctx.value;
        let completed = Arc::new(AtomicBool::new(false));
        let completed_callback = Arc::clone(&completed);
        let callback_name = name.clone();
        let complete = complete_subscriber_callback_on_error(
            callback_id,
            create_js_subscriber_completion_callback(
                &ctx.env,
                callback_name,
                callback_id,
                completed_callback,
            ),
        )?;
        let event = complete_subscriber_callback_on_error(callback_id, unsafe {
            Ok(JsUnknown::from_raw_unchecked(
                ctx.env.raw(),
                Json::to_napi_value(ctx.env.raw(), event)?,
            ))
        })?;
        let complete = unsafe { JsUnknown::from_raw_unchecked(ctx.env.raw(), complete.raw()) };
        Ok(vec![event, complete])
    })
}

fn complete_subscriber_callback_on_error<T>(
    callback_id: u64,
    result: napi::Result<T>,
) -> napi::Result<T> {
    result.inspect_err(|_| {
        complete_js_subscriber_callback(callback_id);
    })
}

fn create_js_subscriber_completion_callback(
    env: &Env,
    callback_name: String,
    callback_id: u64,
    completed_callback: Arc<AtomicBool>,
) -> napi::Result<JsFunction> {
    env.create_function_from_closure("__nemo_relay_complete_subscriber_callback", move |ctx| {
        if completed_callback
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            record_js_subscriber_callback_error(&ctx, &callback_name);
            complete_js_subscriber_callback(callback_id);
        }
        ctx.env.get_undefined()
    })
}

fn record_js_subscriber_callback_error(ctx: &napi::CallContext, callback_name: &str) {
    if ctx.length == 0 {
        return;
    }
    match ctx.get::<String>(0) {
        Ok(message) => record_callback_error(format!(
            "nemo_relay: JS event subscriber '{callback_name}' failed: {message}"
        )),
        Err(error) => record_callback_error(format!(
            "nemo_relay: failed to read JS event subscriber '{callback_name}' failure: {error}"
        )),
    }
}

// ---------------------------------------------------------------------------
// Codec wrappers
// ---------------------------------------------------------------------------

/// A NAPI-RS wrapper that implements the core [`LlmCodec`] trait by delegating
/// `decode` and `encode` to JavaScript functions via `ThreadsafeFunction`.
struct NapiCodec {
    decode: Arc<ThreadsafeFunction<Json, ErrorStrategy::Fatal>>,
    encode: Arc<ThreadsafeFunction<Json, ErrorStrategy::Fatal>>,
    register_thread: std::thread::ThreadId,
    direct_decode: Arc<dyn Fn(Json) -> Result<Json> + Send + Sync>,
    direct_encode: Arc<dyn Fn(Json) -> Result<Json> + Send + Sync>,
}

impl LlmCodec for NapiCodec {
    fn decode(&self, request: &LlmRequest) -> Result<AnnotatedLlmRequest> {
        let req_json = serde_json::to_value(request).unwrap_or(Json::Null);
        if std::thread::current().id() == self.register_thread {
            let result = (self.direct_decode)(req_json)?;
            return serde_json::from_value(result).map_err(|e| {
                FlowError::Internal(format!(
                    "JS codec decode callback: failed to deserialize AnnotatedLlmRequest: {e}"
                ))
            });
        }
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
        if std::thread::current().id() == self.register_thread {
            return serde_json::from_value((self.direct_encode)(arg)?).map_err(|e| {
                FlowError::Internal(format!(
                    "JS codec encode callback: failed to deserialize LlmRequest: {e}"
                ))
            });
        }
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
    register_thread: std::thread::ThreadId,
    direct_decode: Arc<dyn Fn(Json) -> Result<Json> + Send + Sync>,
    direct_encode: Arc<dyn Fn(Json) -> Result<Json> + Send + Sync>,
) -> Arc<dyn LlmCodec> {
    Arc::new(NapiCodec {
        decode: Arc::new(decode),
        encode: Arc::new(encode),
        register_thread,
        direct_decode,
        direct_encode,
    })
}

// ---------------------------------------------------------------------------
// Response codec wrapper
// ---------------------------------------------------------------------------

/// A NAPI-RS wrapper that implements the core [`LlmResponseCodec`] trait by
/// delegating `decode_response` to a JavaScript function via `ThreadsafeFunction`.
struct NapiResponseCodec {
    decode_response: Arc<ThreadsafeFunction<Json, ErrorStrategy::Fatal>>,
    register_thread: std::thread::ThreadId,
    direct_decode_response: Arc<dyn Fn(Json) -> Result<Json> + Send + Sync>,
}

impl LlmResponseCodec for NapiResponseCodec {
    fn decode_response(&self, response: &Json) -> Result<AnnotatedLlmResponse> {
        if std::thread::current().id() == self.register_thread {
            let result = (self.direct_decode_response)(response.clone())?;
            return serde_json::from_value(result).map_err(|e| {
                FlowError::Internal(format!(
                    "decode_response returned invalid AnnotatedLlmResponse: {e}"
                ))
            });
        }
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
    register_thread: std::thread::ThreadId,
    direct_decode_response: Arc<dyn Fn(Json) -> Result<Json> + Send + Sync>,
) -> Arc<dyn LlmResponseCodec> {
    Arc::new(NapiResponseCodec {
        decode_response: Arc::new(decode_response),
        register_thread,
        direct_decode_response,
    })
}

/// Wrap a JS function `(args, next) => { result, annotation?, pendingMarks? }` for tool execution intercept.
///
/// The JS callback receives the tool arguments and a real `next(args)` function
/// that returns a Promise for the downstream result.
pub fn wrap_js_tool_exec_intercept_fn(
    func: Arc<PromiseAwareFn>,
) -> nemo_relay::api::runtime::ToolExecutionFn {
    Arc::new(move |_name: &str, args: Json, next: ToolExecutionNextFn| {
        let func = func.clone();
        let next_json: JsonNextFn = Arc::new(move |next_args| {
            let next = next.clone();
            Box::pin(async move {
                serde_json::to_value(next(next_args).await?)
                    .map_err(|error| FlowError::Internal(error.to_string()))
            })
        });
        Box::pin(async move {
            let result = func.call_with_json_next(args, next_json).await?;
            #[derive(Deserialize)]
            #[serde(rename_all = "camelCase")]
            struct JsOutcome {
                result: Json,
                #[serde(default)]
                annotation: Option<Json>,
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
                annotation: outcome
                    .annotation
                    .filter(|annotation| !annotation.is_null()),
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
/// and a real `next(request)` function whose Promise resolves to a lazy
/// async iterable. The callback can return it directly or wrap it with an async
/// generator without materializing the downstream stream.
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
                    next(next_request).await
                })
            });
            Box::pin(async move { func.call_with_stream_next(req_json, next_stream).await })
        },
    )
}
