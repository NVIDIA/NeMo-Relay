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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use napi::bindgen_prelude::{
    FnArgs, FromNapiValue, Function, FunctionCallContext, JsObjectValue, JsValuesTupleIntoVec,
    Null, Object, Unknown,
};
use napi::threadsafe_function::{ThreadsafeCallContext, ThreadsafeFunctionCallMode};
use napi::{Env, ValueType};
use napi_derive::napi;
use nemo_relay::api::runtime::{
    EventMetadataInjectorFn, EventSanitizeFn, EventSubscriberFn, LlmCodecIdentity, LlmCollector,
    LlmConditionalFn, LlmExecutionContext, LlmExecutionFn, LlmExecutionNextFn, LlmFinalizer,
    LlmRequestInterceptFn, LlmSanitizeRequestContext, LlmSanitizeRequestFn,
    LlmSanitizeResponseContext, LlmSanitizeResponseFn, LlmStreamExecutionFn,
    LlmStreamExecutionNextFn, ToolConditionalFn, ToolExecutionContext, ToolExecutionNextFn,
    ToolInterceptFn, ToolSanitizeFn,
};
use serde::de::DeserializeOwned;
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
use crate::convert::{
    HandledTsfn, JsArg, JsArgs2, JsArgs3, PersistentJsFunction, Tsfn, abort_tsfn, closure_function,
    detach, js_value, record_callback_error,
};
use crate::environment::teardown_flag;
use crate::error::Error;
use crate::promise_call::{JsonNextFn, JsonStreamNextFn, PromiseAwareFn, StreamCancellation};
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

fn envelope_field<T: FromNapiValue>(envelope: &Object, key: &str) -> napi::Result<T> {
    envelope.get::<T>(key)?.ok_or_else(|| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("callback envelope has no {key}"),
        )
    })
}

fn decode_envelope(envelope: Unknown, error_prefix: &str) -> napi::Result<Result<Json>> {
    let envelope = Object::from_unknown(envelope)?;
    if envelope_field::<bool>(&envelope, "ok")? {
        return Ok(match envelope.get::<Unknown>("value")? {
            Some(value) => Json::from_unknown(value).map_err(|error| {
                FlowError::Internal(format!(
                    "{error_prefix}: callback result could not be converted: {error}"
                ))
            }),
            None => Ok(Json::Null),
        });
    }
    if envelope.has_named_property("exception")? {
        let exception = envelope.get_named_property::<Unknown>("exception")?;
        let message: String = envelope_field(&envelope, "message")?;
        let exception_type: String = envelope_field(&envelope, "exceptionType")?;
        return Ok(Err(FlowError::CallbackException {
            message: format!("{error_prefix}: {message}"),
            exception_type,
            source: Some(Arc::new(napi::Error::from_unknown_without_coercion(
                exception,
            ))),
        }));
    }
    let error: String = envelope_field(&envelope, "error")?;
    Ok(Err(FlowError::Internal(format!("{error_prefix}: {error}"))))
}

pub(crate) fn unwrap_envelope(envelope: Unknown, error_prefix: &str) -> Result<Json> {
    decode_envelope(envelope, error_prefix).unwrap_or_else(|error| {
        Err(FlowError::Internal(format!(
            "{error_prefix}: invalid middleware callback result: {error}"
        )))
    })
}

pub(crate) fn tsfn_envelope(
    env: &Env,
    value: napi::Result<JsArg>,
    error_prefix: &str,
) -> Result<Json> {
    match value {
        Ok(value) => unwrap_envelope(unsafe { value.into_unknown(env) }, error_prefix),
        Err(error) => Err(FlowError::Internal(format!("{error_prefix}: {error}"))),
    }
}

async fn await_callback_result(
    rx: tokio::sync::oneshot::Receiver<Result<Json>>,
    error_prefix: &str,
) -> Result<Json> {
    rx.await
        .map_err(|error| FlowError::Internal(format!("{error_prefix}: {error}")))?
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
                    let mut args = env.create_array(2)?;
                    let request = js_value(env, request)?;
                    args.set(0, request)?;
                    args.set(1, js_llm_sanitize_request_context_to_napi(env, context)?)?;
                    Ok(detach(env, &args))
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
                let mut args = env.create_array(2)?;
                let response = js_value(env, response)?;
                args.set(0, response)?;
                args.set(1, js_llm_sanitize_response_context_to_napi(env, context)?)?;
                Ok(detach(env, &args))
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

fn recv_json_result(
    rx: std::sync::mpsc::Receiver<Result<Json>>,
    error_prefix: &str,
) -> Result<Json> {
    rx.recv()
        .map_err(|e| FlowError::Internal(format!("{error_prefix}: {e}")))?
}

/// Wrap a JS function `(name: string, args: object) => object` for tool sanitize/intercept.
pub fn wrap_js_tool_fn(func: Tsfn<(String, Json), JsArgs2, JsArg>) -> ToolSanitizeFn {
    let func = Arc::new(func);
    Arc::new(move |name: String, args: Json| {
        let func = func.clone();
        Box::pin(async move {
            let (tx, rx) = tokio::sync::oneshot::channel();
            let status = func.call_with_return_value(
                (name, args),
                ThreadsafeFunctionCallMode::NonBlocking,
                move |value: napi::Result<JsArg>, env| {
                    let _ = tx.send(tsfn_envelope(
                        &env,
                        value,
                        "nemo_relay: JS tool callback failed",
                    ));
                    Ok(())
                },
            );
            if status != napi::Status::Ok {
                return Err(FlowError::Internal(format!(
                    "failed to queue JS tool callback: {status:?}"
                )));
            }
            await_callback_result(rx, "nemo_relay: JS tool callback failed").await
        })
    })
}

pub(crate) fn parse_tool_execution_result(value: Json) -> Result<ToolExecutionResult> {
    serde_json::from_value(value).map_err(|error| {
        FlowError::InvalidArgument(format!(
            "tool execution callback must return ToolExecutionResult; return {{ result: payload }}, not a raw object: {error}"
        ))
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

fn request_codec_object(env: &Env, codec: Arc<dyn LlmCodec>) -> napi::Result<Object<'_>> {
    let mut object = Object::new(env)?;
    let decode_codec = codec.clone();
    let decode = closure_function(env, "decode", move |ctx| {
        let request = ctx.get::<Json>(0)?;
        let request = serde_json::from_value(request)
            .map_err(|error| Error::invalid_argument(format!("invalid LlmRequest: {error}")))?;
        serde_json::to_value(decode_codec.decode(&request).map_err(Error::from)?)
            .map_err(Error::from)
    })?;
    let encode_codec = codec;
    let encode = closure_function(env, "encode", move |ctx| {
        let annotated = ctx.get::<Json>(0)?;
        let original = ctx.get::<Json>(1)?;
        let annotated = serde_json::from_value(annotated).map_err(|error| {
            Error::invalid_argument(format!("invalid AnnotatedLlmRequest: {error}"))
        })?;
        let original = serde_json::from_value(original)
            .map_err(|error| Error::invalid_argument(format!("invalid LlmRequest: {error}")))?;
        serde_json::to_value(
            encode_codec
                .encode(&annotated, &original)
                .map_err(Error::from)?,
        )
        .map_err(Error::from)
    })?;
    object.set_named_property("decode", decode)?;
    object.set_named_property("encode", encode)?;
    Ok(object)
}

fn response_codec_object(env: &Env, codec: Arc<dyn LlmResponseCodec>) -> napi::Result<Object<'_>> {
    let mut object = Object::new(env)?;
    let decode = closure_function(env, "decodeResponse", move |ctx| {
        let response = ctx.get::<Json>(0)?;
        serde_json::to_value(codec.decode_response(&response).map_err(Error::from)?)
            .map_err(Error::from)
    })?;
    object.set_named_property("decodeResponse", decode)?;
    Ok(object)
}

/// Convert a request sanitizer context into the JavaScript object passed to a callback.
pub(crate) fn js_llm_sanitize_request_context_to_napi(
    env: &Env,
    context: JsLlmSanitizeRequestContext,
) -> napi::Result<JsArg> {
    let mut object = Object::new(env)?;
    let codec = js_value(env, context.codec)?;
    object.set_named_property("codec", codec)?;

    let resolved = context.resolved;
    let resolve_codec = closure_function(env, "resolveCodec", move |ctx| match resolved.clone() {
        Some(codec) => Ok(detach(ctx.env, &request_codec_object(ctx.env, codec)?)),
        None => Ok(js_value(ctx.env, Null)?),
    })?;
    object.set_named_property("resolveCodec", resolve_codec)?;
    Ok(detach(env, &object))
}

/// Convert a response sanitizer context into the JavaScript object passed to a callback.
pub(crate) fn js_llm_sanitize_response_context_to_napi(
    env: &Env,
    context: JsLlmSanitizeResponseContext,
) -> napi::Result<JsArg> {
    let mut object = Object::new(env)?;
    let codec = js_value(env, context.codec)?;
    object.set_named_property("codec", codec)?;

    let resolved = context.resolved;
    let resolve_codec = closure_function(env, "resolveCodec", move |ctx| match resolved.clone() {
        Some(codec) => Ok(detach(ctx.env, &response_codec_object(ctx.env, codec)?)),
        None => Ok(js_value(ctx.env, Null)?),
    })?;
    object.set_named_property("resolveCodec", resolve_codec)?;
    Ok(detach(env, &object))
}

fn js_llm_execution_context_args(
    request: LlmRequest,
    context: LlmExecutionContext,
) -> crate::promise_call::Arg0Builder {
    Box::new(move |env| {
        let mut args = env.create_array(2)?;
        let request = serde_json::to_value(request)
            .map_err(|error| napi::Error::from_reason(error.to_string()))?;
        args.set(0, js_value(env, request)?)?;

        let mut execution = Object::new(env)?;
        execution.set_named_property(
            "requestCodec",
            js_llm_sanitize_request_context_to_napi(
                env,
                js_llm_sanitize_request_context(context.request_codec()),
            )?,
        )?;
        let response = match context.response_codec() {
            Some(context) => js_llm_sanitize_response_context_to_napi(
                env,
                js_llm_sanitize_response_context(context),
            )?,
            None => js_value(env, Null)?,
        };
        execution.set_named_property("responseCodec", response)?;
        args.set(1, detach(env, &execution))?;
        Ok(detach(env, &args))
    })
}

thread_local! {
    static DIRECT_STREAM_CALLBACKS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Depth guard for stream polls on the JavaScript thread, where stream callbacks run directly.
pub(crate) struct DirectStreamCallbacks;

impl DirectStreamCallbacks {
    pub(crate) fn enter() -> Self {
        DIRECT_STREAM_CALLBACKS.with(|depth| depth.set(depth.get() + 1));
        Self
    }

    fn active() -> bool {
        DIRECT_STREAM_CALLBACKS.with(|depth| depth.get() > 0)
    }
}

impl Drop for DirectStreamCallbacks {
    fn drop(&mut self) {
        DIRECT_STREAM_CALLBACKS.with(|depth| depth.set(depth.get() - 1));
    }
}

/// Reply slot for `complete(envelope)`. Calls on one stream callback are sequential, so one
/// slot serves both paths.
type CompletionSlot = Arc<Mutex<Option<tokio::sync::oneshot::Sender<Result<Json>>>>>;

pub(crate) struct JsCollectorCall {
    chunk: Json,
}

pub(crate) struct JsFinalizerCall;

/// Build the wrapper arguments from a call, its completion function, and the direct flag.
type StreamCallArgs<T, Args> = fn(&Env, T, JsArg, JsArg) -> napi::Result<Args>;

/// JS stream callback queued through a threadsafe function from other threads
/// and called directly from a stream poll on its own thread.
///
/// The last argument is `true` for a direct call and `false` for a queued call. A direct
/// call returns an envelope when the callback settles synchronously and `undefined` when it
/// settles through `complete`.
pub(crate) struct JsStreamCallback<T: 'static, Args: 'static + JsValuesTupleIntoVec> {
    tsfn: Tsfn<T, Args, JsArg>,
    teardown: Arc<AtomicBool>,
    function: PersistentJsFunction<Args, JsArg>,
    complete: Arc<PersistentJsFunction<JsArg, ()>>,
    pending: CompletionSlot,
    args: StreamCallArgs<T, Args>,
    thread: std::thread::ThreadId,
    direct: bool,
    error_prefix: &'static str,
}

fn completion_function(
    env: &Env,
    name: &str,
    pending: &CompletionSlot,
    error_prefix: &'static str,
) -> napi::Result<Arc<PersistentJsFunction<JsArg, ()>>> {
    let pending = Arc::clone(pending);
    let callback = closure_function(env, name, move |ctx| {
        let envelope = unwrap_envelope(ctx.get::<Unknown>(0)?, error_prefix);
        if let Some(sender) = pending
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
        {
            let _ = sender.send(envelope);
        }
        Ok(())
    })?;
    Ok(Arc::new(PersistentJsFunction::new(env, &callback)?))
}

impl<T: 'static, Args: 'static + JsValuesTupleIntoVec> JsStreamCallback<T, Args> {
    fn new(
        env: &Env,
        wrapper: Function<Args, JsArg>,
        completion_name: &str,
        error_prefix: &'static str,
        args: StreamCallArgs<T, Args>,
    ) -> napi::Result<Self> {
        let pending = CompletionSlot::default();
        let complete = completion_function(env, completion_name, &pending, error_prefix)?;
        let queued_complete = Arc::clone(&complete);
        // An abandoned stream must not keep the Node event loop alive.
        let tsfn = wrapper
            .build_threadsafe_function::<T>()
            .weak::<true>()
            .build_callback(move |ctx: ThreadsafeCallContext<T>| {
                let complete = queued_complete.function()?;
                let complete = detach(&ctx.env, &complete);
                args(&ctx.env, ctx.value, complete, js_value(&ctx.env, false)?)
            })?;
        Ok(Self {
            tsfn,
            teardown: teardown_flag(env)?,
            function: PersistentJsFunction::new(env, &wrapper)?,
            complete,
            pending,
            args,
            thread: std::thread::current().id(),
            direct: true,
            error_prefix,
        })
    }

    fn queue(&self, call: T, name: &str) -> Result<()> {
        if self.teardown.load(Ordering::Acquire) {
            return Err(FlowError::Internal(format!(
                "JS {name} callback skipped: Node environment is shutting down"
            )));
        }
        let status = self
            .tsfn
            .call(call, ThreadsafeFunctionCallMode::NonBlocking);
        if status == napi::Status::Ok {
            return Ok(());
        }
        let message = format!("nemo_relay: failed to queue JS {name} callback: {status:?}");
        record_callback_error(message.clone());
        Err(FlowError::Internal(message))
    }

    pub(crate) fn without_direct(mut self) -> Self {
        self.direct = false;
        self
    }

    pub(crate) fn abort_tsfn(&self) -> napi::Result<()> {
        abort_tsfn(&self.tsfn)
    }

    fn teardown(&self) -> Arc<AtomicBool> {
        self.teardown.clone()
    }

    fn calls_directly(&self) -> bool {
        self.direct && DirectStreamCallbacks::active() && std::thread::current().id() == self.thread
    }

    fn call_directly(&self, call: T) -> napi::Result<()> {
        let env = self.function.env();
        let complete = self.complete.function()?;
        let args = (self.args)(&env, call, detach(&env, &complete), js_value(&env, true)?)?;
        let envelope = self.function.function()?.call(args)?;
        let envelope = unsafe { envelope.into_unknown(&env) };
        if envelope.get_type()? != ValueType::Undefined
            && let Some(sender) = self
                .pending
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .take()
        {
            let _ = sender.send(unwrap_envelope(envelope, self.error_prefix));
        }
        Ok(())
    }

    fn invoke(
        &self,
        call: T,
        sender: tokio::sync::oneshot::Sender<Result<Json>>,
        name: &str,
    ) -> Result<()> {
        *self
            .pending
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(sender);
        if self.calls_directly() {
            return self.call_directly(call).map_err(|error| {
                let message = format!("nemo_relay: JS {name} callback failed: {error}");
                record_callback_error(message.clone());
                FlowError::Internal(message)
            });
        }
        self.queue(call, name)
    }
}

pub(crate) fn js_collector_tsfn(
    env: &Env,
    func: &Function<JsArg, JsArg>,
) -> napi::Result<JsStreamCallback<JsCollectorCall, JsArgs3>> {
    let wrapper = callback_factory::wrap_collector_callback(env, func)?;
    JsStreamCallback::new(
        env,
        wrapper,
        "__nemo_relay_collector_complete",
        "JS collector callback failed",
        |env, JsCollectorCall { chunk }, complete, direct| {
            Ok(FnArgs::from((js_value(env, chunk)?, complete, direct)))
        },
    )
}

pub(crate) fn js_finalizer_tsfn(
    env: &Env,
    func: &Function<JsArg, JsArg>,
) -> napi::Result<JsStreamCallback<JsFinalizerCall, JsArgs2>> {
    let wrapper = callback_factory::wrap_finalizer_callback(env, func)?;
    JsStreamCallback::new(
        env,
        wrapper,
        "__nemo_relay_finalizer_complete",
        "nemo_relay: JS finalizer callback failed",
        |_env, JsFinalizerCall, complete, direct| Ok(FnArgs::from((complete, direct))),
    )
}

/// Wrap a JS collector `(chunk: object) => void | Promise<void>` for a streaming LLM call.
///
/// If the collector throws or rejects, the stream terminates with that error.
pub fn wrap_js_collector_fn(callback: JsStreamCallback<JsCollectorCall, JsArgs3>) -> LlmCollector {
    LlmCollector::Async(Box::new(move |chunk: Json| {
        let (sender, reply) = tokio::sync::oneshot::channel();
        let queued = callback.invoke(JsCollectorCall { chunk }, sender, "collector");
        Box::pin(async move {
            queued?;
            await_callback_result(reply, "JS collector callback failed")
                .await
                .map(|_| ())
        })
    }))
}

/// Finalizer failure shared with the stream handle so `close()` can report it.
pub(crate) type FinalizerError = Arc<std::sync::Mutex<Option<Error>>>;

/// Wrap a JS finalizer `() => object | Promise<object>` for a streaming LLM call.
///
/// If the finalizer throws, rejects, or returns a non-JSON value, the failure
/// is returned, stored in `error`, and the END event carries no aggregate.
pub fn wrap_js_finalizer_fn(
    callback: JsStreamCallback<JsFinalizerCall, JsArgs2>,
    error: FinalizerError,
) -> LlmFinalizer {
    LlmFinalizer::Async(Box::new(move || {
        let (sender, reply) = tokio::sync::oneshot::channel();
        let queued = callback.invoke(JsFinalizerCall, sender, "finalizer");
        let teardown = callback.teardown();
        Box::pin(async move {
            let result = match queued {
                Ok(()) => {
                    await_callback_result(reply, "nemo_relay: JS finalizer callback failed").await
                }
                Err(failure) => Err(failure),
            };
            result.inspect_err(|failure| {
                if !teardown.load(Ordering::Acquire) {
                    record_callback_error(failure.to_string());
                }
                *error
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()) =
                    Some(Error::from(failure.clone()));
            })
        })
    }))
}

struct JsSubscriberCallbackCall {
    event: Json,
    callback_id: u64,
}

/// Wrap a JS function for event subscriber: `(event: JsEvent) => void | Promise<void>`.
pub fn wrap_js_event_subscriber(
    env: &Env,
    name: String,
    callback: &Function<JsArg, JsArg>,
) -> napi::Result<EventSubscriberFn> {
    let complete = create_js_subscriber_completion_callback(env, name.clone())?;
    let callback: Function<JsArgs2, ()> =
        callback_factory::wrap_subscriber_callback(env, callback, &complete)?;
    let queue_error_name = name;
    let func = Arc::new(create_js_event_subscriber_function(&callback)?);
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
    callback: &Function<JsArgs2, ()>,
) -> napi::Result<HandledTsfn<JsSubscriberCallbackCall, JsArgs2>> {
    callback
        .build_threadsafe_function::<JsSubscriberCallbackCall>()
        .callee_handled::<true>()
        .weak::<true>()
        .build_callback(|ctx: ThreadsafeCallContext<JsSubscriberCallbackCall>| {
            let JsSubscriberCallbackCall { event, callback_id } = ctx.value;
            let args = js_value(&ctx.env, event).and_then(|event| {
                Ok(FnArgs::from((
                    event,
                    js_value(&ctx.env, callback_id as f64)?,
                )))
            });
            args.inspect_err(|_| complete_js_subscriber_callback(callback_id))
        })
}

/// `(callbackId, message?)`: finishes the callback. A message records a failure.
fn create_js_subscriber_completion_callback<'env>(
    env: &'env Env,
    callback_name: String,
) -> napi::Result<Function<'env, JsArg, ()>> {
    closure_function(
        env,
        "__nemo_relay_complete_subscriber_callback",
        move |ctx| {
            let callback_id = ctx.get::<f64>(0)? as u64;
            record_js_subscriber_callback_error(&ctx, &callback_name);
            complete_js_subscriber_callback(callback_id);
            Ok(())
        },
    )
}

fn record_js_subscriber_callback_error(ctx: &FunctionCallContext, callback_name: &str) {
    if ctx.length() < 2 {
        return;
    }
    match ctx.get::<String>(1) {
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

/// Argument for a JavaScript `codecEncode` callback.
#[derive(Serialize)]
struct JsCodecEncodeArgument<'a> {
    annotated: &'a AnnotatedLlmRequest,
    original: &'a LlmRequest,
}

/// A JavaScript codec callback that any thread can call.
///
/// A throw or an unreadable return value comes back as an error.
pub(crate) struct JsCodecCallback {
    direct: PersistentJsFunction<Json, JsArg>,
    queued: Tsfn<Json, Json, JsArg>,
    registration_thread: std::thread::ThreadId,
    error_prefix: &'static str,
}

impl JsCodecCallback {
    pub(crate) fn new(
        env: &Env,
        func: &Function<JsArg, JsArg>,
        error_prefix: &'static str,
    ) -> napi::Result<Self> {
        let wrapper: Function<Json, JsArg> = callback_factory::wrap_execution_callback(env, func)?;
        let direct = PersistentJsFunction::new(env, &wrapper)?;
        let queued = wrapper
            .build_threadsafe_function::<Json>()
            .weak::<true>()
            .build_callback(|ctx: ThreadsafeCallContext<Json>| Ok(ctx.value))?;
        Ok(Self {
            direct,
            queued,
            registration_thread: std::thread::current().id(),
            error_prefix,
        })
    }

    fn call<A: Serialize, T: DeserializeOwned>(&self, argument: &A, expected: &str) -> Result<T> {
        serde_json::to_value(argument)
            .map_err(|error| {
                FlowError::Internal(format!(
                    "{}: could not serialize the argument: {error}",
                    self.error_prefix
                ))
            })
            .and_then(|argument| self.invoke(argument))
            .and_then(|value| {
                serde_json::from_value(value).map_err(|error| {
                    FlowError::Internal(format!(
                        "{}: expected {expected}: {error}",
                        self.error_prefix
                    ))
                })
            })
            .inspect_err(|error| record_callback_error(error.to_string()))
    }

    fn invoke(&self, argument: Json) -> Result<Json> {
        if std::thread::current().id() == self.registration_thread {
            let env = self.direct.env();
            let envelope = self
                .direct
                .function()
                .and_then(|function| function.call(argument))
                .map_err(|error| FlowError::Internal(format!("{}: {error}", self.error_prefix)))?;
            return unwrap_envelope(unsafe { envelope.into_unknown(&env) }, self.error_prefix);
        }
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let error_prefix = self.error_prefix;
        let status = self.queued.call_with_return_value(
            argument,
            ThreadsafeFunctionCallMode::NonBlocking,
            move |value: napi::Result<JsArg>, env| {
                let _ = tx.send(tsfn_envelope(&env, value, error_prefix));
                Ok(())
            },
        );
        if status != napi::Status::Ok {
            return Err(FlowError::Internal(format!(
                "{}: could not queue the callback: {status:?}",
                self.error_prefix
            )));
        }
        recv_json_result(rx, self.error_prefix)
    }
}

/// Request codec backed by JavaScript `decode` and `encode` callbacks.
struct NapiCodec {
    decode: JsCodecCallback,
    encode: JsCodecCallback,
}

impl LlmCodec for NapiCodec {
    fn decode(&self, request: &LlmRequest) -> Result<AnnotatedLlmRequest> {
        self.decode.call(request, "AnnotatedLlmRequest")
    }

    fn encode(&self, annotated: &AnnotatedLlmRequest, original: &LlmRequest) -> Result<LlmRequest> {
        self.encode.call(
            &JsCodecEncodeArgument {
                annotated,
                original,
            },
            "LlmRequest",
        )
    }
}

/// Build a request codec from JavaScript `decode` and `encode` callbacks.
pub(crate) fn wrap_js_codec(decode: JsCodecCallback, encode: JsCodecCallback) -> Arc<dyn LlmCodec> {
    Arc::new(NapiCodec { decode, encode })
}

// ---------------------------------------------------------------------------
// Response codec wrapper
// ---------------------------------------------------------------------------

/// Response codec backed by a JavaScript `decodeResponse` callback.
struct NapiResponseCodec {
    decode_response: JsCodecCallback,
}

impl LlmResponseCodec for NapiResponseCodec {
    fn decode_response(&self, response: &Json) -> Result<AnnotatedLlmResponse> {
        self.decode_response.call(response, "AnnotatedLlmResponse")
    }
}

/// Build a response codec from a JavaScript `decodeResponse` callback.
pub(crate) fn wrap_js_response_codec(
    decode_response: JsCodecCallback,
) -> Arc<dyn LlmResponseCodec> {
    Arc::new(NapiResponseCodec { decode_response })
}

/// Wrap a JS function `(context, next) => outcome` for tool execution intercepts.
///
/// The JS callback receives a plain object carrying `toolName`, `args`,
/// and `toolCallId`, so it can correlate a result it produces itself with the
/// originating tool call.
pub fn wrap_js_tool_exec_intercept_fn(
    func: Arc<PromiseAwareFn>,
) -> nemo_relay::api::runtime::ToolExecutionFn {
    Arc::new(
        move |context: ToolExecutionContext, next: ToolExecutionNextFn| {
            let func = func.clone();
            let js_context = serde_json::json!({
                "toolName": context.tool_name(),
                "args": context.args(),
                "toolCallId": context.tool_call_id(),
            });
            Box::pin(call_js_tool_exec_intercept(func, js_context, next))
        },
    )
}

/// Invoke a JS tool execution intercept and decode its outcome.
///
/// `primary` is the serialized tool execution context.
async fn call_js_tool_exec_intercept(
    func: Arc<PromiseAwareFn>,
    primary: Json,
    next: ToolExecutionNextFn,
) -> Result<ToolExecutionInterceptOutcome> {
    let next_json: JsonNextFn = Arc::new(move |next_args| {
        let next = next.clone();
        Box::pin(async move {
            serde_json::to_value(next(next_args).await?)
                .map_err(|error| FlowError::Internal(error.to_string()))
        })
    });
    let result = func.call_with_json_next(primary, next_json).await?;
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
        FlowError::InvalidArgument(format!(
            "tool execution intercept must return {{ result, annotation?, pendingMarks? }}; return {{ result: downstream.result, annotation: downstream.annotation }} after next(args): {error}"
        ))
    })?;
    Ok(ToolExecutionInterceptOutcome {
        result: outcome.result,
        annotation: outcome
            .annotation
            .filter(|annotation| !annotation.is_null()),
        pending_marks: outcome.pending_marks.into_iter().map(Into::into).collect(),
    })
}

/// Wrap a JS function `(request, context, next) => result` for LLM execution intercept.
///
/// The JS callback receives the `LlmRequest` serialized as a plain JSON object
/// and a real `next(request)` function that returns a Promise for the downstream
/// result.
pub fn wrap_js_llm_exec_intercept_fn(func: Arc<PromiseAwareFn>) -> LlmExecutionFn {
    Arc::new(
        move |_name: &str,
              request: LlmRequest,
              context: LlmExecutionContext,
              next: LlmExecutionNextFn| {
            let func = func.clone();
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
            let args = js_llm_execution_context_args(request, context);
            Box::pin(async move {
                func.call_spread_with_arg0_and_json_next(args, next_json)
                    .await
            })
        },
    )
}

/// Wrap a JS `(request: object, signal: AbortSignal) => AsyncIterable` streaming LLM provider.
///
/// Each call registers its abort in `cancellation`.
pub fn wrap_js_llm_stream_fn(
    func: Arc<PromiseAwareFn>,
    cancellation: StreamCancellation,
) -> LlmStreamExecutionNextFn {
    Arc::new(move |request: LlmRequest| {
        let func = func.clone();
        let cancellation = cancellation.clone();
        Box::pin(async move {
            let request = serde_json::to_value(&request).map_err(|error| {
                FlowError::Internal(format!("failed to serialize LlmRequest: {error}"))
            })?;
            func.call_stream(request, &cancellation).await
        })
    })
}

/// Wrap a JS function `(request, context, next) => result` for LLM stream execution intercept.
///
/// The JS callback receives the `LlmRequest` serialized as a plain JSON object
/// and a real `next(request)` function whose Promise resolves to a lazy
/// async iterable. The callback can return it directly or wrap it with an async
/// generator without materializing the downstream stream.
pub fn wrap_js_llm_stream_exec_intercept_fn(func: Arc<PromiseAwareFn>) -> LlmStreamExecutionFn {
    Arc::new(
        move |_name: &str,
              request: LlmRequest,
              context: LlmExecutionContext,
              next: LlmStreamExecutionNextFn| {
            let func = func.clone();
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
            let args = js_llm_execution_context_args(request, context);
            Box::pin(async move {
                func.call_spread_with_arg0_and_stream_next(args, next_stream)
                    .await
            })
        },
    )
}
