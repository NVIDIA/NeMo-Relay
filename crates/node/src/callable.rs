// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::RecvTimeoutError;
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::Duration;

use napi::bindgen_prelude::{
    FnArgs, FromNapiValue, Function, JsObjectValue, JsValuesTupleIntoVec, Null, Object, Unknown,
};
use napi::threadsafe_function::{ThreadsafeCallContext, ThreadsafeFunctionCallMode};
use napi::{Env, ValueType};
use nemo_relay_runtime::api::runtime::subscriber_dispatcher::in_dispatcher_callback;
use nemo_relay_runtime::api::runtime::{
    EventMetadataInjectorFn, EventSanitizeFn, EventSubscriberFn, LlmCodecIdentity, LlmCollector,
    LlmConditionalFn, LlmExecutionContext, LlmExecutionFn, LlmExecutionNextFn, LlmFinalizer,
    LlmRequestInterceptFn, LlmSanitizeRequestContext, LlmSanitizeRequestFn,
    LlmSanitizeResponseContext, LlmSanitizeResponseFn, LlmStreamExecutionFn,
    LlmStreamExecutionNextFn, ToolConditionalFn, ToolExecutionContext, ToolExecutionFn,
    ToolExecutionNextFn, ToolInterceptFn, ToolSanitizeFn,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;

use nemo_relay_runtime::api::event::{
    CategoryProfile, DataSchema, Event, EventCategory,
    EventSanitizeFields as CoreEventSanitizeFields, LogSeverity, PendingMarkSpec,
};
use nemo_relay_runtime::api::llm::{LlmRequest, LlmRequestInterceptOutcome};
use nemo_relay_runtime::api::tool::{ToolExecutionInterceptOutcome, ToolExecutionResult};
use nemo_relay_types::codec::optimization::LlmOptimizationContribution;
use nemo_relay_types::codec::request::AnnotatedLlmRequest;
use nemo_relay_types::codec::response::AnnotatedLlmResponse;
use nemo_relay_types::codec::traits::{LlmCodec, LlmResponseCodec};
use nemo_relay_types::error::{FlowError, Result};

use crate::callback_factory::{self, Wrapper};
use crate::convert::{
    BoxFuture, HandledTsfn, JsArg, JsArgs2, JsArgs3, PersistentJsFunction, abort_tsfn,
    closure_function, detach, js_value, lock, record_callback_error, report_queue_status,
};
use crate::environment::{ENV_SHUTTING_DOWN, teardown_flag};
use crate::error::Error;
use crate::promise_call::{Invocation, JsonNextFn, PromiseAwareFn, StreamCancellation};
use crate::types::{EventSanitizeFields, event_sanitize_fields_from_json, mirror_fields};

struct JsSubscriberCallbackState {
    next_id: u64,
    pending: BTreeSet<u64>,
}

static JS_SUBSCRIBER_CALLBACKS: (Mutex<JsSubscriberCallbackState>, Condvar) = (
    Mutex::new(JsSubscriberCallbackState {
        next_id: 0,
        pending: BTreeSet::new(),
    }),
    Condvar::new(),
);

fn reserve_js_subscriber_callback() -> u64 {
    let (state, _) = &JS_SUBSCRIBER_CALLBACKS;
    let mut state = lock(state);
    state.next_id += 1;
    let id = state.next_id;
    state.pending.insert(id);
    id
}

fn complete_js_subscriber_callback(id: u64) {
    let (state, completed) = &JS_SUBSCRIBER_CALLBACKS;
    let mut state = lock(state);
    state.pending.remove(&id);
    completed.notify_all();
}

pub(crate) fn flush_js_subscriber_callbacks() {
    let (state, completed) = &JS_SUBSCRIBER_CALLBACKS;
    let mut state = lock(state);
    let watermark = state.next_id;
    while state.pending.range(..=watermark).next().is_some() {
        state = completed
            .wait(state)
            .unwrap_or_else(PoisonError::into_inner);
    }
}

#[derive(Serialize)]
struct JsLlmCodecIdentity {
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
}

trait CodecObject: 'static {
    fn object<'env>(self: Arc<Self>, env: &'env Env) -> napi::Result<Object<'env>>;
}

struct JsLlmCodecContext<C: ?Sized> {
    codec: JsLlmCodecIdentity,
    resolved: Option<Arc<C>>,
}

#[derive(Deserialize, Serialize)]
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

mirror_fields!(JsPendingMarkSpec <=> PendingMarkSpec {
    name, category, category_profile, data, data_schema, metadata, severity,
});

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
        // SAFETY: `value` was returned by the threadsafe call settling now, in `env`.
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

pub(crate) trait ReplySender: Send + 'static {
    fn deliver(self, result: Result<Json>);
}

impl ReplySender for tokio::sync::oneshot::Sender<Result<Json>> {
    fn deliver(self, result: Result<Json>) {
        let _ = self.send(result);
    }
}

impl ReplySender for std::sync::mpsc::SyncSender<Result<Json>> {
    fn deliver(self, result: Result<Json>) {
        let _ = self.send(result);
    }
}

pub(crate) struct Reply<S>(Arc<Mutex<Option<S>>>);

impl<S> Clone for Reply<S> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<S: ReplySender> Reply<S> {
    fn empty() -> Self {
        Self(Arc::default())
    }

    fn new(sender: S) -> Self {
        Self(Arc::new(Mutex::new(Some(sender))))
    }

    fn arm(&self, sender: S) {
        *lock(&self.0) = Some(sender);
    }

    fn deliver(&self, result: Result<Json>) {
        if let Some(sender) = lock(&self.0).take() {
            sender.deliver(result);
        }
    }
}

pub(crate) type AsyncReply = Reply<tokio::sync::oneshot::Sender<Result<Json>>>;

pub(crate) trait JsCall: Send + 'static {
    type Args: 'static + JsValuesTupleIntoVec;

    fn args(self, env: &Env) -> napi::Result<Self::Args>;
}

impl JsCall for Json {
    type Args = JsArg;

    fn args(self, env: &Env) -> napi::Result<JsArg> {
        js_value(env, self)
    }
}

pub(crate) struct Queued<C, S> {
    call: C,
    reply: Reply<S>,
}

pub(crate) struct ReplyTsfn<C: JsCall, S: 'static> {
    tsfn: HandledTsfn<Queued<C, S>, C::Args, JsArg>,
    teardown: Arc<AtomicBool>,
}

pub(crate) type AsyncReplyTsfn<C> = ReplyTsfn<C, tokio::sync::oneshot::Sender<Result<Json>>>;

pub(crate) fn reply_tsfn<C: JsCall, S: ReplySender>(
    env: &Env,
    function: &Function<'_, C::Args, JsArg>,
    error_prefix: &'static str,
) -> napi::Result<ReplyTsfn<C, S>> {
    let teardown = teardown_flag(env)?;
    let dispatch_teardown = Arc::clone(&teardown);
    let tsfn = callback_factory::handled_tsfn(
        env,
        function,
        move |ctx: ThreadsafeCallContext<Queued<C, S>>| {
            let Queued { call, reply } = ctx.value;
            if dispatch_teardown.load(Ordering::Acquire) {
                reply.deliver(Err(FlowError::Internal(format!(
                    "{error_prefix}: {ENV_SHUTTING_DOWN}"
                ))));
                return Err(napi::Error::from_reason(ENV_SHUTTING_DOWN));
            }
            call.args(&ctx.env).inspect_err(|error| {
                reply.deliver(Err(FlowError::Internal(format!("{error_prefix}: {error}"))));
            })
        },
    )?;
    Ok(ReplyTsfn { tsfn, teardown })
}

fn queue_reply<C: JsCall, S: ReplySender>(
    tsfn: &ReplyTsfn<C, S>,
    call: C,
    sender: S,
    error_prefix: &'static str,
) -> Result<()> {
    if tsfn.teardown.load(Ordering::Acquire) {
        return Err(FlowError::Internal(format!(
            "{error_prefix}: could not queue the callback: {ENV_SHUTTING_DOWN}"
        )));
    }
    let reply = Reply::new(sender);
    let settle = reply.clone();
    let status = tsfn.tsfn.call_with_return_value(
        Ok(Queued { call, reply }),
        ThreadsafeFunctionCallMode::NonBlocking,
        move |value: napi::Result<JsArg>, env| {
            settle.deliver(tsfn_envelope(&env, value, error_prefix));
            Ok(())
        },
    );
    if status == napi::Status::Ok {
        return Ok(());
    }
    Err(FlowError::Internal(format!(
        "{error_prefix}: could not queue the callback: {status:?}"
    )))
}

pub(crate) async fn call_tsfn<C: JsCall>(
    tsfn: &AsyncReplyTsfn<C>,
    call: C,
    error_prefix: &'static str,
) -> Result<Json> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    queue_reply(tsfn, call, tx, error_prefix)?;
    rx.await.map_err(|_| {
        FlowError::Internal(format!("{error_prefix}: callback channel disconnected"))
    })?
}

fn recorded_internal(message: String) -> FlowError {
    let error = FlowError::Internal(message);
    record_callback_error(error.to_string());
    error
}

async fn recorded(call: impl Future<Output = Result<Json>>) -> Result<Json> {
    call.await
        .inspect_err(|error| record_callback_error(error.to_string()))
}

fn serialized<T: Serialize>(value: &T, what: &str) -> Result<Json> {
    serde_json::to_value(value)
        .map_err(|error| recorded_internal(format!("failed to serialize {what}: {error}")))
}

fn deserialized<T: DeserializeOwned>(value: Json, what: &str) -> Result<T> {
    serde_json::from_value(value)
        .map_err(|error| recorded_internal(format!("invalid {what}: {error}")))
}

fn conditional_outcome(value: Json, what: &str) -> Result<Option<String>> {
    match value {
        Json::Null => Ok(None),
        Json::String(reason) => Ok(Some(reason)),
        other => Err(recorded_internal(format!(
            "JS {what} conditional callback failed: expected string or null, got {other:?}"
        ))),
    }
}

pub(crate) fn wrap_js_tool_conditional_promise_fn(
    env: &Env,
    func: &Function<JsArg, JsArg>,
) -> napi::Result<ToolConditionalFn> {
    let func = Arc::new(PromiseAwareFn::new(env, func)?);
    Ok(Arc::new(move |name: String, args: Json| {
        let func = func.clone();
        Box::pin(async move {
            let value =
                recorded(func.call(Invocation::spread(vec![Json::String(name), args]))).await?;
            conditional_outcome(value, "tool")
        })
    }))
}

pub(crate) fn wrap_js_tool_request_intercept_promise_fn(
    env: &Env,
    func: &Function<JsArg, JsArg>,
) -> napi::Result<ToolInterceptFn> {
    let func = Arc::new(PromiseAwareFn::new(env, func)?);
    Ok(Arc::new(move |name: String, args: Json| {
        let func = func.clone();
        Box::pin(async move {
            recorded(func.call(Invocation::spread(vec![Json::String(name), args]))).await
        })
    }))
}

pub(crate) fn wrap_js_tool_sanitize_promise_fn(
    env: &Env,
    func: &Function<JsArg, JsArg>,
) -> napi::Result<ToolSanitizeFn> {
    let func = Arc::new(PromiseAwareFn::new(env, func)?);
    Ok(Arc::new(move |name: String, value: Json| {
        let func = func.clone();
        let publication = in_dispatcher_callback();
        Box::pin(async move {
            recorded(
                func.call(
                    Invocation::spread(vec![Json::String(name), value]).publication(publication),
                ),
            )
            .await
        })
    }))
}

pub(crate) fn wrap_js_llm_sanitize_request_promise_fn(
    env: &Env,
    func: &Function<JsArg, JsArg>,
) -> napi::Result<LlmSanitizeRequestFn> {
    let func = Arc::new(PromiseAwareFn::new(env, func)?);
    Ok(Arc::new(
        move |request: LlmRequest, context: LlmSanitizeRequestContext| {
            let func = func.clone();
            let publication = in_dispatcher_callback();
            Box::pin(async move {
                let request = serialized(&request, "JS LLM sanitize request")?;
                let invocation = sanitize_invocation(
                    request,
                    codec_context(context.codec(), context.resolve_codec()),
                );
                let value = recorded(func.call(invocation.publication(publication))).await?;
                if value.is_null() {
                    Ok(None)
                } else {
                    serde_json::from_value(value).map(Some).map_err(|error| {
                        recorded_internal(format!(
                            "JS LLM sanitize request callback failed: failed to deserialize LlmRequest: {error}"
                        ))
                    })
                }
            })
        },
    ))
}

pub(crate) fn wrap_js_llm_sanitize_response_promise_fn(
    env: &Env,
    func: &Function<JsArg, JsArg>,
) -> napi::Result<LlmSanitizeResponseFn> {
    let func = Arc::new(PromiseAwareFn::new(env, func)?);
    Ok(Arc::new(
        move |response: Json, context: LlmSanitizeResponseContext| {
            let func = func.clone();
            let publication = in_dispatcher_callback();
            Box::pin(async move {
                let invocation = sanitize_invocation(
                    response,
                    codec_context(context.codec(), context.resolve_codec()),
                );
                let value = recorded(func.call(invocation.publication(publication))).await?;
                Ok((!value.is_null()).then_some(value))
            })
        },
    ))
}

pub(crate) fn wrap_js_llm_conditional_promise_fn(
    env: &Env,
    func: &Function<JsArg, JsArg>,
) -> napi::Result<LlmConditionalFn> {
    let func = Arc::new(PromiseAwareFn::new(env, func)?);
    Ok(Arc::new(move |request: LlmRequest| {
        let func = func.clone();
        Box::pin(async move {
            let request = serialized(&request, "JS LLM conditional request")?;
            let value = recorded(func.call(Invocation::single(request))).await?;
            conditional_outcome(value, "LLM")
        })
    }))
}

pub(crate) fn wrap_js_llm_request_intercept_promise_fn(
    env: &Env,
    func: &Function<JsArg, JsArg>,
) -> napi::Result<LlmRequestInterceptFn> {
    let func = Arc::new(PromiseAwareFn::new(env, func)?);
    Ok(Arc::new(
        move |name: String, request: LlmRequest, annotated: Option<AnnotatedLlmRequest>| {
            let func = func.clone();
            Box::pin(async move {
                let value = serde_json::json!({
                    "name": name,
                    "request": serialized(&request, "JS LLM request intercept request")?,
                    "annotated": serialized(&annotated, "JS LLM request intercept annotation")?,
                });
                let value = recorded(func.call(Invocation::single(value))).await?;
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
                let outcome: JsOutcome = deserialized(value, "JS LLM request intercept outcome")?;
                Ok(LlmRequestInterceptOutcome {
                    request: outcome.request,
                    annotated_request: outcome.annotated,
                    pending_marks: outcome.pending_marks.into_iter().map(Into::into).collect(),
                    optimization_contributions: outcome.optimization_contributions,
                })
            })
        },
    ))
}

pub(crate) fn wrap_js_event_sanitize_promise_fn(
    env: &Env,
    func: &Function<JsArg, JsArg>,
) -> napi::Result<EventSanitizeFn> {
    let func = Arc::new(PromiseAwareFn::new(env, func)?);
    Ok(Arc::new(
        move |event: Arc<Event>, fields: CoreEventSanitizeFields| {
            let func = func.clone();
            Box::pin(async move {
                let js_fields = EventSanitizeFields {
                    data: fields.data,
                    category_profile: fields
                        .category_profile
                        .as_ref()
                        .map(|profile| serialized(profile, "JS event sanitizer category profile"))
                        .transpose()?,
                    metadata: fields.metadata,
                };
                let args = vec![
                    serialized(&*event, "JS event sanitizer context")?,
                    serialized(&js_fields, "JS event sanitizer fields")?,
                ];
                let publication = in_dispatcher_callback();
                let value =
                    recorded(func.call(Invocation::spread(args).publication(publication))).await?;
                let fields = event_sanitize_fields_from_json(value).map_err(|error| {
                    recorded_internal(format!("invalid JS event sanitizer result: {error}"))
                })?;
                let category_profile = fields
                    .category_profile
                    .map(|profile| deserialized(profile, "JS event sanitizer result"))
                    .transpose()?;
                Ok(CoreEventSanitizeFields {
                    data: fields.data,
                    category_profile,
                    metadata: fields.metadata,
                })
            })
        },
    ))
}

pub(crate) fn wrap_js_event_metadata_injector_promise_fn(
    env: &Env,
    func: &Function<JsArg, JsArg>,
) -> napi::Result<EventMetadataInjectorFn> {
    let func = Arc::new(PromiseAwareFn::new(env, func)?);
    Ok(Arc::new(move |event: Arc<Event>| {
        let func = func.clone();
        Box::pin(async move {
            let event_json = serialized(&*event, "JavaScript event metadata injector context")?;
            let publication = in_dispatcher_callback();
            let value =
                recorded(func.call(Invocation::spread(vec![event_json]).publication(publication)))
                    .await?;
            deserialized::<BTreeMap<String, Json>>(
                value,
                "JavaScript event metadata injector result",
            )
        })
    }))
}

pub(crate) fn recv_callback_result(
    rx: &std::sync::mpsc::Receiver<Result<Json>>,
    timeout: Option<Duration>,
    error_prefix: &str,
) -> Result<Json> {
    let received = match timeout {
        Some(timeout) => rx.recv_timeout(timeout).map_err(|error| match error {
            RecvTimeoutError::Timeout => FlowError::Internal(format!(
                "{error_prefix}: callback timed out after {} seconds",
                timeout.as_secs()
            )),
            RecvTimeoutError::Disconnected => {
                FlowError::Internal(format!("{error_prefix}: callback channel disconnected"))
            }
        }),
        None => rx.recv().map_err(|_| {
            FlowError::Internal(format!("{error_prefix}: callback channel disconnected"))
        }),
    };
    received?
}

pub(crate) fn parse_tool_execution_result(value: Json) -> Result<ToolExecutionResult> {
    serde_json::from_value(value).map_err(|error| {
        FlowError::InvalidArgument(format!(
            "tool execution callback must return ToolExecutionResult; return {{ result: payload }}, not a raw object: {error}"
        ))
    })
}

fn codec_context<C: ?Sized>(
    identity: &LlmCodecIdentity,
    resolved: Option<Arc<C>>,
) -> JsLlmCodecContext<C> {
    let (kind, id) = match identity {
        LlmCodecIdentity::None => ("none", None),
        LlmCodecIdentity::BuiltIn(codec) => ("builtin", Some(codec.id().into())),
        LlmCodecIdentity::Runtime(id) => ("runtime", Some(id.clone())),
        LlmCodecIdentity::Opaque => ("opaque", None),
    };
    JsLlmCodecContext {
        codec: JsLlmCodecIdentity { kind, id },
        resolved,
    }
}

impl CodecObject for dyn LlmCodec {
    fn object<'env>(self: Arc<Self>, env: &'env Env) -> napi::Result<Object<'env>> {
        let mut object = Object::new(env)?;
        let decode_codec = Arc::clone(&self);
        let decode = closure_function(env, "decode", move |ctx| {
            crate::types::decode_request(&*decode_codec, ctx.get(0)?)
        })?;
        let encode = closure_function(env, "encode", move |ctx| {
            crate::types::encode_request(&*self, ctx.get(0)?, ctx.get(1)?)
        })?;
        object.set_named_property("decode", decode)?;
        object.set_named_property("encode", encode)?;
        Ok(object)
    }
}

impl CodecObject for dyn LlmResponseCodec {
    fn object<'env>(self: Arc<Self>, env: &'env Env) -> napi::Result<Object<'env>> {
        let mut object = Object::new(env)?;
        let decode = closure_function(env, "decodeResponse", move |ctx| {
            let response = crate::convert::checked_json(ctx.env, ctx.get(0)?)?;
            crate::types::decode_response(&*self, &response)
        })?;
        object.set_named_property("decodeResponse", decode)?;
        Ok(object)
    }
}

fn codec_context_to_napi<C: CodecObject + ?Sized>(
    env: &Env,
    context: JsLlmCodecContext<C>,
) -> napi::Result<JsArg> {
    let mut object = Object::new(env)?;
    object.set_named_property("codec", env.to_js_value(&context.codec)?)?;
    let resolved = context.resolved;
    let resolve_codec = closure_function(env, "resolveCodec", move |ctx| match resolved.clone() {
        Some(codec) => Ok(detach(ctx.env, &codec.object(ctx.env)?)),
        None => Ok(js_value(ctx.env, Null)?),
    })?;
    object.set_named_property("resolveCodec", resolve_codec)?;
    Ok(detach(env, &object))
}

fn sanitize_invocation<C: CodecObject + Send + Sync + ?Sized>(
    payload: Json,
    context: JsLlmCodecContext<C>,
) -> Invocation {
    Invocation::spread_built(move |env| {
        let mut args = env.create_array(2)?;
        args.set(0, js_value(env, payload)?)?;
        args.set(1, codec_context_to_napi(env, context)?)?;
        Ok(detach(env, &args))
    })
}

fn js_llm_execution_invocation(request: LlmRequest, context: LlmExecutionContext) -> Invocation {
    Invocation::spread_built(move |env| {
        let mut args = env.create_array(2)?;
        let request = serde_json::to_value(request)
            .map_err(|error| napi::Error::from_reason(error.to_string()))?;
        args.set(0, js_value(env, request)?)?;

        let request_codec = context.request_codec();
        let mut execution = Object::new(env)?;
        execution.set_named_property(
            "requestCodec",
            codec_context_to_napi(
                env,
                codec_context(request_codec.codec(), request_codec.resolve_codec()),
            )?,
        )?;
        let response = match context.response_codec() {
            Some(context) => {
                codec_context_to_napi(env, codec_context(context.codec(), context.resolve_codec()))?
            }
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

pub(crate) trait StreamCall: 'static {
    type Args: JsValuesTupleIntoVec + 'static;
    const WRAPPER: Wrapper;
    const NAME: &'static str;
    const COMPLETION: &'static str;
    const ERROR_PREFIX: &'static str;

    fn args(self, env: &Env, complete: JsArg, direct: JsArg) -> napi::Result<Self::Args>;
}

pub(crate) struct JsCollectorCall {
    chunk: Json,
}

impl StreamCall for JsCollectorCall {
    type Args = JsArgs3;
    const WRAPPER: Wrapper = Wrapper::Collector;
    const NAME: &'static str = "collector";
    const COMPLETION: &'static str = "__nemo_relay_collector_complete";
    const ERROR_PREFIX: &'static str = "JS collector callback failed";

    fn args(self, env: &Env, complete: JsArg, direct: JsArg) -> napi::Result<JsArgs3> {
        Ok(FnArgs::from((js_value(env, self.chunk)?, complete, direct)))
    }
}

pub(crate) struct JsFinalizerCall;

impl StreamCall for JsFinalizerCall {
    type Args = JsArgs2;
    const WRAPPER: Wrapper = Wrapper::Finalizer;
    const NAME: &'static str = "finalizer";
    const COMPLETION: &'static str = "__nemo_relay_finalizer_complete";
    const ERROR_PREFIX: &'static str = "nemo_relay: JS finalizer callback failed";

    fn args(self, _env: &Env, complete: JsArg, direct: JsArg) -> napi::Result<JsArgs2> {
        Ok(FnArgs::from((complete, direct)))
    }
}

pub(crate) struct JsStreamCallback<T: StreamCall> {
    tsfn: HandledTsfn<T, T::Args, JsArg>,
    teardown: Arc<AtomicBool>,
    function: PersistentJsFunction<T::Args, JsArg>,
    complete: Arc<PersistentJsFunction<JsArg, ()>>,
    pending: AsyncReply,
}

fn completion_function(
    env: &Env,
    name: &str,
    pending: &AsyncReply,
    error_prefix: &'static str,
) -> napi::Result<Arc<PersistentJsFunction<JsArg, ()>>> {
    let pending = pending.clone();
    let callback = closure_function(env, name, move |ctx| {
        pending.deliver(unwrap_envelope(ctx.get::<Unknown>(0)?, error_prefix));
        Ok(())
    })?;
    Ok(Arc::new(PersistentJsFunction::new(env, &callback)?))
}

impl<T: StreamCall> JsStreamCallback<T> {
    pub(crate) fn new(env: &Env, func: &Function<JsArg, JsArg>) -> napi::Result<Self> {
        let wrapper: Function<T::Args, JsArg> = callback_factory::wrap(env, T::WRAPPER, func)?;
        let pending = AsyncReply::empty();
        let complete = completion_function(env, T::COMPLETION, &pending, T::ERROR_PREFIX)?;
        let queued_complete = Arc::clone(&complete);
        let queued_pending = pending.clone();
        let teardown = teardown_flag(env)?;
        let dispatch_teardown = Arc::clone(&teardown);
        let tsfn =
            callback_factory::handled_tsfn(env, &wrapper, move |ctx: ThreadsafeCallContext<T>| {
                if dispatch_teardown.load(Ordering::Acquire) {
                    queued_pending.deliver(Err(FlowError::Internal(format!(
                        "JS {} callback skipped: {ENV_SHUTTING_DOWN}",
                        T::NAME
                    ))));
                    return Err(napi::Error::from_reason(ENV_SHUTTING_DOWN));
                }
                queued_complete
                    .borrow()
                    .and_then(|(_, complete)| {
                        let complete = detach(&ctx.env, &complete);
                        ctx.value
                            .args(&ctx.env, complete, js_value(&ctx.env, false)?)
                    })
                    .inspect_err(|error| {
                        queued_pending.deliver(Err(FlowError::Internal(format!(
                            "{}: {error}",
                            T::ERROR_PREFIX
                        ))));
                    })
            })?;
        Ok(Self {
            tsfn,
            teardown,
            function: PersistentJsFunction::new(env, &wrapper)?,
            complete,
            pending,
        })
    }

    fn queue(&self, call: T) -> Result<()> {
        if self.teardown.load(Ordering::Acquire) {
            return Err(FlowError::Internal(format!(
                "JS {} callback skipped: {ENV_SHUTTING_DOWN}",
                T::NAME
            )));
        }
        match self
            .tsfn
            .call(Ok(call), ThreadsafeFunctionCallMode::NonBlocking)
        {
            napi::Status::Ok => Ok(()),
            napi::Status::Closing => Err(FlowError::Internal(format!(
                "JS {} callback skipped: {ENV_SHUTTING_DOWN}",
                T::NAME
            ))),
            status => {
                let message = format!(
                    "nemo_relay: failed to queue JS {} callback: {status:?}",
                    T::NAME
                );
                record_callback_error(message.clone());
                Err(FlowError::Internal(message))
            }
        }
    }

    pub(crate) fn abort_tsfn(&self) -> napi::Result<()> {
        abort_tsfn(&self.tsfn)
    }

    fn calls_directly(&self) -> bool {
        DirectStreamCallbacks::active() && self.function.on_creating_thread()
    }

    fn call_directly(&self, call: T) -> napi::Result<()> {
        let (env, function) = self.function.borrow()?;
        let (_, complete) = self.complete.borrow()?;
        let args = call.args(env, detach(env, &complete), js_value(env, true)?)?;
        let envelope = function.call(args)?;
        // SAFETY: `envelope` was returned by the call above, in `env`.
        let envelope = unsafe { envelope.into_unknown(env) };
        if envelope.get_type()? != ValueType::Undefined {
            self.pending
                .deliver(unwrap_envelope(envelope, T::ERROR_PREFIX));
        }
        Ok(())
    }

    fn invoke(&self, call: T, sender: tokio::sync::oneshot::Sender<Result<Json>>) -> Result<()> {
        self.pending.arm(sender);
        if self.calls_directly() {
            return self.call_directly(call).map_err(|error| {
                let message = format!("nemo_relay: JS {} callback failed: {error}", T::NAME);
                record_callback_error(message.clone());
                FlowError::Internal(message)
            });
        }
        self.queue(call)
    }
}

pub(crate) fn wrap_js_collector_fn(callback: JsStreamCallback<JsCollectorCall>) -> LlmCollector {
    LlmCollector::Async(Box::new(move |chunk: Json| {
        let (sender, reply) = tokio::sync::oneshot::channel();
        let queued = callback.invoke(JsCollectorCall { chunk }, sender);
        Box::pin(async move {
            queued?;
            await_callback_result(reply, JsCollectorCall::ERROR_PREFIX)
                .await
                .map(|_| ())
        })
    }))
}

pub(crate) type FinalizerError = Arc<Mutex<Option<Error>>>;

pub(crate) fn wrap_js_finalizer_fn(
    callback: JsStreamCallback<JsFinalizerCall>,
    error: FinalizerError,
) -> LlmFinalizer {
    LlmFinalizer::Async(Box::new(move || {
        let (sender, reply) = tokio::sync::oneshot::channel();
        let queued = callback.invoke(JsFinalizerCall, sender);
        Box::pin(async move {
            let result = match queued {
                Ok(()) => await_callback_result(reply, JsFinalizerCall::ERROR_PREFIX)
                    .await
                    .inspect_err(|failure| record_callback_error(failure.to_string())),
                Err(failure) => Err(failure),
            };
            result.inspect_err(|failure| *lock(&error) = Some(Error::from(failure.clone())))
        })
    }))
}

struct JsSubscriberCallbackCall {
    event: Json,
    callback_id: u64,
}

pub(crate) fn wrap_js_event_subscriber(
    env: &Env,
    name: String,
    callback: &Function<JsArg, JsArg>,
) -> napi::Result<EventSubscriberFn> {
    let complete = create_js_subscriber_completion_callback(env, name.clone())?;
    let callback: Function<JsArgs2, ()> =
        callback_factory::wrap_subscriber_callback(env, callback, &complete)?;
    let queue_error_name = name;
    let func = Arc::new(callback_factory::handled_tsfn(
        env,
        &callback,
        |ctx: ThreadsafeCallContext<JsSubscriberCallbackCall>| {
            let JsSubscriberCallbackCall { event, callback_id } = ctx.value;
            let args = js_value(&ctx.env, event).and_then(|event| {
                Ok(FnArgs::from((
                    event,
                    js_value(&ctx.env, callback_id as f64)?,
                )))
            });
            args.inspect_err(|_| complete_js_subscriber_callback(callback_id))
        },
    )?);
    Ok(Arc::new(move |event: &Event| {
        let event_json = match event.try_to_json_value() {
            Ok(event) => event,
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
            report_queue_status(
                status,
                &format!("JS event subscriber '{queue_error_name}' callback"),
            );
            complete_js_subscriber_callback(callback_id);
        }
    }))
}

fn create_js_subscriber_completion_callback<'env>(
    env: &'env Env,
    callback_name: String,
) -> napi::Result<Function<'env, JsArg, ()>> {
    closure_function(
        env,
        "__nemo_relay_complete_subscriber_callback",
        move |ctx| {
            let callback_id = ctx.get::<f64>(0)? as u64;
            if ctx.length() >= 2 {
                record_callback_error(match ctx.get::<String>(1) {
                    Ok(message) => format!(
                        "nemo_relay: JS event subscriber '{callback_name}' failed: {message}"
                    ),
                    Err(error) => format!(
                        "nemo_relay: failed to read JS event subscriber '{callback_name}' failure: {error}"
                    ),
                });
            }
            complete_js_subscriber_callback(callback_id);
            Ok(())
        },
    )
}

#[derive(Serialize)]
struct JsCodecEncodeArgument<'a> {
    annotated: &'a AnnotatedLlmRequest,
    original: &'a LlmRequest,
}

pub(crate) struct SyncJsCallback<C: JsCall> {
    direct: PersistentJsFunction<C::Args, JsArg>,
    queued: ReplyTsfn<C, std::sync::mpsc::SyncSender<Result<Json>>>,
    error_prefix: &'static str,
    timeout: Option<Duration>,
}

impl<C: JsCall> SyncJsCallback<C> {
    pub(crate) fn new(
        env: &Env,
        wrapper: Function<C::Args, JsArg>,
        error_prefix: &'static str,
        timeout: Option<Duration>,
    ) -> napi::Result<Self> {
        Ok(Self {
            direct: PersistentJsFunction::new(env, &wrapper)?,
            queued: reply_tsfn(env, &wrapper, error_prefix)?,
            error_prefix,
            timeout,
        })
    }

    pub(crate) fn call(&self, call: C) -> Result<Json> {
        if self.direct.on_creating_thread() {
            let (env, envelope) = self
                .direct
                .borrow()
                .and_then(|(env, function)| Ok((env, function.call(call.args(env)?)?)))
                .map_err(|error| FlowError::Internal(format!("{}: {error}", self.error_prefix)))?;
            // SAFETY: `envelope` was returned by the call above, in `env`.
            return unwrap_envelope(unsafe { envelope.into_unknown(env) }, self.error_prefix);
        }
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        queue_reply(&self.queued, call, tx, self.error_prefix)?;
        recv_callback_result(&rx, self.timeout, self.error_prefix)
    }
}

pub(crate) fn js_codec_callback(
    env: &Env,
    func: &Function<JsArg, JsArg>,
    error_prefix: &'static str,
) -> napi::Result<SyncJsCallback<Json>> {
    SyncJsCallback::new(
        env,
        callback_factory::wrap(env, Wrapper::Execution, func)?,
        error_prefix,
        None,
    )
}

fn codec_call<A: Serialize, T: DeserializeOwned>(
    callback: &SyncJsCallback<Json>,
    argument: &A,
    expected: &str,
) -> Result<T> {
    serde_json::to_value(argument)
        .map_err(|error| {
            FlowError::Internal(format!(
                "{}: could not serialize the argument: {error}",
                callback.error_prefix
            ))
        })
        .and_then(|argument| callback.call(argument))
        .and_then(|value| {
            serde_json::from_value(value).map_err(|error| {
                FlowError::Internal(format!(
                    "{}: expected {expected}: {error}",
                    callback.error_prefix
                ))
            })
        })
        .inspect_err(|error| record_callback_error(error.to_string()))
}

pub(crate) struct NapiCodec {
    pub(crate) decode: SyncJsCallback<Json>,
    pub(crate) encode: SyncJsCallback<Json>,
}

impl LlmCodec for NapiCodec {
    fn decode(&self, request: &LlmRequest) -> Result<AnnotatedLlmRequest> {
        codec_call(&self.decode, request, "AnnotatedLlmRequest")
    }

    fn encode(&self, annotated: &AnnotatedLlmRequest, original: &LlmRequest) -> Result<LlmRequest> {
        codec_call(
            &self.encode,
            &JsCodecEncodeArgument {
                annotated,
                original,
            },
            "LlmRequest",
        )
    }
}

pub(crate) struct NapiResponseCodec {
    pub(crate) decode_response: SyncJsCallback<Json>,
}

impl LlmResponseCodec for NapiResponseCodec {
    fn decode_response(&self, response: &Json) -> Result<AnnotatedLlmResponse> {
        codec_call(&self.decode_response, response, "AnnotatedLlmResponse")
    }
}

pub(crate) fn wrap_js_tool_exec_intercept_fn(
    env: &Env,
    func: &Function<JsArg, JsArg>,
) -> napi::Result<ToolExecutionFn> {
    let func = Arc::new(PromiseAwareFn::new(env, func)?);
    Ok(Arc::new(
        move |context: ToolExecutionContext, next: ToolExecutionNextFn| {
            let func = func.clone();
            let js_context = serde_json::json!({
                "toolName": context.tool_name(),
                "args": context.args(),
                "toolCallId": context.tool_call_id(),
            });
            Box::pin(async move {
                let next: JsonNextFn = Arc::new(move |next_args| {
                    let next = next.clone();
                    Box::pin(async move {
                        serde_json::to_value(next(next_args).await?)
                            .map_err(|error| FlowError::Internal(error.to_string()))
                    })
                });
                let result = func
                    .call_with_next(Invocation::single(js_context), next)
                    .await?;
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
            })
        },
    ))
}

fn llm_next<R: Send + 'static>(
    next: Arc<dyn Fn(LlmRequest) -> BoxFuture<Result<R>> + Send + Sync>,
) -> Arc<dyn Fn(Json) -> BoxFuture<Result<R>> + Send + Sync> {
    Arc::new(move |request| {
        let next = next.clone();
        Box::pin(async move {
            let request: LlmRequest = serde_json::from_value(request).map_err(|e| {
                FlowError::Internal(format!("invalid LlmRequest from JS next: {e}"))
            })?;
            next(request).await
        })
    })
}

pub(crate) fn wrap_js_llm_exec_intercept_fn(
    env: &Env,
    func: &Function<JsArg, JsArg>,
) -> napi::Result<LlmExecutionFn> {
    let func = Arc::new(PromiseAwareFn::new(env, func)?);
    Ok(Arc::new(
        move |_name: &str,
              request: LlmRequest,
              context: LlmExecutionContext,
              next: LlmExecutionNextFn| {
            let func = func.clone();
            let invocation = js_llm_execution_invocation(request, context);
            Box::pin(async move { func.call_with_next(invocation, llm_next(next)).await })
        },
    ))
}

pub(crate) fn wrap_js_llm_stream_fn(
    env: &Env,
    func: &Function<JsArg, JsArg>,
    cancellation: StreamCancellation,
) -> napi::Result<LlmStreamExecutionNextFn> {
    let func = Arc::new(PromiseAwareFn::new(env, func)?);
    Ok(Arc::new(move |request: LlmRequest| {
        let func = func.clone();
        let cancellation = cancellation.clone();
        Box::pin(async move {
            let request = serde_json::to_value(&request).map_err(|error| {
                FlowError::Internal(format!("failed to serialize LlmRequest: {error}"))
            })?;
            func.call_stream(Invocation::single(request), &cancellation)
                .await
        })
    }))
}

pub(crate) fn wrap_js_llm_stream_exec_intercept_fn(
    env: &Env,
    func: &Function<JsArg, JsArg>,
) -> napi::Result<LlmStreamExecutionFn> {
    let func = Arc::new(PromiseAwareFn::new(env, func)?);
    Ok(Arc::new(
        move |_name: &str,
              request: LlmRequest,
              context: LlmExecutionContext,
              next: LlmStreamExecutionNextFn| {
            let func = func.clone();
            let invocation = js_llm_execution_invocation(request, context);
            Box::pin(async move { func.call_stream_with_next(invocation, llm_next(next)).await })
        },
    ))
}

#[cfg(test)]
#[path = "../tests/rust/callable_tests.rs"]
mod tests;
